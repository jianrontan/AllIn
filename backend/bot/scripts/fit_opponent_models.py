#!/usr/bin/env python3
"""Fit per-player + population opponent-model COUNTS from exported recaps (Phase 6 / E0->E1).

Replays each recap and, at every HUMAN decision, reconstructs the *blueprint-compatible*
info-set key (via cfr.keys.make_info_set_key, the single source of truth) and records the
human's action char. The output is per-(player, key) action-char counts -- the Layer-2
sufficient statistic the tracker's opponent model is built from. The empirical-Bayes
shrinkage toward the blueprint prior (EXPLOITATION_PLAN.md §4.1) is applied LATER at
serve/model-build time, NOT here -- so these counts are the durable, continuously-growing
artifact and this fitter needs no blueprint.

Keys are reconstructed exactly as the trainer builds them:
  * position 'ip' (seat 0 = SB/button) / 'oop' (seat 1 = BB) -- fixed per hand by seat.
  * preflop key uses the FINE preflop bucket; postflop key uses the per-street postflop
    bucket (make_info_set_key collapses the fine bucket to the coarse class internally).
  * bet_pattern = the action chars on the CURRENT street so far (both players), reset each
    street -- the pattern leading INTO the decision (the human's own action is appended after).
  * postflop board for a street = community[:BOARD_COUNT[street]] (NOT the full revealed
    runout -- an all-in flop decision must bucket on 3 cards, not the 5 that got dealt out).

Run from backend/bot (set a small river-board cache to bound RAM -- the 125MB turn table loads):
  ALLIN_RIVER_CACHE_BOARDS=20000 python scripts/fit_opponent_models.py \
      --in analysis/opponent_models/hands_export.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.abstractions.card_abstractions import CardAbstraction       # noqa: E402
from src.cfr.keys import make_info_set_key, action_char, BOARD_COUNT  # noqa: E402
from src.game.cards import to_engine                                 # noqa: E402
from src.exploitation.replay import replay_hand, _street_idx         # noqa: E402  (shared impl)

_STREET_NAMES = ('preflop', 'flop', 'turn', 'river')


# replay_hand + _street_idx moved to src/exploitation/replay.py (shared with serve-time live
# last-N refits, HumanModel.chardist_from_recent) and imported above -- ONE implementation.


def _coverage(counts, thresh=5):
    """(distinct keys, keys with >= thresh visits, total decisions) for a count table."""
    keys = len(counts)
    rich = sum(1 for c in counts.values() if sum(c.values()) >= thresh)
    total = sum(sum(c.values()) for c in counts.values())
    return keys, rich, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default='analysis/opponent_models/hands_export.jsonl')
    ap.add_argument('--out-dir', default='analysis/opponent_models')
    ap.add_argument('--thresh', type=int, default=5,
                    help='visits-per-key threshold for the "rich keys" coverage column')
    ap.add_argument('--records-out', default='analysis/opponent_models/decisions.jsonl',
                    help='also dump per-decision (player, hand, key, char) records IN ORDER, '
                         'for the hand-boundary-split alpha tuner')
    ap.add_argument('--pop-cap', type=float, default=15.0,
                    help="cap each player's contribution to a population-prior key at this many "
                         "effective visits (defangs the whale AND down-weights 1-shot noise)")
    args = ap.parse_args()

    ca = CardAbstraction()
    per_player = defaultdict(lambda: defaultdict(Counter))   # pid -> key -> Counter(char)
    population = defaultdict(Counter)                        # key -> Counter(char)
    meta = {}                                               # pid -> {handle, hands, recorded, skipped}

    os.makedirs(os.path.dirname(args.records_out) or '.', exist_ok=True)
    rec_fh = open(args.records_out, 'w', encoding='utf-8')
    hidx = 0
    with open(args.inp, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get('playerId')
            if pid is None:
                continue
            hand_recs = []
            rec_n, skip_n = replay_hand(rec, ca, per_player[pid], hand_recs)
            for k, ch in hand_recs:                 # 'h' = a unique hand id so the tuner can
                rec_fh.write(json.dumps({'p': pid, 'h': hidx, 'k': k, 'a': ch},  # split on HAND
                                        separators=(',', ':')) + '\n')           # boundaries
            hidx += 1
            m = meta.setdefault(pid, {'handle': rec.get('_handle', 'anon'),
                                      'hands': 0, 'recorded': 0, 'skipped': 0})
            m['hands'] += 1
            m['recorded'] += rec_n
            m['skipped'] += skip_n
    rec_fh.close()

    # Population counts = per-player CAPPED-weighted (EXPLOITATION_PLAN §5.3). Each player
    # contributes its within-key distribution scaled to mass min(visits, pop_cap): this defangs
    # the whale (capped) AND down-weights a tiny-sample player's 1-shot decisions (which a pure
    # equal-mass-1 scheme would have given the same say as a well-estimated frequency).
    for pid, counts in per_player.items():
        for key, ctr in counts.items():
            tot = sum(ctr.values())
            if tot == 0:
                continue
            w = min(tot, args.pop_cap)                  # capped effective mass for this player
            for ch, n in ctr.items():
                population[key][ch] += (n / tot) * w

    os.makedirs(args.out_dir, exist_ok=True)
    # Per-player count tables (one file each, keyed by playerId).
    for pid, counts in per_player.items():
        out = {'playerId': pid, 'handle': meta[pid]['handle'], 'hands': meta[pid]['hands'],
               'counts': {k: dict(c) for k, c in counts.items()}}
        with open(os.path.join(args.out_dir, f'model_{pid}.json'), 'w', encoding='utf-8') as f:
            json.dump(out, f, separators=(',', ':'))
    with open(os.path.join(args.out_dir, 'model_population.json'), 'w', encoding='utf-8') as f:
        json.dump({'counts': {k: {c: round(v, 4) for c, v in d.items()}
                              for k, d in population.items()}}, f, separators=(',', ':'))

    # Coverage report -- the REAL answer to "is the data enough", at the key level.
    print(f"fit {len(per_player)} players from {args.inp}\n")
    hdr = f"  {'handle':<16}{'hands':>6}{'decisions':>10}{'keys':>7}{'keys>=' + str(args.thresh):>9}{'skip':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for pid in sorted(per_player, key=lambda p: meta[p]['hands'], reverse=True):
        k, rich, total = _coverage(per_player[pid], args.thresh)
        m = meta[pid]
        print(f"  {m['handle']:<16}{m['hands']:>6}{total:>10}{k:>7}{rich:>9}{m['skipped']:>6}")
    pk, prich, ptot = _coverage(population, args.thresh)
    print(f"\n  population (per-player capped @ {args.pop_cap:g}): {pk} distinct keys, {prich} "
          f"with >= {args.thresh} effective visits, {ptot:.0f} total mass")
    print(f"\n  wrote per-player model_<id>.json + model_population.json to {args.out_dir}")


if __name__ == '__main__':
    main()
