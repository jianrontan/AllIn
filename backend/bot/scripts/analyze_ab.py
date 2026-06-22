#!/usr/bin/env python3
"""Live exploitation A/B analysis: split recaps by their recorded `exploitArm` ('on' = exploit,
'off' = pure-blueprint control) and compare the BOT's win-rate per arm.

The arm is assigned per session (`ALLIN_AB_ARM=random` for a prod 50/50 split by player; `on`/`off`
forces it for dev to play each side) and stamped on every recap by `recap_from_session`. This is the
LIVE EV ORACLE the offline scoreboard (`measure_exploit.py`) cannot be -- but raw bb/100 only resolves
a small (sub-BB/hand) delta with MANY hands across MANY players; a single dev player is far too noisy.

Pass --db to ALSO print the AIVAT-reduced estimate (c1 preflop-equity + c3 all-in-runout control
variates -- unbiased, far lower variance, so the exploit-vs-control delta resolves with FAR fewer
hands). Needs recaps carrying the `invested`/`allinStreet` fields (recorded from the AIVAT-wiring
build onward); older recaps fall back to c1-only (still unbiased, less reduction). c2 (river-runout)
is skipped here -- it needs a live RangeTracker snapshot in the recap (future work).

Reads an export JSONL (re-run `scripts/export_hands.py` against the live store first):
  python scripts/analyze_ab.py --in analysis/opponent_models/hands_export.jsonl \
                               --db analysis/blueprints/snapshots/snap_52500000.db
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_BB = 2.0   # SB=1/BB=2; humanDelta is in chips


def _aivat_rec(r):
    """Reconstruct an aivat.AIVATEstimator record from a recap, with the BOT as player A."""
    from src.game.cards import to_engine
    bot_seat = 1 - r.get('humanSeat')
    return {
        'seat_of_A': bot_seat,
        'hand_a': [to_engine(c) for c in r['botHole']],
        'hand_b': [to_engine(c) for c in r['humanHole']],
        'board': [to_engine(c) for c in (r.get('community') or [])],
        'result': -float(r['result']['humanDelta']),     # BOT chip delta = -human
        'invested': r.get('invested') or [0.0, 0.0],
        'allin_street': r.get('allinStreet'),
        'folded': r.get('folder'),
        'events': [],                                     # c2 skipped (no tracker snapshot in the recap)
    }


def _print_delta(stats, label):
    if 'on' in stats and 'off' in stats:
        (m1, s1, _), (m0, s0, _) = stats['on'], stats['off']
        diff = m1 - m0
        sed = math.sqrt(s1 * s1 + s0 * s0)
        t = diff / sed if sed > 0 else 0.0
        print(f"\n  {label} EXPLOIT - CONTROL: {diff * 100:+.1f} bb/100  +/- {sed * 100 * 1.96:.1f} "
              f"(95%)  |t|={abs(t):.2f}   (|t|>=2 ~ significant)")
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default='analysis/opponent_models/hands_export.jsonl')
    ap.add_argument('--db', default=None,
                    help="blueprint DB -> ALSO print the AIVAT-reduced estimate (c1+c3 variance "
                         "reduction). Needs recaps with the invested/allinStreet fields.")
    args = ap.parse_args()

    arms = defaultdict(list)        # arm -> list of BOT bb/hand (raw)
    recaps = defaultdict(list)      # arm -> full recaps (for AIVAT)
    players = defaultdict(set)
    n_total = n_tagged = 0
    with open(args.inp, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            n_total += 1
            arm = r.get('exploitArm')
            if arm not in ('on', 'off'):
                continue            # untagged (pre-A/B) hands
            hd = (r.get('result') or {}).get('humanDelta')
            if hd is None:
                continue
            n_tagged += 1
            arms[arm].append(-float(hd) / _BB)          # BOT bb/hand = -human
            recaps[arm].append(r)
            players[arm].add(r.get('playerId'))

    print(f"{n_total} recaps, {n_tagged} tagged with an A/B arm\n")
    stats = {}
    for arm in ('on', 'off'):
        xs = arms.get(arm, [])
        n = len(xs)
        if n == 0:
            print(f"  arm {arm:3}: 0 hands")
            continue
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / n if n > 1 else 0.0
        se = math.sqrt(var / n) if n > 1 else 0.0
        stats[arm] = (mean, se, n)
        print(f"  arm {arm:3}: {n:6} hands / {len(players[arm])} players | "
              f"bot {mean * 100:+.1f} bb/100  +/- {se * 100 * 1.96:.1f} (95%)")

    if not _print_delta(stats, 'RAW'):
        print("\n  Need BOTH arms with data. Serve some hands on each (ALLIN_AB_ARM=random in prod, or"
              " alternate on/off in dev), re-export, and re-run.")

    # ---- AIVAT-reduced (optional; needs the blueprint for the equity baselines) ----
    if args.db:
        from src.storage.blueprint_db import BlueprintDB
        from src.evaluation.aivat import AIVATEstimator
        print("\n  AIVAT-reduced (c1 preflop-equity + c3 all-in-runout; unbiased, lower variance):")
        est = AIVATEstimator(BlueprintDB(args.db, read_only=True), seed=0)
        astats = {}
        for arm in ('on', 'off'):
            recs = []
            for r in recaps.get(arm, []):
                try:
                    recs.append(_aivat_rec(r))
                except Exception:
                    pass                                 # skip a malformed recap, don't abort the arm
            if not recs:
                print(f"  arm {arm:3}: 0 usable hands")
                continue
            out = est.estimate(recs)
            m = out['aivat_mbb'] / 1000.0                # bb/hand (mbb/hand -> bb/hand)
            se = out['aivat_stderr_mbb'] / 1000.0
            astats[arm] = (m, se, out['num_hands'])
            print(f"  arm {arm:3}: {out['num_hands']:6} hands | bot {m * 100:+.1f} bb/100  "
                  f"+/- {se * 100 * 1.96:.1f} (95%)   [raw {out['raw_mbb'] / 10:+.1f}; "
                  f"variance -{out['var_reduction'] * 100:.0f}%]")
        _print_delta(astats, 'AIVAT')


if __name__ == '__main__':
    main()
