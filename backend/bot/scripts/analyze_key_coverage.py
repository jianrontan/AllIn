#!/usr/bin/env python3
"""Decision support for opponent-model key granularity (Phase 6 / E1).

Loads the per-player count tables fit by fit_opponent_models.py and, WITHOUT re-replaying,
re-aggregates them under several key-COLLAPSE schemes -- all of which KEEP the hand-strength
bucket (humans play heavily on strength) but collapse the betting PATTERN (the real sparsity
driver). Prints coverage (distinct keys, keys with >= N visits) per scheme so we can pick the
finest granularity the data actually supports. Read-only, no abstraction tables.

Run from backend/bot:  python scripts/analyze_key_coverage.py
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.abstractions.card_abstractions import FINE_TO_COARSE      # noqa: E402  (no tables loaded)

_POST_STREETS = ('flop', 'turn', 'river')


def _parse(key):
    """Split a blueprint key into components. Returns (kind, dict)."""
    p = key.split('_')
    if any(s in _POST_STREETS for s in p):                 # postflop: pf_C_S_pos_street_pattern
        return 'post', {'coarse': f"{p[0]}_{p[1]}", 'strength': p[2], 'pos': p[3],
                        'street': p[4], 'pat': p[5] if len(p) > 5 else ''}
    return 'pre', {'fine': f"{p[0]}_{p[1]}", 'pos': p[2],   # preflop: pf_F_pos_pattern
                   'pat': p[3] if len(p) > 3 else ''}


def _coarse_of_fine(fine):
    try:
        return f"pf_{FINE_TO_COARSE[int(fine.split('_')[1])]}"
    except Exception:
        return fine


def _facing(pat):
    """Collapse a street pattern to the single action the player FACES (its last char),
    or 'first' if first-to-act. Keeps the bet-size faced, drops the full sequence."""
    return pat[-1] if pat else 'first'


def collapse(key, scheme):
    """Re-key under a scheme. All schemes keep the strength/hand bucket."""
    kind, d = _parse(key)
    if kind == 'pre':
        s = _coarse_of_fine(d['fine'])                     # preflop 'strength' = hand; fine->coarse
        if scheme == 'full':
            return f"PRE|{d['fine']}|{d['pos']}|{d['pat']}"
        if scheme == 'no-pattern':
            return f"PRE|{s}|{d['pos']}"
        if scheme == 'facing':
            return f"PRE|{s}|{d['pos']}|{_facing(d['pat'])}"
        if scheme == 'no-coarse':                          # preflop has no coarse to drop -> keep fine
            return f"PRE|{d['fine']}|{d['pos']}|{d['pat']}"
        if scheme == 'strength+facing':
            return f"PRE|{s}|{d['pos']}|{_facing(d['pat'])}"
        if scheme == 'strength-only':
            return f"PRE|{s}|{d['pos']}"
    else:
        if scheme == 'full':
            return f"P|{d['coarse']}|{d['strength']}|{d['pos']}|{d['street']}|{d['pat']}"
        if scheme == 'no-pattern':
            return f"P|{d['coarse']}|{d['strength']}|{d['pos']}|{d['street']}"
        if scheme == 'facing':
            return f"P|{d['coarse']}|{d['strength']}|{d['pos']}|{d['street']}|{_facing(d['pat'])}"
        if scheme == 'no-coarse':
            return f"P|{d['strength']}|{d['pos']}|{d['street']}|{d['pat']}"
        if scheme == 'strength+facing':
            return f"P|{d['strength']}|{d['pos']}|{d['street']}|{_facing(d['pat'])}"
        if scheme == 'strength-only':
            return f"P|{d['strength']}|{d['pos']}|{d['street']}"
    return key


def _cov(counts, thresh):
    """(distinct, >=thresh, >=2*thresh) over a {ckey: total} dict."""
    return (len(counts),
            sum(1 for v in counts.values() if v >= thresh),
            sum(1 for v in counts.values() if v >= 2 * thresh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='analysis/opponent_models')
    ap.add_argument('--thresh', type=int, default=5)
    args = ap.parse_args()

    files = [f for f in glob.glob(os.path.join(args.dir, 'model_*.json'))
             if not f.endswith('model_population.json')]
    players = []
    for f in files:
        with open(f, encoding='utf-8') as fh:
            m = json.load(fh)
        # total visits per ORIGINAL key
        per_key = {k: sum(c.values()) for k, c in m['counts'].items()}
        players.append((m['handle'], m['hands'], per_key))
    players.sort(key=lambda x: x[1], reverse=True)

    schemes = ['full', 'no-pattern', 'facing', 'no-coarse', 'strength+facing', 'strength-only']
    show = players[:4]                                     # the 4 highest-volume players

    for handle, hands, per_key in show:
        print(f"\n  {handle} ({hands} hands):")
        print(f"    {'scheme':<14}{'keys':>7}{'>=' + str(args.thresh):>7}"
              f"{'>=' + str(2 * args.thresh):>7}")
        for sch in schemes:
            agg = Counter()
            for k, n in per_key.items():
                agg[collapse(k, sch)] += n
            d, r, rr = _cov(agg, args.thresh)
            print(f"    {sch:<14}{d:>7}{r:>7}{rr:>7}")

    # Pooled (all players, raw) -- a proxy for the population model's coverage.
    print(f"\n  POOLED (all {len(players)} players):")
    print(f"    {'scheme':<14}{'keys':>7}{'>=' + str(args.thresh):>7}{'>=' + str(2 * args.thresh):>7}")
    for sch in schemes:
        agg = Counter()
        for _, _, per_key in players:
            for k, n in per_key.items():
                agg[collapse(k, sch)] += n
        d, r, rr = _cov(agg, args.thresh)
        print(f"    {sch:<14}{d:>7}{r:>7}{rr:>7}")


if __name__ == '__main__':
    main()
