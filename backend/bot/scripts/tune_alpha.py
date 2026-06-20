#!/usr/bin/env python3
"""Tune + validate the EB shrinkage alpha (Phase 6 / E1) -- honest version after audit.

Two held-out evaluations, both with HAND-boundary temporal splits and PER-PLAYER-EQUAL
weighting (so the 45%-of-data whale can't set the dial), using the SHARED estimator
(build_opponent_model.estimate) so the tuned alpha is exactly the served one:

  (1) KNOWN-player path -- per player, fit on the early 70% of their HANDS, test on the
      late 30%. Answers "how bold should we be with a player's OWN history?"
  (2) COLD-START path (leave-one-player-out) -- build the population prior from the OTHER
      N-1 players and predict a held-out player with ZERO personal history. Answers "how
      well does the modal-human prior beat GTO against a STRANGER?" -- the deployment
      reality (most opponents are first-session strangers), which the old tuner never
      measured.

alpha=inf (1e9) is the blueprint-only baseline; the gain vs it is the signal-existence
result (a PREDICTION gain, NOT an EV estimate -- live A/B remains the sole EV oracle).

Run from backend/bot:
    python scripts/tune_alpha.py --db analysis/blueprints/snapshots/snap_52500000.db
"""
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))               # build_opponent_model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src

import build_opponent_model as bom                                  # noqa: E402
from src.storage.blueprint_db import BlueprintDB                    # noqa: E402

ALPHAS = [2, 5, 10, 20, 40, 80, 160, 1e9]        # 1e9 ~ blueprint-only baseline


def _counts(decs):
    """[(key,char)] -> {key: {char: count}}."""
    c = defaultdict(Counter)
    for key, ch in decs:
        c[key][ch] += 1
    return {k: dict(v) for k, v in c.items()}


def _capped_pop(player_counts, exclude, cap):
    """Population counts (capped per player, EXCLUDING `exclude`) -> {key: {char: mass}}."""
    pop = defaultdict(Counter)
    for pid, counts in player_counts.items():
        if pid == exclude:
            continue
        for key, ctr in counts.items():
            tot = sum(ctr.values())
            if tot == 0:
                continue
            w = min(tot, cap)
            for ch, n in ctr.items():
                pop[key][ch] += (n / tot) * w
    return {k: dict(v) for k, v in pop.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', default='analysis/opponent_models/decisions.jsonl')
    ap.add_argument('--db', default='analysis/blueprints/snapshots/snap_52500000.db')
    ap.add_argument('--train-frac', type=float, default=0.7)
    ap.add_argument('--min-test', type=int, default=40, help='min test decisions to include a player')
    ap.add_argument('--pop-cap', type=float, default=15.0)
    args = ap.parse_args()

    by_player = defaultdict(list)                # pid -> [(hand_id, key, char), ...] in order
    with open(args.records, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            by_player[r['p']].append((r['h'], r['k'], r['a']))
    player_counts = {pid: _counts([(k, a) for _, k, a in decs]) for pid, decs in by_player.items()}

    bp = BlueprintDB(args.db, read_only=True)
    bp_cache = {}

    def prior(key):
        if key not in bp_cache:
            bp_cache[key] = bom._bp_chardist(bp, key)
        return bp_cache[key]

    def mean_ll(level_counts, decs, alpha):
        if not decs:
            return None
        s = 0.0
        for key, ch in decs:
            p = bom.estimate(level_counts, key, prior(key), alpha).get(ch, 1e-9)
            s += math.log(max(p, 1e-9))
        return s / len(decs)

    # (1) KNOWN-player path: hand-boundary split, per-player mean test LL.
    known = {}                                   # pid -> {alpha: mean test LL}
    for pid, decs in by_player.items():
        hids = []
        for h, _, _ in decs:
            if not hids or hids[-1] != h:
                hids.append(h)
        split = int(args.train_frac * len(hids))
        train_hids = set(hids[:split])
        train = [(k, a) for h, k, a in decs if h in train_hids]
        test = [(k, a) for h, k, a in decs if h not in train_hids]
        if len(test) < args.min_test:
            continue
        tl = bom._player_levels(_counts(train))
        known[pid] = {a: mean_ll(tl, test, a) for a in ALPHAS}

    # (2) COLD-START LOPO: population-from-others predicts the held-out player (all their decisions).
    cold = {}                                    # pid -> {alpha: mean LL}
    for pid, decs in by_player.items():
        alldec = [(k, a) for _, k, a in decs]
        if len(alldec) < args.min_test:
            continue
        pl = bom._player_levels(_capped_pop(player_counts, pid, args.pop_cap))
        cold[pid] = {a: mean_ll(pl, alldec, a) for a in ALPHAS}
    bp.close()

    def equal_curve(table):
        return {a: sum(d[a] for d in table.values()) / len(table) for a in ALPHAS}

    def report(title, table, handles):
        eq = equal_curve(table)
        base = eq[1e9]
        bestA = max((a for a in ALPHAS if a < 1e9), key=lambda a: eq[a])
        print(f"\n  {title}  ({len(table)} players, per-player-EQUAL weighted)")
        print(f"    {'alpha':>7}{'LL/dec':>9}{'vs GTO':>9}")
        for a in ALPHAS:
            tag = '  <- GTO baseline' if a >= 1e9 else ('  <- best' if a == bestA else '')
            print(f"    {('inf' if a >= 1e9 else f'{a:g}'):>7}{eq[a]:>9.3f}{eq[a] - base:>+9.3f}{tag}")
        spread = sorted(max((a for a in ALPHAS if a < 1e9), key=lambda a: d[a]) for d in table.values())
        print(f"    -> best alpha = {bestA:g}; +{eq[bestA] - base:.3f} nats/dec vs GTO; "
              f"per-player optima span {spread[0]:g}-{spread[-1]:g}")
        return bestA, eq[bestA] - base

    handles = {pid: pid[:6] for pid in by_player}
    ka, kg = report("KNOWN-player (own history, hand-split hold-out)", known, handles)
    ca, cg = report("COLD-START (population prior vs an UNSEEN player, LOPO)", cold, handles)

    print(f"\n  RECOMMENDATION: serve alpha = {ka:g} (known-player optimum; governs the per-player "
          f"tilt once a player has data).")
    print(f"  Cold-start optimum is alpha = {ca:g} (modal-human prior beats GTO by +{cg:.3f} "
          f"nats/dec against a STRANGER -- smaller than the whale-weighted number, the honest "
          f"launch expectation). Prediction gain only; EV proof = live A/B.")


if __name__ == '__main__':
    main()
