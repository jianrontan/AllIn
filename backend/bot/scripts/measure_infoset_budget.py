# backend/bot/scripts/measure_infoset_budget.py
"""
MEASUREMENT B -- info-set budget per candidate abstraction config.

THE QUESTION. Each candidate config in the redesign grid (docs/ROADMAP.md) changes
the postflop info-set count differently. Does the "full" SPR-bucket config fit a
trainable budget, or does it blow the tree up? B prices each config so we don't
commit to one whose info-set count needs weeks of training to cover.

WHY EMPIRICAL, NOT THEORETICAL. A theoretical enumeration (coarse10 x strength x
pos2 x street x pattern) massively over-counts: most (bucket, pattern) combos are
never reachable (you can't be in strength-bucket 19 with a check-check-bet pattern
that the abstraction never produces, etc.). The honest budget is the ACTUAL reachable
key count in a trained blueprint, projected forward by each config's structural
multiplier. So B reads the served blueprint's real per-street key counts and applies
each config's multiplier.

CONFIG MULTIPLIERS (relative to the current key structure):
  #0 control (current menu + all-in node)        : baseline (= served DB)
  #1 menu-cap, drop all-in node                  : ~baseline (REMOVES an action ->
       same key structure, slightly FEWER distinct betting patterns; not more keys)
  #2 SPR-aware sizing, no buckets                 : ~baseline (same # of sizes, just
       different chip amounts -> identical key structure)
  #3 K=2 SPR buckets on flop+turn (current menu)  : flop+turn slice x2, river x1
  #4 K=3 SPR buckets on flop+turn (menu-cap)      : flop+turn slice x3, river x1

Run from backend/bot/:
    python scripts/measure_infoset_budget.py
    python scripts/measure_infoset_budget.py --db analysis/blueprints/blueprint_par_20260529_233056.db
"""
import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import resolve_blueprint_path


def _counts(db_path):
    """Reachable key counts in the blueprint: preflop, and postflop by street."""
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = c.execute("SELECT key FROM info_sets").fetchall()
    finally:
        c.close()
    pre = 0
    street = {'flop': 0, 'turn': 0, 'river': 0}
    for (k,) in rows:
        parts = k.split('_')
        # preflop: pf_<n>_<pos>_<pat> = 4 tokens; postflop: pf_<c>_<s>_<pos>_<street>_<pat> = 6
        if len(parts) == 4:
            pre += 1
        else:
            for s in street:
                if s in parts:
                    street[s] += 1
                    break
    return pre, street


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=None)
    args = ap.parse_args()
    db_path = args.db or str(resolve_blueprint_path())

    pre, street = _counts(db_path)
    flop, turn, river = street['flop'], street['turn'], street['river']
    postflop = flop + turn + river
    base_total = pre + postflop

    print(f"Measurement B -- info-set budget per candidate config")
    print(f"  blueprint (reachable-key baseline): {db_path}")
    print(f"  preflop {pre:,} | flop {flop:,} | turn {turn:,} | river {river:,} "
          f"| postflop {postflop:,} | TOTAL {base_total:,}\n")

    # Each config: (label, flop_mult, turn_mult, river_mult, note). Preflop is never
    # SPR-bucketed (SPR is implicit in the preflop betting pattern), so pre is constant.
    configs = [
        ("#0 control (current menu + all-in node)", 1, 1, 1,
         "= served DB baseline"),
        ("#1 menu-cap, drop all-in node",           1, 1, 1,
         "removes an action -> same key STRUCTURE, ~baseline (slightly fewer patterns)"),
        ("#2 SPR-aware sizing, no buckets",          1, 1, 1,
         "same # sizes, different chips -> identical key structure"),
        ("#3 K=2 SPR buckets flop+turn (cur. menu)", 2, 2, 1,
         "flop+turn x2, river unchanged"),
        ("#4 K=3 SPR buckets flop+turn (menu-cap)",  3, 3, 1,
         "flop+turn x3, river unchanged"),
    ]

    print(f"  {'config':<44}{'postflop':>11}{'total':>11}{'x base':>8}")
    for label, fm, tm, rm, note in configs:
        post = flop * fm + turn * tm + river * rm
        total = pre + post
        print(f"  {label:<44}{post:>11,}{total:>11,}{total / base_total:>7.2f}x")
        print(f"      {note}")

    print("\nTrainability rule of thumb: MCCFR must VISIT each info-set many times to")
    print("converge it, so iters-to-equal-convergence scale ~linearly with the count.")
    print("The served 33.1M-iter run covered the ~baseline tree; a 2x/3x tree needs")
    print("roughly 2x/3x the iters (or proportionally more cloud/parallel time) for the")
    print("SAME per-key convergence. Read against Measurement A's verdict (SPR buckets")
    print("low-leverage -> the 2-3x cost of #3/#4 likely isn't justified vs #1/#2).")


if __name__ == '__main__':
    main()
