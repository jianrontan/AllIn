# backend/bot/scripts/validate_cfv_bucketing.py
"""
M0 steps 2-3: is a BUCKETED turn-leaf adequate, and does it need to be
reach-conditioned (the matrix) rather than a frozen per-bucket scalar?

Against the EXACT per-hand leaf (src/subgame/cfv.turn_leaf_value_exact, validated in
step 1), measure three relative errors over hero hands:

  (c) COARSENESS (no shift): bucketed leaf M.R1_mass vs exact-vs-R1.
      -> how much does collapsing hero hands to turn buckets (+ ignoring fine card
         removal) cost? Small => bucketing the hero is adequate.

  (b) RANGE SHIFT -- the agent-review P0. With the villain range shifted R1 -> R1':
      * FROZEN SCALAR: reuse the R1-calibrated bucketed leaf (M.R1_mass) to predict R1'.
      * MATRIX (reach-conditioned): M.R1'_mass.
      Both vs exact-vs-R1'. If frozen >> matrix, a per-bucket scalar CFV is wrong under
      shift and the reach-conditioned matrix is required (confirms D2). If matrix ~=
      coarseness, the matrix handles the shift and bucketing is the only residual.

DECISION: matrix-shift small & ~= coarseness -> bucketed reach-conditioned leaf is the
v1 leaf. matrix-shift large -> escalate to the per-hand exact leaf (v2).

Run from backend/bot/ (small --rivers for a smoke; same subset used everywhere):
    python scripts/validate_cfv_bucketing.py --rivers 4
    python scripts/validate_cfv_bucketing.py --rivers 16
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.blueprint_db import BlueprintDB, FrozenBlueprint
from src.abstractions.hand_evaluator import HandEvaluator
from src.abstractions.card_abstractions import CardAbstraction
from src.abstractions.sizing import db_menu_mode, postflop_menu_for
from src.subgame.cfv import (turn_hands, turn_leaf_value_exact, turn_leaf_matrix,
                             bucketed_measure_leaf, FULL_DECK)

_BOARD4 = ['CQ', 'SJ', 'H9', 'D5']
_POT = 24.0
_STACKS = (88.0, 88.0)
_BROADWAY = set('TJQKA')


def _latest_capped():
    c = sorted(glob.glob('analysis/blueprints/blueprint_par_capped_*.db'))
    return c[-1] if c else None


def _rel_rmse(pred, exact):
    keys = [h for h in exact if h in pred]
    p = np.array([pred[h] for h in keys])
    e = np.array([exact[h] for h in keys])
    denom = np.sqrt(np.mean(e ** 2)) or 1.0
    return float(np.sqrt(np.mean((p - e) ** 2)) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=None)
    ap.add_argument('--rivers', type=int, default=16)
    args = ap.parse_args()
    path = args.db or _latest_capped()
    rawdb = BlueprintDB(path, read_only=True)
    menu = postflop_menu_for(db_menu_mode(rawdb))
    db = FrozenBlueprint(rawdb)            # consistent snapshot (immune to live training)
    ev, cards = HandEvaluator(), CardAbstraction()

    rivers = [c for c in FULL_DECK if c not in set(_BOARD4)]
    if args.rivers > 0:
        rivers = rivers[:args.rivers]
    hands = turn_hands(_BOARD4)
    full = {h: 1.0 for h in hands}                                   # R1 (wide)
    shifted = {h: 1.0 for h in hands if h[0][1] in _BROADWAY and h[1][1] in _BROADWAY}  # R1'

    print(f"blueprint: {os.path.basename(path)} | menu={db_menu_mode(rawdb)} | board={_BOARD4} "
          f"pot={_POT} | rivers={len(rivers)}")
    print(f"R1=full ({len(full)}) -> R1'=broadway ({len(shifted)})  (a big villain-range shift)")

    M, buckets, bidx, tb = turn_leaf_matrix(_BOARD4, _POT, _STACKS, 0, db, ev, cards,
                                            menu=menu, rivers=rivers)
    exact_R1 = turn_leaf_value_exact(_BOARD4, _POT, _STACKS, 0, full, full, db, ev, cards,
                                     menu=menu, rivers=rivers)
    exact_R1p = turn_leaf_value_exact(_BOARD4, _POT, _STACKS, 0, full, shifted, db, ev, cards,
                                      menu=menu, rivers=rivers)
    rawdb.close()

    pred_R1 = bucketed_measure_leaf(M, bidx, tb, hands, full)        # M . R1_mass
    pred_R1p = bucketed_measure_leaf(M, bidx, tb, hands, shifted)    # M . R1'_mass

    coarseness = _rel_rmse(pred_R1, exact_R1)
    frozen_shift = _rel_rmse(pred_R1, exact_R1p)     # reuse R1-calibrated leaf for R1'
    matrix_shift = _rel_rmse(pred_R1p, exact_R1p)    # reach-conditioned to R1'

    # Isolate PURE hero-bucketing coarseness: bucket-average the EXACT per-hand leaf
    # (no card-removal approximation, no M construction). If this is << `coarseness`,
    # the 26% is mostly the card-removal/total-mass approximation (fixable), not the
    # hero buckets being too coarse.
    by_bucket = {}
    for h, v in exact_R1.items():
        by_bucket.setdefault(tb.get(h), []).append(v)
    bmean = {b: float(np.mean(vs)) for b, vs in by_bucket.items()}
    bavg = {h: bmean[tb[h]] for h in exact_R1 if tb.get(h) in bmean}
    pure_hero = _rel_rmse(bavg, exact_R1)

    print(f"\n  buckets present: {len(buckets)}")
    print(f"  (c0) PURE hero-bucketing (bucket-avg exact)  : {pure_hero:6.1%} rel-RMSE")
    print(f"  (c) COARSENESS (M.R1 vs exact-R1, no shift) : {coarseness:6.1%} rel-RMSE")
    print(f"  (b) FROZEN SCALAR under shift (M.R1 vs R1') : {frozen_shift:6.1%} rel-RMSE")
    print(f"  (b) MATRIX (reach-cond) under shift (M.R1') : {matrix_shift:6.1%} rel-RMSE")
    print(f"\n  reach-conditioning gain: frozen {frozen_shift:.1%} -> matrix {matrix_shift:.1%} "
          f"({frozen_shift / max(matrix_shift, 1e-9):.1f}x better)")
    if matrix_shift <= max(0.15, 1.5 * coarseness) and frozen_shift > 2 * matrix_shift:
        print("  VERDICT: reach-conditioned BUCKETED leaf is adequate AND clearly needed "
              "(matrix tracks the shift; frozen scalar doesn't) -> v1 = bucketed matrix leaf.")
    elif matrix_shift > 0.25:
        print("  VERDICT: even the matrix leaf is inaccurate -> escalate to the per-hand exact leaf (v2).")
    else:
        print("  VERDICT: see numbers -- bucketing may be fine and the shift may be mild.")


if __name__ == '__main__':
    main()
