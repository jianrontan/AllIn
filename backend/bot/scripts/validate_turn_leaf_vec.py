# backend/bot/scripts/validate_turn_leaf_vec.py
"""
M2 Stage-1 validation: the leaf plumbing the turn CFR solver sits on.

Two equivalences (both must be ~exact -- this is plumbing, not approximation):
  (1) turn_leaf_matrix_both(M0,M1) == turn_leaf_matrix(hero_seat=0/1). The both-seats
      one-pass builder must reproduce the validated per-seat builder.
  (2) leaf_value_vec(...) == bucketed_measure_leaf_cr(...). The vectorized solve-time
      leaf evaluator must reproduce the validated dict reconstruction (which M0/2-3
      already validated against the EXACT per-hand rollout leaf).

If both hold, the M2 solver's leaf terminal is correct by transitivity to the M0
rollout reference. Run from backend/bot/:
    python scripts/validate_turn_leaf_vec.py --rivers 8
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
from src.evaluation.showdown_kernel import build_turn_board_arrays
from src.subgame.cfv import (turn_hands, turn_leaf_matrix, turn_leaf_matrix_both,
                             turn_strength, equal_freq_partition, leaf_value_vec,
                             bucketed_measure_leaf_cr, FULL_DECK)

_BOARD4 = ['CQ', 'SJ', 'H9', 'D5']
_POT = 24.0
_STACKS = (88.0, 88.0)


def _latest_capped():
    c = sorted(glob.glob('analysis/blueprints/blueprint_par_capped_*.db'))
    return c[-1] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=None)
    ap.add_argument('--rivers', type=int, default=8)
    ap.add_argument('--nbuckets', type=int, default=128)
    args = ap.parse_args()
    path = args.db or _latest_capped()
    rawdb = BlueprintDB(path, read_only=True)
    menu = postflop_menu_for(db_menu_mode(rawdb))
    db = FrozenBlueprint(rawdb)            # consistent snapshot (immune to live training)
    ev, cards = HandEvaluator(), CardAbstraction()

    rivers = [c for c in FULL_DECK if c not in set(_BOARD4)]
    if args.rivers > 0:
        rivers = rivers[:args.rivers]
    part = equal_freq_partition(turn_strength(_BOARD4, ev, cards, rivers=rivers), args.nbuckets)

    M0b, M1b, buckets, bidx, tb = turn_leaf_matrix_both(
        _BOARD4, _POT, _STACKS, db, ev, cards, menu=menu, rivers=rivers, partition=part)
    M0 = turn_leaf_matrix(_BOARD4, _POT, _STACKS, 0, db, ev, cards,
                          menu=menu, rivers=rivers, partition=part)[0]
    rawdb.close()

    # M0 must match the validated per-seat builder; M1 must be -M0^T EXACTLY (the
    # enforced zero-sum identity that keeps the turn subgame zero-sum -- the
    # independent per-seat M1 deliberately differs, see turn_leaf_matrix_both).
    d0 = float(np.max(np.abs(M0b - M0)))
    d1 = float(np.max(np.abs(M1b + M0b.T)))
    print(f"blueprint: {os.path.basename(path)} | board={_BOARD4} | rivers={len(rivers)} "
          f"| buckets={len(buckets)}")
    print(f"\n(1) M0 vs per-seat builder: max|M0b-M0|={d0:.2e} ; "
          f"M1==-M0^T: max|M1b+M0b^T|={d1:.2e}")

    # (2) vectorized leaf == dict reconstruction, on a non-trivial villain reach.
    ba = build_turn_board_arrays(_BOARD4)
    hands = ba['hands']
    tb_idx = np.array([bidx[tb[h]] for h in hands], dtype=np.int64)
    rng = np.random.default_rng(0)
    reach = rng.random(len(hands))                       # arbitrary villain reach
    reach[rng.random(len(hands)) < 0.3] = 0.0            # some zeros (range holes)
    vill = {h: float(reach[i]) for i, h in enumerate(hands)}

    worst = 0.0
    for M in (M0b, M1b):
        vec = leaf_value_vec(M, tb_idx, ba['c1'], ba['c2'], reach)
        dct = bucketed_measure_leaf_cr(M, bidx, tb, hands, vill)
        diff = max(abs(vec[i] - dct[h]) for i, h in enumerate(hands) if h in dct)
        scale = max(abs(v) for v in dct.values()) or 1.0
        worst = max(worst, diff / scale)
    print(f"(2) vectorized leaf vs dict recon: max rel diff = {worst:.2e}")

    ok = d0 < 1e-9 and d1 < 1e-9 and worst < 1e-9
    print(f"\nStage-1 {'PASS' if ok else 'FAIL'}: leaf plumbing "
          f"{'exact (solver can build on it)' if ok else 'MISMATCH -- bug'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
