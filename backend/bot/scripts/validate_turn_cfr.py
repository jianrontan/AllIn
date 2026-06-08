# backend/bot/scripts/validate_turn_cfr.py
"""
M2 Stage-2 validation: the turn CFR+ solver mechanics (src/subgame/turn_cfr.py).

This validates the SOLVER, not the leaf's accuracy (M0 did that). Four checks:

  (1) LEAF ZERO-SUM STRUCTURE: M1 == -M0^T. The river is zero-sum and positional, so
      seat-1's bucketed leaf must be the negated transpose of seat-0's. If this holds,
      the turn subgame is EXACTLY zero-sum (fold terminals already are), so:
  (2) ROOT ZERO-SUM IDENTITY: E0 + E1 == 0 at the root under any strategy (current and
      average) -- a pure consequence of (1) + zero-sum folds, independent of solve quality.
  (3) CONVERGENCE: internal exploitability (BR vs the solver's OWN average strategy, the
      bucketed-leaf game) decreases toward ~0 as CFR+ runs -- the mechanics work. (This
      is the solver converging on the game its leaf defines; whether that game is the
      RIGHT game is the M0 leaf-accuracy question, and the M2 out-of-leaf gate, separate.)
  (4) finite values, bounded by the pot.

Run from backend/bot/ (small settings -> fast; mechanics don't need fine buckets):
    python scripts/validate_turn_cfr.py --rivers 4 --nbuckets 24 --iters 400
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
from src.evaluation.showdown_kernel import build_turn_board_arrays, compatible_mass
from src.subgame.turn_tree import build_turn_tree
from src.subgame.turn_cfr import TurnCFR
from src.subgame.cfv import (turn_hands, turn_leaf_matrix_both, turn_strength,
                             equal_freq_partition, FULL_DECK)

_BOARD4 = ['CQ', 'SJ', 'H9', 'D5']
_POT = 24.0
_STACKS = (88.0, 88.0)


def _latest_capped():
    c = sorted(glob.glob('analysis/blueprints/blueprint_par_capped_*.db'))
    return c[-1] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=None)
    ap.add_argument('--rivers', type=int, default=4)
    ap.add_argument('--nbuckets', type=int, default=24)
    ap.add_argument('--iters', type=int, default=400)
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
    buckets = sorted(set(part.values()))
    bidx = {b: i for i, b in enumerate(buckets)}

    ba = build_turn_board_arrays(_BOARD4)
    hands = ba['hands']
    tb_idx = np.array([bidx[part[h]] for h in hands], dtype=np.int64)

    # leaf provider: (final_pot, leaf_stacks) -> (M0, M1), cached per distinct leaf.
    cache = {}

    def leaf_fn(pot, stacks):
        key = (round(pot, 6), round(stacks[0], 6), round(stacks[1], 6))
        if key not in cache:
            M0, M1, _, _, _ = turn_leaf_matrix_both(
                _BOARD4, pot, stacks, db, ev, cards, menu=menu, rivers=rivers, partition=part)
            cache[key] = (M0, M1)
        return cache[key]

    tree = build_turn_tree(_POT, _STACKS)
    solver = TurnCFR(tree, ba, tb_idx, leaf_fn)

    reach0 = np.ones(len(hands))
    reach1 = np.ones(len(hands))

    print(f"blueprint: {os.path.basename(path)} | board={_BOARD4} pot={_POT} stacks={_STACKS}")
    print(f"buckets={len(buckets)} rivers={len(rivers)} turn-hands H={ba['H']} "
          f"decision-nodes={len(tree.decision_nodes)}")

    # (3) convergence trend: DEPTH-LIMITED exploitability of the average strategy at
    # log-spaced checkpoints. NOTE: this is the internal gap of the bucketed-leaf game
    # (river frozen to the blueprint inside the leaf) -- a mechanics check that CFR+ is
    # minimizing regret, NOT a solution-quality measure (that's the Stage-3 gate). We
    # report the trend + whether it is still dropping; we do NOT claim "converged".
    print("\n(3) CONVERGENCE TREND (depth-limited exploitability of the avg strategy, mbb/hand):")
    ck = sorted(set(max(1, args.iters * k // 16) for k in (1, 2, 4, 8, 12, 16)))
    prev = None
    monotone = True
    series = []
    done = 0
    for target in ck:
        solver.run(reach0, reach1, iters=target - done)
        done = target
        expl = solver.exploitability(reach0, reach1)
        mbb = expl / 2.0 * 1000.0   # chips/hand -> mbb (BB=2)
        arrow = "" if prev is None else (" v" if expl < prev + 1e-12 else " ^!")
        if prev is not None and expl > prev + 1e-9:
            monotone = False
        print(f"  iter {done:5d}: {mbb:10.1f} mbb/hand{arrow}")
        prev = expl
        series.append(mbb)
    last_expl = prev
    # last-segment relative drop: small => flattening (closer to a floor)
    last_drop = (series[-2] - series[-1]) / series[-2] if len(series) >= 2 and series[-2] else 0.0

    # (1) leaf zero-sum structure + (2) root zero-sum identity, on a built leaf.
    M0, M1 = leaf_fn(_POT, _STACKS)
    zs_leaf = float(np.max(np.abs(M1 + M0.T)))
    v0, v1 = solver.current_values(reach0, reach1)
    e0 = float((reach0 * v0).sum())
    e1 = float((reach1 * v1).sum())
    scale = max(abs(e0), abs(e1), 1.0)
    a0, a1 = solver._eval(tree.root, reach0, reach1, solver.average_strategy)
    e0a, e1a = float((reach0 * a0).sum()), float((reach1 * a1).sum())
    finite = np.all(np.isfinite(v0)) and np.all(np.isfinite(v1))
    rawdb.close()

    print(f"\n(1) leaf zero-sum: max|M1 + M0^T| = {zs_leaf:.2e}")
    print(f"(2) root zero-sum: E0+E1 (current) = {e0 + e1:+.3e}  (avg) = {e0a + e1a:+.3e} "
          f"(|.|/scale = {abs(e0 + e1) / scale:.1e})")
    print(f"(4) finite values: {finite}")

    ok = (zs_leaf < 1e-9 and abs(e0 + e1) / scale < 1e-6 and abs(e0a + e1a) / scale < 1e-6
          and finite and last_expl is not None)
    # MECHANICS gate (not a quality gate): the subgame is exactly zero-sum AND the
    # depth-limited exploitability is monotonically decreasing. We deliberately do NOT
    # assert a convergence floor here -- this is a smoke at low iters; the floor + the
    # solution-quality judgement live in the M2 Stage-3 out-of-leaf gate.
    print(f"\n  trend: {'monotonically decreasing' if monotone else 'NOT monotone (^!)'}; "
          f"last value {series[-1]:.1f} mbb/hand; last-segment drop {last_drop:.0%} "
          f"({'flattening' if last_drop < 0.2 else 'still falling fast'})")
    print(f"\nStage-2 {'PASS' if ok and monotone else 'CHECK'}: "
          f"zero-sum {'exact' if ok else 'OFF'}, mechanics "
          f"{'OK (zero-sum + monotone-decreasing)' if ok and monotone else 'PROBLEM'}. "
          f"NOTE: convergence FLOOR not established here (smoke); quality = Stage-3 gate.")
    sys.exit(0 if ok and monotone else 1)


if __name__ == '__main__':
    main()
