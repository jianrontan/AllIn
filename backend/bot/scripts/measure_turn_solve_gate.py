# backend/bot/scripts/measure_turn_solve_gate.py
"""
M2 Stage-3 GATE: does the turn SOLVE beat the BLUEPRINT, judged out-of-bucket?

For each sampled turn spot (board, pot, stacks), compare the OUT-OF-BUCKET
exploitability of:
  * the BLUEPRINT turn strategy (blueprint_turn_strategy_on_tree), vs
  * the SOLVED turn strategy (TurnCFR average),
where the best-responding adversary values every depth-limit LEAF by the EXACT
blueprint river rollout (ExactLeafTurnCFR -> turn_leaf_value_exact), NOT the bucketed
matrix the solver used. So the solver cannot win by gaming its own leaf bucketing.

GATE: solved exploitability < blueprint exploitability (the turn betting solve removes
a real leak under the true continuation values). This is NECESSARY, not sufficient --
it is still in-model-river (the rollout assumes blueprint river play); a villain that
deviates in the river is only caught by M4's LBR.

HEAVY (exact rollouts in the BR walk) + reads the blueprint -> run on a FREE machine,
not while a BR sweep is using your cores. Reads via FrozenBlueprint. Run from backend/bot/:
    python scripts/measure_turn_solve_gate.py --boards 1 --rivers 4 --nbuckets 32 --iters 100   # smoke
    python scripts/measure_turn_solve_gate.py --boards 4 --rivers 16 --nbuckets 128 --iters 800 # real
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
from src.subgame.turn_tree import build_turn_tree
from src.subgame.turn_cfr import TurnCFR, ExactLeafTurnCFR
from src.subgame.cfv import (turn_strength, equal_freq_partition, turn_leaf_matrix_both,
                             FULL_DECK)
from src.subgame.blueprint_projection import blueprint_turn_strategy_on_tree

_BOARDS = [
    ['CQ', 'SJ', 'H9', 'D5'], ['SK', 'HK', 'D7', 'C2'], ['H9', 'H8', 'D7', 'S2'],
    ['SA', 'HK', 'HQ', 'D5'], ['S8', 'S5', 'S2', 'HK'], ['D6', 'C6', 'S6', 'H2'],
]


def _latest_capped():
    c = sorted(glob.glob('analysis/blueprints/blueprint_par_capped_*.db'))
    return c[-1] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=None)
    ap.add_argument('--boards', type=int, default=4)
    ap.add_argument('--board-idx', type=int, default=None,
                    help="run ONLY this board index (0-based) — for parallelizing the "
                         "gate across processes (one core each); overrides --boards")
    ap.add_argument('--rivers', type=int, default=16)
    ap.add_argument('--nbuckets', type=int, default=128)
    ap.add_argument('--iters', type=int, default=800)
    ap.add_argument('--pot', type=float, default=24.0)
    ap.add_argument('--spr', type=float, default=3.7)
    ap.add_argument('--adversary', choices=['blueprint', 'river_br'], default='blueprint',
                    help="leaf adversary: 'blueprint' = out-of-bucket/in-model-river (M2 gate); "
                         "'river_br' = also best-responds on the river (M3 frozen-range-trap gate)")
    args = ap.parse_args()
    path = args.db or _latest_capped()
    raw = BlueprintDB(path, read_only=True)
    menu = postflop_menu_for(db_menu_mode(raw))
    db = FrozenBlueprint(raw)
    ev, cards = HandEvaluator(), CardAbstraction()
    behind = args.spr * args.pot
    stacks = (behind, behind)

    print(f"blueprint: {os.path.basename(path)} | menu={db_menu_mode(raw)} | "
          f"pot={args.pot} stacks={stacks} (SPR {args.spr:g}) | n={args.nbuckets} "
          f"rivers={args.rivers} iters={args.iters} | adversary={args.adversary}")
    print(f"\n  {'board':30s} {'blueprint':>12s} {'solved':>12s} {'delta':>10s}   gate")

    sel = [_BOARDS[args.board_idx]] if args.board_idx is not None else _BOARDS[:args.boards]
    wins = 0
    rows = []
    for board in sel:
        ba_cache = {}
        rivers = [c for c in FULL_DECK if c not in set(board)]
        if args.rivers > 0:
            rivers = rivers[:args.rivers]
        strength = turn_strength(board, ev, cards, rivers=rivers, ba_cache=ba_cache)
        part = equal_freq_partition(strength, args.nbuckets)
        buckets = sorted(set(part.values()))
        bidx = {b: i for i, b in enumerate(buckets)}
        ba = build_turn_board_arrays(board, cards)             # cards -> projection buckets
        tb_idx = np.array([bidx[part[h]] for h in ba['hands']], dtype=np.int64)

        def leaf_fn(pot, st, _b=board, _p=part, _c=ba_cache):
            M0, M1, _, _, _ = turn_leaf_matrix_both(
                _b, pot, st, db, ev, cards, menu=menu, rivers=rivers,
                partition=_p, ba_cache=_c)
            return M0, M1

        tree = build_turn_tree(args.pot, stacks)
        r0 = np.ones(ba['H'])
        r1 = np.ones(ba['H'])
        solver = TurnCFR(tree, ba, tb_idx, leaf_fn)
        solver.run(r0, r1, iters=args.iters)

        exact = ExactLeafTurnCFR(tree, ba, tb_idx, leaf_fn, board, db, ev, cards,
                                 menu=menu, rivers=rivers, ba_cache=ba_cache,
                                 adversary=args.adversary)
        bp_proj = blueprint_turn_strategy_on_tree(tree, ba, db.get_average_strategy,
                                                  postflop_menu=menu)
        bp = exact.exploitability(r0, r1, strat_fn=lambda nid: bp_proj[nid]) / 2.0 * 1000.0
        sv = exact.exploitability(r0, r1, strat_fn=solver.average_strategy) / 2.0 * 1000.0
        gate = sv < bp
        wins += int(gate)
        rows.append((board, bp, sv))
        print(f"  {str(board):30s} {bp:12.1f} {sv:12.1f} {sv - bp:+10.1f}   "
              f"{'PASS' if gate else 'FAIL'}")
    raw.close()

    bp_m = float(np.mean([r[1] for r in rows]))
    sv_m = float(np.mean([r[2] for r in rows]))
    print(f"\n  MEAN: blueprint {bp_m:.1f} -> solved {sv_m:.1f} mbb/hand "
          f"({(sv_m - bp_m) / bp_m * 100:+.1f}%) | gate {wins}/{len(rows)} boards")
    print(f"\n  GATE {'PASS' if wins == len(rows) and sv_m < bp_m else 'CHECK'}: "
          f"the turn solve {'beats' if sv_m < bp_m else 'does NOT beat'} the blueprint "
          "out-of-bucket (necessary, not sufficient -- M4 LBR is the serve gate).")


if __name__ == '__main__':
    main()
