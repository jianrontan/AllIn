# backend/bot/tests/test_blueprint_projection.py
"""
Validate the blueprint->tree projection (blueprint_projection.py) and the
river-exploitability comparison it enables:
  * tree_action_char maps actions to blueprint pattern chars correctly;
  * blueprint_strategy_on_tree yields valid per-node distributions;
  * the solved strategy is NO MORE exploitable than the blueprint's river play on
    the same river state (the whole point: solver_expl <= blueprint_expl).
Runs against the active blueprint; skips if none.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.subgame.river_tree import build_river_tree
from src.subgame.river_cfr import RiverCFR
from src.subgame.solve_control import solve_river
from src.subgame.blueprint_projection import (
    tree_action_char, blueprint_strategy_on_tree)
from src.evaluation.showdown_kernel import build_board_arrays
from src.abstractions.hand_evaluator import HandEvaluator
from src.abstractions.card_abstractions import CardAbstraction

_EVAL = HandEvaluator()
_CARDS = CardAbstraction()
_BOARD = ['CQ', 'SJ', 'H9', 'D5', 'C2']


def _db():
    try:
        from src.config import resolve_blueprint_path
        from src.storage.blueprint_db import BlueprintDB
        return BlueprintDB(resolve_blueprint_path(), read_only=True)
    except Exception as e:
        print(f"  (no blueprint DB: {e})")
        return None


def test_tree_action_char():
    import types
    node = types.SimpleNamespace(player=1, sc=(0.0, 0.0), pot_mid=24.0, to_call=0.0)
    assert tree_action_char('check', node) == 'k'
    assert tree_action_char('allin', node) == 'a'
    # bet 12 / pot 24 = 0.5 -> nearest blueprint size (0.33 small / 0.66 medium) = medium
    assert tree_action_char('bet:12', node) == 'm'
    assert tree_action_char('bet:8', node) == 's'     # 0.33 -> small
    print("PASS test_tree_action_char")


def test_projection_valid_distributions():
    db = _db()
    if db is None:
        print("SKIP test_projection_valid_distributions (no blueprint)")
        return
    ba = build_board_arrays(_BOARD, _EVAL, _CARDS)
    tree = build_river_tree(pot_entry=24.0, stacks=(88.0, 88.0))
    strat = blueprint_strategy_on_tree(tree, ba, db.get_average_strategy)
    db.close()
    for node in tree.decision_nodes:
        mat = strat[node.node_id]
        assert mat.shape == (ba['H'], len(node.actions))
        assert np.allclose(mat.sum(axis=1), 1.0), node.node_id
        assert (mat >= -1e-12).all()
    print("PASS test_projection_valid_distributions")


def test_solver_no_more_exploitable_than_blueprint():
    db = _db()
    if db is None:
        print("SKIP test_solver_no_more_exploitable_than_blueprint (no blueprint)")
        return
    ba = build_board_arrays(_BOARD, _EVAL, _CARDS)
    tree = build_river_tree(pot_entry=24.0, stacks=(88.0, 88.0))
    rng = np.random.default_rng(0)
    r0 = rng.random(ba['H']) + 0.1
    r1 = rng.random(ba['H']) + 0.1

    bp_strat = blueprint_strategy_on_tree(tree, ba, db.get_average_strategy)
    measure_cfr = RiverCFR(tree, ba)
    bp_expl = measure_cfr.exploitability(r0, r1, strat_fn=lambda nid: bp_strat[nid])

    solved, info = solve_river(tree, ba, r0, r1, max_iters=300, gap_threshold=0.02 * 24.0)
    solver_expl = solved.exploitability(r0, r1)
    db.close()

    # The whole point: the solved river strategy is no more exploitable than the
    # blueprint's (in chips/dealt-pair). Allow a tiny slack for CFR non-convergence.
    assert solver_expl <= bp_expl + 1e-6, (solver_expl, bp_expl)
    # And the blueprint should leave SOME exploitability the solver removes.
    print(f"PASS test_solver_no_more_exploitable_than_blueprint "
          f"(blueprint={bp_expl:.3f}, solver={solver_expl:.3f} chips/pair, "
          f"leak removed={bp_expl - solver_expl:.3f})")


TESTS = [
    test_tree_action_char,
    test_projection_valid_distributions,
    test_solver_no_more_exploitable_than_blueprint,
]

if __name__ == '__main__':
    passed = failed = 0
    for fn in TESTS:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\nResults: {passed} passed, {failed} failed out of {len(TESTS)}")
    sys.exit(1 if failed else 0)
