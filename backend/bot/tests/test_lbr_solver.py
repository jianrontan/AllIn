# backend/bot/tests/test_lbr_solver.py
"""
Smoke test for the solver-as-victim LBR evaluator (lbr_solver.py): it plays full
hands with the RiverSubgameSolver on the river and returns a finite mbb/hand
without crashing. Runs against the active blueprint; skips if none.

This is a SMOKE (few hands) -- the real measurement is a long offline run via
scripts/run_solver_lbr.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.subgame.river_subgame_solver import RiverSubgameSolver
from src.evaluation.lbr_solver import SolverLBREvaluator


def _blueprint_db():
    try:
        from src.config import resolve_blueprint_path
        from src.storage.blueprint_db import BlueprintDB
        return BlueprintDB(resolve_blueprint_path(), read_only=True)
    except Exception as e:
        print(f"  (no blueprint DB: {e})")
        return None


def test_solver_lbr_smoke():
    db = _blueprint_db()
    if db is None:
        print("SKIP test_solver_lbr_smoke (no blueprint)")
        return
    solver = RiverSubgameSolver(db, max_iters=30, check_every=15, time_budget=8.0)
    ev = SolverLBREvaluator(db, solver, seed=1)
    res = ev.evaluate(num_hands=6, progress_every=0)
    db.close()
    assert res['num_hands'] == 6
    assert isinstance(res['lbr_mbb'], float)
    # LBR is a lower bound on exploitability -> non-negative in expectation, but a
    # 6-hand sample can be negative; just assert it's a finite number.
    assert res['lbr_mbb'] == res['lbr_mbb']        # not NaN
    print(f"PASS test_solver_lbr_smoke (lbr_mbb={res['lbr_mbb']:.0f} over 6 hands)")


TESTS = [test_solver_lbr_smoke]

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
