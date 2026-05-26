# backend/bot/tests/test_storage_and_resolve.py
"""
Coverage for blueprint resolution + the checkpoint/resume round-trip — the two
pieces that decide WHICH blueprint loads and whether a resumed run continues
from the right state. Run: python tests/test_storage_and_resolve.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.information_set import InformationSet
from src.storage.blueprint_db import BlueprintDB
from src.config import resolve_blueprint_path

_passed = _failed = 0


def check(name, cond, extra=''):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name} {extra}")


def _write_db(path, total_iterations, key='pf_9_ip_'):
    db = BlueprintDB(path)
    iset = InformationSet()
    iset.legal_actions = ['fold', 'call', 'bet_large']
    iset.cumulative_regrets = {'fold': 1.0, 'call': 2.5, 'bet_large': 0.0}
    iset.cumulative_strategy = {'fold': 3.0, 'call': 7.0, 'bet_large': 1.0}
    iset.visit_count = 5
    iset.last_visited_iteration = total_iterations - 1
    iset.strategy_visit_count = 4
    iset.last_strategy_iteration = total_iterations - 1
    db.save_checkpoint({key: iset},
                       {'total_iterations': total_iterations, 'alpha': 1.5, 'gamma': 2.0})
    db.close()
    return iset


def test_resolve_picks_highest_iterations():
    with tempfile.TemporaryDirectory() as d:
        _write_db(os.path.join(d, 'blueprint_20260101_000000.db'), 1_000_000)
        _write_db(os.path.join(d, 'blueprint_20260102_000000.db'), 5_000_000)
        chosen = resolve_blueprint_path(d)
        check('resolve picks highest total_iterations',
              os.path.basename(str(chosen)) == 'blueprint_20260102_000000.db',
              f'(got {chosen})')


def test_resolve_env_override(monkeypatch=None):
    with tempfile.TemporaryDirectory() as d:
        _write_db(os.path.join(d, 'blueprint_20260101_000000.db'), 1_000_000)
        pin = os.path.join(d, 'blueprint_20260101_000000.db')
        old = os.environ.get('ALLIN_BLUEPRINT_DB')
        os.environ['ALLIN_BLUEPRINT_DB'] = pin
        try:
            chosen = resolve_blueprint_path(d)
            check('ALLIN_BLUEPRINT_DB override honored',
                  os.path.abspath(str(chosen)) == os.path.abspath(pin), f'(got {chosen})')
        finally:
            if old is None:
                os.environ.pop('ALLIN_BLUEPRINT_DB', None)
            else:
                os.environ['ALLIN_BLUEPRINT_DB'] = old


def test_checkpoint_resume_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'blueprint_20260101_000000.db')
        original = _write_db(path, 2_000_000)
        # Reopen read-only and load every info set back.
        db = BlueprintDB(path, read_only=True)
        try:
            loaded = db.load_all_to_memory()
            ti = db.get_metadata('total_iterations', 0)
            alpha = db.get_metadata('alpha', None)
        finally:
            db.close()
        check('resume metadata round-trips', ti == 2_000_000 and float(alpha) == 1.5,
              f'(ti={ti}, alpha={alpha})')
        iset = loaded.get('pf_9_ip_')
        check('resume loaded the info set', iset is not None)
        if iset:
            check('regrets round-trip', iset.cumulative_regrets == original.cumulative_regrets,
                  f'(got {iset.cumulative_regrets})')
            check('strategy round-trip', iset.cumulative_strategy == original.cumulative_strategy)
            check('discount clocks round-trip',
                  iset.visit_count == original.visit_count
                  and iset.strategy_visit_count == original.strategy_visit_count
                  and iset.last_visited_iteration == original.last_visited_iteration)


if __name__ == '__main__':
    test_resolve_picks_highest_iterations()
    test_resolve_env_override()
    test_checkpoint_resume_roundtrip()
    print(f"\nResults: {_passed} passed, {_failed} failed out of {_passed + _failed}")
    sys.exit(1 if _failed else 0)
