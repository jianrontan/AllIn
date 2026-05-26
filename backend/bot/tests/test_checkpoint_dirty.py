# backend/bot/tests/test_checkpoint_dirty.py
"""
Guards the dirty-set incremental checkpoint (C1): the on-disk blueprint after
several incremental checkpoints must be BYTE-IDENTICAL to the trainer's
in-memory state. A missed dirty-mark would leave a stale/ahead row in the DB,
silently corrupting a resume -- this test would catch it.

Run: python tests/test_checkpoint_dirty.py   (loads the abstraction table; ~slow)
"""
import contextlib
import io
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.blueprint_trainer import BlueprintTrainer
from src.storage.blueprint_db import BlueprintDB

_passed = _failed = 0


def check(name, cond, extra=''):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name} {extra}")


def test_db_matches_memory_after_incremental_checkpoints():
    n, every = 900, 300            # -> checkpoints at iters 300/600/900
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'blueprint_ckpt_test.db')
        random.seed(7)
        trainer = BlueprintTrainer()
        db = BlueprintDB(path)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.train_blueprint(n, db=db, checkpoint_every=every)
        finally:
            db.close()

        mem = trainer.info_sets
        # Reopen and load everything that was persisted.
        rdb = BlueprintDB(path, read_only=True)
        try:
            disk = rdb.load_all_to_memory()
            meta_iters = rdb.get_metadata('total_iterations', -1)
            meta_evc = rdb.get_metadata('ev_count', -1)
            meta_evs = rdb.get_metadata('ev_sum', None)
        finally:
            rdb.close()

        check('all dirty marks fired -> no key missing/extra on disk',
              set(disk) == set(mem),
              f"(mem={len(mem)}, disk={len(disk)}, "
              f"missing={len(set(mem) - set(disk))}, extra={len(set(disk) - set(mem))})")

        mismatches = []
        for k in mem:
            if k not in disk:
                continue
            m, s = mem[k], disk[k]
            if (m.cumulative_regrets != s.cumulative_regrets
                    or m.cumulative_strategy != s.cumulative_strategy
                    or m.visit_count != s.visit_count
                    or m.last_visited_iteration != s.last_visited_iteration
                    or m.strategy_visit_count != s.strategy_visit_count
                    or m.last_strategy_iteration != s.last_strategy_iteration):
                mismatches.append(k)
        check('every persisted info set equals in-memory (regrets/strategy/clocks)',
              not mismatches,
              f"({len(mismatches)} stale rows, e.g. {mismatches[:3]})")

        check('metadata persisted (iters/ev_count/ev_sum)',
              meta_iters == n and meta_evc == trainer.ev_count
              and meta_evs == trainer.ev_sum,
              f"(iters={meta_iters}/{n}, evc={meta_evc}/{trainer.ev_count})")

        check('dirty set cleared after final checkpoint', len(trainer._dirty) == 0,
              f"({len(trainer._dirty)} left)")


if __name__ == '__main__':
    test_db_matches_memory_after_incremental_checkpoints()
    print(f"\nResults: {_passed} passed, {_failed} failed out of {_passed + _failed}")
    sys.exit(1 if _failed else 0)
