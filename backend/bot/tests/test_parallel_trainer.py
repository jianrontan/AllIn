# backend/bot/tests/test_parallel_trainer.py
"""
Tests for data-parallel MCCFR+ (src/cfr/parallel_trainer.py).

Two layers:
  1. merge_round() — the master-side block Linear-CFR merge. Pure and
     deterministic, so it is checked exactly against hand-computed values
     (summed increments, per-round discount decay, CFR+ floor).
  2. train_blueprint_parallel() — a small end-to-end run, asserting it produces
     a valid blueprint (populated info sets, normalized average strategies) and
     that its info-set coverage overlaps a single-thread run of the same size.

Run: python tests/test_parallel_trainer.py
Or:  python -m pytest tests/test_parallel_trainer.py -q
"""
import os
import sys
import random
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.information_set import InformationSet
from src.cfr.blueprint_trainer import BlueprintTrainer
from src.cfr import parallel_trainer as pt
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


def _approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# merge_round unit tests (deterministic)
# --------------------------------------------------------------------------- #

def test_merge_new_key_sums_increments():
    """A new key, two workers: cumulative = sum of worker contributions, clock=1,
    no decay (first round the key appears)."""
    info_sets = {}
    w1 = {'regret': {'k': {'a': 2.0, 'b': 1.0}}, 'strategy': {},
          'legal': {'k': ['a', 'b']}, 'ev_sum': 0.0, 'ev_count': 0}
    w2 = {'regret': {'k': {'a': 3.0, 'b': 0.0}}, 'strategy': {},
          'legal': {'k': ['a', 'b']}, 'ev_sum': 0.0, 'ev_count': 0}
    pt.merge_round(info_sets, [w1, w2], alpha=1.5, gamma=2.0)
    r = info_sets['k'].cumulative_regrets
    check('merge new key sum a', _approx(r['a'], 5.0), r)
    check('merge new key sum b', _approx(r['b'], 1.0), r)
    check('merge new key clock', info_sets['k'].visit_count == 1)


def test_merge_existing_key_block_discount():
    """An existing key (clock already 1, regrets present) gets one more round:
    clock -> 2, decay = (1/2)^alpha applied to the OLD cumulative, then the
    summed increment (worker_abs - baseline) added."""
    alpha = 1.5
    info = InformationSet()
    info.cumulative_regrets = {'a': 4.0, 'b': 2.0}
    info.legal_actions = ['a', 'b']
    info.visit_count = 1
    info_sets = {'k': info}

    # Worker returns ABSOLUTE local cumulative = baseline + its delta.
    # Here it drove a up by +1 and b unchanged.
    w = {'regret': {'k': {'a': 5.0, 'b': 2.0}}, 'strategy': {},
         'legal': {'k': ['a', 'b']}, 'ev_sum': 0.0, 'ev_count': 0}
    pt.merge_round(info_sets, [w], alpha=alpha, gamma=2.0)

    decay = (1 / 2) ** alpha
    exp_a = max(0.0, decay * 4.0 + (5.0 - 4.0))
    exp_b = max(0.0, decay * 2.0 + (2.0 - 2.0))
    r = info_sets['k'].cumulative_regrets
    check('merge existing clock', info_sets['k'].visit_count == 2)
    check('merge existing decay a', _approx(r['a'], exp_a), (r['a'], exp_a))
    check('merge existing decay b', _approx(r['b'], exp_b), (r['b'], exp_b))


def test_merge_regret_floor():
    """A negative net increment that overwhelms the decayed baseline floors at 0."""
    info = InformationSet()
    info.cumulative_regrets = {'a': 1.0}
    info.legal_actions = ['a']
    info.visit_count = 1
    info_sets = {'k': info}
    # Worker drove 'a' far negative (abs cumulative 0 after its own floor would
    # be 0, but simulate a reported decrease relative to baseline).
    w = {'regret': {'k': {'a': -10.0}}, 'strategy': {},
         'legal': {'k': ['a']}, 'ev_sum': 0.0, 'ev_count': 0}
    pt.merge_round(info_sets, [w], alpha=1.5, gamma=2.0)
    check('merge floor', _approx(info_sets['k'].cumulative_regrets['a'], 0.0),
          info_sets['k'].cumulative_regrets)


def test_merge_strategy_gamma_independent_clock():
    """Strategy uses its own gamma clock and is NOT floored. A strategy-only
    touch must not advance the regret clock."""
    info = InformationSet()
    info.cumulative_strategy = {'a': 3.0}
    info.legal_actions = ['a']
    info.strategy_visit_count = 1
    info.visit_count = 7  # should be untouched by a strategy-only round
    info_sets = {'k': info}

    gamma = 2.0
    w = {'regret': {}, 'strategy': {'k': {'a': 4.0}},
         'legal': {'k': ['a']}, 'ev_sum': 0.0, 'ev_count': 0}
    pt.merge_round(info_sets, [w], alpha=1.5, gamma=gamma)

    decay = (1 / 2) ** gamma
    exp = decay * 3.0 + (4.0 - 3.0)
    check('merge strat clock', info_sets['k'].strategy_visit_count == 2)
    check('merge strat regret clock untouched', info_sets['k'].visit_count == 7)
    check('merge strat value', _approx(info_sets['k'].cumulative_strategy['a'], exp),
          info_sets['k'].cumulative_strategy)


def test_worker_chunk_disables_discount():
    """A worker run must not touch the discount clocks (master owns them) and
    must populate role-specific dirty sets. Payload carries menu_mode (last field)
    so the worker builds the right action-abstraction arm."""
    baseline = {}
    res = pt._worker_run_chunk((baseline, 0, 40, 12345, 1.5, 2.0, 'control'))
    check('worker returns regret deltas', len(res['regret']) > 0)
    check('worker returns strategy deltas', len(res['strategy']) > 0)
    check('worker ev_count', res['ev_count'] == 40)


def test_worker_chunk_capped_menu():
    """A capped-menu worker must produce the capped abstraction's actions: the
    'overbet2' char '2' appears in some info-set key, and the voluntary all-in node
    is gone (no high-SPR 'a'-only-from-anchor keys). Light check: it runs and yields
    at least one key whose pattern contains '2' (the 2.0x tier the control menu can
    never produce)."""
    baseline = {}
    res = pt._worker_run_chunk((baseline, 0, 400, 999, 1.5, 2.0, 'capped'))
    keys = set(res['regret']) | set(res['strategy'])
    # postflop key pattern is the last underscore-token; '2' there == overbet2 played.
    has_overbet2 = any('2' in k.split('_')[-1] for k in keys)
    check('capped worker can produce overbet2 (char 2)', has_overbet2,
          f"(no '2' in {len(keys)} keys -- raise chunk if flaky)")


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #

def test_parallel_end_to_end():
    """A small parallel run yields a valid, normalized blueprint."""
    trainer = BlueprintTrainer()
    ev = pt.train_blueprint_parallel(
        trainer, iterations=800, workers=2, merge_every=200, seed=7)
    check('e2e populated', len(trainer.info_sets) > 100, len(trainer.info_sets))
    check('e2e ev finite', ev == ev)  # not NaN

    # Every average strategy normalizes to 1.
    ok = True
    for key, info in trainer.info_sets.items():
        strat = info.get_average_strategy(info.legal_actions)
        if info.legal_actions and not _approx(float(sum(strat)), 1.0, 1e-6):
            ok = False
            break
    check('e2e strategies normalize', ok)

    # Discount clocks advanced (block discount actually ran on the master).
    max_clock = max((i.visit_count for i in trainer.info_sets.values()), default=0)
    check('e2e clocks advanced', max_clock >= 1, max_clock)


def test_parallel_overlaps_single_thread():
    """Parallel and single-thread runs of equal size should discover a largely
    overlapping set of info-set keys (same game, same abstraction)."""
    random.seed(1)
    st = BlueprintTrainer()
    st.train_blueprint(800, checkpoint_every=10 ** 9)

    pl = BlueprintTrainer()
    pt.train_blueprint_parallel(pl, iterations=800, workers=2, merge_every=200, seed=1)

    st_keys, pl_keys = set(st.info_sets), set(pl.info_sets)
    overlap = len(st_keys & pl_keys) / max(1, len(st_keys | pl_keys))
    # Threshold lowered 0.7 -> 0.3 (2026-05-29): the decoupled 30-fine/10-coarse
    # abstraction + the widened betting tree mean 800 iters can't come close to
    # saturating the ~24k+ reachable keys, so two equal-size runs legitimately
    # explore a less-overlapping key set (observed Jaccard ~0.46, stable). The test's
    # real job is to confirm the two runs aren't DISJOINT (a wiring bug) -- a
    # comfortably-positive overlap proves that; the old 0.7 was tuned to the small
    # pre-widening tree and is no longer reachable at this iteration budget.
    check('overlap with single-thread', overlap > 0.3, f"jaccard={overlap:.3f}")


def test_parallel_checkpoint_and_resume():
    """Parallel training checkpoints to SQLite and resumes: the resumed run must
    load the prior info sets + iteration counter and continue without error."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, 'bp_parallel.db')

    db = BlueprintDB(db_path)
    t1 = BlueprintTrainer()
    pt.train_blueprint_parallel(t1, iterations=600, db=db, workers=2,
                                merge_every=150, checkpoint_every=300, seed=3)
    n_keys = len(t1.info_sets)
    db.close()

    # Reopen and confirm rows + metadata persisted.
    db2 = BlueprintDB(db_path)
    iters = db2.get_metadata('total_iterations', 0)
    check('checkpoint wrote iterations', iters >= 300, iters)

    # Resume continues from the stored iteration count.
    t2 = BlueprintTrainer()
    start = t2.resume_from_db(db2)
    check('resume loaded info sets', len(t2.info_sets) == n_keys,
          (len(t2.info_sets), n_keys))
    check('resume iteration counter', start >= 300, start)
    pt.train_blueprint_parallel(t2, iterations=300, db=db2, workers=2,
                                merge_every=150, checkpoint_every=300,
                                start_iteration=start, seed=4)
    check('resume grew blueprint', len(t2.info_sets) >= n_keys)
    db2.close()


def test_worker_mode_leaves_clocks_zero():
    """discount_enabled=False must skip ALL discount-clock bookkeeping (the master
    owns the clocks). Run a trainer in worker mode directly and assert every
    info set's regret/strategy clocks stayed at their initial 0."""
    random.seed(99)
    t = BlueprintTrainer()
    t.discount_enabled = False
    for it in range(60):
        t._run_iteration(it)
    bad = [(k, i.visit_count, i.strategy_visit_count, i.last_visited_iteration,
            i.last_strategy_iteration)
           for k, i in t.info_sets.items()
           if i.visit_count or i.strategy_visit_count
           or i.last_visited_iteration != -1 or i.last_strategy_iteration != -1]
    check('worker mode clocks untouched', not bad, bad[:3])
    check('worker mode populated role-dirty',
          len(t._dirty_regret) > 0 and len(t._dirty_strategy) > 0)


def test_parallel_tail_round_is_checkpointed():
    """Regression for H1: when the run length is NOT a clean multiple of the
    checkpoint cadence, the final (sub-checkpoint) rounds' work must still be
    persisted -- the DB's total_iterations must equal the requested total, not an
    undercount that would make a resume replay lost iterations."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, 'bp_tail.db')
    db = BlueprintDB(db_path)
    t = BlueprintTrainer()
    # round_total = merge_every*workers = 400; iterations 900 -> rounds 400,400,100
    # with checkpoint_every=400 the trailing 100-iter round is left unflushed by the
    # in-loop checkpoint and must be caught by the tail flush.
    pt.train_blueprint_parallel(t, iterations=900, db=db, workers=2,
                                merge_every=200, checkpoint_every=400, seed=5)
    in_mem = len(t.info_sets)
    db.close()

    db2 = BlueprintDB(db_path)
    iters = db2.get_metadata('total_iterations', 0)
    # load_all_to_memory row count must match the in-memory table (no lost rows).
    persisted = len(db2.load_all_to_memory())
    db2.close()
    check('tail iterations persisted', iters == 900, iters)
    check('tail rows persisted', persisted == in_mem, (persisted, in_mem))


def test_worker_mode_floors_regret():
    """REVERTED Fix #2 (2026-06-04): a worker (discount_enabled=False) now floors
    cumulative regret on EVERY write -- canonical CFR+, identical to single-thread --
    so NO cumulative regret ever goes negative in either mode. Fix #2's raw storage
    (no worker write-floor) broke CFR+ re-activation: a dominated action stayed
    suppressed and the strategy collapsed onto the largest action (the bot opened
    xlarge with 100% of hands, never folding). A 500k A/B confirmed flooring restores
    sane folds (scripts/ab_fix2_revert.py). This test guards against re-introducing
    the unfloored worker store."""
    for label, disc in (('worker', False), ('single-thread', True)):
        random.seed(123)
        t = BlueprintTrainer()
        t.discount_enabled = disc
        for it in range(120):
            t._run_iteration(it)
        all_nonneg = all(v >= 0 for info in t.info_sets.values()
                         for v in info.cumulative_regrets.values())
        has_pos = any(v > 0 for info in t.info_sets.values()
                      for v in info.cumulative_regrets.values())
        check(f'{label} mode floors regret on write (all >= 0)', all_nonneg)
        check(f'{label} mode still accrues positive regret (not all zero)', has_pos)


def test_merge_cross_worker_cancellation():
    """merge_round must sum SIGNED per-worker increments and floor the total once:
    baseline a=0, worker1 delta +10, worker2 delta -10 -> merged a = max(0, decay*0 +
    10 - 10) = 0. (Workers now floor their own cumulative on write -- per-worker CFR+,
    Fix #2 reverted -- so they don't hand merge_round raw negatives like this directly;
    but a floored cumulative dropping below the pre-round base still yields a negative
    increment, so merge_round must handle signed deltas. This guards that arithmetic.)"""
    info = InformationSet()
    info.cumulative_regrets = {'a': 0.0}
    info.legal_actions = ['a']
    info.visit_count = 1
    info_sets = {'k': info}
    w1 = {'regret': {'k': {'a': 10.0}}, 'strategy': {}, 'legal': {'k': ['a']},
          'ev_sum': 0.0, 'ev_count': 0}
    w2 = {'regret': {'k': {'a': -10.0}}, 'strategy': {}, 'legal': {'k': ['a']},
          'ev_sum': 0.0, 'ev_count': 0}
    pt.merge_round(info_sets, [w1, w2], alpha=1.5, gamma=2.0)
    check('cross-worker cancellation (raw deltas)',
          _approx(info_sets['k'].cumulative_regrets['a'], 0.0),
          info_sets['k'].cumulative_regrets)


# NOTE (2026-06-04): the former test_floor_bias_direction_old_inflates_high_variance
# (+ its _merge_old_floored helper) was REMOVED. It existed only to justify Fix #2's
# raw-regret merge by showing per-worker flooring puts slightly more mass on
# high-variance actions. Fix #2 was reverted (per-worker flooring restored) because
# raw storage broke CFR+ re-activation and collapsed the strategy (open xlarge with
# 100% of hands); see blueprint_trainer.cfr() + scripts/ab_fix2_revert.py. The small
# floor bias is the accepted tradeoff (covered by Fix #4), so a test asserting it is a
# "bug" is backwards. test_worker_mode_floors_regret now guards the restored behavior.


def test_multiround_cross_worker_cancellation_stable():
    """Cross-worker cancellation holds across MULTIPLE rounds (the unit tests are
    single-round). Each round two workers report exactly opposite deltas on 'jam';
    'call' steadily accrues. After several rounds jam must stay floored at 0 while
    call is clearly positive -- i.e. the cancellation doesn't drift or leak round to
    round via the rebroadcast baseline."""
    info = InformationSet()
    info.cumulative_regrets = {'jam': 0.0, 'call': 0.0}
    info.legal_actions = ['jam', 'call']
    info.visit_count = 1
    info_sets = {'k': info}

    for _ in range(5):
        base = dict(info_sets['k'].cumulative_regrets)  # rebroadcast (floored) baseline
        w1 = {'regret': {'k': {'jam': base['jam'] + 12.0, 'call': base['call'] + 4.0}},
              'strategy': {}, 'legal': {'k': ['jam', 'call']}, 'ev_sum': 0.0, 'ev_count': 0}
        w2 = {'regret': {'k': {'jam': base['jam'] - 12.0, 'call': base['call'] + 4.0}},
              'strategy': {}, 'legal': {'k': ['jam', 'call']}, 'ev_sum': 0.0, 'ev_count': 0}
        pt.merge_round(info_sets, [w1, w2], alpha=0.0, gamma=0.0)

    r = info_sets['k'].cumulative_regrets
    check('multiround jam stays floored at 0', _approx(r['jam'], 0.0), r)
    check('multiround call accrues positive', r['call'] > 30.0, r)


def test_evaluate_served_ev_rng_isolated():
    """evaluate_served_ev must NOT perturb the global RNG stream (so calling it
    mid-training doesn't shift the training hand sequence) and must be deterministic
    for a fixed seed (the paired-trend property the checkpoint gauge relies on)."""
    import random as _r
    trainer = BlueprintTrainer()
    _r.seed(123)
    before = _r.getstate()
    v1 = trainer.evaluate_served_ev(n=20, seed=7)
    after = _r.getstate()
    check('evaluate_served_ev preserves global RNG state', before == after)
    v2 = trainer.evaluate_served_ev(n=20, seed=7)
    check('evaluate_served_ev deterministic for fixed seed', v1 == v2, f'{v1} vs {v2}')
    assert before == after, "evaluate_served_ev perturbed the global RNG"
    assert v1 == v2, f"evaluate_served_ev not deterministic: {v1} vs {v2}"


def _run_all():
    test_evaluate_served_ev_rng_isolated()
    test_merge_new_key_sums_increments()
    test_merge_existing_key_block_discount()
    test_merge_regret_floor()
    test_merge_cross_worker_cancellation()
    test_multiround_cross_worker_cancellation_stable()
    test_merge_strategy_gamma_independent_clock()
    test_worker_chunk_disables_discount()
    test_worker_chunk_capped_menu()
    test_worker_mode_floors_regret()
    test_parallel_end_to_end()
    test_parallel_overlaps_single_thread()
    test_parallel_checkpoint_and_resume()
    test_worker_mode_leaves_clocks_zero()
    test_parallel_tail_round_is_checkpointed()
    print(f"\n{_passed} passed, {_failed} failed")
    return _failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
