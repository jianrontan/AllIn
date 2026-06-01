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
    must populate role-specific dirty sets."""
    baseline = {}
    res = pt._worker_run_chunk((baseline, 0, 40, 12345, 1.5, 2.0))
    check('worker returns regret deltas', len(res['regret']) > 0)
    check('worker returns strategy deltas', len(res['strategy']) > 0)
    check('worker ev_count', res['ev_count'] == 40)


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


def test_worker_mode_stores_raw_regret():
    """Fix #2 (the worker-side change): a worker (discount_enabled=False) stores RAW
    signed regret -- it does NOT floor cumulative_regrets on write -- so a losing
    action's negative regret survives to the master, where it can cancel another
    worker's positive regret before the single per-round floor. Single-thread
    (discount_enabled=True) keeps the canonical CFR+ write-floor, so all of ITS
    cumulative regrets stay >= 0. This asymmetry is the whole fix."""
    random.seed(123)
    w = BlueprintTrainer()
    w.discount_enabled = False
    for it in range(120):
        w._run_iteration(it)
    worker_has_neg = any(v < 0 for info in w.info_sets.values()
                         for v in info.cumulative_regrets.values())
    check('worker mode stores raw (negative) regret', worker_has_neg)

    random.seed(123)
    st = BlueprintTrainer()  # discount_enabled True by default
    for it in range(120):
        st._run_iteration(it)
    st_all_nonneg = all(v >= 0 for info in st.info_sets.values()
                        for v in info.cumulative_regrets.values())
    check('single-thread floors on write (all regrets >= 0)', st_all_nonneg)


def test_merge_cross_worker_cancellation():
    """With RAW worker deltas, a losing action's negative increment from one worker
    cancels a winning increment from another WITHIN the round, before the single
    master floor. Baseline a=0; worker1 reports +10, worker2 reports -10 (raw) ->
    merged a = max(0, decay*0 + (10-0) + (-10-0)) = 0. The OLD code floored each
    worker locally, so worker2 would have reported a=0 (its -10 clamped away), giving
    merged a = max(0, 0 + 10 + 0) = 10 -- the inflation Fix #2 removes."""
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


def _merge_old_floored(info_sets, worker_results, alpha, gamma):
    """Re-create the PRE-Fix-#2 behaviour: each worker floors its OWN reported
    cumulative at the baseline before the master sums them. Used only to prove the
    bias DIRECTION against the current (raw) merge_round, as a committed regression
    so a future reintroduction of the worker-local floor is caught cheaply (without
    a 25-min training A/B)."""
    floored = []
    R0 = {}
    for res in worker_results:
        for k, wr in res['regret'].items():
            R0.setdefault(k, dict(info_sets[k].cumulative_regrets) if k in info_sets else {})
    for res in worker_results:
        fr = {}
        for k, wr in res['regret'].items():
            base = R0[k]
            # Old bug: clamp the worker's reported cumulative to >= 0 locally.
            fr[k] = {a: max(0.0, v) for a, v in wr.items()}
            # (base is already >=0; the clamp is what discarded negative deltas.)
        floored.append({**res, 'regret': fr})
    return pt.merge_round(info_sets, floored, alpha, gamma)


def test_floor_bias_direction_old_inflates_high_variance():
    """COMMITTED regression for the bias DIRECTION. A high-variance action gets a
    big WIN from one worker (+30) and a big LOSS from another (-30) in the same
    round -> true net 0. The current (raw) merge cancels them to 0. The old
    worker-local-floor path clamps the -30 worker to 0 first, so it merges to +30 --
    strictly MORE mass on the high-variance action. Asserts new <= old, with a real
    gap, so reinstating the worker floor would fail here in milliseconds."""
    def _fresh():
        info = InformationSet()
        info.cumulative_regrets = {'jam': 0.0, 'fold': 0.0}
        info.legal_actions = ['jam', 'fold']
        info.visit_count = 1
        return {'k': info}

    w_win = {'regret': {'k': {'jam': 30.0, 'fold': 0.0}}, 'strategy': {},
             'legal': {'k': ['jam', 'fold']}, 'ev_sum': 0.0, 'ev_count': 0}
    w_loss = {'regret': {'k': {'jam': -30.0, 'fold': 5.0}}, 'strategy': {},
              'legal': {'k': ['jam', 'fold']}, 'ev_sum': 0.0, 'ev_count': 0}

    new_sets = _fresh()
    pt.merge_round(new_sets, [w_win, w_loss], alpha=0.0, gamma=0.0)
    new_jam = new_sets['k'].cumulative_regrets['jam']

    old_sets = _fresh()
    _merge_old_floored(old_sets, [w_win, w_loss], alpha=0.0, gamma=0.0)
    old_jam = old_sets['k'].cumulative_regrets['jam']

    check('raw merge cancels high-variance to 0', _approx(new_jam, 0.0), new_jam)
    check('old floored path inflates high-variance', old_jam > new_jam + 1.0,
          (old_jam, new_jam))


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


def _run_all():
    test_merge_new_key_sums_increments()
    test_merge_existing_key_block_discount()
    test_merge_regret_floor()
    test_merge_cross_worker_cancellation()
    test_floor_bias_direction_old_inflates_high_variance()
    test_multiround_cross_worker_cancellation_stable()
    test_merge_strategy_gamma_independent_clock()
    test_worker_chunk_disables_discount()
    test_worker_mode_stores_raw_regret()
    test_parallel_end_to_end()
    test_parallel_overlaps_single_thread()
    test_parallel_checkpoint_and_resume()
    test_worker_mode_leaves_clocks_zero()
    test_parallel_tail_round_is_checkpointed()
    print(f"\n{_passed} passed, {_failed} failed")
    return _failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
