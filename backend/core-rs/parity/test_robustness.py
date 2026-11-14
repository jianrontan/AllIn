"""Hostile / malformed inputs to the Rust bindings must raise ValueError (or
be refused) -- never crash the process, hang, or silently corrupt results.
Regression tests for the 2026-11-08 security review (docs/RUST_CORE_PLAN.md).
Runs against the built extension only."""
import os

import numpy as np
import pytest

from conftest import TARGET

if TARGET != 'rust':
    pytest.skip('robustness tests target the Rust extension', allow_module_level=True)
allin_core = pytest.importorskip('allin_core')

CTRL = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'bot',
                                     'analysis', 'abstractions'))


def _board(h=6):
    raw = np.arange(h, dtype=float)
    c1 = np.arange(0, 2 * h, 2, dtype=np.int64)
    c2 = c1 + 1
    return raw, c1, c2


# -- river tree: size/depth limits (was: SIGSEGV / OOM) -----------------------
@pytest.mark.parametrize('args', [
    (10.0, 1e10, [0.5], 10**6),          # deep recursion
    (10.0, 1e10, [0.5], 7),              # max_aggressions over the cap
    (10.0, 200.0, [0.1 * i for i in range(1, 11)], 3),   # menu too long
    (10.0, 200.0, [], 3),
    (10.0, 200.0, [float('nan')], 3),
    (10.0, 200.0, [-0.5], 3),
    (float('nan'), 200.0, [0.5], 3),
    (10.0, float('inf'), [0.5], 3),
    (10.0, -1.0, [0.5], 3),
])
def test_river_tree_rejects_bad_params(args):
    with pytest.raises(ValueError):
        allin_core.river_tree_nodes(*args)


def test_river_tree_node_budget():
    menu = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    with pytest.raises(ValueError, match='too large'):
        allin_core.river_tree_nodes(2.0, 1e6, menu, 6)


def test_river_rejects_too_many_hands():
    raw = np.zeros(1327)
    c = np.zeros(1327, dtype=np.int64)
    with pytest.raises(ValueError):
        allin_core.compatible_mass(c, c + 1, raw)


# -- river CFR: non-finite / negative inputs (was: silent uniform / NaN) --------
@pytest.mark.parametrize('bad', [np.nan, np.inf, -1.0])
def test_river_cfr_rejects_bad_reach(bad):
    raw, c1, c2 = _board()
    cfr = allin_core.RiverCFR(10.0, 100.0, [0.5, 1.0], 3, raw, c1, c2)
    r = np.ones(len(raw))
    r[2] = bad
    with pytest.raises(ValueError):
        cfr.run(r, np.ones(len(raw)), 2)
    with pytest.raises(ValueError):
        cfr.exploitability(np.ones(len(raw)), r)


def test_showdown_rejects_nonfinite_scalars():
    raw, c1, c2 = _board()
    rv = np.ones(len(raw))
    with pytest.raises(ValueError):
        allin_core.showdown_measure(raw, c1, c2, rv, float('nan'), 1.0)
    with pytest.raises(ValueError):
        allin_core.showdown_measure(raw, c1, c2, rv, 10.0, float('inf'))


# -- engine: chip sanity (was: negative cost, counter wrap, 2-card showdown) ---
def _st(**kw):
    e = allin_core.Engine('control')
    st = e.init_node_state(1, 10.0)
    st.update(kw)
    return e, st


def test_engine_rejects_negative_custom_cost():
    e, st = _st(c=[5.0, 0.0])
    with pytest.raises(ValueError):
        e.advance(st, 'bet_custom_1', 1, 0, 100.0, (0.0, 0.0))


def test_engine_rejects_spending_past_stack():
    e, st = _st()
    with pytest.raises(ValueError):
        e.advance(st, 'bet_custom_500', 1, 0, 100.0, (0.0, 0.0))
    with pytest.raises(ValueError):
        e.legal_actions(1, st, 0, -5.0)


def test_engine_counter_overflow_is_an_error():
    e, st = _st(num_br=2**32 - 1, last_kind='aggr', bet_to=4.0, prev_bet_to=2.0)
    with pytest.raises(ValueError):
        e.advance(st, 'allin', 1, 0, 100.0, (0.0, 0.0))


def test_utility_needs_full_board_for_showdown():
    e = allin_core.Engine('control')
    with pytest.raises(ValueError):
        e.utility(['HA', 'DA'], ['C7', 'S2'], [], ['call', 'check'], 0, 3.0, 0.0, 0.0, 4.0, 2.0)


# -- trainer: overflow / paths (was: infinite loop, URI interpretation) --------
def test_parallel_round_overflow_is_an_error():
    t = allin_core.Trainer('capped')
    t.load_abstraction(CTRL)
    with pytest.raises((ValueError, OverflowError)):
        t.train(10, 0, 1, workers=2, merge_every=2**63)


def test_checkpoint_path_is_not_a_uri(tmp_path):
    t = allin_core.Trainer('control')
    name = 'file:uri_test.db?mode=memory'
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        t.checkpoint(name, 0)
        assert os.path.exists(tmp_path / name), 'path must be a plain file name'
    finally:
        os.chdir(cwd)


def test_merge_counter_overflow_is_an_error():
    key = 'pf_1_2_ip_flop_'
    glob = {key: {'legal': ['check', 'bet_small'], 'regret': {'check': 1.0},
                  'strategy': {}, 'visit_count': 2**63 - 1, 'strategy_visit_count': 0}}
    w = {'regret': {key: {'check': 2.0}}, 'strategy': {}, 'legal': {key: ['check', 'bet_small']}}
    with pytest.raises(ValueError):
        allin_core.merge_round(glob, [w], 1.5, 1.0)


def test_river_warm_start_preserves_clock_and_validates_before_mutating():
    raw, c1, c2 = _board()
    cfr = allin_core.RiverCFR(10.0, 100.0, [0.5, 1.0], 3, raw, c1, c2)
    seeds = []
    for nid in range(cfr.n_decision):
        seed = np.zeros_like(cfr.average_strategy(nid))
        seed[:, 0] = 1.0
        seeds.append(seed)
    cfr.warm_start(seeds, 10.0)
    assert cfr.iterations == 0
    for nid, seed in enumerate(seeds):
        np.testing.assert_array_equal(cfr.average_strategy(nid), seed)
    bad = [seed.copy() for seed in seeds]
    bad[0][:] = 0.0
    bad[0][:, -1] = 1.0
    bad[-1][0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        cfr.warm_start(bad, 10.0)
    # An invalid later node must not partially apply earlier seeds.
    np.testing.assert_array_equal(cfr.average_strategy(0), seeds[0])
    with pytest.raises(ValueError, match='shape'):
        cfr.warm_start([seeds[0].T, *seeds[1:]], 10.0)
    with pytest.raises(ValueError, match='finite'):
        cfr.warm_start(seeds, np.inf)


@pytest.mark.parametrize('fault', ['cycle', 'shared', 'orphan', 'duplicate_id', 'nan'])
def test_serving_tree_rejects_malformed_graph(fault):
    nodes = allin_core.river_tree_nodes(10., 20., [0.5], 2)
    if fault == 'cycle':
        nodes[0]['children'][0] = 0
    elif fault == 'shared':
        nodes[0]['children'][1] = nodes[0]['children'][0]
    elif fault == 'orphan':
        nodes.append(dict(terminal=True, final_pot=10., contrib=[5., 5.], folder=None))
    elif fault == 'duplicate_id':
        next(n for n in nodes[1:] if not n['terminal'])['node_id'] = 0
    else:
        next(n for n in nodes if n['terminal'])['final_pot'] = np.nan
    with pytest.raises(ValueError):
        allin_core.RiverCFR.from_tree(10., nodes, *_board())
