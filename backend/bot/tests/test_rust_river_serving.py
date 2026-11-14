"""Live-tree parity, including injected human actions and the safety gadget."""
import numpy as np
import pytest

pytest.importorskip("allin_core", reason="Rust serving tests require the compiled extension")

from .test_solve_control import _setup
from src.subgame.river_cfr import RiverCFR
from src.subgame.rust_river_cfr import RustRiverCFR
from src.subgame.solve_control import solve_river


@pytest.mark.parametrize('gadget', [False, True])
def test_exact_serving_tree_parity(gadget):
    tree, ba = _setup(30., (40., 40.))
    # Inject below a non-root node: appended ids differ from traversal order.
    node = tree.root.children[0]
    assert tree.inject_realized_edge(node, 'bet', 7.123456789) is not None
    py, rs = RiverCFR(tree, ba), RustRiverCFR(tree, ba, exact=True)
    rng = np.random.default_rng(72)
    r0, r1 = rng.random((2, ba['H'])) + .1
    seeds = [py.average_strategy(i) for i in range(len(tree.decision_nodes))]
    for cfr in (py, rs):
        cfr.warm_start(seeds, 2.)
        for _ in range(2):
            if gadget:
                cfr.run_gadget(r0, r1, np.zeros(ba['H']), 1, iters=2)
            else:
                cfr.run(r0, r1, iters=2)
    assert py._iter == rs._iter == 4
    for n in tree.decision_nodes:
        np.testing.assert_array_equal(py.average_strategy(n.node_id), rs.average_strategy(n.node_id))
    np.testing.assert_array_equal(py.node_action_values(node, r0, r1),
                                  rs.node_action_values(node, r0, r1))
    np.testing.assert_array_equal(py.reach_into([0], r0, r1), rs.reach_into([0], r0, r1))


def test_solve_backend_selection(monkeypatch):
    monkeypatch.setenv("ALLIN_RIVER_ALGORITHM", "cfrplus")
    tree, ba = _setup(30., (10., 10.))
    r = np.ones(ba['H'])
    monkeypatch.setenv('ALLIN_RIVER_BACKEND', 'rust')
    cfr, info = solve_river(tree, ba, r, r, max_iters=2, check_every=1)
    assert isinstance(cfr, RustRiverCFR)
    assert info['iters'] > 0
    monkeypatch.setenv('ALLIN_RIVER_BACKEND', 'typo')
    with pytest.raises(ValueError, match='Unknown ALLIN_RIVER_BACKEND'):
        solve_river(tree, ba, r, r)


def test_backend_checks_extension_before_serving(monkeypatch):
    monkeypatch.setenv("ALLIN_RIVER_ALGORITHM", "cfrplus")
    import sys
    from types import SimpleNamespace
    from src.subgame.solve_control import river_backend
    monkeypatch.setenv('ALLIN_RIVER_BACKEND', 'rust')
    monkeypatch.setitem(sys.modules, 'allin_core', SimpleNamespace(RiverCFR=object))
    with pytest.raises(RuntimeError, match='Rebuild allin_core'):
        river_backend()
    monkeypatch.setenv('ALLIN_RIVER_BACKEND', 'python')
    assert river_backend() == 'python'


@pytest.mark.parametrize('villain_seat', [None, 0, 1])
def test_experimental_dcfr_matches_reference(villain_seat):
    import importlib.util
    import sys
    from pathlib import Path
    bench = Path(__file__).resolve().parents[2] / 'core-rs' / 'bench'
    sys.path.insert(0, str(bench))
    try:
        spec = importlib.util.spec_from_file_location('dcfr_reference', bench / 'compare_dcfr.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(bench))
    tree, ba = _setup(30., (40., 40.))
    py, rs = module.DCFR(tree, ba), RustRiverCFR(tree, ba, exact=True)
    rs._native.enable_dcfr()
    rng = np.random.default_rng(19)
    r0, r1 = rng.random((2, ba['H'])) + .1
    for _ in range(2):
        for solver in (py, rs):
            if villain_seat is None:
                solver.run(r0, r1, 5)
            else:
                solver.run_gadget(r0, r1, np.zeros(ba['H']), villain_seat, iters=5)
        for n in tree.decision_nodes:
            np.testing.assert_allclose(rs._native.regret(n.node_id), py.regret[n.node_id], rtol=1e-11, atol=1e-8)
            np.testing.assert_allclose(rs.average_strategy(n.node_id), py.average_strategy(n.node_id), rtol=1e-11, atol=1e-10)
    if villain_seat is not None:
        np.testing.assert_allclose(rs._native.gadget_regret(), py._g_regret, rtol=1e-11, atol=1e-8)
    with pytest.raises(ValueError, match='cannot change'):
        rs._native.enable_dcfr()
