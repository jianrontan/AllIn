# backend/bot/tests/test_turn_cfr.py
"""
Direct tests for the turn CFR+ solver (src/subgame/turn_cfr.py), the subclass of
RiverCFR. These use a SYNTHETIC leaf_matrix_fn (no blueprint DB, no rollout) so they
are fast and deterministic and exercise the ENGINE, not the leaf's accuracy.

What they pin (the subclassing hazards from review):
  * The solver runs on the 4-card turn basis (build_turn_board_arrays: H=1128, NO
    showdown raw/g/G fields) without any inherited method touching a missing field.
  * The leaf matrices are built once per DISTINCT leaf (the _leaf_cache contract).
  * With M1 = -M0^T the subgame is EXACTLY zero-sum: root v0+v1 == 0 (current AND
    average strategy), independent of solve quality.
  * CFR+ converges: internal (depth-limited) exploitability decreases.
  * _terminal dispatches leaf vs fold correctly.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.showdown_kernel import build_turn_board_arrays
from src.subgame.turn_tree import build_turn_tree, is_leaf
from src.subgame.turn_cfr import TurnCFR

_BOARD4 = ['CQ', 'SJ', 'H9', 'D5']
_POT = 20.0
_STACKS = (60.0, 60.0)
_B = 8                      # synthetic bucket count


def _setup(pot=_POT, stacks=_STACKS):
    """Build a TurnCFR with a synthetic zero-sum leaf (M1 = -M0^T). leaf_fn returns a
    pot/stacks-seeded random M0 so distinct leaves get distinct matrices; `calls`
    counts actual builds (to check caching)."""
    ba = build_turn_board_arrays(_BOARD4)
    H = ba['H']
    tb_idx = np.arange(H, dtype=np.int64) % _B          # synthetic partition
    calls = {'n': 0}

    def leaf_fn(p, s):
        calls['n'] += 1
        r = np.random.default_rng(int(round(p * 1000)) + 7919 * int(round(s[0] * 1000)))
        M0 = r.standard_normal((_B, _B)) * p
        return M0, -M0.T                                  # enforced zero-sum

    tree = build_turn_tree(pot, stacks)
    solver = TurnCFR(tree, ba, tb_idx, leaf_fn)
    return solver, tree, ba, calls


def _distinct_leaves(tree):
    seen = set()

    def walk(n):
        if n.terminal:
            if is_leaf(n):
                seen.add((round(n.final_pot, 6),
                          round(n.leaf_stacks[0], 6), round(n.leaf_stacks[1], 6)))
            return
        for c in n.children:
            walk(c)
    walk(tree.root)
    return seen


def test_runs_on_4card_basis():
    """The inherited CFR+ traversal + BR/exploitability must run on the 4-card ba
    (no raw/g/G) without KeyError -- the core subclassing-safety check."""
    solver, tree, ba, _ = _setup()
    assert 'raw' not in ba and 'g' not in ba, "turn ba must omit showdown fields"
    r0 = np.ones(ba['H'])
    r1 = np.ones(ba['H'])
    solver.run(r0, r1, iters=20)
    _ = solver.exploitability(r0, r1)          # BR walk over leaves + folds
    print("PASS test_runs_on_4card_basis")


def test_leaf_cache_one_build_per_distinct_leaf():
    solver, tree, ba, calls = _setup()
    r0 = np.ones(ba['H'])
    r1 = np.ones(ba['H'])
    solver.run(r0, r1, iters=30)               # many traversals, each hits every leaf
    distinct = _distinct_leaves(tree)
    assert len(solver._leaf_cache) == len(distinct), (len(solver._leaf_cache), len(distinct))
    assert calls['n'] == len(distinct), ("leaf_fn must be called once per distinct leaf",
                                         calls['n'], len(distinct))
    print(f"PASS test_leaf_cache_one_build_per_distinct_leaf "
          f"({len(distinct)} distinct leaves, {calls['n']} builds)")


def test_root_zero_sum():
    """M1 = -M0^T (synthetic) + zero-sum folds => root E0+E1 == 0 identically."""
    solver, tree, ba, _ = _setup()
    r0 = np.ones(ba['H'])
    r1 = np.ones(ba['H'])
    solver.run(r0, r1, iters=50)
    for name, sf in (('current', solver._strategy), ('average', solver.average_strategy)):
        v0, v1 = solver._eval(tree.root, r0, r1, sf)
        e = float((r0 * v0).sum() + (r1 * v1).sum())
        scale = max(abs(float((r0 * v0).sum())), 1.0)
        assert abs(e) / scale < 1e-9, (name, e, scale)
    print("PASS test_root_zero_sum (current + average)")


def test_converges():
    """Internal (depth-limited) exploitability should decrease with iterations."""
    solver, tree, ba, _ = _setup()
    r0 = np.ones(ba['H'])
    r1 = np.ones(ba['H'])
    solver.run(r0, r1, iters=50)
    e_early = solver.exploitability(r0, r1)
    solver.run(r0, r1, iters=450)
    e_late = solver.exploitability(r0, r1)
    assert e_late < e_early, (e_early, e_late)
    assert np.isfinite(e_late)
    print(f"PASS test_converges (expl {e_early:.3g} -> {e_late:.3g})")


def test_terminal_dispatch():
    """_terminal returns leaf values on a leaf node and fold-transfer values on a fold
    node; both finite and on the H-vector shape."""
    solver, tree, ba, _ = _setup()
    H = ba['H']
    r0 = np.ones(H)
    r1 = np.ones(H)
    leaf = next(n for n, _ in _walk(tree.root) if is_leaf(n))
    fold = next(n for n, p in _walk(tree.root) if n.terminal and n.folder is not None)
    lv0, lv1 = solver._terminal(leaf, r0, r1)
    fv0, fv1 = solver._terminal(fold, r0, r1)
    assert lv0.shape == (H,) and fv0.shape == (H,)
    assert np.all(np.isfinite(lv0)) and np.all(np.isfinite(fv0))
    print("PASS test_terminal_dispatch")


def _walk(node, path=()):
    yield node, path
    if not node.terminal:
        for c in node.children:
            yield from _walk(c, path + (1,))


TESTS = [
    test_runs_on_4card_basis,
    test_leaf_cache_one_build_per_distinct_leaf,
    test_root_zero_sum,
    test_converges,
    test_terminal_dispatch,
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
