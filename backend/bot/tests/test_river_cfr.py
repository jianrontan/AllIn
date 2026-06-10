# backend/bot/tests/test_river_cfr.py
"""
Validate the two-sided river CFR+ solver (src/subgame/river_cfr.py):

  * zero-sum invariant: under any strategy, sum(reach0*v0) + sum(reach1*v1) == 0
    at the root (a tight float-precision check that the value propagation +
    terminal kernel are consistent across both players).
  * forced check-check tree (no chips behind): the only line is check/check ->
    showdown, so the root value must equal the kernel showdown directly, and
    exploitability must be ~0 with no iterations (nothing to exploit).
  * convergence: on a real river board with betting, exploitability (the BR gap)
    drops sharply and becomes small relative to the pot as CFR+ iterates.
  * average strategy rows are valid distributions.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.subgame.river_tree import build_river_tree
from src.subgame.river_cfr import RiverCFR
from src.evaluation.showdown_kernel import build_board_arrays, showdown_measure
from src.abstractions.hand_evaluator import HandEvaluator
from src.abstractions.card_abstractions import CardAbstraction

_EVAL = HandEvaluator()
_CARDS = CardAbstraction()
_BOARD = ['HA', 'DK', 'CQ', 'SJ', 'H9']    # a dry-ish river


def _setup(pot_entry, stacks):
    ba = build_board_arrays(_BOARD, _EVAL, _CARDS)
    tree = build_river_tree(pot_entry=pot_entry, stacks=stacks)
    return tree, ba


def test_zero_sum_invariant():
    tree, ba = _setup(30.0, (100.0, 100.0))
    cfr = RiverCFR(tree, ba)
    rng = np.random.default_rng(0)
    r0 = rng.random(ba['H'])
    r1 = rng.random(ba['H'])
    # Run a few iterations so the strategy is non-trivial, then check the
    # read-only value propagation is exactly zero-sum.
    cfr.run(r0, r1, iters=20)
    v0, v1 = cfr.current_values(r0, r1)
    s = float((r0 * v0).sum()) + float((r1 * v1).sum())
    scale = 30.0 * float((r0.sum()) * (r1.sum())) ** 0.5
    assert abs(s) < 1e-6 * max(1.0, scale), (s, scale)
    print(f"PASS test_zero_sum_invariant (residual={s:.3e})")


def test_forced_checkdown_matches_kernel():
    """Stacks=0 behind -> the only line is check/check -> showdown. Root value
    must equal the kernel showdown over the entry pot; exploitability ~0."""
    tree, ba = _setup(30.0, (0.0, 0.0))
    cfr = RiverCFR(tree, ba)
    H = ba['H']
    r0 = np.ones(H)
    r1 = np.ones(H)
    v0, v1 = cfr.current_values(r0, r1)
    # No river chips: final_pot = 30, each contributed 15.
    want0 = showdown_measure(ba, r1, 30.0, 15.0)
    want1 = showdown_measure(ba, r0, 30.0, 15.0)
    assert np.allclose(v0, want0) and np.allclose(v1, want1)
    expl = cfr.exploitability(r0, r1)
    assert abs(expl) < 1e-9, expl
    print(f"PASS test_forced_checkdown_matches_kernel (expl={expl:.2e})")


def test_convergence_reduces_exploitability():
    tree, ba = _setup(30.0, (100.0, 100.0))
    H = ba['H']
    # Asymmetric, non-uniform ranges so the spot is non-trivial.
    rng = np.random.default_rng(7)
    r0 = rng.random(H) + 0.1
    r1 = rng.random(H) + 0.1

    cfr = RiverCFR(tree, ba)
    expl_start = cfr.exploitability(r0, r1)        # uniform average strategy
    t0 = time.time()
    cfr.run(r0, r1, iters=300)
    dt = time.time() - t0
    expl_end = cfr.exploitability(r0, r1)

    assert expl_end < expl_start, (expl_start, expl_end)
    # Should converge to a small fraction of the pot (chips per dealt pair).
    assert expl_end < 0.02 * 30.0, f"exploitability still high: {expl_end}"
    assert expl_end < expl_start / 5.0, (expl_start, expl_end)
    print(f"PASS test_convergence_reduces_exploitability "
          f"({expl_start:.3f} -> {expl_end:.4f} chips/pair in {dt:.1f}s)")


def test_average_strategy_is_distribution():
    tree, ba = _setup(30.0, (80.0, 80.0))
    cfr = RiverCFR(tree, ba)
    H = ba['H']
    cfr.run(np.ones(H), np.ones(H), iters=50)
    for n in tree.decision_nodes:
        avg = cfr.average_strategy(n.node_id)
        assert avg.shape == (H, len(n.actions))
        assert np.allclose(avg.sum(axis=1), 1.0), n.node_id
        assert (avg >= -1e-12).all()
    print("PASS test_average_strategy_is_distribution")


def test_incremental_run_matches_single():
    """The Linear-CFR clock persists across run() calls, so run(100)+run(100)
    must be bit-identical to run(200). Step 5's early-stop relies on this."""
    tree, ba = _setup(30.0, (100.0, 100.0))
    H = ba['H']
    rng = np.random.default_rng(3)
    r0 = rng.random(H) + 0.1
    r1 = rng.random(H) + 0.1

    one = RiverCFR(tree, ba)
    one.run(r0, r1, iters=200)

    split = RiverCFR(tree, ba)
    split.run(r0, r1, iters=100)
    split.run(r0, r1, iters=100)

    for n in tree.decision_nodes:
        a = one.average_strategy(n.node_id)
        b = split.average_strategy(n.node_id)
        assert np.array_equal(a, b), f"node {n.node_id} differs between split/single run"
    print("PASS test_incremental_run_matches_single")


def test_hand_action_evs_none_on_zero_villain_mass():
    """M1: with ~no compatible villain mass for the bot's exact hand the per-action chip
    EVs are undefined (tiny/tiny) -> hand_action_evs returns None, so the EV-gate keeps
    the blueprint baseline instead of comparing garbage-scaled values."""
    from src.subgame.solve_control import hand_action_evs
    tree, ba = _setup(30.0, (100.0, 100.0))
    cfr = RiverCFR(tree, ba)
    H = ba['H']
    node = tree.root
    if node.player == 0:
        r0, r1 = np.ones(H), np.zeros(H)       # villain (P1) has zero mass
    else:
        r0, r1 = np.zeros(H), np.ones(H)       # villain (P0) has zero mass
    assert hand_action_evs(cfr, node, 0, r0, r1) is None, "expected None on zero villain mass"
    out = hand_action_evs(cfr, node, 0, np.ones(H), np.ones(H))   # villain mass present
    assert out is not None and len(out) == len(node.actions), out
    print("PASS test_hand_action_evs_none_on_zero_villain_mass")


def test_average_strategy_uniform_on_unreached_row():
    """H2 precondition: average_strategy returns UNIFORM 1/A for a hand whose strat_sum row
    never moved (zero reach into the node). solve_for_action's node-reach guard exists to
    avoid reading off that uniform as the solve's recommendation."""
    tree, ba = _setup(30.0, (100.0, 100.0))
    cfr = RiverCFR(tree, ba)
    H = ba['H']
    r_self = np.ones(H)
    r_self[0] = 0.0                            # zero the acting seat's reach for hand 0
    if tree.root.player == 0:
        r0, r1 = r_self, np.ones(H)
    else:
        r0, r1 = np.ones(H), r_self
    cfr.run(r0, r1, iters=20)
    strat = cfr.average_strategy(tree.root.node_id)
    A = len(tree.root.actions)
    assert np.allclose(strat[0], 1.0 / A), f"unreached row should be uniform, got {strat[0]}"
    assert not np.allclose(strat[1], 1.0 / A), "a reached row should NOT be uniform after 20 iters"
    print("PASS test_average_strategy_uniform_on_unreached_row")


def test_blueprint_cfv_is_zero_sum():
    """Phase 5a piece 1: blueprint_cfv returns each seat's per-hand counterfactual
    value at the river-entry root with BOTH seats on the blueprint. The two seats'
    CFVs must net to zero against their reaches (a poker game is zero-sum), which
    also checks the projection -> _eval value walk is consistent. raw_strategy=None
    (untrained) -> uniform tree distributions, a valid strategy for the check."""
    from src.subgame.blueprint_projection import blueprint_cfv
    tree, ba = _setup(30.0, (100.0, 100.0))
    H = ba['H']
    rng = np.random.default_rng(11)
    r0 = rng.random(H) + 0.1
    r1 = rng.random(H) + 0.1
    raw = lambda key: None                          # untrained -> uniform projection
    v0 = blueprint_cfv(tree, ba, raw, r0, r1, villain_seat=0)
    v1 = blueprint_cfv(tree, ba, raw, r0, r1, villain_seat=1)
    assert v0.shape == (H,) and v1.shape == (H,), (v0.shape, v1.shape)
    s = float((r0 * v0).sum()) + float((r1 * v1).sum())
    scale = 30.0 * (r0.sum() * r1.sum()) ** 0.5
    assert abs(s) < 1e-6 * max(1.0, scale), (s, scale)
    print(f"PASS test_blueprint_cfv_is_zero_sum (residual={s:.3e})")


def test_gadget_no_more_exploitable_than_blueprint():
    """Phase 5a safety property: the re-solve gadget yields a HERO strategy that is
    no-more-exploitable than the blueprint it is anchored to. Measure the villain's
    best-response value (over its FULL range) vs (a) the blueprint hero strategy and
    (b) the gadget-solved hero strategy; (b) must be <= (a). The gadget caps the
    villain's BR at its blueprint opt-out CFV per hand, which is itself <= the
    villain's BR vs the blueprint -- so the inequality holds at convergence. Here the
    'blueprint' is the untrained (uniform) projection: a valid, exploitable baseline."""
    from src.subgame.blueprint_projection import (
        blueprint_cfv, blueprint_strategy_on_tree)
    tree, ba = _setup(30.0, (100.0, 100.0))
    H = ba['H']
    rng = np.random.default_rng(5)
    r0 = rng.random(H) + 0.1
    r1 = rng.random(H) + 0.1
    raw = lambda key: None                          # untrained -> uniform blueprint
    villain_seat = 1
    hero_reach, villain_reach = r0, r1

    bp_strat = blueprint_strategy_on_tree(tree, ba, raw)
    optout = blueprint_cfv(tree, ba, raw, r0, r1, villain_seat=villain_seat)

    base = RiverCFR(tree, ba)
    bp_br = base._br_value(tree.root, villain_seat, hero_reach, lambda nid: bp_strat[nid])
    bp_expl = float((villain_reach * bp_br).sum())

    g = RiverCFR(tree, ba)
    g.run_gadget(hero_reach, villain_reach, optout, villain_seat, iters=400)
    g_br = g._br_value(tree.root, villain_seat, hero_reach, g.average_strategy)
    g_expl = float((villain_reach * g_br).sum())

    assert g_expl <= bp_expl + 1e-6 * (abs(bp_expl) + 1.0), (bp_expl, g_expl)
    print(f"PASS test_gadget_no_more_exploitable_than_blueprint "
          f"(villain BR: blueprint={bp_expl:.2f} -> gadget={g_expl:.2f} chips)")


TESTS = [
    test_zero_sum_invariant,
    test_blueprint_cfv_is_zero_sum,
    test_gadget_no_more_exploitable_than_blueprint,
    test_incremental_run_matches_single,
    test_forced_checkdown_matches_kernel,
    test_convergence_reduces_exploitability,
    test_average_strategy_is_distribution,
    test_hand_action_evs_none_on_zero_villain_mass,
    test_average_strategy_uniform_on_unreached_row,
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
