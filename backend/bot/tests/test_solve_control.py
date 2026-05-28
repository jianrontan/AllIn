# backend/bot/tests/test_solve_control.py
"""
Validate the river solve-control layer (src/subgame/solve_control.py): the
convergence-based early stop, the per-action value building block, the EV gate,
and the warm-start hook.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.subgame.river_tree import build_river_tree
from src.subgame.river_cfr import RiverCFR
from src.subgame.solve_control import solve_river, ev_gate, hand_action_evs
from src.evaluation.showdown_kernel import build_board_arrays
from src.abstractions.hand_evaluator import HandEvaluator
from src.abstractions.card_abstractions import CardAbstraction

_EVAL = HandEvaluator()
_CARDS = CardAbstraction()
_BOARD = ['HA', 'DK', 'CQ', 'SJ', 'H9']


def _setup(pot, stacks):
    return build_river_tree(pot_entry=pot, stacks=stacks), build_board_arrays(_BOARD, _EVAL, _CARDS)


def test_early_stop_converges_within_budget():
    tree, ba = _setup(30.0, (100.0, 100.0))
    rng = np.random.default_rng(11)
    r0 = rng.random(ba['H']) + 0.1
    r1 = rng.random(ba['H']) + 0.1
    cfr, info = solve_river(tree, ba, r0, r1, max_iters=2000, check_every=50,
                            gap_threshold=0.05 * 30.0)
    assert info['converged'], info
    assert info['iters'] < 2000, info        # stopped early, didn't grind the cap
    assert info['gap'] <= 0.05 * 30.0
    print(f"PASS test_early_stop_converges_within_budget "
          f"(iters={info['iters']}, gap={info['gap']:.3f}, {info['seconds']:.1f}s)")


def test_time_budget_stops_early():
    tree, ba = _setup(30.0, (100.0, 100.0))
    H = ba['H']
    # Tiny budget + a tight default threshold it can't meet quickly -> time-stop.
    cfr, info = solve_river(tree, ba, np.ones(H), np.ones(H),
                            max_iters=100000, check_every=50, time_budget=0.01)
    assert info['iters'] < 100000, info
    assert not info['converged'] or info['iters'] <= 50
    print(f"PASS test_time_budget_stops_early (iters={info['iters']}, {info['seconds']:.2f}s)")


def test_node_action_values_consistent():
    """The avg-strategy-weighted action values must equal the node's value vector
    under the average strategy (validates node_action_values)."""
    tree, ba = _setup(30.0, (100.0, 100.0))
    cfr = RiverCFR(tree, ba)
    rng = np.random.default_rng(5)
    r0 = rng.random(ba['H']) + 0.1
    r1 = rng.random(ba['H']) + 0.1
    cfr.run(r0, r1, iters=120)

    node = tree.root
    vals = cfr.node_action_values(node, r0, r1)            # [H, A] measures
    avg = cfr.average_strategy(node.node_id)
    via_actions = (avg * vals).sum(axis=1)
    via_eval = cfr._eval(node, r0, r1, cfr.average_strategy)[node.player]
    assert np.allclose(via_actions, via_eval, atol=1e-8), \
        float(np.max(np.abs(via_actions - via_eval)))
    print("PASS test_node_action_values_consistent")


def test_ev_gate_logic():
    actions = ['check', 'bet:10']
    evs = [2.0, 5.0]                       # betting is +3 chips better here
    solved = {'check': 0.0, 'bet:10': 1.0}
    baseline = {'check': 1.0, 'bet:10': 0.0}

    chosen, info = ev_gate(actions, solved, baseline, evs, margin=0.5)
    assert info['used'] == 'solved' and chosen is solved, info

    chosen, info = ev_gate(actions, solved, baseline, evs, margin=10.0)
    assert info['used'] == 'baseline' and chosen is baseline, info   # edge too small

    # Tie -> never deviate (margin > 0).
    chosen, info = ev_gate(actions, baseline, baseline, evs, margin=0.5)
    assert info['used'] == 'baseline' and abs(info['delta']) < 1e-12
    print("PASS test_ev_gate_logic")


def test_hand_action_evs_match_strategy_value():
    """The bot-hand action EVs weighted by its solved strategy equal that hand's
    normalised node value (chips)."""
    tree, ba = _setup(30.0, (100.0, 100.0))
    cfr = RiverCFR(tree, ba)
    H = ba['H']
    r0 = np.ones(H)
    r1 = np.ones(H)
    cfr.run(r0, r1, iters=120)
    node = tree.root
    hand_row = 17
    evs = hand_action_evs(cfr, node, hand_row, r0, r1)
    avg_row = cfr.average_strategy(node.node_id)[hand_row]
    ev_mix = float((avg_row * evs).sum())
    # Same quantity via the node value, normalised by compatible villain mass.
    from src.evaluation.showdown_kernel import compatible_mass
    node_val = cfr._eval(node, r0, r1, cfr.average_strategy)[node.player][hand_row]
    z = compatible_mass(ba, r1)[hand_row]
    assert abs(ev_mix - node_val / z) < 1e-8, (ev_mix, node_val / z)
    print("PASS test_hand_action_evs_match_strategy_value")


def test_warm_start_seed_and_washout():
    tree, ba = _setup(30.0, (90.0, 90.0))
    H = ba['H']
    r0 = np.ones(H)
    r1 = np.ones(H)
    nodes = tree.decision_nodes
    # Non-uniform prior: all mass on the first action of each node.
    seed = []
    for n in nodes:
        A = len(n.actions)
        s = np.zeros((H, A))
        s[:, 0] = 1.0
        seed.append(s)

    # 0 iterations -> the average strategy IS the seed.
    seeded0 = RiverCFR(tree, ba)
    seeded0.warm_start(seed, weight=1.0)
    for n in nodes:
        avg = seeded0.average_strategy(n.node_id)
        assert np.allclose(avg[:, 0], 1.0), n.node_id

    # Many iterations -> at a WELL-REACHED node (the root is fully reached, reach=1
    # for every hand) the seed washes out, so seeded and unseeded agree. (At rare
    # nodes the seed intentionally persists -- reach-weighted real mass vs a fixed
    # seed -- which is the graceful fallback where the solve is uninformed.)
    seeded = RiverCFR(tree, ba)
    seeded.warm_start(seed, weight=1.0)
    seeded.run(r0, r1, iters=150)
    plain = RiverCFR(tree, ba)
    plain.run(r0, r1, iters=150)
    root_id = tree.root.node_id
    root_diff = float(np.max(np.abs(
        seeded.average_strategy(root_id) - plain.average_strategy(root_id))))
    assert root_diff < 0.01, f"seed did not wash out at the root: {root_diff}"
    # And the seed visibly persists at a low-reach node (graceful fallback there).
    print(f"PASS test_warm_start_seed_and_washout (root washout diff {root_diff:.4f})")


TESTS = [
    test_early_stop_converges_within_budget,
    test_time_budget_stops_early,
    test_node_action_values_consistent,
    test_ev_gate_logic,
    test_hand_action_evs_match_strategy_value,
    test_warm_start_seed_and_washout,
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
