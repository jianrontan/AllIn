# backend/bot/tests/test_range_inputs.py
"""
Validate the river solver's range inputs (src/subgame/range_inputs.py): tracker
-> board-basis projection, the confidence/temper widening blend, and reading the
bot's solved action back out for its actual hand. Plus an end-to-end smoke test:
ranges -> RiverCFR solve -> action.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.subgame.range_inputs import (
    project_tracker, blend_villain, temper, read_action_strategy,
    hand_index_map, hand_row)
from src.subgame.river_tree import build_river_tree
from src.subgame.river_cfr import RiverCFR
from src.evaluation.showdown_kernel import build_board_arrays
from src.game.range_tracker import RangeTracker
from src.abstractions.hand_evaluator import HandEvaluator
from src.abstractions.card_abstractions import CardAbstraction

_EVAL = HandEvaluator()
_CARDS = CardAbstraction()
# Board chosen so the bot's cards (HA, DK) are NOT on it.
_BOARD = ['CQ', 'SJ', 'H9', 'D5', 'C2']
_BOT = ('HA', 'DK')


def _ba():
    return build_board_arrays(_BOARD, _EVAL, _CARDS)


def test_project_excludes_bot_and_board_cards():
    ba = _ba()
    idx = hand_index_map(ba)
    tr = RangeTracker(_BOT, _CARDS)     # villain tracker: bot's cards removed
    tr.reveal(_BOARD)
    proj = project_tracker(tr, ba, idx)

    # A villain hand using neither a board card nor a bot card is present.
    keep = ('SA', 'CA')
    rk = hand_row(ba, keep, idx)
    assert rk is not None and proj[rk] > 0.0

    # Every hand using a bot card has zero villain weight (villain can't hold it).
    for i, h in enumerate(ba['hands']):
        if _BOT[0] in h or _BOT[1] in h:
            assert proj[i] == 0.0, h
    # No board card appears in the basis at all (sanity).
    assert all(c not in h for h in ba['hands'] for c in _BOARD)
    print("PASS test_project_excludes_bot_and_board_cards")


def test_blend_confidence_endpoints():
    ba = _ba()
    tr = RangeTracker(_BOT, _CARDS)
    tr.reveal(_BOARD)
    # Make the belief non-uniform so flattening is observable.
    tr.w[:] = 0.0
    live = [i for i, h in enumerate(tr.hands)
            if all(c not in _BOARD for c in h)]
    tr.w[live[0]] = 10.0
    tr.w[live[1]] = 1.0
    for i in live[2:12]:
        tr.w[i] = 1.0
    tracked = project_tracker(tr, ba)

    # confidence = 1 -> exactly the normalised tracked belief.
    hi = blend_villain(tracked, confidence=1.0)
    assert np.allclose(hi, tracked / tracked.sum())

    # confidence = 0, beta = 0 -> uniform over the support (max flattening).
    lo = blend_villain(tracked, confidence=0.0, beta=0.0)
    support = tracked > 0
    assert np.allclose(lo[support], 1.0 / support.sum())
    assert np.allclose(lo[~support], 0.0)        # zeros preserved (card removal)
    assert abs(lo.sum() - 1.0) < 1e-9

    # Both endpoints are valid distributions.
    assert abs(hi.sum() - 1.0) < 1e-9
    print("PASS test_blend_confidence_endpoints")


def test_temper_monotonic_flattening():
    """beta between 0 and 1 softens peaks: the ratio of a strong to a weak hand
    shrinks toward 1 as beta -> 0, and equals the raw ratio at beta = 1."""
    r = np.array([0.0, 9.0, 1.0, 1.0])
    raw_ratio = 9.0 / 1.0
    t_half = temper(r, 0.5)
    half_ratio = t_half[1] / t_half[2]
    t_one = temper(r, 1.0)
    assert np.allclose(t_one, r / r.sum())              # beta=1 -> identity
    assert abs(half_ratio - np.sqrt(raw_ratio)) < 1e-9  # sqrt softening
    assert 1.0 < half_ratio < raw_ratio
    assert t_half[0] == 0.0                              # zero stays zero
    print("PASS test_temper_monotonic_flattening")


def test_read_action_strategy_valid():
    ba = _ba()
    tree = build_river_tree(pot_entry=30.0, stacks=(80.0, 80.0))
    cfr = RiverCFR(tree, ba)
    cfr.run(np.ones(ba['H']), np.ones(ba['H']), iters=30)
    dist = read_action_strategy(cfr, tree.root, _BOT, ba)
    assert set(dist.keys()) == set(tree.root.actions)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert all(p >= -1e-12 for p in dist.values())
    print(f"PASS test_read_action_strategy_valid (root dist: "
          f"{ {k: round(v,2) for k,v in dist.items()} })")


def test_end_to_end_ranges_to_action():
    """Build BOTH ranges as RangeTrackers (villain = bot's-cards-removed; hero =
    spans all hands), project, blend the villain, solve, read off the bot action."""
    ba = _ba()
    idx = hand_index_map(ba)

    # Villain range: a tracker over the opponent (bot's cards removed), board revealed.
    villain_tr = RangeTracker(_BOT, _CARDS)
    villain_tr.reveal(_BOARD)
    villain_tr.confidence = 0.4                      # partially off-model -> some widening
    villain_reach = blend_villain(project_tracker(villain_tr, ba, idx),
                                  villain_tr.confidence)

    # Hero (bot) range: a tracker spanning all hands (hero_hole=()), board revealed.
    # In real use it would have observed the BOT's blueprint actions; here uniform
    # blueprint reach is enough to exercise the full path.
    hero_tr = RangeTracker((), _CARDS)
    hero_tr.reveal(_BOARD)
    hero_reach = project_tracker(hero_tr, ba, idx)

    assert villain_reach.sum() > 0 and hero_reach.sum() > 0
    # The bot here is SEAT 0 (its range is reach0=hero_reach). The tree root is the
    # OOP=seat-1 node (the opponent), where the bot's hand has ZERO reach -- so its
    # strategy-sum never moves there and reading the root would return a meaningless
    # uniform fallback. The bot's real decision node is a CHILD of the root, after
    # the opponent acts. This is the seat/path read-off rule step 6 must honor: the
    # node must match the bot's seat AND the realized betting path.
    tree = build_river_tree(pot_entry=24.0, stacks=(90.0, 90.0))
    cfr = RiverCFR(tree, ba)
    cfr.run(reach0=hero_reach, reach1=villain_reach, iters=300)

    assert tree.root.player == 1, "root is the OOP/seat-1 (opponent) node"
    bot_node = tree.root.children[tree.root.actions.index('check')]  # OOP checks -> IP acts
    assert bot_node.player == 0, "the bot's (seat 0) first decision is after OOP checks"

    dist = read_action_strategy(cfr, bot_node, _BOT, ba, idx)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    # Reading at the right node gives a real (non-uniform) strategy, not the 1/n
    # fallback the wrong node would have produced.
    n = len(dist)
    assert not np.allclose(list(dist.values()), 1.0 / n), \
        f"strategy is uniform -- likely read at the wrong node: {dist}"
    expl = cfr.exploitability(hero_reach, villain_reach)
    assert expl >= -1e-9
    print(f"PASS test_end_to_end_ranges_to_action "
          f"(seat-0 node dist { {k: round(v,2) for k,v in dist.items()} }, expl={expl:.3f})")


TESTS = [
    test_project_excludes_bot_and_board_cards,
    test_blend_confidence_endpoints,
    test_temper_monotonic_flattening,
    test_read_action_strategy_valid,
    test_end_to_end_ranges_to_action,
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
