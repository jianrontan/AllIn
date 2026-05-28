# backend/bot/tests/test_river_subgame_solver.py
"""
Validate the RiverSubgameSolver assembly (step 6a): the end-to-end
ranges->tree->solve->read-off path produces a valid action for the bot's hand at
its actual decision node (both seats), and decide() falls back to the blueprint
off-river / when solver inputs are absent.

Runs against the active blueprint DB (read-only); skips cleanly if none exists.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.subgame.river_subgame_solver import RiverSubgameSolver, blueprint_to_tree_dist
from src.subgame.river_tree import build_river_tree
from src.game.range_tracker import RangeTracker
from src.abstractions.card_abstractions import CardAbstraction

_CARDS = CardAbstraction()
_BOARD = ['CQ', 'SJ', 'H9', 'D5', 'C2']
_HOLE = ['HA', 'DK']


def _blueprint_db():
    try:
        from src.config import resolve_blueprint_path
        from src.storage.blueprint_db import BlueprintDB
        path = resolve_blueprint_path()
        return BlueprintDB(path, read_only=True)
    except Exception as e:
        print(f"  (no blueprint DB available: {e})")
        return None


def _trackers():
    villain = RangeTracker(_HOLE, _CARDS)   # opponent: bot's cards removed
    villain.reveal(_BOARD)
    hero = RangeTracker((), _CARDS)         # bot's range: spans all hands
    hero.reveal(_BOARD)
    return villain, hero


def _solver(db):
    # Small iteration budget keeps the test fast; correctness, not convergence.
    return RiverSubgameSolver(db, max_iters=80, check_every=40, time_budget=30.0)


def test_solve_for_action_oop():
    db = _blueprint_db()
    if db is None:
        print("SKIP test_solve_for_action_oop (no blueprint)")
        return
    villain, hero = _trackers()
    solver = _solver(db)
    # Bot is OOP (seat 1): with an empty river path, the root IS the bot's node.
    dist, node, info = solver.solve_for_action(
        board=_BOARD, pot_entry=24.0, stacks=(88.0, 88.0), bot_seat=1, hole=_HOLE,
        villain_tracker=villain, hero_tracker=hero, confidence=1.0, river_path=[])
    db.close()
    assert node.player == 1
    assert set(dist.keys()) == set(node.actions)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert all(p >= -1e-12 for p in dist.values())
    print(f"PASS test_solve_for_action_oop (iters={info['iters']}, gap={info['gap']:.3f}, "
          f"dist={ {k: round(v,2) for k,v in dist.items()} })")


def test_solve_for_action_ip_after_villain_checks():
    db = _blueprint_db()
    if db is None:
        print("SKIP test_solve_for_action_ip_after_villain_checks (no blueprint)")
        return
    villain, hero = _trackers()
    solver = _solver(db)
    # Bot is IP (seat 0): its decision is AFTER OOP acts. Path = OOP checks.
    dist, node, info = solver.solve_for_action(
        board=_BOARD, pot_entry=24.0, stacks=(88.0, 88.0), bot_seat=0, hole=_HOLE,
        villain_tracker=villain, hero_tracker=hero, confidence=1.0,
        river_path=['check'])
    db.close()
    assert node.player == 0, "after OOP checks it's the IP (seat-0) bot's turn"
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    # The bot's hand should have a real (non-uniform) strategy at its own node.
    n = len(dist)
    assert not np.allclose(list(dist.values()), 1.0 / n), dist
    print(f"PASS test_solve_for_action_ip_after_villain_checks "
          f"(node {node.node_id}, dist={ {k: round(v,2) for k,v in dist.items()} })")


def test_navigate_snaps_offmenu_bet():
    db = _blueprint_db()
    if db is None:
        print("SKIP test_navigate_snaps_offmenu_bet (no blueprint)")
        return
    villain, hero = _trackers()
    solver = _solver(db)
    # Villain (OOP) makes an off-menu bet of 13 chips; the bot (IP) faces it. The
    # navigator should snap 13 to the nearest sized edge and land on a seat-0 node
    # that is facing a bet (fold/call/... available).
    dist, node, info = solver.solve_for_action(
        board=_BOARD, pot_entry=24.0, stacks=(88.0, 88.0), bot_seat=0, hole=_HOLE,
        villain_tracker=villain, hero_tracker=hero, confidence=1.0,
        river_path=[('bet', 13.0)])
    db.close()
    assert node.player == 0 and node.to_call > 0, "bot faces the (snapped) bet"
    assert 'fold' in node.actions and 'call' in node.actions
    print(f"PASS test_navigate_snaps_offmenu_bet (to_call={node.to_call:.1f})")


def test_decide_falls_back_off_river():
    db = _blueprint_db()
    if db is None:
        print("SKIP test_decide_falls_back_off_river (no blueprint)")
        return
    solver = _solver(db)
    legal = ['check', 'bet_small', 'bet_medium', 'bet_large']
    # Flop -> must delegate to the blueprint (returns a legal action, no solve).
    a = solver.decide('pf_9_5_ip_flop_', legal, {'street': 'flop'})
    assert a in legal
    # River but missing the solver inputs -> also falls back.
    a2 = solver.decide('pf_9_5_ip_river_', legal, {'street': 'river', 'community': _BOARD})
    assert a2 in legal
    db.close()
    print("PASS test_decide_falls_back_off_river")


def test_blueprint_to_tree_dist_mapping():
    """Blueprint actions redistribute onto the tree menu by nearest size fraction;
    no blueprint DB needed."""
    tree = build_river_tree(pot_entry=24.0, stacks=(88.0, 88.0))
    root = tree.root                                   # OOP, to_call=0: check + bets + allin
    # tree root bets: 0.5/0.75/1.0/1.5 * 24 = 12/18/24/36
    bp = {'check': 0.5, 'bet_small': 0.3, 'bet_large': 0.2}   # fracs 0.33 / 1.0
    out = blueprint_to_tree_dist(bp, root)
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert abs(out['check'] - 0.5) < 1e-9
    assert abs(out['bet:12'] - 0.3) < 1e-9             # 0.33 -> nearest 0.5 -> bet:12
    assert abs(out['bet:24'] - 0.2) < 1e-9             # 1.0 -> bet:24

    # A facing-bet node: fold/call direct, raise mapped to nearest raise edge.
    ip = root.children[root.actions.index('bet:24')]   # IP faces a pot bet
    assert ip.to_call > 0
    bp2 = {'fold': 0.4, 'call': 0.4, 'raise_medium': 0.2}
    out2 = blueprint_to_tree_dist(bp2, ip)
    assert abs(out2['fold'] - 0.4) < 1e-9 and abs(out2['call'] - 0.4) < 1e-9
    raise_mass = sum(v for a, v in out2.items() if a.startswith('raise:') or a == 'allin')
    assert abs(raise_mass - 0.2) < 1e-9
    print("PASS test_blueprint_to_tree_dist_mapping")


def test_blueprint_to_tree_dist_reraise_node():
    """Re-raise node where the dropped-sc offset STRADDLES a nearest-neighbour
    boundary, so the fixed and buggy formulas give DIFFERENT answers (the prior
    test was vacuous -- both formulas agreed on it). Node: sc=(54,34), actor
    (player 1) has 34 in, faces 20, pot 80 -> offset delta = 34/100 = 0.34.

      raise edge:   raise:36  raise:61  raise:86  raise:136
      fixed frac :   0.50      0.75      1.00      1.50
      buggy frac :   0.16      0.41      0.66      1.16    (= fixed - 0.34)

    A blueprint raise_medium (frac 0.66) maps to raise:61 (nearest 0.75) under the
    FIX, but to raise:86 (exact 0.66) under the BUG -- so this discriminates."""
    import types
    node = types.SimpleNamespace(
        player=1, sc=(54.0, 34.0), pot_mid=80.0, to_call=20.0,
        actions=['fold', 'call', 'raise:36', 'raise:61', 'raise:86', 'raise:136', 'allin'])
    out = blueprint_to_tree_dist({'raise_medium': 1.0}, node)
    assert abs(out['raise:61'] - 1.0) < 1e-9, ("fix->raise:61, bug->raise:86", out)
    print("PASS test_blueprint_to_tree_dist_reraise_node (discriminates fix vs bug)")


def test_hero_zero_reach_falls_back():
    """A bot hand with ~zero hero reach must raise (so decide() falls back) rather
    than silently read off a uniform strategy."""
    db = _blueprint_db()
    if db is None:
        print("SKIP test_hero_zero_reach_falls_back (no blueprint)")
        return
    villain, hero = _trackers()
    hi = next(i for i, h in enumerate(hero.hands) if set(h) == set(_HOLE))
    hero.w[hi] = 0.0                                   # bot's actual hand: zero reach
    solver = _solver(db)
    try:
        solver.solve_for_action(
            board=_BOARD, pot_entry=24.0, stacks=(88.0, 88.0), bot_seat=1, hole=_HOLE,
            villain_tracker=villain, hero_tracker=hero, confidence=1.0, river_path=[])
        raise AssertionError("expected zero hero reach to raise")
    except ValueError as e:
        assert 'reach' in str(e), e
    db.close()
    print("PASS test_hero_zero_reach_falls_back")


def test_allin_emits_shove_not_check_at_deep_stack():
    """At a deep-stack node the engine offers no discrete 'allin' (sized bets are
    affordable), but the solver may still choose to shove. The mapping must emit a
    full-stack custom shove, NOT silently fall through to 'check'."""
    import types
    solver = RiverSubgameSolver(None)        # no DB needed for action mapping
    legal = ['check', 'bet_small', 'bet_medium', 'bet_large']   # NB: no 'allin'
    node = types.SimpleNamespace(to_call=0.0, sc=(0.0, 0.0))
    spec = {'bot_seat': 1, 'stacks': (88.0, 88.0)}
    out = solver._pick_engine_action({'check': 0.0, 'allin': 1.0}, legal, spec, node)
    assert out != 'check', "a chosen shove must not degrade to check"
    assert out.startswith('bet_custom_'), out          # no bet to call -> bet shove
    assert abs(float(out.rsplit('_', 1)[1]) - 88.0) < 1e-9, out   # full stack
    # Facing a bet -> raise shove.
    node2 = types.SimpleNamespace(to_call=20.0, sc=(40.0, 10.0))
    out2 = solver._pick_engine_action(
        {'allin': 1.0}, ['fold', 'call', 'raise_small'], spec, node2)
    assert out2.startswith('raise_custom_'), out2
    print("PASS test_allin_emits_shove_not_check_at_deep_stack")


def test_decide_river_emits_action_with_ev_gate():
    """The full decide() path on a river state: solve + EV gate + emit. The action
    is a valid engine action -- check/call/fold/allin, an abstract size, or an
    exact custom_ size (the solver's edge)."""
    db = _blueprint_db()
    if db is None:
        print("SKIP test_decide_river_emits_action_with_ev_gate (no blueprint)")
        return
    villain, hero = _trackers()
    solver = _solver(db)
    legal = ['check', 'bet_small', 'bet_medium', 'bet_large', 'allin']
    ps = {
        'street': 'river', 'community': _BOARD, 'hole_cards': _HOLE,
        'riverEntryPot': 24.0, 'riverEntryStacks': (88.0, 88.0), 'botSeat': 1,
        'opp_range': villain, 'hero_range': hero, 'riverPath': [],
    }
    a = solver.decide('pf_9_5_oop_river_', legal, ps)
    db.close()
    ok = (a in legal) or a.startswith('bet_custom_') or a.startswith('raise_custom_')
    assert ok, f"unexpected action: {a}"
    print(f"PASS test_decide_river_emits_action_with_ev_gate (action={a})")


TESTS = [
    test_blueprint_to_tree_dist_mapping,
    test_blueprint_to_tree_dist_reraise_node,
    test_hero_zero_reach_falls_back,
    test_allin_emits_shove_not_check_at_deep_stack,
    test_decide_river_emits_action_with_ev_gate,
    test_solve_for_action_oop,
    test_solve_for_action_ip_after_villain_checks,
    test_navigate_snaps_offmenu_bet,
    test_decide_falls_back_off_river,
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
