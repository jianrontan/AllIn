# backend/bot/tests/test_turn_subgame_solver.py
"""
Validate the TurnSubgameSolver assembly (M4): the end-to-end ranges -> turn tree ->
leaf matrices -> TurnCFR -> read-off path produces a valid action for the bot's hand
at its actual turn decision node, the SPR gate skips deep/small-pot turns, and decide()
routes non-turn streets to the inherited (river/blueprint) path.

Runs against the active blueprint DB (read-only); skips cleanly if none exists. Uses a
TINY fidelity (n=16, 3 rivers, 50 iters) -- correctness/plumbing, not solve quality
(the quality gate is the offline LBR run).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.subgame.turn_subgame_solver import TurnSubgameSolver
from src.game.range_tracker import RangeTracker
from src.abstractions.card_abstractions import CardAbstraction

_CARDS = CardAbstraction()
_BOARD4 = ['CQ', 'SJ', 'H9', 'D5']        # 4-card turn board
_HOLE = ['HA', 'DK']


def _blueprint_db():
    try:
        from src.config import resolve_blueprint_path
        from src.storage.blueprint_db import BlueprintDB
        return BlueprintDB(resolve_blueprint_path(), read_only=True)
    except Exception as e:
        print(f"  (no blueprint DB available: {e})")
        return None


def _trackers():
    villain = RangeTracker(_HOLE, _CARDS)
    villain.reveal(_BOARD4)
    hero = RangeTracker((), _CARDS)
    hero.reveal(_BOARD4)
    return villain, hero


def _solver(db):
    return TurnSubgameSolver(db, n_buckets=16, leaf_rivers=3, turn_max_iters=50)


def test_solve_turn_for_action_oop():
    db = _blueprint_db()
    if db is None:
        print("SKIP test_solve_turn_for_action_oop (no blueprint)")
        return
    s = _solver(db)
    villain, hero = _trackers()
    dist, node, info = s.solve_turn_for_action(
        board=_BOARD4, pot_entry=24.0, stacks=(88.0, 88.0), bot_seat=1, hole=_HOLE,
        villain_tracker=villain, hero_tracker=hero, confidence=1.0, turn_path=[])
    db.close()
    assert node.player == 1, node.player
    assert abs(sum(dist.values()) - 1.0) < 1e-6, dist
    assert set(dist.keys()) == set(node.actions)
    assert all(p >= -1e-9 for p in dist.values())
    print(f"PASS test_solve_turn_for_action_oop (actions={list(node.actions)})")


def test_solve_turn_for_action_ip_after_check():
    db = _blueprint_db()
    if db is None:
        print("SKIP test_solve_turn_for_action_ip_after_check (no blueprint)")
        return
    s = _solver(db)
    villain, hero = _trackers()
    # OOP checks -> IP (seat 0) to act.
    dist, node, info = s.solve_turn_for_action(
        board=_BOARD4, pot_entry=24.0, stacks=(88.0, 88.0), bot_seat=0, hole=_HOLE,
        villain_tracker=villain, hero_tracker=hero, confidence=1.0, turn_path=['check'])
    db.close()
    assert node.player == 0, node.player
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    print(f"PASS test_solve_turn_for_action_ip_after_check (actions={list(node.actions)})")


def test_decide_turn_emits_action():
    db = _blueprint_db()
    if db is None:
        print("SKIP test_decide_turn_emits_action (no blueprint)")
        return
    s = _solver(db)
    villain, hero = _trackers()
    legal = ['check', 'bet_small', 'bet_medium', 'bet_large', 'allin']
    ps = {
        'street': 'turn', 'community': _BOARD4, 'hole_cards': _HOLE,
        'seat': 1, 'botSeat': 1, 'to_call': 0.0, 'pot': 24.0,
        'p0_stack': 88.0, 'p1_stack': 88.0,
        'turnEntryPot': 24.0, 'turnEntryStacks': (88.0, 88.0), 'turnPath': [],
        'opp_range': villain, 'hero_range': hero,
    }
    a = s.decide('pf_9_5_oop_turn_', legal, ps)
    db.close()
    assert isinstance(a, str) and a, a
    assert s.last_debug is not None
    assert s.last_debug['mode'] in ('turn_solver', 'fallback', 'blueprint', 'allin_guard')
    print(f"PASS test_decide_turn_emits_action (action={a!r}, mode={s.last_debug['mode']})")


def test_decide_falls_back_off_turn():
    db = _blueprint_db()
    if db is None:
        print("SKIP test_decide_falls_back_off_turn (no blueprint)")
        return
    s = _solver(db)
    legal = ['check', 'bet_medium', 'fold']
    # flop -> parent path -> blueprint (no river fields); must not attempt a turn solve.
    a = s.decide('pf_9_5_ip_flop_', legal, {'street': 'flop'})
    db.close()
    assert a in legal, a
    assert s.last_debug is None or s.last_debug.get('mode') != 'turn_solver'
    print(f"PASS test_decide_falls_back_off_turn (action={a!r})")


def test_turn_spr_gate():
    db = _blueprint_db()
    if db is None:
        print("SKIP test_turn_spr_gate (no blueprint)")
        return
    s = _solver(db)
    base = dict(street='turn', community=_BOARD4, hole_cards=_HOLE, botSeat=1,
                opp_range=object(), hero_range=object(), turnPath=[])
    # High-SPR (deep stacks, small pot) -> skip the solve.
    assert s._turn_solver_inputs(
        {**base, 'turnEntryPot': 8.0,
         'turnEntryStacks': (8.0 * (s.max_spr_turn + 1),) * 2}) is None
    # Healthy SPR -> inputs returned.
    assert s._turn_solver_inputs(
        {**base, 'turnEntryPot': 80.0, 'turnEntryStacks': (80.0, 80.0)}) is not None
    db.close()
    print("PASS test_turn_spr_gate")


TESTS = [
    test_decide_falls_back_off_turn,
    test_turn_spr_gate,
    test_solve_turn_for_action_oop,
    test_solve_turn_for_action_ip_after_check,
    test_decide_turn_emits_action,
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
