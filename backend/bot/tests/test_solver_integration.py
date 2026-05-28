# backend/bot/tests/test_solver_integration.py
"""
Step 6c integration: the RiverSubgameSolver wired into GameSession as the bot.
Plays real hands (passive human) and checks that the river-entry snapshots are
built, the solver's public_state inputs are exposed on the river, and hands
complete without crashing (incl. the bot emitting exact custom sizes).

Runs against the active blueprint DB (read-only); skips cleanly if none.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.subgame.river_subgame_solver import RiverSubgameSolver
from src.game.game_session import GameSession, advance_bot_turns
from src.game.bot_strategy import BlueprintStrategy


def _blueprint_db():
    try:
        from src.config import resolve_blueprint_path
        from src.storage.blueprint_db import BlueprintDB
        return BlueprintDB(resolve_blueprint_path(), read_only=True)
    except Exception as e:
        print(f"  (no blueprint DB: {e})")
        return None


def _passive_human(session):
    legal = session.legal_actions()
    for a in ('check', 'call', 'fold'):
        if a in legal:
            return a
    return legal[0]


def test_river_solver_plays_hands():
    db = _blueprint_db()
    if db is None:
        print("SKIP test_river_solver_plays_hands (no blueprint)")
        return
    strat_fn = BlueprintStrategy(db).range_model_fn()
    solver = RiverSubgameSolver(db, max_iters=40, check_every=20, time_budget=10.0)
    session = GameSession.new('it', 'p', strategy_fn=strat_fn)

    saw_river = False
    river_inputs_ok = False
    hands = 0
    while hands < 10 and not (saw_river and river_inputs_ok):
        guard = 0
        while session.data['status'] == 'in_hand' and guard < 80:
            if session.is_human_turn():
                session.apply_action(_passive_human(session))
            else:
                # Inspect the solver's inputs at a bot river decision.
                ps = session.bot_public_state()
                if ps.get('street') == 'river' and ps.get('riverEntryPot') is not None:
                    river_inputs_ok = True
                    assert ps['hero_range'] is not None
                    assert ps['opp_range'] is not None
                    assert isinstance(ps['riverPath'], list)
                    assert ps['botSeat'] in (0, 1)
                    s0, s1 = ps['riverEntryStacks']
                    assert abs(s0 - s1) < 1e-6, "river-entry stacks must be equal"
                advance_bot_turns(session, solver)
            guard += 1

        if session.data.get('river_entry_bot') is not None:
            saw_river = True
            assert isinstance(session.data['river_entry_bot'], dict)
            assert isinstance(session.data['river_entry_opp'], dict)

        assert session.data['status'] in ('in_hand', 'hand_over')
        if session.data['status'] == 'hand_over':
            session.start_next_hand()
        hands += 1

    db.close()
    assert saw_river, "no hand reached the river in 10 tries"
    assert river_inputs_ok, "solver river inputs were never exposed on a bot turn"
    print(f"PASS test_river_solver_plays_hands (reached river; inputs exposed; "
          f"{hands} hands, no crash)")


def test_serialization_survives_new_fields():
    """The new session fields (bot_range, river_entry_*) keep to_dict JSON-clean."""
    import json
    db = _blueprint_db()
    if db is None:
        print("SKIP test_serialization_survives_new_fields (no blueprint)")
        return
    strat_fn = BlueprintStrategy(db).range_model_fn()
    session = GameSession.new('ser', 'p', strategy_fn=strat_fn)
    # Play a couple of actions, then round-trip.
    guard = 0
    while session.data['status'] == 'in_hand' and guard < 4:
        if session.is_human_turn():
            session.apply_action(_passive_human(session))
        else:
            break
        guard += 1
    json.dumps(session.to_dict())          # must not raise
    restored = GameSession.from_dict(json.loads(json.dumps(session.to_dict())),
                                     strategy_fn=strat_fn)
    assert restored.data['bot_range'] is not None
    db.close()
    print("PASS test_serialization_survives_new_fields")


TESTS = [
    test_river_solver_plays_hands,
    test_serialization_survives_new_fields,
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
