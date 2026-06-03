"""
Tests for the GameSession game core (backend/bot/src/game).

Run from backend/bot/:
    python tests/test_game_session.py
"""
import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.game_session import GameSession, advance_bot_turns, GameError
from src.game.bot_strategy import BotStrategy
from src.game.session_store import InMemorySessionStore
from src.game.cards import to_engine, to_display


class RandomBot(BotStrategy):
    """Deterministic-ish random opponent — no DB needed for core tests."""
    def decide(self, info_set_key, legal_actions, public_state):
        return random.choice(legal_actions)


def _passed(msg):
    print(f"  PASS: {msg}")


def test_card_conversion():
    assert to_engine('Ah') == 'HA'
    assert to_engine('AH') == 'HA'
    assert to_engine('HA') == 'HA'
    assert to_display('HA') == 'Ah'
    assert to_display(to_engine('Td')) == 'Td'
    try:
        to_engine('XX')
        assert False, "expected ValueError"
    except ValueError:
        pass
    _passed("card format conversion round-trips")


def test_full_hands_terminate():
    """Every hand must reach a terminal state within a bounded action count."""
    bot = RandomBot()
    random.seed(7)
    for trial in range(200):
        session = GameSession.new(f"s{trial}", "p")
        advance_bot_turns(session, bot)
        steps = 0
        while session.data['status'] == 'in_hand':
            session.apply_action(random.choice(session.legal_actions()))
            advance_bot_turns(session, bot)
            steps += 1
            assert steps < 60, f"hand {trial} did not terminate"
        assert session.data['result'] is not None
    _passed("200 random hands all terminate with a result")


def test_zero_sum_and_net():
    """Each hand's reported delta must equal the actual change in human_net."""
    bot = RandomBot()
    random.seed(11)
    session = GameSession.new("s", "p")
    prev_net = 0.0
    for hand in range(30):
        if hand > 0:
            session.start_next_hand()
        advance_bot_turns(session, bot)
        while session.data['status'] == 'in_hand':
            session.apply_action(random.choice(session.legal_actions()))
            advance_bot_turns(session, bot)
        net = session.data['human_net']
        # result.humanDelta is rounded to 2dp for display; allow that slack.
        assert abs((net - prev_net) - session.data['result']['humanDelta']) < 0.01, \
            f"reported delta mismatch at hand {hand}"
        # No hand can lose more than a full starting stack.
        assert abs(net - prev_net) <= 200.0 + 1e-6, f"impossible swing hand {hand}"
        prev_net = net
    _passed("per-hand delta matches human_net change over 30 hands")


def test_seat_alternates():
    bot = RandomBot()
    random.seed(3)
    session = GameSession.new("s", "p")
    seats = [session.data['human_seat']]
    for _ in range(5):
        advance_bot_turns(session, bot)
        while session.data['status'] == 'in_hand':
            session.apply_action(random.choice(session.legal_actions()))
            advance_bot_turns(session, bot)
        session.start_next_hand()
        seats.append(session.data['human_seat'])
    assert seats == [0, 1, 0, 1, 0, 1], seats
    _passed("human seat alternates each hand")


def test_serialization_roundtrip():
    """A session must survive a full JSON round-trip mid-hand."""
    bot = RandomBot()
    random.seed(5)
    session = GameSession.new("s", "p")
    advance_bot_turns(session, bot)
    for _ in range(3):
        if session.data['status'] != 'in_hand':
            break
        revived = GameSession.from_dict(json.loads(json.dumps(session.to_dict())))
        assert revived.public_view() == session.public_view()
        session = revived
        session.apply_action(random.choice(session.legal_actions()))
        advance_bot_turns(session, bot)
    _passed("session JSON round-trips with identical public view")


def test_illegal_action_rejected():
    session = GameSession.new("s", "p")
    try:
        session.apply_action("bet_enormous")
        assert False, "expected GameError"
    except GameError:
        pass
    _passed("illegal action raises GameError")


def test_info_set_key_format():
    session = GameSession.new("s", "p")
    key = session.info_set_key(0)
    parts = key.split('_')
    # preflop: pf_<n>_<pos>_<pattern>  -> at least 4 underscore-joined parts
    assert key.startswith('pf_'), key
    assert parts[2] in ('ip', 'oop'), key
    _passed(f"preflop info-set key well-formed ({key!r})")


def test_store():
    store = InMemorySessionStore()
    assert store.get("missing") is None
    store.put("a", {"x": 1})
    assert store.get("a") == {"x": 1}
    store.delete("a")
    assert store.get("a") is None
    _passed("InMemorySessionStore get/put/delete")


def test_live_engine_uncapped_training_capped():
    """The LIVE GameSession engine uncaps re-raises (5-bet/6-bet+); the default
    engine used by training/eval keeps the 3-aggression cap. After 1 bet + 2 raises
    the capped engine offers only fold/call; the live engine still offers a raise."""
    from src.cfr.poker_game import PokerGame, STARTING_STACK
    hist = ['bet_medium', 'raise_medium', 'raise_medium']     # 3 aggressions = cap
    big = STARTING_STACK

    capped = PokerGame()                                       # training/eval default
    assert capped.max_raises_per_street == 2
    legal_capped = capped.get_legal_actions(1, hist, 6, 0, p0_stack=big, p1_stack=big)
    assert set(legal_capped) == {'fold', 'call'}, legal_capped

    session = GameSession.new("s", "p")
    assert session.game.max_raises_per_street == float('inf')
    legal_live = session.game.get_legal_actions(1, hist, 6, 0, p0_stack=big, p1_stack=big)
    assert any(a.startswith(('bet_', 'raise_')) for a in legal_live), legal_live
    _passed("live engine uncapped; default (training/eval) engine capped at 3 aggressions")


def test_uncapped_custom_raise_past_cap():
    """A human can custom-raise past the trained cap in live play; the capped engine
    offers no custom raise once aggression is closed."""
    from src.cfr.poker_game import PokerGame, STARTING_STACK
    hist = ['bet_medium', 'raise_medium', 'raise_medium']
    s = STARTING_STACK
    live = PokerGame(max_raises_per_street=float('inf'))
    bounds = live.custom_bet_bounds(1, hist, 6, 0, s, s)
    assert bounds is not None and bounds[0] < bounds[1], bounds   # a 4th raise is legal
    capped = PokerGame()
    assert capped.custom_bet_bounds(1, hist, 6, 0, s, s) is None
    _passed("uncapped engine offers a custom 5-bet past the cap; capped engine does not")


def test_untrained_key_passive_fallback():
    """An untrained key must map to PASSIVE actions only (check/call/fold) -- never a
    raise/jam -- so the uncapped live engine can't make the bot stray-raise from a
    node it never trained (the BUG-011 class)."""
    from src.game.bot_strategy import BlueprintStrategy

    class _NoDB:
        def get_average_strategy(self, k):
            return None                                       # every key 'untrained'

    bs = BlueprintStrategy.__new__(BlueprintStrategy)
    bs.db = _NoDB()
    deep_legal = ['fold', 'call', 'raise_small', 'raise_medium', 'raise_large', 'allin']
    dist = bs._distribution("UNTRAINED_DEEP_KEY", deep_legal)
    assert dist and all(a in ('check', 'call', 'fold')
                        for a, p in dist.items() if p > 0), dist
    _passed("untrained key -> passive only (no stray raise/jam)")


if __name__ == "__main__":
    tests = [
        test_card_conversion,
        test_full_hands_terminate,
        test_zero_sum_and_net,
        test_seat_alternates,
        test_serialization_roundtrip,
        test_illegal_action_rejected,
        test_info_set_key_format,
        test_store,
        test_live_engine_uncapped_training_capped,
        test_uncapped_custom_raise_past_cap,
        test_untrained_key_passive_fallback,
    ]
    print("Running GameSession tests...\n")
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll {len(tests)} test groups passed.")
