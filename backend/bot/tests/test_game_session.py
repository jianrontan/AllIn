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


def test_voluntary_free_fold():
    """LIVE play lets the HUMAN fold even when checking is free (a poker-client
    affordance), but the BOT never sees that free fold (its option set must stay
    byte-identical to training). Drive to a free-check node for each seat and check
    the gating; then confirm a free-fold actually resolves the hand as a forfeit."""
    session = GameSession.new("s", "p")          # human_seat = 0 (button/SB)
    assert session.current_player() == 0
    session.apply_action("call")                 # SB limps -> BB(bot) to act, can check
    assert session.current_player() == 1
    bot_legal = session.legal_actions()
    assert "check" in bot_legal and "fold" not in bot_legal, bot_legal   # bot: no free fold
    session.apply_action("check")                # -> flop; BB(bot, OOP) acts first
    assert session.data["street"] == 1 and session.current_player() == 1
    session.apply_action("check")                # bot checks -> human (button) to act
    assert session.current_player() == 0
    human_legal = session.legal_actions()
    assert "check" in human_legal and "fold" in human_legal, human_legal  # human: free fold
    net_before = session.data["human_net"]
    session.apply_action("fold")                 # voluntary free-fold
    assert session.data["status"] == "hand_over"
    assert session.data["human_net"] < net_before, "folding must forfeit invested chips"

    # Exact forfeit: human (SB) limped (1 blind + 1 call = 2 chips in) then folded,
    # so the loss is exactly those 2 chips -- not an arbitrary number.
    from src.cfr.poker_game import STARTING_STACK
    result = session.data["result"]
    assert result["winner"] == "bot" and result["reason"] == "fold"
    assert result["humanDelta"] == -2.0, result
    assert session.data["human_net"] == -2.0
    # Chip conservation: stacks + final pot == both starting stacks (each hand
    # resets stacks to STARTING_STACK; the pot is the chips committed this hand).
    total = (session.data["p0_stack"] + session.data["p1_stack"]
             + result["finalPot"])
    assert abs(total - 2 * STARTING_STACK) < 1e-6, total
    # Serialization survives the free-fold terminal state (the leaderboard hook and
    # the inactivity sweeper both round-trip the session through the store).
    rt = GameSession.from_dict(session.to_dict())
    assert rt.public_view() == session.public_view()
    assert rt.legal_actions() == []
    _passed("voluntary free-fold: human-only, exact forfeit, conserves chips, round-trips")


def test_bot_never_offered_free_fold_any_street():
    """The free-fold must be gated to the human on EVERY street -- a regression that
    leaked it to the bot would let an untrained-key fallback fold for free. Drive a
    full check-down and assert the gating at every free-check node on both seats."""
    session = GameSession.new("s", "p")          # human_seat = 0
    human_seat = session.data["human_seat"]
    human_freechecks = bot_freechecks = 0
    guard = 0
    while session.data["status"] == "in_hand" and guard < 40:
        guard += 1
        legal = session.legal_actions()
        cur = session.current_player()
        if "check" in legal:                     # a free-check node
            if cur == human_seat:
                assert "fold" in legal, (cur, legal)     # human: free fold offered
                human_freechecks += 1
            else:
                assert "fold" not in legal, (cur, legal)  # bot: never
                bot_freechecks += 1
            session.apply_action("check")
        elif "call" in legal:
            session.apply_action("call")
        else:
            session.apply_action(legal[0])
    assert session.data["status"] == "hand_over"
    # A full check-down exercises a free-check node for each seat on flop/turn/river.
    assert human_freechecks >= 3 and bot_freechecks >= 3, (human_freechecks, bot_freechecks)
    _passed("bot never offered a free fold on any street; human always is")


def test_describe_hand_preflop_labels():
    """Preflop hand labels match the postflop vocabulary (generic 'High card' /
    'Pair'), guarding against drift back to 'Ace-King suited high'."""
    s = GameSession.__new__(GameSession)         # describe_hand preflop needs no state
    assert s.describe_hand(("HA", "CK"), []) == "High card"
    assert s.describe_hand(("HA", "HK"), []) == "High card"   # suited, still generic
    assert s.describe_hand(("HA", "CA"), []) == "Pair"
    _passed("preflop hand labels are the generic postflop 'High card' / 'Pair'")


def test_iter_active_snapshot():
    """The session store enumerates non-expired sessions (the inactivity sweeper's
    input). Expired entries are excluded; the result is a detached snapshot list."""
    import time
    store = InMemorySessionStore(ttl_seconds=3600)
    store.put("live", {"session_id": "live", "status": "in_hand"})
    active = dict(store.iter_active())
    assert "live" in active and active["live"]["status"] == "in_hand"
    # Force-expire and confirm it drops out.
    with store._data_guard:
        _, data = store._data["live"]
        store._data["live"] = (time.time() - 1, data)
    assert dict(store.iter_active()) == {}, "expired session must not be enumerated"
    _passed("iter_active snapshots only non-expired sessions")


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
        test_voluntary_free_fold,
        test_bot_never_offered_free_fold_any_street,
        test_describe_hand_preflop_labels,
        test_iter_active_snapshot,
    ]
    print("Running GameSession tests...\n")
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll {len(tests)} test groups passed.")
