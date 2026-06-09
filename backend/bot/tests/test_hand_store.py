# backend/bot/tests/test_hand_store.py
"""
HandStore round-trip tests:
  - recap_from_session captures the right fields off a hand-over GameSession
  - InMemoryHandStore put/list_for_player/get round-trips and orders newest-first
  - DynamoDBHandStore is exercised against moto if installed (skipped otherwise)
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from src.game.hand_store import (             # noqa: E402
    InMemoryHandStore, DynamoDBHandStore, make_hand_store,
    recap_from_session, _hand_key,
)
from src.game.game_session import GameSession  # noqa: E402


def _play_one_hand_to_fold():
    """Drive a session to hand_over via a preflop fold, deterministically."""
    s = GameSession.new('sess-test', 'player-test', max_raises_per_street=2)
    # The bot may act first (when it's SB). If it's the human's turn, we fold
    # immediately. If the bot fires first, fold once it's our turn.
    while s.data['status'] == 'in_hand':
        if s.is_human_turn():
            s.apply_action('fold')
            break
        # Without a strategy_fn the bot's `advance_bot_turns` is a no-op; force a
        # bot action manually by calling apply_action with a check/call.
        legal = s.legal_actions()
        s.apply_action('call' if 'call' in legal else 'check')
    return s


class TestRecapBuilder(unittest.TestCase):
    def test_recap_shape_off_a_finished_hand(self):
        s = _play_one_hand_to_fold()
        recap = recap_from_session(s, blueprint_name='snap_test.db', ts_ms=1_700_000_000_000)

        self.assertEqual(recap['playerId'], 'player-test')
        self.assertEqual(recap['sessionId'], 'sess-test')
        self.assertEqual(recap['handNumber'], 1)
        self.assertEqual(recap['ts'], 1_700_000_000_000)
        self.assertEqual(recap['handKey'],
                         '1700000000000#sess-test#1')

        # Cards are display format (Rank+suit lowercase), 2 hole each + 0..5 board.
        self.assertEqual(len(recap['humanHole']), 2)
        self.assertEqual(len(recap['botHole']), 2)
        self.assertTrue(0 <= len(recap['community']) <= 5)
        for c in recap['humanHole'] + recap['botHole'] + recap['community']:
            self.assertRegex(c, r'^[2-9TJQKA][cdhs]$')

        # Result is present + carries the four expected keys.
        for k in ('humanDelta', 'winner', 'reason', 'finalPot'):
            self.assertIn(k, recap['result'])

        # Net P/L bracketing: before + delta = after.
        self.assertAlmostEqual(
            recap['humanNetBefore'] + recap['result']['humanDelta'],
            recap['humanNetAfter'], places=2)

        self.assertTrue(len(recap['actionLog']) >= 1)
        for entry in recap['actionLog']:
            for k in ('player', 'street', 'action', 'chips'):
                self.assertIn(k, entry)

    def test_recap_rejects_in_hand_session(self):
        s = GameSession.new('s', 'p', max_raises_per_street=2)
        # `new` leaves the session at 'in_hand' (bot may have acted but didn't fold).
        # If by chance the test session is somehow at hand_over already, skip.
        if s.data.get('status') == 'hand_over':
            self.skipTest("session went straight to hand_over (unlikely)")
        with self.assertRaises(ValueError):
            recap_from_session(s)


class TestInMemoryHandStore(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryHandStore()

    def _recap(self, pid, ts, hand_no, session_id='s1'):
        return {
            'playerId': pid,
            'sessionId': session_id,
            'handNumber': hand_no,
            'handKey': _hand_key(ts, session_id, hand_no),
            'ts': ts,
            'result': {'humanDelta': 0.0, 'winner': 'split',
                       'reason': 'fold', 'finalPot': 3.0},
        }

    def test_put_get_roundtrip(self):
        r = self._recap('alice', 1, 1)
        self.store.put(r)
        got = self.store.get('alice', r['handKey'])
        self.assertEqual(got['playerId'], 'alice')
        self.assertEqual(got['handNumber'], 1)
        self.assertEqual(got['handKey'], r['handKey'])

    def test_list_newest_first(self):
        for ts, hn in [(1, 1), (5, 2), (3, 3), (9, 4)]:
            self.store.put(self._recap('alice', ts, hn))
        rows = self.store.list_for_player('alice', n=3)
        self.assertEqual([r['handNumber'] for r in rows], [4, 2, 3])

    def test_list_isolates_players(self):
        self.store.put(self._recap('alice', 1, 1))
        self.store.put(self._recap('bob', 1, 1))
        self.assertEqual(len(self.store.list_for_player('alice', n=99)), 1)
        self.assertEqual(len(self.store.list_for_player('bob', n=99)), 1)
        self.assertEqual(self.store.list_for_player('alice')[0]['playerId'], 'alice')

    def test_put_idempotent_on_same_key(self):
        r = self._recap('alice', 1, 1)
        self.store.put(r); self.store.put(r); self.store.put(r)
        self.assertEqual(len(self.store.list_for_player('alice', n=99)), 1)

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.store.get('nobody', '0#x#0'))


class TestMakeHandStore(unittest.TestCase):
    def test_default_is_memory(self):
        old = os.environ.pop('ALLIN_STORE_BACKEND', None)
        try:
            self.assertIsInstance(make_hand_store(), InMemoryHandStore)
        finally:
            if old is not None:
                os.environ['ALLIN_STORE_BACKEND'] = old


class TestDynamoDBHandStore(unittest.TestCase):
    """moto-backed exercise of the DynamoDB store. Skipped if moto is absent."""

    def setUp(self):
        try:
            from moto import mock_aws         # moto 5.x
        except ImportError:
            self.skipTest("moto not installed (pip install -r requirements-dev.txt)")
        self._patcher = mock_aws()
        self._patcher.start()
        DynamoDBHandStore.create_table_if_missing('test-hands', region='us-east-1')
        self.store = DynamoDBHandStore('test-hands', region='us-east-1')

    def tearDown(self):
        try:
            self._patcher.stop()
        except Exception:
            pass

    def test_roundtrip_and_ordering(self):
        for ts, hn in [(1, 1), (5, 2), (3, 3)]:
            self.store.put({
                'playerId': 'alice', 'sessionId': 's1',
                'handNumber': hn, 'handKey': _hand_key(ts, 's1', hn),
                'ts': ts,
                # Include a float so the Decimal coercion path is exercised.
                'result': {'humanDelta': 1.5, 'winner': 'you',
                           'reason': 'fold', 'finalPot': 7.5},
            })
        rows = self.store.list_for_player('alice', n=10)
        self.assertEqual([r['handNumber'] for r in rows], [2, 3, 1])
        # Float survived the round-trip as a Python float.
        self.assertIsInstance(rows[0]['result']['humanDelta'], float)


if __name__ == '__main__':
    unittest.main()
