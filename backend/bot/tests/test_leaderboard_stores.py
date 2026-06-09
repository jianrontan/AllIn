# backend/bot/tests/test_leaderboard_stores.py
"""
PlayerStore + GlobalStatsStore (the +EV leaderboard datastores).

InMemory is exercised fully (the dev/test default). The DynamoDB implementations
are exercised against moto if installed (pip install -r requirements-dev.txt);
they're skipped otherwise so the suite stays green in a no-AWS environment.

Covers: handle validation (regex + profanity), upsert/round-trip, hand-result +
rolling-window cap, top() ordering + min_hands + accounts_only, link_account
non-destructive bind + cross-device merge, and the global counter.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.player_store import (
    InMemoryPlayerStore, validate_handle, InvalidHandle, HAND_CAP,
    HAND_WINDOW_SECONDS, make_player_store)
from src.game.global_stats_store import (
    InMemoryGlobalStatsStore, make_global_stats_store)


# -- handle validation --------------------------------------------------------

def test_handle_validation():
    for ok in ('abc', 'Player_1', 'a-b_C9', 'x' * 20):
        assert validate_handle(ok) == ok
    for bad in ('', 'x' * 21, 'has space', 'emoji😀', 'semi;colon', 'slash/y'):
        with pytest.raises(InvalidHandle):
            validate_handle(bad)
    # profanity (built-in fallback list) is rejected even if regex-valid.
    with pytest.raises(InvalidHandle):
        validate_handle('shit_lord')
    with pytest.raises(InvalidHandle):
        validate_handle('NaZi88')


# -- InMemory player store ----------------------------------------------------

def test_player_roundtrip_and_handle():
    s = InMemoryPlayerStore()
    assert s.get('p1') is None
    assert s.create_if_absent('p1') is True
    assert s.create_if_absent('p1') is False        # idempotent
    row = s.upsert_handle('p1', 'Ron')
    assert row['handle'] == 'Ron' and row['playerId'] == 'p1'
    with pytest.raises(InvalidHandle):
        s.upsert_handle('p1', 'bad handle')


def test_record_hand_and_top_ordering():
    s = InMemoryPlayerStore()
    # p_win: +2 bb/hand over 100 hands; p_lose: -1 bb/hand over 60; p_few: +50 but 10 hands
    for _ in range(100):
        s.record_hand_result('p_win', 2.0)
    for _ in range(60):
        s.record_hand_result('p_lose', -1.0)
    for _ in range(10):
        s.record_hand_result('p_few', 50.0)
    s.upsert_handle('p_win', 'Winner')
    s.upsert_handle('p_lose', 'Loser')

    board = s.top(n=10, min_hands=50)
    handles = [r['handle'] for r in board]
    assert 'Winner' in handles and 'Loser' in handles
    assert 'p_few' not in [r.get('handle') for r in board]   # below min_hands, excluded
    # ordered by bb/100 desc: Winner (+200) before Loser (-100)
    assert handles.index('Winner') < handles.index('Loser')
    assert board[0]['bbPer100'] == 200.0
    # public rows are redacted (no playerId/email)
    assert 'playerId' not in board[0] and 'email' not in board[0]


def test_accounts_only_filter():
    s = InMemoryPlayerStore()
    for _ in range(60):
        s.record_hand_result('anon', 1.0)
        s.record_hand_result('acct', 1.0)
    s.link_account('acct', email='a@b.c', auth_provider='google', provider_sub='sub-1')
    all_board = s.top(min_hands=50, accounts_only=False)
    ranked = s.top(min_hands=50, accounts_only=True)
    assert any(r['isRegistered'] for r in ranked) and all(r['isRegistered'] for r in ranked)
    assert len(all_board) == 2 and len(ranked) == 1


def test_link_account_non_destructive_and_idempotent():
    s = InMemoryPlayerStore()
    for _ in range(30):
        s.record_hand_result('p1', 3.0)
    s.upsert_handle('p1', 'Keeper')
    before = s.get('p1')
    r = s.link_account('p1', email='a@b.c', auth_provider='google', provider_sub='sub-9')
    assert r['hands'] == before['hands'] and r['netBB'] == before['netBB']   # stats survive
    assert r['handle'] == 'Keeper' and r['isRegistered'] and r['email'] == 'a@b.c'
    # idempotent: same sub + same player -> no change to stats
    r2 = s.link_account('p1', email='a@b.c', auth_provider='google', provider_sub='sub-9')
    assert r2['hands'] == before['hands'] and r2['netBB'] == before['netBB']


def test_link_account_does_not_set_handle():
    # link_account never sets a username (the player picks a unique one separately).
    s = InMemoryPlayerStore()
    r = s.link_account('p', email='jane@x.com', auth_provider='google', provider_sub='s')
    assert r['isRegistered'] and r['handle'] is None and r['providerSub'] == 's'


def test_link_account_adopts_canonical_no_sum():
    # First device signs in -> binds + absorbs its own anon history.
    s = InMemoryPlayerStore()
    for _ in range(80):
        s.record_hand_result('dev1', 1.5)
    a = s.link_account('dev1', email='a@b.c', auth_provider='google', provider_sub='sub-X')
    assert a['playerId'] == 'dev1' and a['hands'] == 80
    # A SECOND device (own anon history) signs in with the SAME Google sub:
    # it ADOPTS the canonical account (dev1), its own anon hands are NOT summed.
    for _ in range(5):
        s.record_hand_result('dev2', -1.0)
    adopted = s.link_account('dev2', email='a@b.c', auth_provider='google', provider_sub='sub-X')
    assert adopted['playerId'] == 'dev1'          # canonical id returned
    assert adopted['hands'] == 80 and adopted['netBB'] == 120.0   # NOT 85, no sum
    # dev2's anon row is untouched (still anonymous, not merged/registered)
    assert s.get('dev2')['hands'] == 5 and not s.get('dev2').get('isRegistered')


def test_link_account_conflict_on_second_account():
    from src.game.player_store import AccountConflict
    s = InMemoryPlayerStore()
    s.link_account('p', email='a@b.c', auth_provider='google', provider_sub='sub-A')
    # Same browser/player tries to bind a DIFFERENT Google account -> refused.
    with pytest.raises(AccountConflict):
        s.link_account('p', email='c@d.e', auth_provider='google', provider_sub='sub-B')


def test_username_uniqueness():
    from src.game.player_store import HandleTaken
    s = InMemoryPlayerStore()
    s.upsert_handle('p1', 'Ace')
    with pytest.raises(HandleTaken):
        s.upsert_handle('p2', 'ace')               # case-insensitive collision
    with pytest.raises(HandleTaken):
        s.upsert_handle('p2', 'Ace')
    assert s.upsert_handle('p2', 'Ace2')['handle'] == 'Ace2'   # a free name is fine
    assert s.upsert_handle('p1', 'Ace')['handle'] == 'Ace'     # re-set own name ok


def test_hand_cap_window(monkeypatch):
    import src.game.player_store as ps
    s = InMemoryPlayerStore()
    t = [1000]
    monkeypatch.setattr(ps, '_now', lambda: t[0])
    for _ in range(HAND_CAP):
        s.record_hand_result('p', 1.0)
    allowed, retry = s.hand_cap_status('p')
    assert allowed is False and 0 < retry <= HAND_WINDOW_SECONDS
    # advance past the window -> allowed again
    t[0] += HAND_WINDOW_SECONDS + 1
    allowed, retry = s.hand_cap_status('p')
    assert allowed is True and retry == 0
    # recording after expiry resets the window counter
    s.record_hand_result('p', 1.0)
    assert s.get('p')['hands_in_window'] == 1 and s.get('p')['hands'] == HAND_CAP + 1


# -- global stats -------------------------------------------------------------

def test_global_stats():
    g = InMemoryGlobalStatsStore()
    assert g.get() == {'totalHands': 0, 'totalNetBB': 0.0, 'totalPlayers': 0}
    g.record_new_player()
    g.record_hand_result(2.5)
    g.record_hand_result(-1.0)
    d = g.get()
    assert d['totalHands'] == 2 and d['totalNetBB'] == 1.5 and d['totalPlayers'] == 1


def test_factories_default_memory():
    os.environ.pop('ALLIN_STORE_BACKEND', None)
    assert isinstance(make_player_store(), InMemoryPlayerStore)
    assert isinstance(make_global_stats_store(), InMemoryGlobalStatsStore)
    os.environ['ALLIN_STORE_BACKEND'] = 'bogus'
    with pytest.raises(ValueError):
        make_player_store()
    os.environ.pop('ALLIN_STORE_BACKEND', None)


# -- DynamoDB (moto) — skipped if moto/boto3 absent ---------------------------

@pytest.fixture
def dynamo():
    moto = pytest.importorskip("moto")
    pytest.importorskip("boto3")
    import boto3
    mock = getattr(moto, 'mock_aws', None) or getattr(moto, 'mock_dynamodb')
    with mock():
        from src.game.player_store import DynamoDBPlayerStore
        from src.game.global_stats_store import DynamoDBGlobalStatsStore
        DynamoDBPlayerStore.create_table_if_missing('players-test', region='us-east-1')
        DynamoDBGlobalStatsStore.create_table_if_missing('global-test', region='us-east-1')
        yield (DynamoDBPlayerStore('players-test', region='us-east-1'),
               DynamoDBGlobalStatsStore('global-test', region='us-east-1'))


def test_dynamodb_player_roundtrip(dynamo):
    players, _ = dynamo
    assert players.create_if_absent('p1') is True
    assert players.create_if_absent('p1') is False
    for _ in range(60):
        players.record_hand_result('p1', 2.0)
    players.upsert_handle('p1', 'Ron')
    row = players.get('p1')
    assert row['hands'] == 60 and row['netBB'] == 120.0 and row['handle'] == 'Ron'
    board = players.top(min_hands=50)
    assert board and board[0]['handle'] == 'Ron' and board[0]['bbPer100'] == 200.0
    r = players.link_account('p1', email='a@b.c', auth_provider='google', provider_sub='s1')
    assert r['isRegistered'] and r['hands'] == 60          # non-destructive


def test_dynamodb_global(dynamo):
    _, g = dynamo
    g.record_new_player()
    g.record_hand_result(3.0)
    d = g.get()
    assert d['totalHands'] == 1 and d['totalNetBB'] == 3.0 and d['totalPlayers'] == 1


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
