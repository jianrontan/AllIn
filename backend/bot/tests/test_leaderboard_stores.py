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


def test_top_ranks_by_net_bb():
    """Leaderboard ranks by NET BB (total winnings), NOT by rate: a high-rate / low-volume player
    ranks BELOW a higher total-winnings player. This is the change that fixed the 'BB/hand column not
    aligned with Net BB' look (rate-ranking made the Net BB column non-monotonic)."""
    s = InMemoryPlayerStore()
    for _ in range(60):
        s.record_hand_result('p_rate', 3.0)        # +180 net, 300 bb/100 (high rate, low volume)
    for _ in range(300):
        s.record_hand_result('p_big', 1.0)         # +300 net, 100 bb/100 (lower rate, more winnings)
    s.upsert_handle('p_rate', 'Rate')
    s.upsert_handle('p_big', 'Big')
    board = s.top(n=10, min_hands=50)
    handles = [r['handle'] for r in board]
    assert handles.index('Big') < handles.index('Rate')   # +300 net outranks +180 net
    assert board[0]['handle'] == 'Big' and board[0]['netBB'] == 300.0


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


def test_link_account_merges_anon_stats_on_second_device():
    # First device signs in -> binds + absorbs its own anon history.
    s = InMemoryPlayerStore()
    for _ in range(80):
        s.record_hand_result('dev1', 1.5)
    a = s.link_account('dev1', email='a@b.c', auth_provider='google', provider_sub='sub-X')
    assert a['playerId'] == 'dev1' and a['hands'] == 80
    # A SECOND device (own anon history) signs in with the SAME Google sub:
    # it ADOPTS the canonical account (dev1) AND merges its own anon stats in
    # (the non-destructive-upgrade rule: never lose a user's hands).
    for _ in range(5):
        s.record_hand_result('dev2', -1.0)
    adopted = s.link_account('dev2', email='a@b.c', auth_provider='google', provider_sub='sub-X')
    assert adopted['playerId'] == 'dev1'              # canonical id returned
    assert adopted['hands'] == 85                     # 80 + 5 merged
    assert adopted['netBB'] == 115.0                  # 120 + (-5) merged
    # dev2's anon row is marked merged so it's excluded from leaderboard scans.
    dev2 = s.get('dev2')
    assert dev2['merged_into'] == 'dev1' and 'mergedAt' in dev2
    # Top() already filters merged_into; confirm dev2 is gone from the board.
    # public_row() redacts the playerId, so assert by row count: pre-merge there
    # were two qualifying rows (dev1 + dev2), post-merge only one.
    top = s.top(n=10, min_hands=0)
    assert len(top) == 1 and top[0]['hands'] == 85


def test_link_account_merge_is_idempotent():
    # A retry of the same sign-in (or a second sign-in on the already-merged
    # device) must NOT double-add the anon stats to canonical.
    s = InMemoryPlayerStore()
    for _ in range(80):
        s.record_hand_result('dev1', 1.0)
    s.link_account('dev1', email='a@b.c', auth_provider='google', provider_sub='sub-X')
    for _ in range(10):
        s.record_hand_result('dev2', 1.0)
    a1 = s.link_account('dev2', email='a@b.c', auth_provider='google', provider_sub='sub-X')
    assert a1['hands'] == 90                          # 80 + 10
    a2 = s.link_account('dev2', email='a@b.c', auth_provider='google', provider_sub='sub-X')
    assert a2['hands'] == 90                          # NOT 100; merged_into already set


def test_link_account_merge_skips_zero_hands():
    # A device that signs in WITHOUT having played any anon hands first should
    # not get a stray merged_into stamp -- nothing to merge, nothing to mark.
    s = InMemoryPlayerStore()
    for _ in range(40):
        s.record_hand_result('dev1', 2.0)
    s.link_account('dev1', email='a@b.c', auth_provider='google', provider_sub='sub-X')
    a = s.link_account('dev2', email='a@b.c', auth_provider='google', provider_sub='sub-X')
    assert a['playerId'] == 'dev1' and a['hands'] == 40 and a['netBB'] == 80.0
    # dev2 had no row to begin with (no hands played); it stays absent rather
    # than getting created just to be marked merged.
    assert s.get('dev2') is None


def test_link_account_merge_skips_row_bound_to_different_account():
    # dev2 is already signed in to a DIFFERENT Google account (own canonical).
    # Signing in with sub-X on the SAME browser would have hit AccountConflict
    # in the bind branch, but here we're testing the adopt path: another
    # provider entirely, and dev2 just HAPPENS to also be a canonical for
    # provider sub-Y. The "skip merge if dev2.providerSub set" guard prevents
    # double-counting stats across two separate accounts.
    s = InMemoryPlayerStore()
    for _ in range(10):
        s.record_hand_result('canon-X', 1.0)
    s.link_account('canon-X', email='x@x.x', auth_provider='google', provider_sub='sub-X')
    for _ in range(20):
        s.record_hand_result('dev2', 1.0)
    s.link_account('dev2', email='y@y.y', auth_provider='google', provider_sub='sub-Y')
    # Now dev2 has providerSub='sub-Y'. Signing in from dev2 with sub-X must
    # NOT merge dev2's 20 hands into canon-X (those hands belong to another
    # account). Implementation actually hits AccountConflict on the bind
    # branch because canon-X already exists -- but the adopt branch returns
    # canon-X without merging dev2's stats.
    adopted = s.link_account('dev2', email='x@x.x', auth_provider='google',
                             provider_sub='sub-X')
    assert adopted['playerId'] == 'canon-X'
    assert adopted['hands'] == 10           # canon-X's 10, NOT 10 + 20
    # dev2's own account is untouched.
    dev2 = s.get('dev2')
    assert dev2['providerSub'] == 'sub-Y' and not dev2.get('merged_into')
    assert dev2['hands'] == 20


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
    assert g.get() == {'totalHands': 0, 'totalNetBB': 0.0, 'totalPlayers': 0, 'byVersion': {}}
    g.record_new_player()
    g.record_hand_result(2.5)
    g.record_hand_result(-1.0)
    d = g.get()
    assert d['totalHands'] == 2 and d['totalNetBB'] == 1.5 and d['totalPlayers'] == 1


def test_global_stats_version_counters():
    """Per-version running counters: live, no scan; sum reconciles with the totals."""
    g = InMemoryGlobalStatsStore()
    g.record_hand_result(5.0, version='v2')
    g.record_hand_result(-3.0, version='v2')
    g.record_hand_result(2.0, version='v1')
    g.record_hand_result(1.0)                      # version-less still bumps the totals only
    d = g.get()
    assert d['totalHands'] == 4 and abs(d['totalNetBB'] - 5.0) < 1e-9
    assert d['byVersion']['v2'] == {'hands': 2, 'netBB': 2.0}
    assert d['byVersion']['v1'] == {'hands': 1, 'netBB': 2.0}
    # sum across versions <= totals (a version-less hand isn't in any bucket)
    assert sum(b['hands'] for b in d['byVersion'].values()) == 3


def test_global_version_snapshot_roundtrip():
    """Shared version snapshot: None until set, then round-trips; the refresh lease admits exactly
    one holder per window (so only one worker runs the slow recap scan, keeping the board coherent)."""
    g = InMemoryGlobalStatsStore()
    assert g.get_version_snapshot() is None
    data = {'totals': {'v2': {'hands': 3, 'humanNetBB': 1.5}},
            'byPlayer': {'v2': {'p1': {'hands': 3, 'humanNetBB': 1.5}}}}
    g.put_version_snapshot(data, 1000)
    snap = g.get_version_snapshot()
    assert snap['computedAt'] == 1000 and snap['data'] == data
    assert g.try_acquire_version_refresh(lease_seconds=60) is True    # first wins
    assert g.try_acquire_version_refresh(lease_seconds=60) is False   # locked out this window


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
    assert d['byVersion'] == {}
    # Version-keyed counters on real DynamoDB (moto): the vh_<v>/vn_<v> ADD + the get()
    # prefix-reassembly into byVersion -- the new, riskiest path (Decimal coercion, k[3:] split).
    g.record_hand_result(5.0, version='v2')
    g.record_hand_result(-1.0, version='v2')
    d = g.get()
    assert d['byVersion']['v2'] == {'hands': 2, 'netBB': 4.0}
    assert d['totalHands'] == 3                       # totals still bump for versioned hands
    # Shared version snapshot on real DynamoDB (moto): the JSON blob round-trips (separate item) and
    # the refresh lease is a working cross-worker lock (conditional write admits one holder/window).
    assert g.get_version_snapshot() is None
    snap_data = {'totals': {'v2': {'hands': 2, 'humanNetBB': 4.0}}, 'byPlayer': {}}
    g.put_version_snapshot(snap_data, 12345)
    got = g.get_version_snapshot()
    assert got['computedAt'] == 12345 and got['data'] == snap_data
    assert g.try_acquire_version_refresh(lease_seconds=120) is True
    assert g.try_acquire_version_refresh(lease_seconds=120) is False


def test_dynamodb_link_account_merges_anon_stats(dynamo):
    # The DDB merge path is the one that runs in prod (TransactWriteItems:
    # mark anon merged + ADD stats to canonical, atomically). Mirror of the
    # InMemory test_link_account_merges_anon_stats_on_second_device.
    players, _ = dynamo
    for _ in range(80):
        players.record_hand_result('dev1', 1.5)
    a = players.link_account('dev1', email='a@b.c', auth_provider='google',
                             provider_sub='sub-X')
    assert a['playerId'] == 'dev1' and a['hands'] == 80
    for _ in range(5):
        players.record_hand_result('dev2', -1.0)
    adopted = players.link_account('dev2', email='a@b.c', auth_provider='google',
                                   provider_sub='sub-X')
    assert adopted['playerId'] == 'dev1'
    assert adopted['hands'] == 85 and adopted['netBB'] == 115.0
    dev2 = players.get('dev2')
    assert dev2['merged_into'] == 'dev1' and 'mergedAt' in dev2


def test_dynamodb_link_account_merge_idempotent(dynamo):
    # A duplicate sign-in from the already-merged device must not double-add
    # (the transaction's conditional cancels; the except path swallows it).
    players, _ = dynamo
    for _ in range(80):
        players.record_hand_result('dev1', 1.0)
    players.link_account('dev1', email='a@b.c', auth_provider='google',
                         provider_sub='sub-X')
    for _ in range(10):
        players.record_hand_result('dev2', 1.0)
    a1 = players.link_account('dev2', email='a@b.c', auth_provider='google',
                              provider_sub='sub-X')
    assert a1['hands'] == 90
    a2 = players.link_account('dev2', email='a@b.c', auth_provider='google',
                              provider_sub='sub-X')
    assert a2['hands'] == 90                       # NOT 100


def test_dynamodb_handle_reservation_blocks_second_claim(dynamo):
    # The reservation item (PK handle#<lower>) is the race-proof uniqueness
    # layer: a second player claiming the same handle (any case) must get
    # HandleTaken even though the legacy Scan check also covers this -- the
    # reservation is what closes the concurrent-claim window.
    from src.game.player_store import HandleTaken
    players, _ = dynamo
    players.create_if_absent('p1')
    players.upsert_handle('p1', 'Ron')
    players.create_if_absent('p2')
    with pytest.raises(HandleTaken):
        players.upsert_handle('p2', 'ron')         # case-insensitive collision
    # Re-claiming your OWN handle is idempotent (ownerId condition passes).
    assert players.upsert_handle('p1', 'Ron')['handle'] == 'Ron'
    # A rename releases the old reservation, freeing the name for others.
    players.upsert_handle('p1', 'Ron2')
    assert players.upsert_handle('p2', 'Ron')['handle'] == 'Ron'
    # Reservation items never surface on the leaderboard (no hands attribute).
    board = players.top(n=10, min_hands=0)
    assert all('handle#' not in (r.get('handle') or '') for r in board)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
