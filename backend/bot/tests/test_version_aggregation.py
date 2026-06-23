# backend/bot/tests/test_version_aggregation.py
"""
Unit tests for the recap -> bot-version aggregation that powers the +EV card's
`byVersion`, the version-filtered leaderboard, and /api/me's recap-based stats.
These were added with the version-filter feature and had NO coverage at any layer
(the source of truth for every public number, so a sign/tag bug corrupts it all).

Pure functions + in-memory stores only -- no boto3/AWS/moto needed.

Run from backend/bot/:
    python tests/test_version_aggregation.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.hand_store import recap_version, _aggregate_versions
from src.game.player_store import InMemoryPlayerStore
from src.game.global_stats_store import InMemoryGlobalStatsStore


def _passed(msg):
    print(f"  PASS: {msg}")


def test_recap_version_tagging():
    """Explicit botVersion wins; else the blueprint name is matched (specific-first);
    else 'unknown'."""
    assert recap_version({'botVersion': 'v2'}) == 'v2'                       # explicit tag
    assert recap_version({'blueprint': 'snap_52500000.db'}) == 'v2'          # blueprint fallback
    assert recap_version({'blueprint': 'blueprint_final.db'}) == 'v1'
    assert recap_version({'blueprint': 'blueprint_final_v2.db'}) == 'v2'     # specific stem first
    assert recap_version({}) == 'unknown'
    # explicit tag beats a (contradictory) blueprint name
    assert recap_version({'botVersion': 'v2', 'blueprint': 'blueprint_final.db'}) == 'v2'
    _passed("recap_version: tag > blueprint-stem > unknown")


def test_aggregate_versions_totals_and_byplayer():
    """humanNetBB = humanDelta(chips)/2 = BB, summed per version + per (version, player)."""
    recaps = [
        {'playerId': 'A', 'botVersion': 'v1', 'result': {'humanDelta': 10.0}},   # +5 BB
        {'playerId': 'A', 'botVersion': 'v1', 'result': {'humanDelta': -4.0}},   # -2 BB
        {'playerId': 'A', 'botVersion': 'v2', 'result': {'humanDelta': 6.0}},    # +3 BB
        {'playerId': 'B', 'botVersion': 'v2', 'result': {'humanDelta': 2.0}},    # +1 BB
    ]
    agg = _aggregate_versions(recaps)
    assert agg['totals']['v1'] == {'hands': 2, 'humanNetBB': 3.0}   # (10-4)/2
    assert agg['totals']['v2'] == {'hands': 2, 'humanNetBB': 4.0}   # (6+2)/2
    assert agg['byPlayer']['v1']['A'] == {'hands': 2, 'humanNetBB': 3.0}
    assert agg['byPlayer']['v2']['A'] == {'hands': 1, 'humanNetBB': 3.0}
    assert agg['byPlayer']['v2']['B'] == {'hands': 1, 'humanNetBB': 1.0}
    # 'all' (sum across versions) reconciles with v1+v2 -- the property the leaderboard relies on
    all_hands = sum(t['hands'] for t in agg['totals'].values())
    assert all_hands == 4
    _passed("_aggregate_versions: BB units, per-version + per-player totals, all=v1+v2")


def test_record_merged_player_decrements_and_floors():
    """The merge decrement reduces totalPlayers and never goes negative."""
    g = InMemoryGlobalStatsStore()
    g.record_new_player(); g.record_new_player()
    assert g.get()['totalPlayers'] == 2
    g.record_merged_player()
    assert g.get()['totalPlayers'] == 1
    g.record_merged_player(); g.record_merged_player()      # over-decrement
    assert g.get()['totalPlayers'] == 0                     # floored, not negative
    _passed("record_merged_player decrements and floors at 0")


def test_link_account_merged_this_call_idempotent():
    """`_mergedThisCall` is True ONLY on the real merge transition, so the API decrements
    totalPlayers exactly once (the '67 vs 66' fix). A bind sets no flag; a replay sets False."""
    p = InMemoryPlayerStore()
    p.create_if_absent('anon')
    p.record_hand_result('anon', 5.0)
    p.record_hand_result('anon', 3.0)                       # anon has hands > 0

    # First sign-in on a NEW device -> binds it (no merge of itself).
    r1 = p.link_account('dev', email='e', auth_provider='google', provider_sub='g1')
    assert not r1.get('_mergedThisCall')                    # bind path -> None/False

    # Sign in from the anon device, same sub -> merges anon into canonical 'dev'.
    r2 = p.link_account('anon', email='e', auth_provider='google', provider_sub='g1')
    assert r2['_mergedThisCall'] is True                    # the actual merge transition
    assert p.get('anon')['merged_into'] == 'dev'

    # Replay the same sign-in -> already merged -> NO second decrement.
    r3 = p.link_account('anon', email='e', auth_provider='google', provider_sub='g1')
    assert not r3.get('_mergedThisCall')                    # idempotent
    _passed("_mergedThisCall: set once on merge, never on bind/replay")


if __name__ == '__main__':
    tests = [
        test_recap_version_tagging,
        test_aggregate_versions_totals_and_byplayer,
        test_record_merged_player_decrements_and_floors,
        test_link_account_merged_this_call_idempotent,
    ]
    print("Running version-aggregation tests...\n")
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll {len(tests)} tests passed.")
