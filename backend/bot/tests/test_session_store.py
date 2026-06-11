# backend/bot/tests/test_session_store.py
"""
Unit tests for SessionStore enumeration (iter_active), the input to the inactivity
sweeper. The InMemory snapshot path is covered in test_game_session.py; this file
covers the DynamoDBSessionStore.iter_active scan/pagination/skip/error logic with a
fake table (no boto3/AWS/moto needed), so pagination follow-through and the
partial-on-error guard are exercised locally.

Run from backend/bot/:
    python tests/test_session_store.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.session_store import DynamoDBSessionStore


class _FakeClientError(Exception):
    """Stand-in for botocore ClientError (a transient scan failure)."""


class _FakeTable:
    """Minimal DynamoDB Table double: returns canned scan pages in order. A page
    that is an exception instance is raised instead of returned."""
    def __init__(self, pages):
        self._pages = pages
        self.scan_calls = []

    def scan(self, **kwargs):
        i = len(self.scan_calls)
        self.scan_calls.append(kwargs)
        page = self._pages[i]
        if isinstance(page, Exception):
            raise page
        return page


def _store(pages):
    # Bypass __init__ (which needs boto3); wire just what iter_active touches.
    store = DynamoDBSessionStore.__new__(DynamoDBSessionStore)
    store._ClientError = _FakeClientError
    store._table = _FakeTable(pages)
    return store


def _passed(msg):
    print(f"  PASS: {msg}")


def test_dynamo_iter_active_filters_and_paginates():
    """Live, well-formed sessions are yielded across pages; lock items, TTL-expired
    items, and malformed-data items are skipped; pagination follows LastEvaluatedKey."""
    now = int(time.time())
    live = {'session_id': 's1', 'status': 'in_hand'}
    pages = [
        {'Items': [
            {'session_id': 's1', 'data': json.dumps(live), 'expiry': now + 1000},
            {'session_id': '__lock__s1', 'owner': 'tok', 'expiry': now + 1000},  # lock: skip
         ],
         'LastEvaluatedKey': {'session_id': 's1'}},
        {'Items': [
            {'session_id': 's2', 'data': json.dumps({'x': 1}), 'expiry': now - 1},  # expired: skip
            {'session_id': 's3', 'data': '{not json', 'expiry': now + 1000},        # malformed: skip
         ]},
    ]
    store = _store(pages)
    out = store.iter_active()
    assert out == [('s1', live)], out
    assert len(store._table.scan_calls) == 2, "must follow LastEvaluatedKey to page 2"
    assert store._table.scan_calls[1].get('ExclusiveStartKey') == {'session_id': 's1'}
    _passed("dynamo iter_active filters lock/expired/malformed and paginates")


def test_dynamo_iter_active_partial_on_scan_error():
    """A transient scan error mid-pagination returns what was collected so far
    (a partial pass) rather than raising and losing the whole sweep (M2)."""
    now = int(time.time())
    live = {'session_id': 's1', 'status': 'in_hand'}
    pages = [
        {'Items': [{'session_id': 's1', 'data': json.dumps(live), 'expiry': now + 1000}],
         'LastEvaluatedKey': {'session_id': 's1'}},
        _FakeClientError("throttled"),          # page 2 blows up
    ]
    store = _store(pages)
    out = store.iter_active()                    # must NOT raise
    assert out == [('s1', live)], out
    assert len(store._table.scan_calls) == 2
    _passed("dynamo iter_active returns partial on a transient scan error")


if __name__ == "__main__":
    tests = [
        test_dynamo_iter_active_filters_and_paginates,
        test_dynamo_iter_active_partial_on_scan_error,
    ]
    print("Running SessionStore tests...\n")
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll {len(tests)} test groups passed.")
