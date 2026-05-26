# backend/bot/tests/test_postflop_table_stamp.py
"""Tests for the baked-table centroid stamp (C2/M3): a table must match the
centroids it was baked from. Run from backend/bot/:
    python tests/test_postflop_table_stamp.py
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.abstractions.postflop_v2 import PostflopV2
from src.abstractions.postflop_features import load_centroids, centroid_hash


class _FakeNpz:
    """Minimal stand-in for a loaded .npz (has .files and __getitem__)."""
    def __init__(self, **kw):
        self._d = kw
        self.files = list(kw)

    def __getitem__(self, k):
        return self._d[k]


def _stamped(street, hash_str=None, k=None, bins=None):
    c, b = load_centroids(street)
    return _FakeNpz(
        ids=np.array([1, 2, 3]), buckets=np.array([0, 1, 2]),
        centroid_hash=np.array(hash_str if hash_str is not None else centroid_hash(c, b)),
        n_buckets=np.array(k if k is not None else len(c)),
        bins=np.array(bins if bins is not None else b))


def test_hash_deterministic_and_sensitive():
    c, b = load_centroids('flop')
    assert centroid_hash(c, b) == centroid_hash(c, b)            # deterministic
    assert centroid_hash(c, b) != centroid_hash(c, b + 1)        # bins matter
    c2 = c.copy(); c2[0, 0] += 0.01
    assert centroid_hash(c, b) != centroid_hash(c2, b)           # centroids matter
    print("PASS test_hash_deterministic_and_sensitive")


def test_matching_stamp_ok():
    PostflopV2()._verify_stamp('flop', _stamped('flop'))         # no raise
    print("PASS test_matching_stamp_ok")


def test_mismatched_hash_raises():
    try:
        PostflopV2()._verify_stamp('flop', _stamped('flop', hash_str='deadbeef'))
        raise AssertionError("expected ValueError on stale stamp")
    except ValueError as e:
        assert 'stale' in str(e).lower() or 'different centroids' in str(e).lower()
    print("PASS test_mismatched_hash_raises")


def test_wrong_bucket_count_raises():
    c, _ = load_centroids('flop')
    try:
        PostflopV2()._verify_stamp('flop', _stamped('flop', k=len(c) + 1))
        raise AssertionError("expected ValueError on K mismatch")
    except ValueError:
        pass
    print("PASS test_wrong_bucket_count_raises")


def test_legacy_no_stamp_warns_not_raises():
    d = _FakeNpz(ids=np.array([1]), buckets=np.array([0]))       # no stamp keys
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PostflopV2()._verify_stamp('flop', d)                    # must NOT raise
        assert any('no centroid stamp' in str(x.message) for x in w)
    print("PASS test_legacy_no_stamp_warns_not_raises")


TESTS = [
    test_hash_deterministic_and_sensitive,
    test_matching_stamp_ok,
    test_mismatched_hash_raises,
    test_wrong_bucket_count_raises,
    test_legacy_no_stamp_warns_not_raises,
]

if __name__ == '__main__':
    passed = failed = 0
    for fn in TESTS:
        try:
            fn(); passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(TESTS)} tests")
    sys.exit(1 if failed else 0)
