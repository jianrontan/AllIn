# backend/bot/tests/test_river_board_cache.py
"""
Correctness of the canonical-board river-equity cache (PostflopV2._river_equity).

The cache runs board_winrates once per CANONICAL board and reuses it across the
whole suit-isomorphic orbit by relabeling the hero hand. These tests prove that
shortcut is exact:

  1. Cached river equity == a direct board_winrates computation on the CONCRETE
     board, for many random (hole, board) pairs.
  2. Suit-isomorphic boards yield identical equity for the correspondingly
     relabeled hole (the property the cache relies on).
  3. The resulting river BUCKET is unchanged vs a from-scratch (cache-cleared)
     computation -- i.e. the optimization cannot shift a blueprint bucket.

Run: python tests/test_river_board_cache.py
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.abstractions.postflop_v2 import PostflopV2, clear_river_board_cache
from src.abstractions.postflop_features import DECK, board_winrates, _CARD_IDX, assign

# Keep the shared module-global cache from leaking state between tests when run
# under pytest in one process (the cache is process-wide by design).
try:
    import pytest

    @pytest.fixture(autouse=True)
    def _isolate_river_cache():
        clear_river_board_cache()
        yield
        clear_river_board_cache()
except ImportError:
    pass

_passed = _failed = 0


def check(name, cond, extra=''):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name} {extra}")


def _direct_equity(hole, board):
    """Equity of `hole` vs uniform range on `board`, computed directly on the
    CONCRETE board (the reference the cache must match)."""
    c1, c2, wr = board_winrates(list(board))
    a, b = _CARD_IDX[hole[0]], _CARD_IDX[hole[1]]
    target = (a, b) if a < b else (b, a)
    for x, y, w in zip(c1.tolist(), c2.tolist(), wr.tolist()):
        if ((x, y) if x < y else (y, x)) == target:
            return w
    raise AssertionError("hand not found on board (hole overlaps board?)")


def _deal(rng):
    cards = rng.sample(DECK, 7)
    return tuple(cards[:2]), tuple(cards[2:])      # (hole, 5-card board)


def _ref_river_bucket(pv, hole, board):
    """River bucket computed with a FLOAT64 equity (no uint16 cache) -- the
    reference the cached path must match. This is the check that would FAIL under
    the earlier float32 storage (which flipped ~1/5000 bin-edge buckets)."""
    centroids, bins = pv._centroid('river')
    eq = _direct_equity(hole, board)               # float64, direct from board_winrates
    hist = np.zeros(bins)
    hist[min(int(eq * bins), bins - 1)] = 1.0
    return assign(hist, centroids)


def test_cached_equity_matches_direct():
    clear_river_board_cache()
    pv = PostflopV2()
    rng = random.Random(20260530)
    worst = 0.0
    for _ in range(3000):
        hole, board = _deal(rng)
        cached = pv._river_equity(hole, board)
        direct = _direct_equity(hole, board)
        worst = max(worst, abs(cached - direct))
    # float32 storage of a float64 equity -> ~1e-6 worst case; well under 1e-4.
    check('cached equity == direct (3000 random)', worst < 1e-4, f"max|err|={worst:.2e}")


def test_suit_isomorphism():
    """Relabel suits on BOTH hole and board; equity must be identical, and the
    second lookup must be served from the SAME cache entry (no new board pass)."""
    clear_river_board_cache()
    pv = PostflopV2()
    rng = random.Random(7)
    perm = {'H': 'S', 'S': 'H', 'D': 'C', 'C': 'D'}   # an arbitrary suit swap
    ok = True
    worst = 0.0
    for _ in range(500):
        hole, board = _deal(rng)
        e1 = pv._river_equity(hole, board)
        r_hole = tuple(perm[c[0]] + c[1] for c in hole)
        r_board = tuple(perm[c[0]] + c[1] for c in board)
        e2 = pv._river_equity(r_hole, r_board)
        worst = max(worst, abs(e1 - e2))
        if abs(e1 - e2) > 1e-6:
            ok = False
    check('suit-isomorphic boards give equal equity', ok, f"max|err|={worst:.2e}")


def test_bucket_unchanged_vs_uncached():
    """The river bucket from the (warm) cache must equal the bucket computed with
    a freshly cleared cache -- the optimization must not move any bucket."""
    rng = random.Random(99)
    deals = [_deal(rng) for _ in range(800)]

    clear_river_board_cache()
    pv_warm = PostflopV2()
    for hole, board in deals:          # warm the shared cache
        pv_warm.bucket(list(hole), list(board))
    warm = [pv_warm.bucket(list(h), list(b)) for h, b in deals]

    mism = 0
    for (hole, board), wb in zip(deals, warm):
        clear_river_board_cache()      # force a cold recompute for this one
        cold = PostflopV2().bucket(list(hole), list(board))
        if cold != wb:
            mism += 1
    check('river bucket cached == cold', mism == 0, f"{mism}/{len(deals)} mismatched")


def test_bucket_matches_float64_reference():
    """The cached river bucket must equal the bucket from a FLOAT64 equity for
    every situation -- not just match itself. This is the test the float32 store
    failed (~1/5000 bin-edge flips); the uint16 store is exact, so 0 mismatches."""
    clear_river_board_cache()
    pv = PostflopV2()
    rng = random.Random(2024)
    mism = []
    for _ in range(5000):
        hole, board = _deal(rng)
        got = pv.bucket(list(hole), list(board))   # cache (uint16) path
        ref = _ref_river_bucket(pv, hole, board)   # float64 reference
        if got != ref:
            mism.append((hole, board, got, ref))
    check('cached river bucket == float64 reference (5000)', not mism, mism[:3])


def test_hole_on_board_raises():
    """A hero card sitting on the board is invalid input; it must fail loud, not
    silently negative-index the equity array."""
    clear_river_board_cache()
    pv = PostflopV2()
    board = ('SA', 'SK', 'SQ', 'SJ', 'S9')
    raised = False
    try:
        pv._river_equity(('SA', 'H2'), board)      # 'SA' is on the board
    except ValueError:
        raised = True
    check('hole-on-board raises ValueError', raised)


def test_lru_eviction_bounded_and_correct():
    """With a small cap, the LRU cache stays bounded (no wholesale clear / no
    unbounded growth) and still returns exact equities on a post-eviction miss."""
    import src.abstractions.postflop_v2 as pv2mod
    clear_river_board_cache()
    pv = PostflopV2()
    old_cap = pv2mod._RIVER_BOARD_CACHE_CAP
    pv2mod._RIVER_BOARD_CACHE_CAP = 50
    try:
        rng = random.Random(5)
        deals = [_deal(rng) for _ in range(400)]   # ~400 distinct canonical boards
        for hole, board in deals:
            pv.bucket(list(hole), list(board))
        check('lru size stays <= cap', len(pv2mod._RIVER_BOARD_CACHE) <= 50,
              len(pv2mod._RIVER_BOARD_CACHE))
        # Recompute-on-miss after eviction must still be exact.
        worst = max(abs(pv._river_equity(h, b) - _direct_equity(h, b))
                    for h, b in deals[-15:])
        check('lru correctness after eviction', worst < 1e-4, f"max|err|={worst:.2e}")
    finally:
        pv2mod._RIVER_BOARD_CACHE_CAP = old_cap
        clear_river_board_cache()


def _run_all():
    test_cached_equity_matches_direct()
    test_suit_isomorphism()
    test_bucket_unchanged_vs_uncached()
    test_bucket_matches_float64_reference()
    test_hole_on_board_raises()
    test_lru_eviction_bounded_and_correct()
    print(f"\n{_passed} passed, {_failed} failed")
    return _failed == 0


if __name__ == '__main__':
    sys.exit(0 if _run_all() else 1)
