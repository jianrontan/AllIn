# backend/bot/src/abstractions/postflop_v2.py
"""
Version-2 postflop bucketing: distribution-aware (potential-aware) buckets.

This is the ONLY postflop scheme now -- CardAbstraction delegates to it directly
(the legacy BoardTextureEvaluator heuristic is dead). It changes the postflop
info-set keys versus v1, so v1-trained blueprints are incompatible and must be
retrained.

How a postflop bucket is produced:
  * flop / turn : canonicalise (hole, board) -> integer id -> O(log n) lookup in
                  the pre-baked table (scripts/bake_postflop_table.py). On a miss
                  (or absent table) fall back to computing the equity-distribution
                  feature and assigning the nearest centroid -- correct but slow,
                  so the baked table is what keeps training fast.
  * river       : NO table by design -- equity vs a uniform range is a single
                  number, so we build its (spike) histogram and assign the
                  nearest river centroid, cached per canonical situation. A full
                  river table is impractical (~90M canonical situations); the
                  per-situation cache plus a vectorized per-board equity pass
                  (board_winrates, shared by both players) keeps this cheap.

Bucket counts follow the centroids (12 flop / 12 turn / 10 river by default).
"""
import os
import random
import warnings
from collections import OrderedDict

import numpy as np

from .canonical import canonical_key, canonical_board_perm
from .postflop_features import (
    load_centroids, encode_situation, equity_distribution, assign,
    board_winrates, centroid_hash, _CARD_IDX, _ABSTRACTIONS_DIR)

_STREET = {3: 'flop', 4: 'turn', 5: 'river'}

# Per-board river equities, keyed on the CANONICAL (suit-isomorphic) board so all
# 19.3 concrete boards in a suit-orbit share one board_winrates() pass. There are
# only 134,459 canonical 5-card boards (vs 2,598,960 concrete), so a long run does
# board_winrates ~134k times TOTAL instead of ~once per concrete board.
#
# This is a MODULE-LEVEL cache on purpose: it must survive across PostflopV2
# instances. The parallel trainer builds a fresh BlueprintTrainer (hence a fresh
# PostflopV2) every merge round, so an instance-level cache would go cold every
# round and never warm up. A module global lives for the worker PROCESS's
# lifetime, so each persistent worker warms it across rounds. Equity is a pure
# function of the board, so sharing across instances is always sound.
#
# Eviction is LRU (OrderedDict): on overflow the least-recently-used board is
# dropped, NOT the whole cache -- so a cap BELOW the 134,459 canonical boards
# degrades gracefully (keeps the hottest boards) instead of thrashing on a
# wholesale clear.
#
# Memory: each entry is ~2.6 KB effective (a 52-slot int8 position map + a
# 1081-uint16 equity array + numpy/dict overhead). The default 100k cap is
# ~0.26 GB per process; in parallel that is x(workers) -- ~2.1 GB at 8 workers,
# ~1.6 GB at 6. Lower ALLIN_RIVER_CACHE_BOARDS to fit a tighter RAM budget (LRU
# keeps it effective even when small); raise toward 134,459 to cache every
# canonical board on a roomy box. board_winrates is suit-blind, so a cap >=
# 134,459 never evicts.
_RIVER_BOARD_CACHE = OrderedDict()
_RIVER_BOARD_CACHE_CAP = int(os.environ.get('ALLIN_RIVER_CACHE_BOARDS', 100_000))
# Combination-index constants for the 47 live cards on a complete (river) board.
_RIVER_LIVE = 47
_RIVER_2NM1 = 2 * _RIVER_LIVE - 1            # = 93, used in the pair->slot formula
# Equity on a complete board is (win + 0.5*tie)/990 over a CONSTANT 990 disjoint
# opponents, so 2*990*equity is an exact integer in [0, 1980] -- the scale we
# store equities at (uint16) for exactness + compactness.
_RIVER_EQ_SCALE = 1980.0


def clear_river_board_cache():
    """Drop the shared per-board river-equity cache (for tests / memory control)."""
    _RIVER_BOARD_CACHE.clear()


class PostflopV2:
    # Cap the per-instance river-BUCKET cache: canonical river situations number
    # ~90M, so over a long run dealt hands rarely repeat and an uncapped cache
    # would grow without bound. Cleared wholesale on overflow. (The expensive
    # per-board equity pass is cached separately and shared -- see
    # _RIVER_BOARD_CACHE above -- so a bucket-cache miss is now cheap.)
    _RIVER_CACHE_CAP = 500_000

    def __init__(self, seed=0, lazy_runouts=120, lazy_opp=150):
        self.rng = random.Random(seed)
        self.lazy_runouts = lazy_runouts
        self.lazy_opp = lazy_opp
        self._tables = {}        # street -> (ids_sorted, buckets) or None
        self._centroids = {}     # street -> (centroids, bins)
        self._river_cache = {}   # canonical_key -> bucket
        self._warned = set()

    def _centroid(self, street):
        if street not in self._centroids:
            self._centroids[street] = load_centroids(street)
        return self._centroids[street]

    def _table(self, street):
        if street not in self._tables:
            path = os.path.join(_ABSTRACTIONS_DIR, f'postflop_table_{street}.npz')
            if os.path.exists(path):
                d = np.load(path)
                self._verify_stamp(street, d)
                self._tables[street] = (d['ids'], d['buckets'])
            else:
                self._tables[street] = None
        return self._tables[street]

    def _verify_stamp(self, street, d):
        """Guard against a stale table: confirm it was baked from the centroids
        in use (centroid hash) with the same K and bins (C2/M3). A legacy table
        with no stamp warns once and proceeds (so pre-stamp bakes keep working
        until the next re-bake); a stamp MISMATCH is a hard error."""
        if 'centroid_hash' not in getattr(d, 'files', []):
            if street not in self._warned:
                warnings.warn(
                    f"postflop_table_{street}.npz has no centroid stamp (legacy "
                    f"bake) -- cannot verify it matches the current centroids. "
                    f"Re-bake to enable the consistency check.")
                self._warned.add(street)
            return
        centroids, bins = self._centroid(street)
        want = centroid_hash(centroids, bins)
        got = str(d['centroid_hash'])
        if got != want:
            raise ValueError(
                f"postflop_table_{street}.npz was baked from DIFFERENT centroids "
                f"(stamp {got[:8]}... != current {want[:8]}...). The table is stale; "
                f"re-bake it: python scripts/bake_postflop_table.py --street {street}")
        nb, tb = int(d['n_buckets']), int(d['bins'])
        if nb != len(centroids) or tb != bins:
            raise ValueError(
                f"postflop_table_{street}.npz K/bins ({nb}/{tb}) != current centroids "
                f"({len(centroids)}/{bins}); re-bake --street {street}.")

    # ------------------------------------------------------------------
    def bucket(self, hole, board):
        street = _STREET[len(board)]
        centroids, bins = self._centroid(street)
        ck = canonical_key(tuple(hole), tuple(board))
        sid = encode_situation(ck)

        # All three streets use the same baked-table fast path; only the
        # lazy fallback (on a table miss or absent table) differs by street.
        tab = self._table(street)
        if tab is not None:
            ids, buckets = tab
            i = int(np.searchsorted(ids, sid))
            if i < len(ids) and ids[i] == sid:
                return int(buckets[i])

        if street == 'river':
            return self._river_bucket_lazy(ck, hole, board, centroids, bins)
        self._warn_once(street)
        feat = equity_distribution(list(hole), list(board), bins,
                                   self.lazy_runouts, self.lazy_opp, self.rng)
        return assign(feat, centroids)

    def _river_bucket_lazy(self, ck, hole, board, centroids, bins):
        """River bucketing (no baked table -- by design). Equity vs a uniform
        range is a single number, so build its spike histogram and assign the
        nearest river centroid. Cached per canonical situation. A full river
        table is impractical (~90M canonical situations); the per-situation
        cache plus the per-board equity cache keeps this cheap at runtime."""
        cached = self._river_cache.get(ck)
        if cached is not None:
            return cached
        eq = self._river_equity(hole, board)
        hist = np.zeros(bins)
        hist[min(int(eq * bins), bins - 1)] = 1.0      # spike at the equity value
        b = assign(hist, centroids)
        if len(self._river_cache) >= self._RIVER_CACHE_CAP:
            self._river_cache.clear()
        self._river_cache[ck] = b
        return b

    def _river_equity(self, hole, board):
        """Equity (win + 0.5*tie) vs a uniform range on a complete 5-card board.

        board_winrates ranks EVERY hand on the board in one vectorized pass. We
        run it once per CANONICAL board and share the result across the whole
        suit-orbit (19.3 concrete boards each): the concrete board is mapped to
        its canonical representative, the hero's hole cards are relabeled by the
        SAME suit permutation, and the cached equity is looked up. Equity is
        suit-invariant, so this is exact."""
        canon_board, smap = canonical_board_perm(board)
        entry = _RIVER_BOARD_CACHE.get(canon_board)
        if entry is None:
            entry = self._compute_board_entry(canon_board)
            _RIVER_BOARD_CACHE[canon_board] = entry              # most-recently-used
            if len(_RIVER_BOARD_CACHE) > _RIVER_BOARD_CACHE_CAP:
                _RIVER_BOARD_CACHE.popitem(last=False)           # evict least-recently-used
        else:
            _RIVER_BOARD_CACHE.move_to_end(canon_board)          # mark recently used
        pos, eq = entry
        # Relabel the hero hand by the board's suit permutation, then index the
        # canonical board's equity array.
        ca = _CARD_IDX[smap[hole[0][0]] + hole[0][1]]
        cb = _CARD_IDX[smap[hole[1][0]] + hole[1][1]]
        i, j = int(pos[ca]), int(pos[cb])
        if i < 0 or j < 0:
            # A hero card sits on the board -- an invalid (hole, board). Fail loud
            # instead of negative-indexing the equity array and returning garbage.
            raise ValueError(f"river hole {tuple(hole)} overlaps board {tuple(board)}")
        if i > j:
            i, j = j, i
        slot = i * (_RIVER_2NM1 - i) // 2 + (j - i - 1)
        return float(eq[slot]) / _RIVER_EQ_SCALE

    @staticmethod
    def _compute_board_entry(canon_board):
        """Build the compact cache entry for one canonical river board:
          pos : int8[52], global card-idx -> position in the 47 live cards (-1 if
                on the board); used to address the packed equity array.
          eq  : uint16[1081], the EXACT per-hand equity numerator in combination
                (i<j over live cards) order -- the order the pos/slot formula
                reproduces. On a complete (5-card) board every hand faces a
                CONSTANT 990 disjoint opponents, so equity = (win + 0.5*tie)/990
                and 2*990*equity = 2*win + tie is an integer in [0, 1980]. Storing
                that integer is EXACT (no float rounding that could flip a bin-edge
                river bucket) and uses half the memory of float32.
        board_winrates returns hands in combinations(live cards, 2) order with the
        live cards already in ascending card-idx order, so wr is already in slot
        order and needs no reordering."""
        c1, c2, wr = board_winrates(list(canon_board))
        on_board = {_CARD_IDX[c] for c in canon_board}
        live = [idx for idx in range(52) if idx not in on_board]   # ascending
        pos = np.full(52, -1, dtype=np.int8)
        for p, idx in enumerate(live):
            pos[idx] = p
        eq = np.rint(wr.astype(np.float64) * _RIVER_EQ_SCALE).astype(np.uint16)
        return pos, eq

    def _warn_once(self, street):
        if street not in self._warned:
            warnings.warn(
                f"postflop_table_{street}.npz missing/incomplete; bucketing lazily "
                f"(slow). Bake it with scripts/bake_postflop_table.py.")
            self._warned.add(street)
