"""
Research the river-bucketing cache for the Lever-B alternative.

Answers three questions with real numbers:
  (A) How many distinct CANONICAL 5-card boards are there? -> the steady-state
      number of board_winrates() computations if we key the board-equity cache on
      the canonical (suit-isomorphic) board instead of the concrete board.
  (B) During real training, how many river bucket() calls happen per iteration,
      and what fraction currently MISS the concrete-board cache (= board_winrates
      calls)?
  (C) Projected board_winrates calls + cache hit-rate for a long run under the
      current (concrete) key vs a canonical-board key.

Run from backend/bot/:  python scripts/measure_river_cache.py --iters 4000
"""
import argparse
import os
import sys
import time
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---- (A) exact canonical 5-card board count via a fast suit signature -------- #
# A board's suit-isomorphism class is fully described by the multiset of
# {sorted ranks present in each suit}. Sorting those 4 per-suit rank-tuples makes
# the signature invariant to suit relabeling -- the same equivalence canonical.py
# uses for a board with no hole cards, but O(1) instead of 24 permutations.
def canonical_board_count():
    from src.abstractions.postflop_features import DECK, RANKS
    rank_ord = {r: i for i, r in enumerate(RANKS)}
    sigs = set()
    t0 = time.time()
    n = 0
    for board in combinations(DECK, 5):
        by_suit = {}
        for c in board:                      # c = 'HA' (suit, rank)
            by_suit.setdefault(c[0], []).append(rank_ord[c[1]])
        sig = tuple(sorted(tuple(sorted(v)) for v in by_suit.values()))
        sigs.add(sig)
        n += 1
    return len(sigs), n, time.time() - t0


# ---- (B) instrument PostflopV2 during real training -------------------------- #
def measure_training(iters):
    import random
    import src.abstractions.postflop_v2 as pv2
    from src.cfr.blueprint_trainer import BlueprintTrainer

    stats = {'bw_calls': 0, 'river_calls': 0, 'river_cache_hits': 0}
    concrete_boards = set()
    canonical_boards = set()

    rank_ord = {r: i for i, r in enumerate(
        ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'])}

    def board_sig(board):
        by_suit = {}
        for c in board:
            by_suit.setdefault(c[0], []).append(rank_ord[c[1]])
        return tuple(sorted(tuple(sorted(v)) for v in by_suit.values()))

    orig_bw = pv2.board_winrates
    def counting_bw(board):
        stats['bw_calls'] += 1
        return orig_bw(board)
    pv2.board_winrates = counting_bw

    orig_lazy = pv2.PostflopV2._river_bucket_lazy
    def counting_lazy(self, ck, hole, board, centroids, bins):
        stats['river_calls'] += 1
        if ck in self._river_cache:
            stats['river_cache_hits'] += 1
        concrete_boards.add(frozenset(board))
        canonical_boards.add(board_sig(board))
        return orig_lazy(self, ck, hole, board, centroids, bins)
    pv2.PostflopV2._river_bucket_lazy = counting_lazy

    random.seed(0)
    t = BlueprintTrainer()
    t0 = time.time()
    for i in range(iters):
        t._run_iteration(i)
    dt = time.time() - t0

    pv2.board_winrates = orig_bw
    pv2.PostflopV2._river_bucket_lazy = orig_lazy

    stats['distinct_concrete'] = len(concrete_boards)
    stats['distinct_canonical'] = len(canonical_boards)
    stats['secs'] = dt
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iters', type=int, default=4000)
    p.add_argument('--skip-count', action='store_true',
                   help="skip the (slow) full canonical-board enumeration")
    args = p.parse_args()

    print("=== (B) river bucketing during training ===")
    s = measure_training(args.iters)
    rc, bw = s['river_calls'], s['bw_calls']
    print(f"  iters                       : {args.iters:,}  ({s['secs']:.1f}s)")
    print(f"  river bucket() calls        : {rc:,}  ({rc/args.iters:.3f} per iter)")
    print(f"  river_cache hits            : {s['river_cache_hits']:,}")
    print(f"  board_winrates calls (misses): {bw:,}")
    print(f"  distinct CONCRETE boards    : {s['distinct_concrete']:,}")
    print(f"  distinct CANONICAL boards   : {s['distinct_canonical']:,}")
    if s['distinct_concrete']:
        print(f"  concrete->canonical reduction: "
              f"{s['distinct_concrete']/max(1,s['distinct_canonical']):.2f}x")

    if not args.skip_count:
        print("\n=== (A) total canonical 5-card boards (full enumeration) ===")
        ncanon, ntotal, dt = canonical_board_count()
        print(f"  concrete 5-card boards      : {ntotal:,}")
        print(f"  canonical 5-card boards     : {ncanon:,}  ({dt:.1f}s)")
        print(f"  global reduction factor     : {ntotal/ncanon:.2f}x")

        print("\n=== (C) projection for a long run ===")
        for total_iters in (5_000_000, 30_000_000):
            rcalls = (rc / args.iters) * total_iters
            # concrete key: ~one board_winrates per river call until the (capped,
            # cleared) cache happens to hold it -- but with 2.6M concrete boards
            # and a 20k cap it thrashes, so misses ~ min(rcalls, 2.6M) and the
            # cache barely helps. canonical key: at most `ncanon` computations ever.
            canon_calls = min(rcalls, ncanon)
            print(f"  at {total_iters:,} iters: ~{rcalls:,.0f} river calls")
            print(f"     canonical-key board_winrates : ~{canon_calls:,.0f}  "
                  f"(hit-rate ~{100*(1-canon_calls/rcalls):.1f}%)")


if __name__ == '__main__':
    main()
