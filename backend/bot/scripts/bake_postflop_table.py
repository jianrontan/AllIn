# backend/bot/scripts/bake_postflop_table.py
"""
Bake a canonical (hole, board) -> bucket lookup table for the flop or turn,
using the Phase-A centroids. River is NOT baked (1-D equity, thresholded at
runtime). One-time OFFLINE step; the artifact is reused by every training run.

BOARD-CENTRIC (fast): we iterate canonical BOARDS, and for each board compute
every completed-board's win-rate-vs-uniform for ALL hands at once
(postflop_features.board_winrates -- one ranking pass shared across all holes),
then histogram each hero's win-rates over runouts into its equity distribution
and assign the nearest centroid. This reuses each board's ranking across ~1000+
holes, turning a ~6-day naive bake into ~1-2 hours.

Saved as two parallel sorted arrays in analysis/abstractions/postflop_table_<street>.npz:
    ids[i] : int64 canonical situation id (sorted) ; buckets[i] : uint8 bucket
Runtime lookup is np.searchsorted(ids, sid); misses fall back to lazy compute,
so a partial/interrupted bake is still usable.

Run from backend/bot/ (long-running -- use your own terminal, like training):
    python scripts/bake_postflop_table.py --street flop
    python scripts/bake_postflop_table.py --street turn
Smoke (process N boards, report, do NOT save):
    python scripts/bake_postflop_table.py --street flop --limit-boards 30
"""
import argparse
import os
import random
import sys
import time
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.abstractions.postflop_features import (   # noqa: E402
    DECK, STREET_BOARD, _CARD_IDX, _ABSTRACTIONS_DIR,
    board_winrates, assign, load_centroids, encode_situation, centroid_hash)
from src.abstractions.canonical import canonical_key   # noqa: E402


def _canonical_boards(nboard):
    """One representative concrete board per canonical board class."""
    reps = {}
    for board in combinations(DECK, nboard):
        k = canonical_key((), board)
        if k not in reps:
            reps[k] = list(board)
    return list(reps.values())


def _process_board(street, board, centroids, bins, flop_runouts):
    """Yield (situation_id, bucket) for every canonical hole on this board."""
    nboard = STREET_BOARD[street]
    remaining = [c for c in DECK if c not in set(board)]
    heroes = list(combinations(remaining, 2))                 # 49->1176 / 48->1128
    # hero id (c1*52+c2) -> row, via a dense lookup array
    hero_row = np.full(52 * 52, -1, dtype=np.int64)
    for row, h in enumerate(heroes):
        hero_row[_CARD_IDX[h[0]] * 52 + _CARD_IDX[h[1]]] = row

    # runouts: river -> board already complete, a single empty runout (the
    # hand's equity is one number, so its distribution is a spike). turn ->
    # each of the ~48 rivers (exact, cheap). flop -> turn+river pairs, SAMPLED
    # (shared across all holes) unless flop_runouts == 0 (enumerate).
    if nboard == 5:
        runouts = [()]
    elif nboard == 4:
        runouts = [(c,) for c in remaining]
    elif flop_runouts and flop_runouts < len(remaining) * (len(remaining) - 1) // 2:
        # #8 fix: seed the runout sample PER BOARD (deterministic, independent of
        # board processing order and of any global seed), so a re-bake is
        # bit-identical given the same centroids/flop_runouts. That makes the
        # centroid stamp sufficient to detect a stale table. (Previously a single
        # shared RNG made each board's sample depend on processing order, so two
        # bakes with different seeds/orders silently produced different tables
        # under the same centroid stamp.) A str seed hashes deterministically
        # across processes (unlike Python's salted hash()).
        board_rng = random.Random('flop-runouts|' + ''.join(sorted(board)))
        runouts = [tuple(board_rng.sample(remaining, 2)) for _ in range(flop_runouts)]
    else:
        runouts = list(combinations(remaining, 2))

    W = np.full((len(heroes), len(runouts)), np.nan)
    for j, ro in enumerate(runouts):
        c1, c2, wr = board_winrates(board + list(ro))
        rows = hero_row[c1 * 52 + c2]                         # all valid (subset of heroes)
        W[rows, j] = wr

    seen = set()
    for i, h in enumerate(heroes):
        vals = W[i][~np.isnan(W[i])]
        hist = np.histogram(vals, bins=bins, range=(0.0, 1.0))[0].astype(float)
        s = hist.sum()
        if s > 0:
            hist /= s
        sid = encode_situation(canonical_key(h, board))
        if sid in seen:
            continue
        seen.add(sid)
        yield sid, assign(hist, centroids)


def _save(street, ids, buckets):
    os.makedirs(_ABSTRACTIONS_DIR, exist_ok=True)
    order = np.argsort(ids)
    out = os.path.join(_ABSTRACTIONS_DIR, f'postflop_table_{street}.npz')
    # Stamp the table with the centroids it was baked from (+ K, bins) so a load
    # can detect a stale table after centroids are regenerated (C2/M3).
    centroids, bins = load_centroids(street)
    np.savez(out, ids=np.asarray(ids, np.int64)[order],
             buckets=np.asarray(buckets, np.uint8)[order],
             centroid_hash=np.array(centroid_hash(centroids, bins)),
             n_buckets=np.array(len(centroids)),
             bins=np.array(int(bins)))
    return out


def main():
    p = argparse.ArgumentParser(description="Bake canonical postflop bucket table (board-centric).")
    p.add_argument('--street', choices=['flop', 'turn', 'river'], required=True)
    p.add_argument('--save-every', type=int, default=500, help="Checkpoint every N boards.")
    p.add_argument('--limit-boards', type=int, default=0, help="Smoke: N boards, no save.")
    p.add_argument('--flop-runouts', type=int, default=200,
                   help="Sampled turn+river runouts per flop board (0 = enumerate all 1176).")
    p.add_argument('--seed', type=int, default=42,
                   help="(unused) the flop runout sample is now seeded deterministically "
                        "per board (#8); kept for CLI back-compat.")
    args = p.parse_args()

    centroids, bins = load_centroids(args.street)
    print(f"Baking {args.street}: K={len(centroids)} buckets, bins={bins}")
    boards = _canonical_boards(STREET_BOARD[args.street])
    print(f"canonical {args.street} boards: {len(boards)}")

    ids, buckets = [], []
    counts = np.zeros(len(centroids), dtype=np.int64)
    t0 = time.time()
    for bi, board in enumerate(boards):
        for sid, b in _process_board(args.street, board, centroids, bins,
                                     args.flop_runouts):
            ids.append(sid)
            buckets.append(b)
            counts[b] += 1
        n = bi + 1
        if args.limit_boards and n >= args.limit_boards:
            break
        if n % args.save_every == 0:
            el = time.time() - t0
            eta = el / n * (len(boards) - n) / 60
            print(f"  board {n}/{len(boards)} | {len(ids)} sits | "
                  f"{el/60:.1f}m elapsed | ETA {eta:.0f}m", flush=True)
            if not args.limit_boards:
                _save(args.street, ids, buckets)

    print(f"\n{len(ids)} canonical situations in {(time.time()-t0)/60:.1f}m")
    print("bucket distribution:", (counts / max(1, counts.sum())).round(3).tolist())
    if args.limit_boards:
        print("(--limit-boards: smoke run, not saved)")
    else:
        print(f"saved -> {os.path.relpath(_save(args.street, ids, buckets))}  ({len(ids)} entries)")


if __name__ == '__main__':
    main()
