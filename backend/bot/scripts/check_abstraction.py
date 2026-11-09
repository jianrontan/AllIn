# backend/bot/scripts/check_abstraction.py
"""
Check an abstraction's bucket files before training on them (the V3.2/V3.3 gates).

  1. Its tables pass `verify_tables` (stamps, K/bins, manifest, complete bake).
  2. The flop and turn tables cover exactly the root's canonical situations
     (when the root tables are present to compare with).
  3. Independent re-bucketing: on random boards (not the bake's representative
     ones), every hand's histogram over ALL runouts, assigned to the nearest
     centroid, equals the table entry. Turn always (the bake enumerates the 46
     rivers); flop only when the manifest says flop_runouts = 0 (a sampled bake
     depends on which representative board was baked, so it can't be redone
     from another board).
  4. Rust (`allin_core.Buckets`) buckets the same hands and random river spots
     exactly like Python, if the module is built.

Run from backend/bot/ (under a minute with the defaults):
    python scripts/check_abstraction.py --abstraction v3
"""
import argparse
import os
import random
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.abstractions.abstraction_set import (   # noqa: E402
    abstraction_dir, abstraction_id, read_manifest, verify_tables)
from src.abstractions.canonical import canonical_key   # noqa: E402
from src.abstractions.postflop_features import (   # noqa: E402
    DECK, _CARD_IDX, assign, board_winrates, encode_situation, load_centroids)
from src.abstractions.postflop_v2 import PostflopV2   # noqa: E402


def _rows(hands, board):
    return np.array([[_CARD_IDX[c] for c in list(h) + board] for h in hands], dtype=np.uint8)


def _rebucket(board, cent, bins):
    """Each hand's bucket from its histogram over every runout to the river."""
    rem = [c for c in DECK if c not in board]
    heroes = list(combinations(rem, 2))
    row = np.full(52 * 52, -1, dtype=np.int64)
    for r, h in enumerate(heroes):
        row[_CARD_IDX[h[0]] * 52 + _CARD_IDX[h[1]]] = r
    runouts = list(combinations(rem, 5 - len(board)))
    W = np.full((len(heroes), len(runouts)), np.nan)
    for j, ro in enumerate(runouts):
        c1, c2, wr = board_winrates(board + list(ro))
        W[row[c1 * 52 + c2], j] = wr
    out = []
    for i in range(len(heroes)):
        v = W[i][~np.isnan(W[i])]
        hist = np.histogram(v, bins=bins, range=(0.0, 1.0))[0].astype(float)
        out.append(assign(hist / hist.sum(), cent))
    return heroes, out


def main():
    p = argparse.ArgumentParser(description="Check an abstraction's bucket files.")
    p.add_argument('--abstraction', default=None, help="name (default: the root, v2)")
    p.add_argument('--flops', type=int, default=10, help="random flops to re-bucket")
    p.add_argument('--turns', type=int, default=30, help="random turns to re-bucket")
    p.add_argument('--river-spots', type=int, default=20_000)
    p.add_argument('--seed', type=int, default=3)
    args = p.parse_args()

    adir = abstraction_dir(args.abstraction)
    manifest = read_manifest(adir)
    print(f"{adir}\n  abstraction_id {abstraction_id(adir)}  manifest {manifest}")
    print(f"  verify_tables: {verify_tables(adir)} ok")
    try:
        import allin_core
        rs = allin_core.Buckets(adir)
        rs.preload()
    except ImportError:
        rs = None
        print("  allin_core not built: skipping the Rust comparisons")
    py = PostflopV2(abstraction_dir=adir)
    failures = 0

    root = abstraction_dir(None)
    for street in ('flop', 'turn'):
        here = os.path.join(adir, f'postflop_table_{street}.npz')
        ref = os.path.join(root, f'postflop_table_{street}.npz')
        if not os.path.exists(here):
            continue
        with np.load(here) as d:
            ids = d['ids']
        if adir != root and os.path.exists(ref):
            with np.load(ref) as r:
                same = np.array_equal(r['ids'], ids)
            print(f"  {street} table: {len(ids):,} situations, same set as the root: {same}")
            failures += not same
        else:
            print(f"  {street} table: {len(ids):,} situations")

    rng = random.Random(args.seed)
    plan = [('turn', 4, args.turns)]
    if manifest['flop_runouts'] == 0:
        plan.insert(0, ('flop', 3, args.flops))
    else:
        print(f"  flop: sampled bake (flop_runouts={manifest['flop_runouts']}), not re-bucketed")
    for street, nboard, n in plan:
        cent, bins = load_centroids(street, adir)
        boards = [rng.sample(DECK, nboard) for _ in range(n)]
        if street == 'flop':
            boards[0] = ['SQ', 'C9', 'C3']              # the BUG-027 flop
        bad_py = bad_rs = tot = 0
        for board in boards:
            heroes, want = _rebucket(board, cent, bins)
            got_py = [py.bucket(list(h), board) for h in heroes]
            bad_py += sum(a != b for a, b in zip(got_py, want))
            if rs is not None:
                got_rs = rs.bucket_batch(_rows(heroes, board))
                bad_rs += sum(int(a) != b for a, b in zip(got_rs, want))
            tot += len(heroes)
        print(f"  {street}: re-bucketed {tot:,} hands on {n} random boards: "
              f"{bad_py} Python mismatches" + ('' if rs is None else f", {bad_rs} Rust mismatches"))
        failures += bad_py + bad_rs

    if rs is not None:
        hands = [rng.sample(DECK, 7) for _ in range(args.river_spots)]
        want = np.array([py.bucket(c[:2], c[2:]) for c in hands])
        got = np.asarray(rs.bucket_batch(np.array([[_CARD_IDX[x] for x in c] for c in hands],
                                                  dtype=np.uint8)))
        k = rs.n_buckets(2)
        occ = np.bincount(want, minlength=k)
        print(f"  river: {args.river_spots:,} random spots, K={k}, "
              f"{int((got != want).sum())} Rust mismatches, empty buckets "
              f"{np.flatnonzero(occ == 0).tolist()}")
        failures += int((got != want).sum()) + int((occ == 0).sum())

    print("OK" if not failures else f"FAILED ({failures})")
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
