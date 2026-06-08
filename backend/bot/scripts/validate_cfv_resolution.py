# backend/bot/scripts/validate_cfv_resolution.py
"""
M0 step 4: does a FINER leaf partition fix the coarseness?

Steps 2-3 (validate_cfv_bucketing.py) found, on one turn board, that bucketing the
leaf at the blueprint's TURN resolution (~13 buckets) costs ~23-26% (pure hero
coarseness ~= full coarseness, so it is REAL hero-bucketing loss, not a card-removal
artifact), and degrades to ~40% under a big villain-range shift -- while a FROZEN
per-bucket scalar is ~440% wrong (reach-conditioning via the matrix is essential).

The leaf-matrix resolution is a FREE lever: the matrix need NOT use the blueprint's
turn buckets. Here we re-partition hero/villain by an OBSERVABLE, range-independent
strength scalar (mean showdown rank over runouts, `turn_strength`) at increasing bucket
counts, and ask: how fine must the leaf be to get coarseness down to ~<=10%?

  Phase A (cheap): pure-hero coarseness (bucket-average the EXACT leaf) vs n_buckets.
                   No matrix rebuild -- isolates the resolution needed.
  Phase B (confirm): rebuild the actual reach-conditioned MATRIX at the chosen n and
                   re-measure the UNDER-SHIFT error (the real solver metric).

DECISION: if a modest n (say <=64) gets coarseness AND matrix-shift to ~<=10-15%, the
v1 leaf is a finer-bucketed reach-conditioned matrix (still O(1) at runtime, just a
bigger M). If even fine n stays high, the 1-D strength partition is insufficient ->
either a 2-D feature (strength x draw/potential) or the per-hand exact leaf.

Run from backend/bot/:
    python scripts/validate_cfv_resolution.py --rivers 16
    python scripts/validate_cfv_resolution.py --rivers 16 --confirm-n 64
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.blueprint_db import BlueprintDB, FrozenBlueprint
from src.abstractions.hand_evaluator import HandEvaluator
from src.abstractions.card_abstractions import CardAbstraction
from src.abstractions.sizing import db_menu_mode, postflop_menu_for
from src.subgame.cfv import (turn_hands, turn_leaf_value_exact, turn_leaf_matrix,
                             turn_strength, equal_freq_partition,
                             bucketed_measure_leaf, bucketed_measure_leaf_cr,
                             turn_bucket, FULL_DECK)

_BOARD4 = ['CQ', 'SJ', 'H9', 'D5']
_POT = 24.0
_STACKS = (88.0, 88.0)
_BROADWAY = set('TJQKA')


def _latest_capped():
    c = sorted(glob.glob('analysis/blueprints/blueprint_par_capped_*.db'))
    return c[-1] if c else None


def _rel_rmse(pred, exact):
    keys = [h for h in exact if h in pred]
    p = np.array([pred[h] for h in keys])
    e = np.array([exact[h] for h in keys])
    denom = np.sqrt(np.mean(e ** 2)) or 1.0
    return float(np.sqrt(np.mean((p - e) ** 2)) / denom)


def _pure_coarseness(exact, part):
    """Bucket-average the EXACT leaf by partition `part`, vs the exact leaf."""
    by = {}
    for h, v in exact.items():
        by.setdefault(part.get(h), []).append(v)
    bmean = {b: float(np.mean(vs)) for b, vs in by.items()}
    bavg = {h: bmean[part[h]] for h in exact if part.get(h) in bmean}
    return _rel_rmse(bavg, exact)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=None)
    ap.add_argument('--rivers', type=int, default=16)
    ap.add_argument('--confirm-n', type=int, default=64,
                    help="bucket count at which to rebuild the matrix + measure under-shift")
    ap.add_argument('--board', default=None,
                    help="comma-separated 4-card turn board (SuitRank, e.g. SK,HK,D7,C2); "
                         "default the canonical M0 board")
    args = ap.parse_args()
    board4 = args.board.split(',') if args.board else list(_BOARD4)
    path = args.db or _latest_capped()
    rawdb = BlueprintDB(path, read_only=True)
    menu = postflop_menu_for(db_menu_mode(rawdb))
    db = FrozenBlueprint(rawdb)            # consistent snapshot (immune to live training)
    ev, cards = HandEvaluator(), CardAbstraction()

    rivers = [c for c in FULL_DECK if c not in set(board4)]
    if args.rivers > 0:
        rivers = rivers[:args.rivers]
    hands = turn_hands(board4)
    full = {h: 1.0 for h in hands}
    shifted = {h: 1.0 for h in hands if h[0][1] in _BROADWAY and h[1][1] in _BROADWAY}

    print(f"blueprint: {os.path.basename(path)} | menu={db_menu_mode(rawdb)} | board={board4} "
          f"pot={_POT} | rivers={len(rivers)}")
    print(f"R1=full ({len(full)}) -> R1'=broadway ({len(shifted)})  (a big villain-range shift)")

    # exact references (computed once, reused for every resolution)
    exact_R1 = turn_leaf_value_exact(board4, _POT, _STACKS, 0, full, full, db, ev, cards,
                                     menu=menu, rivers=rivers)
    exact_R1p = turn_leaf_value_exact(board4, _POT, _STACKS, 0, full, shifted, db, ev, cards,
                                      menu=menu, rivers=rivers)
    strength = turn_strength(board4, ev, cards, rivers=rivers)

    # baseline: the blueprint's own turn buckets
    bp_part = {h: turn_bucket(h, board4, cards) for h in hands}
    nbp = len(set(bp_part.values()))

    print(f"\nPHASE A -- pure hero-bucketing coarseness vs leaf resolution:")
    print(f"  blueprint turn buckets (n={nbp:3d}) : {_pure_coarseness(exact_R1, bp_part):6.1%}")
    for n in (16, 32, 64, 128, 256):
        part = equal_freq_partition(strength, n)
        print(f"  strength bins         (n={n:3d}) : {_pure_coarseness(exact_R1, part):6.1%}")

    # Phase B: rebuild the actual reach-conditioned matrix and measure under-shift, with
    # BOTH reconstructions (total-mass vs card-removal-aware). cr should track the pure floor.
    print(f"\nPHASE B -- reach-conditioned MATRIX under shift "
          f"(recon: total-mass | card-removal-aware):")
    print(f"  {'n':>4} {'coarse':>8} {'coarse_cr':>10} {'frozen':>8} "
          f"{'matrix':>8} {'matrix_cr':>10}")
    best = None
    for n in sorted(set([64, 128, args.confirm_n])):
        part = equal_freq_partition(strength, n)
        M, buckets, bidx, tb = turn_leaf_matrix(board4, _POT, _STACKS, 0, db, ev, cards,
                                                menu=menu, rivers=rivers, partition=part)
        pr1 = bucketed_measure_leaf(M, bidx, tb, hands, full)
        pr1c = bucketed_measure_leaf_cr(M, bidx, tb, hands, full)
        pr1p = bucketed_measure_leaf(M, bidx, tb, hands, shifted)
        pr1pc = bucketed_measure_leaf_cr(M, bidx, tb, hands, shifted)
        coarse = _rel_rmse(pr1, exact_R1)
        coarse_cr = _rel_rmse(pr1c, exact_R1)
        frozen = _rel_rmse(pr1, exact_R1p)
        matrix = _rel_rmse(pr1p, exact_R1p)
        matrix_cr = _rel_rmse(pr1pc, exact_R1p)
        print(f"  {n:>4} {coarse:>8.1%} {coarse_cr:>10.1%} {frozen:>8.1%} "
              f"{matrix:>8.1%} {matrix_cr:>10.1%}")
        if best is None or matrix_cr < best[1]:
            best = (n, matrix_cr, coarse_cr)
    rawdb.close()

    bn, bm, bc = best
    print(f"\n  best under-shift (card-removal-aware): {bm:.1%} at n={bn} "
          f"(coarseness {bc:.1%})")
    if bm <= 0.15 and bc <= 0.12:
        print("  VERDICT: finer card-removal-aware matrix leaf is ADEQUATE -> v1 = finer matrix.")
    elif bm <= 0.25:
        print("  VERDICT: borderline; a 2-D feature (strength x potential) or larger n may close it.")
    else:
        print("  VERDICT: 1-D strength insufficient -> 2-D feature or per-hand exact leaf.")


if __name__ == '__main__':
    main()
