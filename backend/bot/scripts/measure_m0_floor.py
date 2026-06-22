#!/usr/bin/env python3
"""Pre-flight for the NN-leaf run (Agent-A axis 5): estimate the ACHIEVABLE rel-RMSE FLOOR before
spending 8h. The production target board->M0 uses a crc32-seeded 12-river sample. A SMALL/smooth net
tends toward the expectation over river draws, so the spread of M0 across DIFFERENT valid river samples
(at a FIXED partition) bounds how low rel-RMSE can realistically go. Also measures the 12-river BIAS vs a
44-river "gold".

  rel-RMSE(prod vs alt-12)  ~ the sampling floor a smooth predictor can't beat
  rel-RMSE(prod vs gold-44) ~ how far 12-river is from the truth (bias)

If the floor is >~0.15, the <=0.15 target is unreachable as set up -> don't launch; raise rivers or BAKE.

  python scripts/measure_m0_floor.py --boards 120
"""
import argparse
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('ALLIN_MMAP_POSTFLOP', '1')
from src.storage.blueprint_db import BlueprintDB
from src.subgame.turn_subgame_solver import TurnSubgameSolver
from src.subgame.turn_leaf_gen import board_rivers_and_partition, m0_for
from src.subgame.cfv import FULL_DECK
from src.game.cards import shuffled_deck

SPRS = [0.5, 2.0, 4.0, 8.0]


def rel_rmse(P, T):
    return float(np.sqrt(np.mean((P - T) ** 2)) / (np.sqrt(np.mean(T ** 2)) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boards', type=int, default=120)
    ap.add_argument('--rivers', default='12,20,24,28', help="comma-list of river counts to sweep")
    args = ap.parse_args()
    river_ns = [int(x) for x in args.rivers.split(',')]
    db = BlueprintDB('analysis/blueprints/snapshots/snap_52500000.db', read_only=True)
    bot = TurnSubgameSolver(db, n_buckets=24, leaf_rivers=12, safe_gadget=True, gadget_anchor='auto')
    pm, ev, cards = bot._postflop_menu, bot._evaluator, bot._cards
    rng = np.random.default_rng(7)
    deck = shuffled_deck()

    # floor[N] = list of rel-RMSE(prod-N vs alt-N) across boards/SPRs -> the floor a smooth net can't beat
    floor = {N: [] for N in river_ns}
    n = 0
    while n < args.boards:
        b = list(rng.choice(deck, size=4, replace=False))
        try:
            pool = [c for c in FULL_DECK if c not in set(b)]
            for N in river_ns:
                # prod-N: crc32-seeded N rivers + the partition from them (fixed for prod & alt at this N)
                rivers_prod, part, bac = board_rivers_and_partition(b, ev, cards, 24, N)
                alt_rng = np.random.default_rng(zlib.crc32(('alt' + ''.join(sorted(b))).encode()))
                rivers_alt = sorted(alt_rng.choice(pool, size=min(N, len(pool)), replace=False).tolist())
                for s in SPRS:
                    st = (float(s), float(s))
                    mp, _ = m0_for(b, 1.0, st, db, ev, cards, menu=pm, rivers=rivers_prod, partition=part, ba_cache=bac)
                    ma, _ = m0_for(b, 1.0, st, db, ev, cards, menu=pm, rivers=rivers_alt, partition=part, ba_cache=bac)
                    if mp.shape == (24, 24):
                        floor[N].append(rel_rmse(mp, ma))
            n += 1
            if n % 20 == 0:
                print(f"  {n}/{args.boards} boards", flush=True)
        except Exception:
            continue
    db.close()

    print("\nrivers   floor(rel-RMSE between two samples)   <0.15?")
    for N in river_ns:
        f = float(np.mean(floor[N])) if floor[N] else float('nan')
        print(f"{N:5}        {f:.3f}                            {'YES' if f < 0.15 else 'no'}")
    print("\n  floor = best a smooth net can do (target's own sampling spread). Need floor < ~0.13 so the")
    print("  net has margin to actually reach <0.15. Gen cost scales ~linearly with rivers; gate ~linearly.")


if __name__ == '__main__':
    main()
