"""
Measure the steady-state training speedup from the canonical river-board cache.

The benefit only shows when the cache is WARM (boards repeat), so we replay the
SAME seeded iterations under two regimes and compare wall time:

  WARM     : cache enabled, second pass over boards already seen -> ~all hits
             (the new steady state on a long run).
  BASELINE : cache cleared at the START OF EACH ITERATION -> within-iteration
             reuse kept (both players share a board, as before) but NO cross-
             iteration reuse -- i.e. the old steady state, where random concrete
             boards almost never repeat across iterations.

speedup = baseline_time / warm_time.

Run from backend/bot/:  python scripts/measure_river_speedup.py --iters 2500
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.abstractions.postflop_v2 as pv2
from src.abstractions.postflop_v2 import clear_river_board_cache
from src.cfr.blueprint_trainer import BlueprintTrainer


def _run(iters, seed, clear_each_iter):
    random.seed(seed)
    t = BlueprintTrainer()
    bw = {'n': 0}
    orig = pv2.board_winrates
    def counting(board):
        bw['n'] += 1
        return orig(board)
    pv2.board_winrates = counting
    try:
        t0 = time.time()
        for i in range(iters):
            if clear_each_iter:
                clear_river_board_cache()
            t._run_iteration(i)
        dt = time.time() - t0
    finally:
        pv2.board_winrates = orig
    return dt, bw['n']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iters', type=int, default=2500)
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()
    N, S = args.iters, args.seed

    # Warm-up pass (fills the shared cache with this workload's boards).
    clear_river_board_cache()
    _run(N, S, clear_each_iter=False)

    # WARM timed pass: same seed, cache already holds these boards.
    warm_dt, warm_bw = _run(N, S, clear_each_iter=False)

    # BASELINE timed pass: clear cross-iteration reuse (old steady state).
    clear_river_board_cache()
    base_dt, base_bw = _run(N, S, clear_each_iter=True)

    print(f"\n=== river-cache training speedup ({N:,} iters, seed {S}) ===")
    print(f"  BASELINE (no cross-iter reuse): {base_dt:6.2f}s  "
          f"{N/base_dt:6.1f} it/s   board_winrates={base_bw:,}")
    print(f"  WARM     (canonical cache hot): {warm_dt:6.2f}s  "
          f"{N/warm_dt:6.1f} it/s   board_winrates={warm_bw:,}")
    print(f"  speedup                       : {base_dt/warm_dt:.3f}x  "
          f"(board_winrates calls {base_bw:,} -> {warm_bw:,})")


if __name__ == '__main__':
    main()
