"""Compare complete solve-control calls; run under Docker --cpus=.25 for quota tests.

Reports wall-clock samples and p95, including construction and convergence checks.
These synthetic spots measure solver latency, not HTTP end-to-end latency.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.abstractions.card_abstractions import CardAbstraction
from src.abstractions.hand_evaluator import HandEvaluator
from src.evaluation.showdown_kernel import build_board_arrays
from src.subgame.river_tree import build_river_tree
from src.subgame.solve_control import solve_river, solve_river_gadget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['python', 'rust'], default='rust')
    parser.add_argument('--algorithm', choices=['cfrplus', 'dcfr'], default='cfrplus')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--iters', type=int, default=275)
    args = parser.parse_args()
    if args.repeats < 1 or args.iters < 1:
        parser.error('repeats and iters must be positive')
    os.environ['ALLIN_RIVER_BACKEND'] = args.backend
    os.environ['ALLIN_RIVER_ALGORITHM'] = args.algorithm
    ba = build_board_arrays(['HA', 'DK', 'CQ', 'SJ', 'H9'], HandEvaluator(),
                            CardAbstraction(), streets=(3,))
    rng = np.random.default_rng(371)
    r0, r1 = rng.random((2, ba['H'])) + .1
    for pot, stack in [(12., 194.), (30., 185.), (80., 160.)]:
        tree = build_river_tree(pot, (stack, stack))
        for gadget in (False, True):
            samples, gaps, cpu_samples = [], [], []
            for _ in range(args.repeats):
                start = time.perf_counter()
                cpu_start = time.process_time()
                if gadget:
                    _, info = solve_river_gadget(tree, ba, r0, r1, np.zeros(ba['H']), 1,
                                                 max_iters=args.iters, check_every=40)
                else:
                    _, info = solve_river(tree, ba, r0, r1, max_iters=args.iters,
                                          check_every=40, gap_threshold=-1)
                samples.append(time.perf_counter() - start)
                cpu_samples.append(time.process_time() - cpu_start)
                gaps.append(info['gap'])
            print(json.dumps(dict(backend=args.backend, algorithm=args.algorithm, pot=pot, stack=stack,
                                  gadget=gadget, iterations=args.iters,
                                  seconds=samples, cpu_seconds=cpu_samples, p95=float(np.percentile(samples, 95)),
                                  gaps=gaps)), flush=True)


if __name__ == '__main__':
    main()
