"""
Step-0 profiler for the Lever-B decision: how much of training / inference wall
time is actually spent in the hand evaluator (postflop_features.rank7 ->
phevaluator._evaluate_cards)?

Prints the evaluator's share of total time for:
  1. TRAINING  — N single-thread CFR iterations (the real inner loop).
  2. INFERENCE — a small best-response walk (eval-heavy: board_winrates etc.).

Run from backend/bot/:
    python scripts/profile_evaluator.py --train-iters 2000 --br-samples 20
"""
import argparse
import cProfile
import os
import pstats
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.blueprint_trainer import BlueprintTrainer

# Functions that ARE the evaluator (cumtime in these = evaluator cost).
_EVAL_NAMES = ('rank7', '_evaluate_cards', 'board_winrates', 'hand_winrate')


def _eval_share(stats):
    """Sum tottime of evaluator functions and report vs total tottime."""
    total_tt = 0.0
    eval_tt = 0.0
    rows = []
    for (fn, line, name), (cc, nc, tt, ct, callers) in stats.stats.items():
        total_tt += tt
        if name in _EVAL_NAMES:
            eval_tt += tt
            rows.append((name, tt, ct, nc))
    return total_tt, eval_tt, sorted(rows, key=lambda r: -r[1])


def profile_training(iters):
    random.seed(0)
    t = BlueprintTrainer()
    for i in range(200):            # warm caches / lazy tables before timing
        t._run_iteration(i)
    pr = cProfile.Profile()
    pr.enable()
    for i in range(200, 200 + iters):
        t._run_iteration(i)
    pr.disable()
    return pstats.Stats(pr)


def profile_inference(db_path, samples):
    from src.storage.blueprint_db import BlueprintDB
    from src.evaluation.best_response import BestResponseEvaluator
    db = BlueprintDB(db_path, read_only=True)
    try:
        ev = BestResponseEvaluator(db, seed=1)
        pr = cProfile.Profile()
        pr.enable()
        ev.evaluate(num_samples=samples, progress_every=10 ** 9)
        pr.disable()
    finally:
        db.close()
    return pstats.Stats(pr)


def _report(label, stats):
    total, eval_tt, rows = _eval_share(stats)
    print(f"\n=== {label} ===")
    print(f"  total tottime across all funcs : {total:8.3f} s")
    print(f"  evaluator tottime              : {eval_tt:8.3f} s  "
          f"({100 * eval_tt / total:.1f}% of total)")
    print(f"  {'function':<22}{'tottime':>10}{'cumtime':>10}{'ncalls':>12}")
    for name, tt, ct, nc in rows:
        print(f"  {name:<22}{tt:>10.3f}{ct:>10.3f}{nc:>12,}")
    print("  top 8 by tottime:")
    stats.sort_stats('tottime')
    stats.print_stats(8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-iters', type=int, default=2000)
    p.add_argument('--br-samples', type=int, default=20)
    p.add_argument('--db', default=None, help="blueprint DB for the inference profile")
    p.add_argument('--skip-inference', action='store_true')
    args = p.parse_args()

    _report(f"TRAINING ({args.train_iters} iters)", profile_training(args.train_iters))

    if not args.skip_inference:
        db = args.db
        if db is None:
            from src.config import resolve_blueprint_path
            db = str(resolve_blueprint_path())
        print(f"\n(inference profile using blueprint: {os.path.basename(db)})")
        _report(f"INFERENCE / best-response ({args.br_samples} samples)",
                profile_inference(db, args.br_samples))


if __name__ == '__main__':
    main()
