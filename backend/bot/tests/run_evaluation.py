# backend/bot/tests/run_evaluation.py
"""
Score a blueprint's exploitability.

Usage (from backend/bot/):
    python tests/run_evaluation.py                 # active blueprint, 400 samples
    python tests/run_evaluation.py --samples 1000
    python tests/run_evaluation.py --db analysis/blueprints/blueprint_20260518_160906.db
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import resolve_blueprint_path
from src.storage.blueprint_db import BlueprintDB
from src.evaluation.best_response import BestResponseEvaluator


def main():
    parser = argparse.ArgumentParser(description="Blueprint exploitability via best response.")
    parser.add_argument('--db', default=None, help="Path to a blueprint .db (default: active).")
    parser.add_argument('--samples', type=int, default=400, help="Monte Carlo board samples.")
    parser.add_argument('--seed', type=int, default=42, help="RNG seed for reproducibility.")
    parser.add_argument('--workers', type=int, default=1,
                        help="Parallel worker processes for board samples (>1 = "
                             "parallel). Bit-identical to serial for the same "
                             "(seed, samples) -- only faster. Default 1 (serial). "
                             "Each worker opens its own read-only DB connection.")
    parser.add_argument('--purify', type=float, default=0.0,
                        help="Strategy purification threshold for the A/B (drop blueprint "
                             "actions below this prob, renormalise; >max => argmax). "
                             "0.0 = off (default). Try 0.01 (1%%) / 0.05 / 1.0 (full).")
    args = parser.parse_args()

    db_path = args.db or resolve_blueprint_path()
    print(f"Blueprint : {db_path}")
    print(f"Samples   : {args.samples}  (seed {args.seed}, workers {args.workers})")
    print(f"Purify    : {args.purify}  ({'OFF' if args.purify <= 0 else 'threshold'})")

    db = BlueprintDB(db_path, read_only=True)
    try:
        # menu_mode auto-derived from the DB metadata (control for a pre-stamp DB)
        # so a capped blueprint is walked on its own tree.
        ev = BestResponseEvaluator(db, seed=args.seed, purify_threshold=args.purify)
        print(f"Menu mode : {ev.menu_mode}")
        t0 = time.time()
        if args.workers and args.workers > 1:
            result = ev.evaluate_parallel(num_samples=args.samples, workers=args.workers)
        else:
            result = ev.evaluate(num_samples=args.samples)
        elapsed = time.time() - t0
    finally:
        db.close()

    print("\n--- Exploitability (lower = closer to unexploitable) ---")
    print(f"  BR as seat 0 (IP/SB) : {result['br_seat0_mbb']:+.1f} mbb/hand")
    print(f"  BR as seat 1 (OOP/BB): {result['br_seat1_mbb']:+.1f} mbb/hand")
    print(f"  Exploitability       : {result['exploitability_mbb']:.1f} mbb/hand")
    print(f"  (computed in {elapsed:.1f}s)")


if __name__ == '__main__':
    main()
