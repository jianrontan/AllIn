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
    args = parser.parse_args()

    db_path = args.db or resolve_blueprint_path()
    print(f"Blueprint : {db_path}")
    print(f"Samples   : {args.samples}  (seed {args.seed})")

    db = BlueprintDB(db_path, read_only=True)
    try:
        ev = BestResponseEvaluator(db, seed=args.seed)
        t0 = time.time()
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
