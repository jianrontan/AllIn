# backend/bot/tests/run_lbr.py
"""
Score a blueprint with Local Best Response (LBR): a greedy off-tree exploiter.
The reported mbb/hand is a LOWER bound on real exploitability -- compare it to
the in-abstraction best-response number (run_evaluation.py). LBR coming in
HIGHER means the abstraction (not just convergence) is leaking.

Usage (from backend/bot/):
    python tests/run_lbr.py                       # active blueprint, 3000 hands
    python tests/run_lbr.py --hands 5000
    python tests/run_lbr.py --db analysis/blueprint_20260521_170429.db

NOTE: this is Monte Carlo over real deals, so it is high-variance -- use several
thousand hands. AIVAT (a later harness piece) will cut the variance sharply.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import resolve_blueprint_path
from src.storage.blueprint_db import BlueprintDB
from src.evaluation.lbr import LBREvaluator


def main():
    parser = argparse.ArgumentParser(description="Blueprint Local Best Response (LBR).")
    parser.add_argument('--db', default=None, help="Path to a blueprint .db (default: active).")
    parser.add_argument('--hands', type=int, default=3000, help="Monte Carlo hands.")
    parser.add_argument('--seed', type=int, default=42, help="RNG seed.")
    parser.add_argument('--flop-samples', type=int, default=20,
                        help="Board-runout samples used in equity estimates.")
    args = parser.parse_args()

    db_path = args.db or resolve_blueprint_path()
    print(f"Blueprint : {db_path}")
    print(f"Hands     : {args.hands}  (seed {args.seed}, flop-samples {args.flop_samples})")

    db = BlueprintDB(db_path, read_only=True)
    try:
        ev = LBREvaluator(db, seed=args.seed, flop_runout_samples=args.flop_samples)
        t0 = time.time()
        result = ev.evaluate(num_hands=args.hands, progress_every=500)
        elapsed = time.time() - t0
    finally:
        db.close()

    print("\n--- Local Best Response (lower bound on exploitability) ---")
    print(f"  LBR value : {result['lbr_mbb']:+.1f} mbb/hand  over {result['num_hands']} hands")
    print(f"  (computed in {elapsed:.1f}s)")


if __name__ == '__main__':
    main()
