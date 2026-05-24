# backend/bot/tests/run_match.py
"""
Head-to-head: play blueprint A vs blueprint B and report A's win rate, both RAW
and AIVAT-corrected (variance-reduced). Use this to compare blueprint versions
(e.g. old gamma=0 vs new gamma=2) far more cheaply than raw.

Usage (from backend/bot/):
    # New 6M gamma=2 (A) vs old 6.5M gamma=0 (B):
    python tests/run_match.py \
        --db-a analysis/blueprint_20260523_171956.db \
        --db-b analysis/blueprint_20260521_170429.db --hands 20000

    # Sanity: a blueprint vs itself (expect ~0):
    python tests/run_match.py --hands 10000
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import resolve_blueprint_path
from src.storage.blueprint_db import BlueprintDB
from src.evaluation.match import HeadToHeadMatch
from src.evaluation.aivat import AIVATEstimator


def main():
    p = argparse.ArgumentParser(description="Head-to-head match with AIVAT.")
    p.add_argument('--db-a', default=None, help="Player A blueprint (default: active).")
    p.add_argument('--db-b', default=None, help="Player B blueprint (default: active).")
    p.add_argument('--hands', type=int, default=20000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    a_path = args.db_a or resolve_blueprint_path()
    b_path = args.db_b or resolve_blueprint_path()
    print(f"A : {a_path}")
    print(f"B : {b_path}")
    print(f"Hands: {args.hands}  (seed {args.seed})  [A's perspective]")

    da = BlueprintDB(a_path, read_only=True)
    db = BlueprintDB(b_path, read_only=True)
    try:
        rec = []
        t0 = time.time()
        HeadToHeadMatch(da, db, seed=args.seed).evaluate(
            num_hands=args.hands, record=rec, progress_every=max(1, args.hands // 10))
        res = AIVATEstimator(db, seed=args.seed).estimate(
            rec, progress_every=max(1, args.hands // 10))
        elapsed = time.time() - t0
    finally:
        da.close()
        db.close()

    print("\n--- A vs B  (mbb/hand, A's perspective; >0 means A wins) ---")
    print(f"  raw   : {res['raw_mbb']:+8.1f}  +/- {res['raw_stderr_mbb']:6.1f}")
    print(f"  AIVAT : {res['aivat_mbb']:+8.1f}  +/- {res['aivat_stderr_mbb']:6.1f}"
          f"   ({100 * res['var_reduction']:.0f}% variance reduced)")
    print(f"  (computed in {elapsed:.1f}s)")


if __name__ == '__main__':
    main()
