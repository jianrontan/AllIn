# backend/bot/scripts/check_strategy_shape.py
"""
Run the strategy-shape sanity probe (src/cfr/strategy_shape.py) against any
blueprint DB -- a live run, a snapshot, or an archived blueprint. Catches the
BUG-014 "open xlarge with 100% of hands, never fold" collapse (and the BB-vs-open
variant) that no aggregate metric (EV / LBR / AIVAT) detects. READ-ONLY.

Run from backend/bot/:
    python scripts/check_strategy_shape.py                    # active blueprint
    python scripts/check_strategy_shape.py --db analysis/blueprints/blueprint_par_capped_20260604_114512.db
    python scripts/check_strategy_shape.py --db ... --verbose # per-bucket fold table

Exit code 0 if OK/WARN, 2 if COLLAPSE -- so it can gate a script/CI loop.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import resolve_blueprint_path
from src.storage.blueprint_db import BlueprintDB
from src.cfr.strategy_shape import strategy_shape_report, format_shape_line, COLLAPSE
from src.abstractions.card_abstractions import NUM_PREFLOP_BUCKETS


def main():
    p = argparse.ArgumentParser(description="Strategy-shape sanity probe for a blueprint DB.")
    p.add_argument('--db', default=None, help="Blueprint DB path (default: active blueprint).")
    p.add_argument('--verbose', action='store_true', help="Print per-bucket fold%% tables.")
    args = p.parse_args()

    path = args.db or str(resolve_blueprint_path())
    db = BlueprintDB(path, read_only=True)
    iters = db.get_metadata('total_iterations', 0)
    menu = db.get_metadata('menu_mode')
    print(f"blueprint: {os.path.basename(path)} | iters={iters:,} | menu={menu}")

    rep = strategy_shape_report(db.get_average_strategy, num_preflop_buckets=NUM_PREFLOP_BUCKETS)
    print(format_shape_line(rep))
    print(f"  probed: {rep['n_open']} open nodes, {rep['n_bbx']} BB-vs-5BB nodes | "
          f"pf_0 open fold={_pct(rep['pf0_open_fold'])}  "
          f"strongest open fold={_pct(rep['strong_open_fold'])}")
    for r in rep['reasons']:
        print(f"    - {r}")

    if args.verbose:
        print("\n  per-bucket open fold%  (pf_N_ip_):")
        for n in sorted(rep['open_fold_by_bucket']):
            print(f"    pf_{n:<2} {rep['open_fold_by_bucket'][n]:.0%}")
        print("  per-bucket BB-vs-5BB fold%  (pf_N_oop_x):")
        for n in sorted(rep['bbx_fold_by_bucket']):
            print(f"    pf_{n:<2} {rep['bbx_fold_by_bucket'][n]:.0%}")

    db.close()
    print(f"\nVERDICT: {rep['verdict']}")
    sys.exit(2 if rep['verdict'] == COLLAPSE else 0)


def _pct(x):
    return f"{x:.0%}" if x is not None else "n/a"


if __name__ == '__main__':
    main()
