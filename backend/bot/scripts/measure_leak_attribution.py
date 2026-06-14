# backend/bot/scripts/measure_leak_attribution.py
"""
Attribute the blueprint's best-response exploitability to STREET and BUCKET, so a
retrain's extra postflop buckets are TARGETED at the leak instead of guessed.

Two modes (both reuse the parallel BR walk in best_response.py):

  * default -- ONE full-BR pass with per-hero-node GAIN logging. The gain at a node
    = best-action value minus the blueprint-weighted value, per hero hand, summed by
    (street, hero bucket). Summed by STREET -> the per-street leak share; by BUCKET
    within a street -> which buckets the exploiter milks most. The gain is hero-reach-
    UNWEIGHTED, so read it as a HEURISTIC ranking (the --restricted numbers are the
    clean per-street cross-check).

  * --restricted -- the clean per-street cross-check (~5x the cost): a RESTRICTED best
    response per street -- the hero may deviate ONLY on street S, blueprint elsewhere,
    so exploitability(S) is the leak reachable by deviating only on S. Also prints the
    no-deviation baseline (~0, a zero-sum sanity check) and the full BR.

Cost: a BR pass is ~30s/sample on 16 cores (see run_evaluation / br-eval-cost-warning).
SHARES are far more stable than absolute exploitability, so ~30-50 samples suffice.
Default mode = one BR pass; --restricted = ~6 passes (full + 4 streets + baseline).

Run from backend/bot/:
    python scripts/measure_leak_attribution.py --samples 40 --workers 16
    python scripts/measure_leak_attribution.py --samples 40 --workers 16 --restricted
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import resolve_blueprint_path
from src.storage.blueprint_db import BlueprintDB
from src.evaluation.best_response import BestResponseEvaluator

_STREET = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}


def _gain_report(gain):
    """Print per-street shares + top buckets per street from the gain dict."""
    per_street = {s: 0.0 for s in _STREET}
    for (s, _bk), g in gain.items():
        per_street[s] += g
    total = sum(per_street.values()) or 1.0
    print("\n  Per-STREET leak share (gain decomposition, single full-BR pass):")
    for s in _STREET:
        print(f"    {_STREET[s]:<8} {100.0 * per_street[s] / total:6.1f}%   "
              f"(gain {per_street[s]:+.1f})")
    print("\n  Top buckets per street (share OF THAT STREET's gain):")
    for s in _STREET:
        rows = sorted(((bk, g) for (ss, bk), g in gain.items() if ss == s),
                      key=lambda kv: kv[1], reverse=True)
        st = per_street[s] or 1.0
        top = rows[:6]
        cells = "  ".join(f"{str(bk)}={100.0 * g / st:.0f}%" for bk, g in top)
        print(f"    {_STREET[s]:<8} {cells}")


def main():
    p = argparse.ArgumentParser(description="Per-street / per-bucket BR leak attribution.")
    p.add_argument('--db', default=None)
    p.add_argument('--samples', type=int, default=40)
    p.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--purify', type=float, default=0.0,
                   help="Match the served purify threshold to attribute the SERVED bot.")
    p.add_argument('--restricted', action='store_true',
                   help="Also run the clean per-street restricted-BR cross-check (~5x cost).")
    args = p.parse_args()

    path = args.db or resolve_blueprint_path()
    db = BlueprintDB(path, read_only=True)
    ev = BestResponseEvaluator(db, seed=args.seed, purify_threshold=args.purify)
    print(f"Blueprint : {path}")
    print(f"Menu mode : {ev.menu_mode}")
    print(f"Samples   : {args.samples}  (seed {args.seed}, workers {args.workers}, "
          f"purify {args.purify})")

    t0 = time.time()
    # Default: one full-BR pass with gain logging -> per-street + per-bucket.
    full = ev.evaluate_attribution(args.samples, args.workers, br_streets=None,
                                   log_gain=True)
    print(f"\nFull exploitability: {full['exploitability_mbb']:.0f} mbb/hand  "
          f"(seat0 {full['br_seat0_mbb']:+.0f} / seat1 {full['br_seat1_mbb']:+.0f}; "
          f"BB/OOP = seat1)")
    _gain_report(full['gain'])

    if args.restricted:
        print("\n  Per-STREET restricted best response (clean cross-check):")
        base = ev.evaluate_attribution(args.samples, args.workers, br_streets=set(),
                                       log_gain=False)
        print(f"    baseline (no deviation, expect ~0): "
              f"{base['exploitability_mbb']:+.0f} mbb")
        for s in _STREET:
            r = ev.evaluate_attribution(args.samples, args.workers, br_streets={s},
                                        log_gain=False)
            share = 100.0 * r['exploitability_mbb'] / (full['exploitability_mbb'] or 1.0)
            print(f"    {_STREET[s]:<8} {r['exploitability_mbb']:8.0f} mbb  "
                  f"({share:5.1f}% of full)")

    db.close()
    print(f"\n(computed in {time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
