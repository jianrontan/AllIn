# backend/bot/scripts/compare_jam_fingerprint.py
"""
TEST-2 (C measurement) -- jam-fingerprint COMPARISON across two blueprint arms.

An earlier throwaway probe filtered `if 'allin' not in legal:
continue`, which is UNFAIR across the Fix-#4 A/B: the capped arm has no voluntary
all-in in its menu (all-in only EMERGES at low SPR), so that filter would skip every
high-SPR capped node and under-sample it. This script measures menu-agnostic,
visit-weighted quantities that are comparable across the control (4-size + voluntary
all-in) and capped (5-size + emergent-only all-in) arms.

Two headline numbers per arm, per street, visit-weighted over postflop decision
nodes that COULD act aggressively (some bet_/raise_ or allin in the menu):

  * P(all-in): average probability mass the blueprint puts on the literal 'allin'
    action. The over-jamming symptom. Lower under capped = the fix worked.
  * P(top-aggr): mass on the BIGGEST proposal the arm has -- 'allin' for control;
    'allin' OR the 2.0x 'overbet2' for capped. This is the fair cross-arm analogue
    ("how much mass piles onto max aggression"): the cap is SUPPOSED to move stray
    high-SPR jam mass DOWN into overbet2, not eliminate aggression -- so P(all-in)
    should drop a lot while P(top-aggr) drops less (mass relocated, not deleted).

Visit-weighted so well-trained nodes dominate (raw per-key averages over-weight the
long tail of barely-visited keys -- and the capped arm has more keys at equal iters).

Run from backend/bot/:
    python scripts/compare_jam_fingerprint.py \
        --control analysis/blueprints/blueprint_par_20260601_191133.db \
        --capped  analysis/blueprints/blueprint_par_capped_20260601_204425.db
"""
import argparse
import os
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.storage.blueprint_db import BlueprintDB

_STREETS = ('flop', 'turn', 'river')
# "top aggression" actions per arm (the biggest proposal the menu offers).
_TOP_AGGR = {'control': {'allin'},
             'capped': {'allin', 'bet_overbet2', 'raise_overbet2'}}


def _scan(db_path, arm, min_visits=0):
    db = BlueprintDB(db_path, read_only=True)
    con = sqlite3.connect('file:' + db_path + '?mode=ro', uri=True)
    keys = [r[0] for r in con.execute('select key from info_sets')]
    con.close()

    # per street: visit-weighted sums
    agg = {s: {'visits': 0.0, 'allin': 0.0, 'top': 0.0, 'nodes': 0} for s in _STREETS}
    top_set = _TOP_AGGR[arm]
    for k in keys:
        parts = k.split('_')
        street = next((s for s in _STREETS if s in parts), None)
        if street is None:
            continue
        rec = db.get_record(k)
        if not rec:
            continue
        legal = rec.get('legalActions') or []
        # node must permit aggression (a sized bet/raise OR an emergent all-in)
        if not any(a.startswith(('bet_', 'raise_')) or a == 'allin' for a in legal):
            continue
        vc = float(rec.get('visitCount') or 0)
        if vc <= 0 or vc < min_visits:
            continue
        strat = rec.get('strategy') or {}
        p_allin = float(strat.get('allin', 0.0))
        p_top = sum(float(strat.get(a, 0.0)) for a in top_set)
        a = agg[street]
        a['visits'] += vc
        a['allin'] += vc * p_allin
        a['top'] += vc * p_top
        a['nodes'] += 1
    db.close()
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--control', required=True)
    ap.add_argument('--capped', required=True)
    ap.add_argument('--min-visits', type=int, default=0,
                    help="only count keys with >= this visit_count (converged-node "
                         "subset; cuts the under-convergence confound at low budget)")
    args = ap.parse_args()

    ctrl = _scan(args.control, 'control', args.min_visits)
    capd = _scan(args.capped, 'capped', args.min_visits)
    if args.min_visits:
        print(f"[only keys with visit_count >= {args.min_visits}]")

    print(f"Jam-fingerprint comparison (visit-weighted, postflop aggressive nodes)\n")
    print(f"  control: {args.control}")
    print(f"  capped : {args.capped}\n")
    print(f"  {'street':<7} | {'arm':<8} | {'nodes':>7} | {'P(all-in)':>10} | {'P(top-aggr)':>12}")
    print(f"  {'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*10}-+-{'-'*12}")
    tot = {'control': {'v': 0.0, 'ai': 0.0, 'tp': 0.0},
           'capped': {'v': 0.0, 'ai': 0.0, 'tp': 0.0}}
    for s in _STREETS:
        for arm, data in (('control', ctrl), ('capped', capd)):
            d = data[s]
            v = d['visits'] or 1.0
            print(f"  {s:<7} | {arm:<8} | {d['nodes']:>7,} | "
                  f"{d['allin']/v:>10.4f} | {d['top']/v:>12.4f}")
            t = tot[arm]
            t['v'] += d['visits']; t['ai'] += d['allin']; t['tp'] += d['top']
        print(f"  {'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*10}-+-{'-'*12}")
    print()
    for arm in ('control', 'capped'):
        t = tot[arm]
        v = t['v'] or 1.0
        print(f"  OVERALL {arm:<8}: P(all-in) {t['ai']/v:.4f}   "
              f"P(top-aggr) {t['tp']/v:.4f}")
    ca, cp = tot['control'], tot['capped']
    dv_ai = cp['ai']/(cp['v'] or 1) - ca['ai']/(ca['v'] or 1)
    print(f"\n  P(all-in) delta (capped - control): {dv_ai:+.4f}"
          f"   <- want NEGATIVE (the cap pulled stray jam mass down)")
    print(f"  Note: capped P(top-aggr) includes its 2.0x tier, so compare it to "
          f"control's P(all-in)\n  to see if max-aggression mass was RELOCATED (good) "
          f"vs ELIMINATED.")


if __name__ == '__main__':
    main()
