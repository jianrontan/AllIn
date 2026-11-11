# backend/bot/scripts/br_attr.py
"""
Best response with per-(street, bucket) gain attribution, one blueprint, one seed.
Same boards as tests/run_evaluation.py for the same (seed, samples). The gains are
weighted by the hero's blueprint reach, so per seat they sum to BR minus the
blueprint's own value, and over both seats to the exploitability.

Run from backend/bot/ (~10 min per board per worker):
    python scripts/br_attr.py DB SEED OUT.json --samples 24 --workers 12
"""
import argparse
import json
import os
import sys
import time

BOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, BOT)
from src.storage.blueprint_db import BlueprintDB  # noqa: E402
from src.evaluation.best_response import BestResponseEvaluator  # noqa: E402

def main():
    p = argparse.ArgumentParser()
    p.add_argument('db')
    p.add_argument('seed', type=int)
    p.add_argument('out')
    p.add_argument('--samples', type=int, default=24)
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    db = BlueprintDB(a.db, read_only=True)
    t0 = time.time()
    ev = BestResponseEvaluator(db, seed=a.seed)
    r = ev.evaluate_attribution(num_samples=a.samples, workers=a.workers, seed=a.seed)
    gain = r.pop('gain')
    to_mbb = r.pop('gain_mbb_to')
    seat_street = r.pop('gain_mbb_by_seat_street')
    by_street = {}
    for (_seat, st), v in seat_street.items():
        by_street[st] = by_street.get(st, 0.0) + v
    r.update(db=a.db, seed=a.seed, menu_mode=ev.menu_mode, seconds=time.time() - t0,
             gain_mbb_by_street={str(k): v for k, v in sorted(by_street.items())},
             gain_mbb_by_seat_street={f'{s}-{st}': v for (s, st), v in sorted(seat_street.items())},
             gain_mbb_sum=sum(by_street.values()),
             gain_mbb=[[st, bk, v * to_mbb] for (st, bk), v in sorted(gain.items())])
    with open(a.out, 'w') as f:
        json.dump(r, f, indent=1)
    print(a.out, 'exploitability', round(r['exploitability_mbb'], 1), 'gain sum',
          round(r['gain_mbb_sum'], 1), r['gain_mbb_by_street'])


if __name__ == '__main__':      # Pool workers re-import this module (spawn)
    main()
