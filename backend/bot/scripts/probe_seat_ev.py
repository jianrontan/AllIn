# backend/bot/scripts/probe_seat_ev.py
"""
READ-ONLY probe contrasting the two strategies inside a CFR blueprint:

  EV(current) : value of the CURRENT regret-matched strategy -- THIS is what the
                trainer prints as EV(round)/EV(cum). By CFR theory the current
                iterate need NOT converge; it can oscillate forever. Expected to
                be noisy/high.
  EV(average) : value of the AVERAGE strategy = the blueprint we actually SERVE.
                This is the iterate CFR guarantees converges to equilibrium. For
                a healthy blueprint its P0 self-play value is SMALL (near the
                game value), even while EV(current) sits high.

If EV(average) is small while EV(current) is high -> the high EV gauge is
measuring the wrong object; the served blueprint is fine. If EV(average) is ALSO
high (and not shrinking across snapshots) -> a real problem.

Writes NOTHING to disk -- safe against a blueprint a training run holds open.

Run from backend/bot/ (oldest -> newest to read the trend):
    python scripts/probe_seat_ev.py --iters 20000 \
        analysis/blueprints/snapshots/snap_20260601_204425_4000000.db \
        analysis/blueprints/snapshots/snap_20260601_204425_7008000.db \
        analysis/blueprints/blueprint_par_capped_20260601_204425.db \
        analysis/blueprints/blueprint_par_20260529_233056.db
"""
import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.blueprint_db import BlueprintDB
from src.cfr.blueprint_trainer import BlueprintTrainer
from src.abstractions.sizing import db_menu_mode


def probe(path, iters, seed):
    db = BlueprintDB(path, read_only=True)
    mode = db_menu_mode(db)
    total = db.get_metadata('total_iterations', 0)
    trainer = BlueprintTrainer(menu_mode=mode)
    trainer.resume_from_db(db)
    db.close()

    # --- EV(current): what the trainer prints. _run_iteration uses get_strategy
    #     (current/regret-matched). Mutates the in-memory base negligibly; db=None
    #     so nothing is written. Split by traverser parity (P0 vs P1).
    random.seed(seed); np.random.seed(seed)
    cur_even, cur_odd = [], []
    for i in range(iters):
        u = trainer._run_iteration(i)
        (cur_even if i % 2 == 0 else cur_odd).append(u)
    cur = np.array(cur_even + cur_odd)

    # --- EV(average): the SERVED blueprint. Uses the trainer's own
    #     evaluate_served_ev (the SAME code path training prints), so the probe and
    #     the live gauge can't drift.
    avg_mean = trainer.evaluate_served_ev(n=iters, seed=seed)

    def ci(a):
        se = a.std(ddof=1) / len(a) ** 0.5 if len(a) > 1 else 0.0
        return 1.96 * se

    print(f"\n{os.path.basename(path)} | menu={mode} | iters={total:,} | "
          f"infosets={len(trainer.info_sets):,}")
    print(f"  EV(current, regret-match): {cur.mean():+7.2f}  (+/-{ci(cur):.2f})   "
          f"[P0 {np.mean(cur_even):+.2f} / P1 {np.mean(cur_odd):+.2f}]  <- the trainer's EV gauge")
    print(f"  EV(average, SERVED bp)   : {avg_mean:+7.2f}   "
          f"<- the blueprint we actually play (evaluate_served_ev)")
    return total, cur.mean(), avg_mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dbs', nargs='+')
    p.add_argument('--iters', type=int, default=20000)
    p.add_argument('--seed', type=int, default=12345)
    args = p.parse_args()
    print(f"probe: {args.iters:,} iters/arm, seed={args.seed}")
    rows = [probe(path, args.iters, args.seed) for path in args.dbs]
    print("\n=== trend: does the SERVED blueprint's EV settle while the gauge stays high? ===")
    print(f"  {'iters':>12} | {'EV(current gauge)':>18} | {'EV(average=served)':>18}")
    for total, c, a in rows:
        print(f"  {total:>12,} | {c:>18.2f} | {a:>18.2f}")


if __name__ == '__main__':
    main()
