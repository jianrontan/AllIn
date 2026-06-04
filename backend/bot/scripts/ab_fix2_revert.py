# backend/bot/scripts/ab_fix2_revert.py
"""
*** HISTORICAL -- DO NOT RE-RUN (kept as the record of how BUG-014 was confirmed). ***
The env flag this relied on (ALLIN_WORKER_FLOOR_REGRET) was REMOVED when the Fix-#2
revert landed: workers now ALWAYS floor (canonical CFR+). So both arms below now floor
identically -- re-running prints a meaningless null result (both fold). The original
decisive run (RAW pf_0 fold 1% vs FLOORED 74%) is recorded in BUG-014 / the
fix2-parallel-floor-reverted memo. Left here only for provenance.

A/B test for the Fix-#2 open-collapse regression (2026-06-04).

Trains two SHORT capped parallel runs from scratch and compares the preflop OPEN
strategy of the weakest buckets:
  RAW      = current trainer (Fix #2: workers store raw regret) -- expected to
             collapse to "always bet_xlarge, never fold" (the bug).
  FLOORED  = ALLIN_WORKER_FLOOR_REGRET=1 -> per-worker CFR+ flooring restored (the
             pre-Fix-#2 path) -- expected to FOLD trash like blueprint 233056 did
             (pf_0 folded ~79% by 500k).

If FLOORED folds trash and RAW doesn't, the Fix-#2 raw-regret merge is confirmed as
the cause and per-worker flooring as the fix.

Run from backend/bot/:
    python scripts/ab_fix2_revert.py 500000 8
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.blueprint_trainer import BlueprintTrainer
from src.cfr.parallel_trainer import train_blueprint_parallel
from src.storage.blueprint_db import BlueprintDB


def run_arm(floored, iters, workers, label):
    if floored:
        os.environ['ALLIN_WORKER_FLOOR_REGRET'] = '1'      # candidate fix
    else:
        os.environ.pop('ALLIN_WORKER_FLOOR_REGRET', None)  # current (raw) default
    tmp = os.path.join(tempfile.gettempdir(), f'ab_fix2_{label}.db')
    for ext in ('', '-wal', '-shm'):
        try:
            os.remove(tmp + ext)
        except OSError:
            pass
    print(f"\n=== ARM {label} (floored={floored}) : {iters:,} iters, {workers} workers ===",
          flush=True)
    db = BlueprintDB(tmp, read_only=False)
    t = BlueprintTrainer(menu_mode='capped')
    train_blueprint_parallel(t, iterations=iters, db=db, workers=workers,
                             merge_every=2000, checkpoint_every=max(250000, iters // 2),
                             seed=1)
    db.close()
    return tmp


def report(path, label):
    db = BlueprintDB(path, read_only=True)

    def g(k, a):
        return (db.get_average_strategy(k) or {}).get(a, 0.0)
    print(f"\n[{label}] iters={db.get_metadata('total_iterations')}")
    for n in (0, 2, 4, 6):
        k = f'pf_{n}_ip_'
        print(f"  pf_{n}: xlarge={g(k,'bet_xlarge'):.0%} large={g(k,'bet_large'):.0%} "
              f"med={g(k,'bet_medium'):.0%} small={g(k,'bet_small'):.0%} "
              f"call={g(k,'call'):.0%} FOLD={g(k,'fold'):.0%}")
    f0 = g('pf_0_ip_', 'fold')
    db.close()
    return f0


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 500000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    print(f"A/B Fix-#2 revert: {iters:,} iters/arm, {workers} workers, capped menu")
    raw_db = run_arm(floored=False, iters=iters, workers=workers, label='RAW_current')
    flr_db = run_arm(floored=True, iters=iters, workers=workers, label='FLOORED_fix')
    print("\n" + "=" * 64)
    raw_fold = report(raw_db, 'RAW_current (Fix #2)')
    flr_fold = report(flr_db, 'FLOORED_fix (per-worker CFR+)')
    print("\n=== VERDICT ===")
    print(f"  pf_0 (trash) FOLD%:  RAW={raw_fold:.0%}   FLOORED={flr_fold:.0%}")
    if flr_fold > 0.30 and raw_fold < 0.10:
        print("  -> CONFIRMED: per-worker flooring folds trash; raw collapses. Fix #2 is the cause.")
    elif flr_fold > raw_fold + 0.15:
        print("  -> LIKELY: flooring folds notably more trash than raw (run longer to be sure).")
    else:
        print("  -> INCONCLUSIVE at this iter count -- run more iterations.")


if __name__ == '__main__':
    main()
