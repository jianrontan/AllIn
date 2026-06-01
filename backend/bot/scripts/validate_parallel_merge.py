# backend/bot/scripts/validate_parallel_merge.py
"""
Validation harness for Fix #2 (bias-corrected parallel merge).

The unit tests in tests/test_parallel_trainer.py prove the MECHANISM deterministically
(workers now store raw signed regret; opposite-signed deltas cancel in merge_round
before the single master floor). This harness is the STATISTICAL companion: it trains
the real game both ways for the same iteration budget and checks that the parallel
average strategy agrees with the single-thread oracle within Monte Carlo noise -- and,
crucially, that the parallel trainer no longer INFLATES the all-in / jam frequency
(the high-variance upward bias Fix #2 targets).

It is not bit-identity: the two paths seed the RNG differently (single-thread runs one
continuous stream; the parallel path re-seeds per worker per round), so the sampled
hands differ. The claim being validated is convergence agreement, not determinism.

Run from backend/bot/ (single-thread is slow -- it is the reason parallel exists, so
keep N modest for a smoke and large only when you mean it; ~200k iters ~ 25 min):

    python scripts/validate_parallel_merge.py --iters 200000 --workers 8 --merge-every 2000

Reports, over info-sets both paths visited >= --min-visits times:
  - mean / p95 / max total-variation distance between the two average strategies
  - mean all-in (jam) probability under each path, and parallel-minus-single delta
    (the bias scoreboard: it should be ~0, not a large positive number)
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.cfr.blueprint_trainer import BlueprintTrainer, _format_duration
from src.cfr.parallel_trainer import train_blueprint_parallel

# Pattern char / action names that denote a voluntary all-in / jam, for the bias score.
_JAM_ACTIONS = ('all_in', 'allin', 'jam')


def _train_single_thread(iters, seed):
    """Drive the single-thread reference by looping _run_iteration (the same call the
    parallel worker uses), so this harness needs no assumption about the
    train_blueprint() wrapper. Discount stays enabled (the CFR+ oracle)."""
    random.seed(seed)
    trainer = BlueprintTrainer()
    t0 = time.time()
    for i in range(iters):
        trainer._run_iteration(i)
    return trainer, time.time() - t0


def _train_parallel(iters, seed, workers, merge_every):
    trainer = BlueprintTrainer()
    t0 = time.time()
    train_blueprint_parallel(trainer, iters, db=None, workers=workers,
                             merge_every=merge_every, seed=seed)
    return trainer, time.time() - t0


def _avg_strategy(info):
    la = list(info.cumulative_strategy.keys()) or list(info.legal_actions)
    if not la:
        return {}
    return dict(zip(la, info.get_average_strategy(la)))


def _jam_prob(strat):
    return sum(p for a, p in strat.items() if a in _JAM_ACTIONS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=100_000)
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--merge-every', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20260531)
    ap.add_argument('--min-mass', type=float, default=30.0,
                    help="only compare info-sets whose cumulative-strategy mass is "
                         ">= this in BOTH runs. NOTE: do NOT gate on visit_count -- "
                         "it counts iterations in single-thread but merge-ROUNDS in "
                         "parallel (only ~iters/(merge_every*workers) of them), so a "
                         "fixed visit threshold excludes every parallel key. Strategy "
                         "mass accumulates comparably in both modes.")
    args = ap.parse_args()

    print(f"Validation: single-thread vs parallel ({args.workers} workers, "
          f"merge_every={args.merge_every}), {args.iters:,} iters each.\n")

    st, st_secs = _train_single_thread(args.iters, args.seed)
    print(f"  single-thread: {len(st.info_sets):,} info-sets in {_format_duration(st_secs)}")
    par, par_secs = _train_parallel(args.iters, args.seed + 1, args.workers, args.merge_every)
    print(f"  parallel     : {len(par.info_sets):,} info-sets in {_format_duration(par_secs)}")
    if par_secs > 0:
        print(f"  speedup      : {st_secs / par_secs:.2f}x\n")

    def _mass(info):
        return float(sum(info.cumulative_strategy.values()))

    tvs, jam_st, jam_par = [], [], []
    shared = 0
    for key, ist in st.info_sets.items():
        ipar = par.info_sets.get(key)
        if ipar is None:
            continue
        if _mass(ist) < args.min_mass or _mass(ipar) < args.min_mass:
            continue
        a_st, a_par = _avg_strategy(ist), _avg_strategy(ipar)
        actions = set(a_st) | set(a_par)
        if not actions:
            continue
        shared += 1
        tvs.append(0.5 * sum(abs(a_st.get(a, 0.0) - a_par.get(a, 0.0)) for a in actions))
        if any(a in _JAM_ACTIONS for a in actions):
            jam_st.append(_jam_prob(a_st))
            jam_par.append(_jam_prob(a_par))

    if not tvs:
        print("No shared high-mass info-sets -- raise --iters or lower --min-mass.")
        return

    tvs = np.array(tvs)
    print(f"Compared {shared:,} shared info-sets (cumulative-strategy mass >= "
          f"{args.min_mass:g} in both):")
    print(f"  total-variation distance  mean {tvs.mean():.4f} | "
          f"p95 {np.percentile(tvs, 95):.4f} | max {tvs.max():.4f}")
    if jam_st:
        jam_st, jam_par = np.array(jam_st), np.array(jam_par)
        print(f"\n  all-in/jam frequency (over {len(jam_st):,} keys with a jam action):")
        print(f"    single-thread mean P(jam) : {jam_st.mean():.4f}")
        print(f"    parallel      mean P(jam) : {jam_par.mean():.4f}")
        print(f"    BIAS (parallel - single)  : {jam_par.mean() - jam_st.mean():+.4f}"
              f"   <- should be ~0; a large POSITIVE value = residual bias")
    else:
        print("  (no info-sets with a jam action met the visit threshold)")


if __name__ == '__main__':
    main()
