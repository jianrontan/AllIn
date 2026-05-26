# backend/bot/tests/perf_seed_compare.py
"""
Seed-compare harness for performance refactors of the CFR trainer.

A speed change is only safe if it leaves the trained blueprint UNCHANGED. This
runs train_blueprint(N) in-memory under a fixed RNG seed and snapshots every
info set (regrets + cumulative strategy + visit counters) and (ev_sum, ev_count).
Capture a baseline on the current code, make the change, then re-run and compare:

    python tests/perf_seed_compare.py save  baseline.pkl   # before the change
    python tests/perf_seed_compare.py check baseline.pkl   # after the change

`check` exits nonzero if anything differs. By default it requires EXACT bit
equality (the small wins must be trajectory-preserving). Pass --tol 1e-9 only for
a change that knowingly reorders a floating-point sum.
"""
import argparse
import contextlib
import io
import os
import pickle
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.blueprint_trainer import BlueprintTrainer

DEFAULT_N = 3000
DEFAULT_SEED = 7


def run_snapshot(n, seed):
    random.seed(seed)
    trainer = BlueprintTrainer()
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):       # silence the trainer's logging
        trainer.train_blueprint(n, db=None)
    elapsed = time.time() - t0
    info = {
        key: {
            'regrets': dict(s.cumulative_regrets),
            'strategy': dict(s.cumulative_strategy),
            'visit_count': s.visit_count,
            'last_visited_iteration': s.last_visited_iteration,
            'strategy_visit_count': s.strategy_visit_count,
            'last_strategy_iteration': s.last_strategy_iteration,
        }
        for key, s in trainer.info_sets.items()
    }
    meta = {'ev_sum': trainer.ev_sum, 'ev_count': trainer.ev_count,
            'n': n, 'seed': seed}
    return info, meta, elapsed


def _diff_value(a, b, tol):
    if tol <= 0:
        return a != b
    try:
        return abs(float(a) - float(b)) > tol
    except (TypeError, ValueError):
        return a != b


def compare(base, cur, tol):
    base_info, base_meta = base
    cur_info, cur_meta = cur
    problems = []
    if set(base_info) != set(cur_info):
        only_b = set(base_info) - set(cur_info)
        only_c = set(cur_info) - set(base_info)
        problems.append(f"info-set key sets differ: {len(only_b)} only-baseline, "
                        f"{len(only_c)} only-current (e.g. {list(only_b)[:3] or list(only_c)[:3]})")
        return problems
    for key in base_info:
        b, c = base_info[key], cur_info[key]
        for field in ('visit_count', 'last_visited_iteration',
                      'strategy_visit_count', 'last_strategy_iteration'):
            if b[field] != c[field]:
                problems.append(f"{key}.{field}: {b[field]} != {c[field]}")
        for d in ('regrets', 'strategy'):
            if set(b[d]) != set(c[d]):
                problems.append(f"{key}.{d} action sets differ")
                continue
            for a in b[d]:
                if _diff_value(b[d][a], c[d][a], tol):
                    problems.append(f"{key}.{d}[{a}]: {b[d][a]!r} != {c[d][a]!r}")
        if len(problems) > 40:
            problems.append("... (truncated)")
            break
    for k in ('ev_sum', 'ev_count'):
        if _diff_value(base_meta[k], cur_meta[k], tol):
            problems.append(f"meta.{k}: {base_meta[k]!r} != {cur_meta[k]!r}")
    return problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['save', 'check'])
    p.add_argument('path')
    p.add_argument('-n', type=int, default=DEFAULT_N)
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)
    p.add_argument('--tol', type=float, default=0.0,
                   help="abs tolerance (0 = exact). Use only for a known sum reorder.")
    args = p.parse_args()

    info, meta, elapsed = run_snapshot(args.n, args.seed)
    rate = args.n / elapsed if elapsed else 0.0
    print(f"ran N={args.n} seed={args.seed}: {len(info)} info sets, "
          f"{elapsed:.1f}s ({rate:.1f} it/s)")

    if args.mode == 'save':
        with open(args.path, 'wb') as f:
            pickle.dump((info, meta), f)
        print(f"baseline saved -> {args.path}")
        return 0

    with open(args.path, 'rb') as f:
        base = pickle.load(f)
    problems = compare(base, (info, meta), args.tol)
    if problems:
        print(f"\nFAIL: {len(problems)} difference(s) (tol={args.tol}):")
        for pr in problems[:40]:
            print(f"  {pr}")
        return 1
    print(f"\nPASS: blueprint is identical to baseline (tol={args.tol}).")
    return 0


if __name__ == '__main__':
    sys.exit(main())
