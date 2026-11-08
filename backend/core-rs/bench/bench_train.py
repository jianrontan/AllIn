"""Rust MCCFR training throughput (iterations/s), single-thread and parallel.

Run from backend/core-rs/ after `maturin develop --release`:
    ../../.venv/bin/python bench/bench_train.py [--iters 20000] [--workers 1,4,8,12]
Writes bench/rust_train.json and prints the speedup over the Python baseline
(bench/python_baseline.json, single core) when present.
"""
import argparse
import json
import os
import time

import allin_core

HERE = os.path.dirname(os.path.abspath(__file__))
ABS_DIR = os.path.normpath(os.path.join(HERE, '..', '..', 'bot', 'analysis', 'abstractions'))


def run(mode, iters, workers, merge_every, seed, warm):
    t = allin_core.Trainer(mode)
    t.load_abstraction(ABS_DIR)
    if warm:
        # Warm the canonical river-board cache like a long run would be.
        t.train(warm, 0, seed + 1, workers=max(workers, 1), merge_every=merge_every)
        t = _fresh(t, mode)
    t0 = time.time()
    s = t.train(iters, 0, seed, workers=workers, merge_every=merge_every)
    dt = time.time() - t0
    return {'mode': mode, 'workers': workers, 'iterations': s['iterations'],
            'seconds': dt, 'it_per_s': s['iterations'] / dt, 'info_sets': t.num_info_sets}


def _fresh(t, mode):
    return t   # cache lives in the loaded abstraction; reuse the same trainer object


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--workers', default='1,4,8,12')
    ap.add_argument('--merge-every', type=int, default=2000)
    ap.add_argument('--mode', default='capped')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    out = []
    for w in [int(x) for x in args.workers.split(',')]:
        r = run(args.mode, args.iters if w == 1 else args.iters * w, w, args.merge_every,
                args.seed, warm=0)
        out.append(r)
        print(f"  {r['mode']:8s} workers={w:2d}  {r['iterations']:>8,} it in {r['seconds']:7.2f}s"
              f"  = {r['it_per_s']:>9,.0f} it/s  ({r['info_sets']:,} info sets)", flush=True)
    base = os.path.join(HERE, 'python_baseline.json')
    if os.path.exists(base):
        with open(base) as f:
            py = json.load(f)
        ips = (py.get('training') or {}).get('it_per_s')
        if ips:
            for r in out:
                print(f"  speedup vs Python single-core ({ips:.0f} it/s): "
                      f"workers={r['workers']}: {r['it_per_s'] / ips:,.1f}x")
    with open(os.path.join(HERE, 'rust_train.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
