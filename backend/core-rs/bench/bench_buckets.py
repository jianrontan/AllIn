"""Phase 2 benchmark: Rust evaluator + bucketing vs the Python reference.

Run from backend/core-rs/ after `maturin develop --release`:
    ../../.venv/bin/python bench/bench_buckets.py

Measures (Python numbers are taken live, same process, same inputs):
  * rank7 throughput (7-card hands/s): Rust batch + Python rank7 loop
    (the latter also compared with bench/python_baseline.json if present)
  * river bucketing of ONE board, all 1081 hands: cold (board-equity cache
    empty) vs cached (board cached; Python's per-situation cache dropped)
  * flop / turn table lookup throughput (Rust batch vs Python bucket loop)
Writes bench/rust_buckets.json (git-ignored: machine-specific).
"""
import json
import os
import sys
import time
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import allin_core  # noqa: E402
from bench_baseline import BOT, machine  # noqa: E402  (puts backend/bot on sys.path)
from src.abstractions import postflop_v2  # noqa: E402
from src.abstractions.postflop_features import rank7  # noqa: E402
from src.evaluation.showdown_kernel import _FULL_DECK  # noqa: E402

ABS = os.path.join(BOT, 'analysis', 'abstractions')
RIVER_BOARD = [48, 45, 42, 39, 29]          # A K Q J 9 (mixed suits)


def best_of(fn, reps):
    best = float('inf')
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def deal(rng, n, k):
    return np.array([rng.choice(52, k, replace=False) for _ in range(n)], dtype=np.uint8)


def main():
    rng = np.random.default_rng(7)
    out = {'machine': machine(), 'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S')}

    # --- rank7 -------------------------------------------------------------
    n = 1_000_000
    cards = deal(rng, n, 7)
    rust_s = best_of(lambda: allin_core.rank_batch(cards), 5)
    sub = [[_FULL_DECK[i] for i in r] for r in cards[:100_000]]
    py_s = best_of(lambda: [rank7(h) for h in sub], 1) * (n / len(sub))
    base_path = os.path.join(HERE, 'python_baseline.json')
    base = json.load(open(base_path)).get('eval', {}).get('rank7_per_s') if os.path.exists(base_path) else None
    out['rank7'] = {'rust_per_s': round(n / rust_s), 'python_per_s_live': round(n / py_s),
                    'python_per_s_baseline': base, 'speedup_vs_live': round(py_s / rust_s, 1)}

    # --- river, one board, all 1081 hands ------------------------------------
    b = allin_core.Buckets(ABS)
    live = [c for c in range(52) if c not in RIVER_BOARD]
    rows = np.array([[x, y] + RIVER_BOARD for x, y in combinations(live, 2)], dtype=np.uint8)

    def rust_cold():
        b.clear_river_cache()
        b.bucket_batch(rows)

    r_cold = best_of(rust_cold, 20)
    b.bucket_batch(rows)
    r_warm = best_of(lambda: b.bucket_batch(rows), 50)

    hb = [([_FULL_DECK[r[0]], _FULL_DECK[r[1]]], [_FULL_DECK[c] for c in r[2:]]) for r in rows]

    def py_cold():
        postflop_v2.clear_river_board_cache()
        pv = postflop_v2.PostflopV2()
        for h, bd in hb:
            pv.bucket(h, bd)

    def py_warm():
        pv = postflop_v2.PostflopV2()             # fresh per-situation cache; board cache warm
        for h, bd in hb:
            pv.bucket(h, bd)

    p_cold = best_of(py_cold, 3)
    py_warm()
    p_warm = best_of(py_warm, 3)
    out['river_one_board_1081'] = {
        'rust_cold_ms': round(r_cold * 1e3, 3), 'rust_cached_ms': round(r_warm * 1e3, 3),
        'python_cold_ms': round(p_cold * 1e3, 2), 'python_cached_ms': round(p_warm * 1e3, 2),
        'speedup_cold': round(p_cold / r_cold, 1), 'speedup_cached': round(p_warm / r_warm, 1)}

    # --- flop / turn table lookups -------------------------------------------
    t = time.perf_counter()
    b.preload()
    out['table_load_s'] = round(time.perf_counter() - t, 3)
    for street, nb in (('flop', 3), ('turn', 4)):
        m = 200_000
        rr = deal(rng, m, 2 + nb)
        rs = best_of(lambda: b.bucket_batch(rr), 3)
        pv = postflop_v2.PostflopV2()
        sub = [([_FULL_DECK[r[0]], _FULL_DECK[r[1]]], [_FULL_DECK[c] for c in r[2:]]) for r in rr[:20_000]]
        pv.bucket(*sub[0])                         # load the table outside the timing
        ps = best_of(lambda: [pv.bucket(h, bd) for h, bd in sub], 1) * (m / len(sub))
        out[f'{street}_lookup'] = {'rust_per_s': round(m / rs), 'python_per_s': round(m / ps),
                                   'speedup': round(ps / rs, 1)}

    print(json.dumps({k: v for k, v in out.items() if k != 'machine'}, indent=2))
    with open(os.path.join(HERE, 'rust_buckets.json'), 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
