"""Time the Rust river solver on the same spots as bench_baseline.py and print
the speedup against bench/python_baseline.json (run that first). Run from
backend/core-rs/ after `maturin develop --release`:
    ../../.venv/bin/python bench/bench_rust.py

Writes bench/rust_river.json (git-ignored: machine-specific).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import allin_core  # noqa: E402
from bench_baseline import BOARD, RIVER_CONFIGS, machine  # noqa: E402  (also puts backend/bot on sys.path)
from src.abstractions.card_abstractions import CardAbstraction  # noqa: E402
from src.abstractions.hand_evaluator import HandEvaluator  # noqa: E402
from src.evaluation.showdown_kernel import build_board_arrays  # noqa: E402
from src.subgame.river_tree import DEFAULT_MENU, MAX_AGGRESSIONS  # noqa: E402


def timed(fn, iters):
    t = time.perf_counter()
    fn(iters)
    return (time.perf_counter() - t) / iters * 1000


def main():
    ba = build_board_arrays(BOARD, HandEvaluator(), CardAbstraction(), streets=(3,))
    H = ba['H']
    raw, c1, c2 = ba['raw'], ba['c1'], ba['c2']
    rng = np.random.default_rng(0)
    base_path = os.path.join(HERE, 'python_baseline.json')
    base = json.load(open(base_path))['river']['configs'] if os.path.exists(base_path) else []
    out = {'machine': machine(), 'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'configs': []}
    for pot, stack in RIVER_CONFIGS:
        row = {'pot_entry': pot, 'stack': stack}
        for mode, exact in (('fast', False), ('exact', True)):
            mk = lambda: allin_core.RiverCFR(pot, stack, list(DEFAULT_MENU), MAX_AGGRESSIONS,  # noqa: E731
                                             raw, c1, c2, exact=exact)
            r0, r1 = np.ones(H), np.ones(H)
            c = mk()
            c.run(r0, r1, 2)
            n = 400 if not exact else 100
            cfr_ms = timed(lambda k: c.run(r0, r1, k), n)
            t = time.perf_counter()
            c.exploitability(r0, r1)
            expl_ms = (time.perf_counter() - t) * 1000
            g = mk()
            hero, vill = rng.random(H), rng.random(H)
            optout = rng.uniform(-0.3, 0.6, H) * pot * 100
            g.run_gadget(hero, vill, optout, 1, 2)
            gad_ms = timed(lambda k: g.run_gadget(hero, vill, optout, 1, k), n)
            row['decision_nodes'] = c.n_decision
            row[mode] = {'cfr_ms_per_iter': round(cfr_ms, 3), 'gadget_ms_per_iter': round(gad_ms, 3),
                         'exploitability_ms': round(expl_ms, 3),
                         'solve_400_iters_s': round(cfr_ms * 0.4, 3)}
        py = next((b for b in base if b['pot_entry'] == pot and b['stack'] == stack), None)
        if py:
            row['speedup_fast_vs_python'] = {
                'cfr': round(py['cfr_ms_per_iter'] / row['fast']['cfr_ms_per_iter'], 1),
                'gadget': round(py['gadget_ms_per_iter'] / row['fast']['gadget_ms_per_iter'], 1),
                'exploitability': round(py['exploitability_ms'] / row['fast']['exploitability_ms'], 1)}
        out['configs'].append(row)
        print(json.dumps(row), flush=True)
    with open(os.path.join(HERE, 'rust_river.json'), 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
