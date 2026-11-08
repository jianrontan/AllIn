"""Python performance baseline for the components being ported to Rust.

Every phase's speedup claim is measured against the JSON this writes (same
machine, same configs). Run from backend/core-rs/:
    ../../.venv/bin/python bench/bench_baseline.py            # ~1-2 min
    ../../.venv/bin/python bench/bench_baseline.py --quick

Writes bench/python_baseline.json (git-ignored: machine-specific).
Rust columns get added per phase (the same configs, timed through allin_core).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.normpath(os.path.join(HERE, '..', '..', 'bot'))
sys.path.insert(0, BOT)

from src.abstractions.card_abstractions import CardAbstraction  # noqa: E402
from src.abstractions.hand_evaluator import HandEvaluator  # noqa: E402
from src.evaluation.showdown_kernel import build_board_arrays, _FULL_DECK  # noqa: E402
from src.subgame.river_cfr import RiverCFR  # noqa: E402
from src.subgame.river_tree import build_river_tree  # noqa: E402

BOARD = ['HA', 'DK', 'CQ', 'SJ', 'H9']
# (pot_entry, stack): served-path sizes -- deep (194 decision nodes), mid (106), short (24).
RIVER_CONFIGS = [(12.0, 194.0), (30.0, 185.0), (80.0, 160.0)]


def _sysctl(name):
    try:
        return subprocess.check_output(['sysctl', '-n', name], text=True).strip()
    except Exception:
        return None


def machine():
    return {'platform': platform.platform(), 'python': platform.python_version(),
            'numpy': np.__version__, 'cpu': _sysctl('machdep.cpu.brand_string'),
            'cores': {lvl: _sysctl(f'hw.perflevel{i}.physicalcpu')
                      for i, lvl in enumerate(('super', 'performance', 'efficiency'))},
            'mem_gb': round(int(_sysctl('hw.memsize') or 0) / 2**30, 1)}


def bench_river(iters):
    ev, ca = HandEvaluator(), CardAbstraction()
    t = time.perf_counter()
    ba = build_board_arrays(BOARD, ev, ca, streets=(3,))
    out = {'board_arrays_s': round(time.perf_counter() - t, 4), 'configs': []}
    H = ba['H']
    rng = np.random.default_rng(0)
    for pot, stack in RIVER_CONFIGS:
        tree = build_river_tree(pot_entry=pot, stacks=(stack, stack))
        cfr = RiverCFR(tree, ba)
        r0, r1 = np.ones(H), np.ones(H)
        cfr.run(r0, r1, iters=2)                              # warm-up
        t = time.perf_counter()
        cfr.run(r0, r1, iters=iters)
        cfr_ms = (time.perf_counter() - t) / iters * 1000
        t = time.perf_counter()
        cfr.exploitability(r0, r1)
        expl_ms = (time.perf_counter() - t) * 1000
        g = RiverCFR(tree, ba)
        hero, vill = rng.random(H), rng.random(H)
        optout = rng.uniform(-0.3, 0.6, H) * pot * 100
        g.run_gadget(hero, vill, optout, 1, iters=2)
        t = time.perf_counter()
        g.run_gadget(hero, vill, optout, 1, iters=iters)
        gadget_ms = (time.perf_counter() - t) / iters * 1000
        out['configs'].append({'pot_entry': pot, 'stack': stack,
                               'decision_nodes': len(tree.decision_nodes),
                               'cfr_ms_per_iter': round(cfr_ms, 3),
                               'gadget_ms_per_iter': round(gadget_ms, 3),
                               'exploitability_ms': round(expl_ms, 3),
                               'solve_400_iters_s': round(cfr_ms * 400 / 1000, 3)})
    return out


def bench_eval(n):
    from src.abstractions.postflop_features import rank7
    rng = np.random.default_rng(1)
    hands = [[_FULL_DECK[i] for i in rng.choice(52, 7, replace=False)] for _ in range(n)]
    t = time.perf_counter()
    for h in hands:
        rank7(h)
    return {'rank7_per_s': round(n / (time.perf_counter() - t))}


def bench_training(warm, timed, mode='capped'):
    from src.cfr.blueprint_trainer import BlueprintTrainer
    tr = BlueprintTrainer(menu_mode=mode)
    t = time.perf_counter()
    for i in range(warm):
        tr._run_iteration(i)
    warm_s = time.perf_counter() - t
    t = time.perf_counter()
    for i in range(warm, warm + timed):
        tr._run_iteration(i)
    dt = time.perf_counter() - t
    return {'mode': mode, 'single_core': True,
            'warmup_iters': warm, 'warmup_it_per_s': round(warm / warm_s, 1),
            'timed_iters': timed, 'it_per_s': round(timed / dt, 1),
            'info_sets': len(tr.info_sets),
            'note': 'early-run throughput; long runs differ (cache warmth, table size)'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default=os.path.join(HERE, 'python_baseline.json'))
    a = ap.parse_args()
    res = {'machine': machine(), 'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    print('river ...', flush=True)
    res['river'] = bench_river(20 if a.quick else 100)
    print('eval ...', flush=True)
    res['eval'] = bench_eval(20_000 if a.quick else 200_000)
    print('training ...', flush=True)
    res['training'] = bench_training(500 if a.quick else 2_000, 1_000 if a.quick else 5_000)
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != 'machine'}, indent=2))
    print(f"-> {a.out}")


if __name__ == '__main__':
    main()
