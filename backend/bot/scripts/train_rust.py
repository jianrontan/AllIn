# backend/bot/scripts/train_rust.py
"""
Train a blueprint with the Rust MCCFR+ trainer (backend/core-rs, `allin_core`).

Same algorithm, info-set keys, DB schema and metadata as the Python trainer
(tests/run_blueprint_trainer.py), so the API serves the result unchanged and
either trainer can resume the other's blueprint (same training mode). The
single-thread path is bit-identical to Python's given the same deals and
uniforms (parity-tested); the parallel path is Python's block-Linear-CFR scheme
on threads. Defaults match the served v2 blueprint: capped menu, parallel,
alpha=1.5, gamma=1.0.

Run from backend/bot/ (after `maturin develop --release` in backend/core-rs):
    python scripts/train_rust.py --iterations 50000000
    python scripts/train_rust.py --iterations 10000000 --resume blueprint_par_capped_rs_20261108_120000.db
    python scripts/train_rust.py --iterations 200000 --workers 1 --out-dir /tmp/bp   # single-thread

Ctrl+C stops at the next round boundary and checkpoints what completed.
New blueprints land in analysis/blueprints/ by default, where
config.resolve_blueprint_path() auto-selects the one with the most iterations
for LOCAL serving -- use --out-dir for experiments.
"""
import argparse
import os
import random
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.dirname(HERE)
sys.path.insert(0, BOT)

import allin_core  # noqa: E402

BLUEPRINTS = os.path.join(BOT, 'analysis', 'blueprints')
ABSTRACTIONS = os.path.join(BOT, 'analysis', 'abstractions')


def _fmt(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _shape_line(trainer):
    """Strategy-shape collapse probe (BUG-014) on the in-memory average strategy;
    never allowed to kill a run."""
    try:
        from src.cfr.strategy_shape import strategy_shape_report, format_shape_line
        from src.abstractions.card_abstractions import NUM_PREFLOP_BUCKETS
        return format_shape_line(strategy_shape_report(
            trainer.cumulative_strategy, num_preflop_buckets=NUM_PREFLOP_BUCKETS))
    except Exception as e:                        # noqa: BLE001
        return f"  [shape probe skipped: {type(e).__name__}: {e}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--iterations', type=int, required=True)
    ap.add_argument('--resume', default=None, help='blueprint file name in --out-dir to continue')
    ap.add_argument('--out-dir', default=BLUEPRINTS)
    ap.add_argument('--abstractions-dir', default=ABSTRACTIONS)
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 1,
                    help='1 = single-thread (per-iteration discount); >1 = parallel')
    ap.add_argument('--merge-every', type=int, default=2000, help='parallel: iterations per worker per round')
    ap.add_argument('--checkpoint-every', type=int, default=1_000_000)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--alpha', type=float, default=1.5)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--menu-mode', choices=['control', 'capped', 'capped_no2'], default='capped')
    ap.add_argument('--river-cache-boards', type=int, default=None,
                    help='canonical river-board cache cap (default: the crate default)')
    args = ap.parse_args()
    if args.iterations <= 0 or args.checkpoint_every <= 0 or args.workers <= 0:
        ap.error('iterations, checkpoint-every and workers must be positive')

    mode = 'parallel' if args.workers > 1 else 'single'
    seed = args.seed if args.seed is not None else random.randrange(1 << 62)
    os.makedirs(args.out_dir, exist_ok=True)
    trainer = allin_core.Trainer(args.menu_mode, args.alpha, args.gamma)
    start = 0
    if args.resume:
        if os.path.basename(args.resume) != args.resume:
            ap.error('--resume takes a file name inside --out-dir, not a path')
        db_path = os.path.join(args.out_dir, args.resume)
        if not os.path.exists(db_path):
            ap.error(f'not found: {db_path}')
        start = trainer.resume(db_path, mode)
        print(f"Resumed {args.resume}: {trainer.num_info_sets:,} info sets, iteration {start:,}")
    else:
        prefix = 'blueprint_par' if mode == 'parallel' else 'blueprint'
        tag = '' if args.menu_mode == 'control' else f"_{args.menu_mode.replace('_', '')}"
        name = f"{prefix}{tag}_rs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        db_path = os.path.join(args.out_dir, name)
        print(f"New run: {db_path}")
    trainer.stamp_mode(db_path, mode)

    print(f"Loading abstraction from {args.abstractions_dir} ...", flush=True)
    t_load = time.time()
    trainer.load_abstraction(args.abstractions_dir, args.river_cache_boards)
    print(f"  loaded in {time.time() - t_load:.1f}s")
    print(f"menu={args.menu_mode} mode={mode} workers={args.workers} merge_every={args.merge_every} "
          f"alpha={args.alpha} gamma={args.gamma} seed={seed}")

    total_target = start + args.iterations
    done = 0
    t0 = time.time()
    state = {'t': time.time(), 'it': 0}

    def progress(d):
        now = time.time()
        if now - state['t'] < 10 and d['iterations'] < chunk:
            return
        rate = (d['iterations'] - state['it']) / max(now - state['t'], 1e-9)
        state['t'], state['it'] = now, d['iterations']
        it = start + done + d['iterations']
        overall = (done + d['iterations']) / max(now - t0, 1e-9)
        eta = (args.iterations - done - d['iterations']) / max(overall, 1e-9)
        # (The trainer is busy inside train(); read totals from the callback dict.)
        ev = d['total_ev_sum'] / d['total_ev_count'] if d['total_ev_count'] else 0.0
        print(f"  iter {it:>12,}/{total_target:,} | {rate:>8,.0f} it/s | info sets {d['info_sets']:>9,} | "
              f"EV(cum{',lagged' if mode == 'parallel' else ''}) {ev:+.5f} | "
              f"elapsed {_fmt(now - t0)} | ETA {_fmt(eta)}", flush=True)

    interrupted = False
    while done < args.iterations and not interrupted:
        chunk = min(args.checkpoint_every, args.iterations - done)
        state['it'] = 0
        s = trainer.train(chunk, start + done, seed, workers=args.workers,
                          merge_every=args.merge_every, progress=progress)
        done += s['iterations']
        interrupted = s['interrupted']
        n = trainer.checkpoint(db_path, start + done - 1)
        print(f"Checkpoint: {n:,} info sets written ({trainer.num_info_sets:,} total) "
              f"at iteration {start + done:,}")
        print(_shape_line(trainer), flush=True)
    print(f"\n{'Interrupted' if interrupted else 'Done'}: {done:,} iterations in {_fmt(time.time() - t0)} "
          f"-> {db_path} (total {start + done:,})")


if __name__ == '__main__':
    main()
