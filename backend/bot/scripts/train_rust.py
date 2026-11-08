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
    python scripts/train_rust.py --iterations 100000000 --checkpoint-every 5000000 --snapshot-every 5000000

Ctrl+C stops at the next round boundary and checkpoints what completed.
New blueprints land in analysis/blueprints_rs/ by default -- deliberately NOT
analysis/blueprints/, where config.resolve_blueprint_path() auto-selects the
blueprint with the most iterations for local serving. Move a finished run there
(or point ALLIN_BLUEPRINT_DB at it) when you want to serve it.
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

BLUEPRINTS = os.path.join(BOT, 'analysis', 'blueprints_rs')
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


def _snapshot(db_path, total):
    """Consistent copy of the just-checkpointed blueprint (SQLite backup API, so
    WAL contents are included) to <run>_snaps/snap_<total>.db. Never overwrites."""
    import sqlite3
    snap_dir = os.path.splitext(db_path)[0] + '_snaps'
    os.makedirs(snap_dir, exist_ok=True)
    dest = os.path.join(snap_dir, f'snap_{total}.db')
    if os.path.exists(dest):
        return f'skipped, {dest} exists'
    from pathlib import Path
    src = sqlite3.connect(Path(db_path).resolve().as_uri() + '?mode=ro', uri=True)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return dest


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
    ap.add_argument('--snapshot-every', type=int, default=0,
                    help='also copy the blueprint to <run>_snaps/snap_<iter>.db each time the total '
                         'crosses a multiple of this (0 = off); keep it a multiple of --checkpoint-every')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--alpha', type=float, default=1.5)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--menu-mode', choices=['control', 'capped', 'capped_no2'], default='capped')
    ap.add_argument('--key-format', choices=['v2', 'spr5'], default='v2',
                    help="info-set key format: v2 (served) or spr5 (Phase 6 SPR buckets on flop/turn)")
    ap.add_argument('--river-cache-boards', type=int, default=None,
                    help='canonical river-board cache cap (default: the crate default)')
    args = ap.parse_args()
    if args.iterations <= 0 or args.checkpoint_every <= 0 or args.workers <= 0:
        ap.error('iterations, checkpoint-every and workers must be positive')
    if args.snapshot_every < 0:
        ap.error('snapshot-every must be >= 0')

    mode = 'parallel' if args.workers > 1 else 'single'
    seed = args.seed if args.seed is not None else random.randrange(1 << 62)
    trainer = allin_core.Trainer(args.menu_mode, args.alpha, args.gamma, key_format=args.key_format)
    # Load the abstraction FIRST, so a bad tables dir fails before any DB exists.
    print(f"Loading abstraction from {args.abstractions_dir} ...", flush=True)
    t_load = time.time()
    trainer.load_abstraction(args.abstractions_dir, args.river_cache_boards)
    print(f"  loaded in {time.time() - t_load:.1f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    start = 0
    if args.resume:
        if os.path.basename(args.resume) != args.resume:
            ap.error('--resume takes a file name inside --out-dir, not a path')
        if args.resume.startswith('blueprint_final'):
            ap.error('refusing to train into a served blueprint (blueprint_final*); copy it first')
        db_path = os.path.join(args.out_dir, args.resume)
        if not os.path.exists(db_path):
            ap.error(f'not found: {db_path}')
        start = trainer.resume(db_path, mode)
        print(f"Resumed {args.resume}: {trainer.num_info_sets:,} info sets, iteration {start:,}")
    else:
        prefix = 'blueprint_par' if mode == 'parallel' else 'blueprint'
        tag = '' if args.menu_mode == 'control' else f"_{args.menu_mode.replace('_', '')}"
        if args.key_format != 'v2':
            tag += f"_{args.key_format}"
        # Timestamp + random suffix: two runs started in the same second must
        # never share (and corrupt) one DB.
        name = (f"{prefix}{tag}_rs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_{random.randrange(16**6):06x}.db")
        db_path = os.path.join(args.out_dir, name)
        if os.path.exists(db_path):
            ap.error(f'refusing to overwrite {db_path}')
        print(f"New run: {db_path}")
    trainer.stamp_mode(db_path, mode)
    print(f"menu={args.menu_mode} key_format={args.key_format} mode={mode} workers={args.workers} merge_every={args.merge_every} "
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
        total = start + done
        if args.snapshot_every and total // args.snapshot_every > (total - s['iterations']) // args.snapshot_every:
            print(f"Snapshot: {_snapshot(db_path, total)}", flush=True)
    print(f"\n{'Interrupted' if interrupted else 'Done'}: {done:,} iterations in {_fmt(time.time() - t0)} "
          f"-> {db_path} (total {start + done:,})")


if __name__ == '__main__':
    main()
