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
    WAL contents are included) to <run>_snaps/snap_<total>.db. Written to a temp
    name and renamed into place, so a failed copy never leaves a partial
    snap_<total>.db behind. Never overwrites."""
    import sqlite3
    from pathlib import Path
    snap_dir = os.path.splitext(db_path)[0] + '_snaps'
    os.makedirs(snap_dir, exist_ok=True)
    dest = os.path.join(snap_dir, f'snap_{total}.db')
    if os.path.exists(dest):
        return f'skipped, {dest} exists'
    tmp = dest + '.partial'
    if os.path.exists(tmp):
        os.remove(tmp)
    src = sqlite3.connect(Path(db_path).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        dst = sqlite3.connect(tmp)
        try:
            src.backup(dst)
        finally:
            dst.close()
        os.replace(tmp, dest)
    finally:
        src.close()
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest


def _read_abstraction_stamp(db_path, keys=('abstraction', 'abstraction_id')):
    """The blueprint's training_metadata values for `keys` (default: abstraction name
    and abstraction_id), None when absent."""
    import json
    import sqlite3
    from pathlib import Path
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        out = []
        for key in keys:
            row = conn.execute("SELECT value FROM training_metadata WHERE key = ?",
                               (key,)).fetchone()
            out.append(json.loads(row[0]) if row else None)
        return tuple(out)
    finally:
        conn.close()


def _write_abstraction_stamp(db_path, meta):
    """Stamp the blueprint with the abstraction it is trained on. The Rust trainer's
    checkpoints use INSERT OR REPLACE per key, so these rows survive them."""
    import json
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            for key, value in meta.items():
                conn.execute("INSERT OR REPLACE INTO training_metadata (key, value) "
                             "VALUES (?, ?)", (key, json.dumps(value)))
    finally:
        conn.close()


def _is_served_blueprint(path):
    """True if `path` is (or links to) a served blueprint: any blueprint_final*
    name in any letter case (APFS/NTFS are case-insensitive), or the same file as
    one in analysis/blueprints/."""
    if os.path.basename(path).lower().startswith('blueprint_final'):
        return True
    real = os.path.realpath(path)
    if os.path.basename(real).lower().startswith('blueprint_final'):
        return True
    served_dir = os.path.join(BOT, 'analysis', 'blueprints')
    try:
        names = os.listdir(served_dir)
    except OSError:
        return False
    for name in names:
        if name.lower().startswith('blueprint_final'):
            try:
                if os.path.samefile(path, os.path.join(served_dir, name)):
                    return True
            except OSError:
                pass
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--iterations', type=int, required=True)
    ap.add_argument('--resume', default=None, help='blueprint file name in --out-dir to continue')
    ap.add_argument('--out-dir', default=BLUEPRINTS)
    ap.add_argument('--abstractions-dir', default=ABSTRACTIONS,
                    help='abstraction directory: analysis/abstractions (v2) or a direct '
                         'subdirectory of it (see src/abstractions/abstraction_set.py)')
    ap.add_argument('--abstraction', default=None,
                    help='named abstraction (analysis/abstractions/<name>/); overrides '
                         '--abstractions-dir')
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
    ap.add_argument('--menu-mode', choices=['control', 'capped', 'capped_no2', 'capped_pj'], default='capped')
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
    from src.abstractions.abstraction_set import (
        abstraction_dir, check_resume, stamp_for, verify_tables)
    if args.abstraction is not None:
        args.abstractions_dir = abstraction_dir(args.abstraction)
    # Fingerprint the abstraction and check its tables' bake stamps before any DB
    # exists: the Rust loader verifies the centroid stamp but not the bake settings.
    abs_meta = stamp_for(args.abstractions_dir)
    verified = verify_tables(args.abstractions_dir)
    print(f"Abstraction {abs_meta.get('abstraction', 'root (v2)')} "
          f"id {abs_meta['abstraction_id'][:12]}... (tables verified: {', '.join(verified) or 'none'})")
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
        db_path = os.path.join(args.out_dir, args.resume)
        if not os.path.exists(db_path):
            ap.error(f'not found: {db_path}')
        if _is_served_blueprint(db_path):
            ap.error('refusing to train into a served blueprint (blueprint_final*); copy it first')
        stored = _read_abstraction_stamp(db_path)
        check_resume(stored[0], stored[1], abs_meta, 'train_rust')
        if mode == 'parallel':
            (old_round,) = _read_abstraction_stamp(db_path, ('merge_round_size',))
            if old_round is not None and old_round != args.workers * args.merge_every:
                print(f"WARNING: this run was trained with {old_round:,}-iteration merge rounds; "
                      f"--workers {args.workers} x --merge-every {args.merge_every} = "
                      f"{args.workers * args.merge_every:,}. The discount clocks tick once per "
                      f"round, so each tick now weighs a different number of iterations.",
                      flush=True)
        start = trainer.resume(db_path, mode)
        print(f"Resumed {args.resume}: {trainer.num_info_sets:,} info sets, iteration {start:,}")
    else:
        prefix = 'blueprint_par' if mode == 'parallel' else 'blueprint'
        tag = '' if args.menu_mode == 'control' else f"_{args.menu_mode.replace('_', '')}"
        if args.key_format != 'v2':
            tag += f"_{args.key_format}"
        if 'abstraction' in abs_meta:
            tag += f"_{abs_meta['abstraction']}"
        # Timestamp + random suffix: two runs started in the same second must
        # never share (and corrupt) one DB.
        name = (f"{prefix}{tag}_rs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_{random.randrange(16**6):06x}.db")
        db_path = os.path.join(args.out_dir, name)
        if os.path.exists(db_path):
            ap.error(f'refusing to overwrite {db_path}')
        print(f"New run: {db_path}")
    trainer.stamp_mode(db_path, mode)
    _write_abstraction_stamp(db_path, abs_meta)
    if mode == 'parallel':
        round_size = args.workers * args.merge_every
        _write_abstraction_stamp(db_path, {'merge_round_size': round_size})
        if args.checkpoint_every % round_size:
            # Each train() call ends in a partial round that still advances the discount
            # clocks a full tick (the trainer can't carry a remainder across calls).
            fixed = max(round_size, args.checkpoint_every // round_size * round_size)
            print(f"WARNING: --checkpoint-every {args.checkpoint_every:,} is not a multiple of "
                  f"the {round_size:,}-iteration merge round, so every checkpoint chunk ends in "
                  f"a partial round weighted as a full one. Use e.g. --checkpoint-every "
                  f"{fixed:,} (or adjust --workers / --merge-every).", flush=True)
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
        try:
            print(f"  iter {it:>12,}/{total_target:,} | {rate:>8,.0f} it/s | info sets {d['info_sets']:>9,} | "
                  f"EV(cum{',lagged' if mode == 'parallel' else ''}) {ev:+.5f} | "
                  f"elapsed {_fmt(now - t0)} | ETA {_fmt(eta)}", flush=True)
        except Exception:                             # noqa: BLE001
            # A display failure (closed pipe / terminal) must not abort training:
            # an exception escaping the callback ends train() without a checkpoint.
            pass

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
            try:
                print(f"Snapshot: {_snapshot(db_path, total)}", flush=True)
            except Exception as e:                    # noqa: BLE001
                # The checkpoint is already safe in the DB; a failed copy (e.g. disk
                # full) must not kill a multi-hour run.
                print(f"  [snapshot at {total:,} FAILED, training continues: "
                      f"{type(e).__name__}: {e}]", flush=True)
    print(f"\n{'Interrupted' if interrupted else 'Done'}: {done:,} iterations in {_fmt(time.time() - t0)} "
          f"-> {db_path} (total {start + done:,})")


if __name__ == '__main__':
    main()
