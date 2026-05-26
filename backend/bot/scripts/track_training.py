# backend/bot/scripts/track_training.py
"""
Watch a RUNNING blueprint training job and build its convergence curve.

It polls the active (highest-iteration) blueprint DB; each time training crosses
the next iteration milestone it takes a CONSISTENT snapshot of the live DB
(SQLite online-backup -- safe while training is writing, thanks to WAL), scores
the snapshot's exploitability with the best-response harness (and optionally
LBR), appends a row to a CSV, and reprints the curve so far. Run it in its own
terminal alongside training:

    cd backend/bot
    python scripts/track_training.py                      # auto-detect active DB
    python scripts/track_training.py --every 1000000 --samples 60 --lbr

Snapshots land in analysis/blueprints/snapshots/ (a SUBFOLDER, so they are NOT
picked up by resolve_blueprint_path's analysis/blueprints/blueprint_*.db glob --
they won't ever shadow the active blueprint). Stop with Ctrl+C.

Notes
-----
* BR is CPU-heavy and competes with training for a core, so milestones are spaced
  (default 1M iters); keep --samples modest. Each point is a frozen snapshot, so
  the measurement is stable even though training keeps advancing.
* Run this from the SAME code the training is using (the abstraction/keys must
  match the blueprint). Lever A work on another branch is fine -- it doesn't
  change the abstraction -- but to be safe, run the tracker from `main`.
"""
import argparse
import csv
import glob
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.blueprint_db import BlueprintDB
from src.evaluation.best_response import BestResponseEvaluator

_SNAP_DIR = os.path.join('analysis', 'blueprints', 'snapshots')


def _active_run_db():
    """The blueprint being TRAINED right now = the most-recently-modified
    analysis/blueprints/blueprint_*.db. (NOT resolve_blueprint_path, which picks
    the highest-iteration DB -- a fresh run starts at few iterations, so that
    would wrongly select an older, larger, completed run.)"""
    cands = glob.glob(os.path.join('analysis', 'blueprints', 'blueprint_*.db'))
    if not cands:
        raise SystemExit("no analysis/blueprints/blueprint_*.db found")
    return max(cands, key=os.path.getmtime)


def _live_iterations(db_path):
    """total_iterations from a live DB, read-only (safe while training writes)."""
    try:
        con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=5)
        try:
            row = con.execute(
                "SELECT value FROM training_metadata WHERE key='total_iterations'"
            ).fetchone()
            return int(float(row[0])) if row else 0
        finally:
            con.close()
    except sqlite3.Error:
        return -1     # transient (mid-checkpoint); caller retries


def _snapshot(db_path, iters):
    """Consistent frozen copy of the live DB via SQLite online backup."""
    os.makedirs(_SNAP_DIR, exist_ok=True)
    dst_path = os.path.join(_SNAP_DIR, f'snap_{iters}.db')
    src = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=30)
    dst = sqlite3.connect(dst_path)
    try:
        src.backup(dst)          # online backup: consistent even while training writes
    finally:
        dst.close()
        src.close()
    return dst_path


def _measure(snap_path, samples, seed, do_lbr, lbr_hands):
    db = BlueprintDB(snap_path, read_only=True)
    try:
        # progress_every > 0 so BR/LBR print live sample/hand progress while the
        # (slow) measurement runs, instead of going dark for tens of minutes.
        br = BestResponseEvaluator(db, seed=seed).evaluate(
            num_samples=samples, progress_every=max(1, samples // 10))
        row = {'br_seat0_mbb': round(br['br_seat0_mbb'], 1),
               'br_seat1_mbb': round(br['br_seat1_mbb'], 1),
               'exploitability_mbb': round(br['exploitability_mbb'], 1),
               'lbr_mbb': ''}
        if do_lbr:
            from src.evaluation.lbr import LBREvaluator
            lbr = LBREvaluator(db, seed=seed).evaluate(
                num_hands=lbr_hands, progress_every=max(1, lbr_hands // 10))
            row['lbr_mbb'] = round(lbr['lbr_mbb'], 1)
        return row
    finally:
        db.close()


def _print_curve(csv_path):
    if not os.path.exists(csv_path):
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    print("\n  iters        BR(expl)     BR.ip     BR.oop    LBR")
    for r in rows:
        print(f"  {int(r['iterations']):>10,}  {r['exploitability_mbb']:>10}  "
              f"{r['br_seat0_mbb']:>8}  {r['br_seat1_mbb']:>8}  {r['lbr_mbb'] or '-':>8}")
    print()


def main():
    p = argparse.ArgumentParser(description="Track a running blueprint's convergence curve.")
    p.add_argument('--db', default=None,
                   help="Blueprint DB to watch (default: the actively-training one = "
                        "most-recently-modified blueprint_*.db).")
    p.add_argument('--every', type=int, default=1_000_000, help="Measure every N iterations.")
    p.add_argument('--first', type=int, default=None, help="First milestone (default: --every).")
    p.add_argument('--samples', type=int, default=60, help="BR board samples (modest = faster).")
    p.add_argument('--lbr', action='store_true', help="Also run LBR at each milestone (slower).")
    p.add_argument('--lbr-hands', type=int, default=1000)
    p.add_argument('--seed', type=int, default=1, help="Fixed seed so points are comparable.")
    p.add_argument('--poll', type=int, default=300, help="Seconds between DB polls.")
    p.add_argument('--out', default=os.path.join('analysis', 'training_curve.csv'))
    args = p.parse_args()

    db_path = args.db or _active_run_db()
    next_at = args.first if args.first is not None else args.every
    print(f"Tracking {db_path}")
    print(f"  milestone every {args.every:,} iters (first at {next_at:,}), "
          f"BR samples={args.samples}{', +LBR' if args.lbr else ''}, seed={args.seed}")
    print(f"  curve -> {args.out} | snapshots -> {_SNAP_DIR}/  | Ctrl+C to stop")

    fieldnames = ['iterations', 'exploitability_mbb', 'br_seat0_mbb',
                  'br_seat1_mbb', 'lbr_mbb', 'snapshot', 'utc']
    if not os.path.exists(args.out):
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames).writeheader()

    start_t = None                  # wall-clock when tracking began (sentinel)
    start_it = 0                    # iters when tracking began (for a cumulative rate)
    last_print = -1                 # last checkpointed iters we printed a heartbeat for
    while True:
        iters = _live_iterations(db_path)
        if iters < 0:
            time.sleep(args.poll)
            continue
        if start_t is None:
            start_t, start_it = time.time(), iters
        if iters < next_at:
            # Heartbeat ONLY when a new checkpoint has landed (iters changed),
            # so it prints ~once per checkpoint (~20 min), not every poll. We poll
            # often (--poll) to catch the milestone promptly, but stay quiet
            # between checkpoints. `iters` reflects the last checkpoint, not the
            # live counter (total_iterations is written only at checkpoints).
            if iters != last_print:
                el = time.time() - start_t
                rate = (iters - start_it) / el if el > 5 and iters > start_it else 0
                eta = (f", next BR ~{(next_at - iters) / rate / 60:.0f}m"
                       if rate > 0 else "")
                print(f"[{time.strftime('%H:%M:%S')}] training at {iters:,} "
                      f"({100*iters/next_at:.0f}% to {next_at:,})"
                      f"{f' | ~{rate:.0f} it/s avg' if rate > 0 else ''}{eta}", flush=True)
                last_print = iters
            time.sleep(args.poll)
            continue
        if iters >= next_at:
            stamp_iters = (iters // args.every) * args.every if args.first is None else iters
            print(f"\n[{time.strftime('%H:%M:%S')}] milestone {iters:,} -> snapshot + BR "
                  f"({args.samples} samples"
                  f"{', LBR ' + str(args.lbr_hands) + ' hands' if args.lbr else ''})...",
                  flush=True)
            t0 = time.time()
            snap = _snapshot(db_path, iters)
            row = _measure(snap, args.samples, args.seed, args.lbr, args.lbr_hands)
            row.update({'iterations': iters, 'snapshot': os.path.basename(snap),
                        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
            with open(args.out, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames).writerow(row)
            print(f"  done in {(time.time()-t0)/60:.1f}m: "
                  f"BR={row['exploitability_mbb']} mbb"
                  f"{'  LBR=' + str(row['lbr_mbb']) if args.lbr else ''}", flush=True)
            _print_curve(args.out)
            # Advance to the next uncrossed milestone (handles big jumps).
            while next_at <= iters:
                next_at += args.every
        time.sleep(args.poll)


if __name__ == '__main__':
    main()
