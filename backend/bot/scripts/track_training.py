# backend/bot/scripts/track_training.py
"""
Watch a RUNNING blueprint training job and build its convergence curve.

BR/LBR are MUCH slower than training (BR ~2 min/sample -> tens of minutes to hours
per milestone), so measuring inline would fall behind and miss milestones. The
default flow therefore DECOUPLES snapshotting from measuring:

  * snapshot mode (fast): poll the live DB; each time training crosses the next
    milestone, take a CONSISTENT frozen copy (SQLite online-backup -- safe while
    training writes, thanks to WAL) and append it to a manifest CSV. Seconds per
    milestone, so it keeps perfect cadence with training and never blocks on a
    measurement.
  * measure mode (slow): read the manifest, and for any snapshot not yet in the
    curve, score BR (+ optional LBR) and append to the curve CSV. Run it
    concurrently OR after training finishes -- the snapshots are frozen, so the
    numbers are valid whenever they're computed. It can drain a backlog and exits
    when caught up (or --follow to keep waiting for new snapshots).

  * watch mode (legacy): the original inline snapshot+measure-at-each-milestone
    loop, kept for small runs where measurement keeps up.

Run from backend/bot, each in its own terminal:

    # 1) snapshot every 2M iters starting at 500M (fast; runs with training)
    python scripts/track_training.py --mode snapshot --first 500000000 --every 2000000
    # 2) measure the snapshots whenever (slow; can start later / after training)
    python scripts/track_training.py --mode measure --samples 60 --lbr --follow
    # legacy all-in-one:
    python scripts/track_training.py --mode watch --every 1000000 --samples 60 --lbr

Snapshots land in analysis/blueprints/snapshots/ (a SUBFOLDER, so they are NOT
picked up by resolve_blueprint_path's analysis/blueprints/blueprint_*.db glob --
they won't ever shadow the active blueprint). Manifest + curve live in
analysis/training_curve/, both tied to the run's timestamp. Stop with Ctrl+C.

Notes
-----
* The manifest is written ONLY by the snapshotter and the curve ONLY by the
  measurer, so the two processes never race on a write.
* Each point is a frozen snapshot, so the measurement is stable even though
  training keeps advancing past it.
* Run this from the SAME code the training is using (the abstraction/keys must
  match the blueprint). To be safe, run the tracker from the training branch.
"""
import argparse
import csv
import glob
import os
import re
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.blueprint_db import BlueprintDB
from src.evaluation.best_response import BestResponseEvaluator

_SNAP_DIR = os.path.join('analysis', 'blueprints', 'snapshots')
_CURVE_DIR = os.path.join('analysis', 'training_curve')

# Decoupled snapshot/measure flow (default): the snapshotter appends to a MANIFEST
# as it crosses milestones (cheap, keeps cadence with training); the measurer reads
# the manifest and fills in the CURVE. Single-writer per file (snapshotter owns the
# manifest, measurer owns the curve), so the two processes never race on a write.
_MANIFEST_NAME = 'snapshots_{stamp}.csv'      # in _CURVE_DIR
_CURVE_NAME = 'training_curve_{stamp}.csv'    # in _CURVE_DIR
_MANIFEST_FIELDS = ['iterations', 'snapshot', 'utc']
_CURVE_FIELDS = ['iterations', 'exploitability_mbb', 'br_seat0_mbb',
                 'br_seat1_mbb', 'lbr_mbb', 'snapshot', 'utc']


def _active_run_db():
    """The blueprint being TRAINED right now = the most-recently-modified
    analysis/blueprints/blueprint_*.db. (NOT resolve_blueprint_path, which picks
    the highest-iteration DB -- a fresh run starts at few iterations, so that
    would wrongly select an older, larger, completed run.) Matches both
    blueprint_* and blueprint_par_* (parallel) runs."""
    cands = glob.glob(os.path.join('analysis', 'blueprints', 'blueprint_*.db'))
    if not cands:
        raise SystemExit("no analysis/blueprints/blueprint_*.db found")
    return max(cands, key=os.path.getmtime)


def _run_stamp(db_path):
    """The YYYYMMDD_HHMMSS stamp embedded in the run's DB filename, used to name
    the curve CSV and snapshots so they are tied to the exact run being tracked
    (and don't collide across runs). Handles blueprint_<stamp>.db and
    blueprint_par_<stamp>.db; falls back to 'unknown' if no stamp is present."""
    m = re.search(r'\d{8}_\d{6}', os.path.basename(db_path))
    return m.group(0) if m else 'unknown'


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


def _snapshot(db_path, iters, stamp):
    """Consistent frozen copy of the live DB via SQLite online backup. Named with
    the run's timestamp so snapshots from different runs don't overwrite."""
    os.makedirs(_SNAP_DIR, exist_ok=True)
    dst_path = os.path.join(_SNAP_DIR, f'snap_{stamp}_{iters}.db')
    src = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=30)
    dst = sqlite3.connect(dst_path)
    try:
        src.backup(dst)          # online backup: consistent even while training writes
    finally:
        dst.close()
        src.close()
    return dst_path


def _measure_br(snap_path, samples, seed):
    """Best-response exploitability for one snapshot. Returns the BR row (lbr_mbb
    left blank -- LBR, if requested, fills it in afterward)."""
    db = BlueprintDB(snap_path, read_only=True)
    try:
        # progress_every > 0 so BR prints live sample progress while the (slow)
        # measurement runs, instead of going dark for tens of minutes.
        br = BestResponseEvaluator(db, seed=seed).evaluate(
            num_samples=samples, progress_every=max(1, samples // 10))
        return {'br_seat0_mbb': round(br['br_seat0_mbb'], 1),
                'br_seat1_mbb': round(br['br_seat1_mbb'], 1),
                'exploitability_mbb': round(br['exploitability_mbb'], 1),
                'lbr_mbb': ''}
    finally:
        db.close()


def _measure_lbr(snap_path, lbr_hands, seed):
    """LBR (off-tree lower bound) for one snapshot. Run AFTER BR is already saved,
    so a slow/failed LBR never costs us the BR point."""
    from src.evaluation.lbr import LBREvaluator
    db = BlueprintDB(snap_path, read_only=True)
    try:
        lbr = LBREvaluator(db, seed=seed).evaluate(
            num_hands=lbr_hands, progress_every=max(1, lbr_hands // 10))
        return round(lbr['lbr_mbb'], 1)
    finally:
        db.close()


def _append_row(csv_path, fieldnames, row):
    with open(csv_path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames).writerow(row)


def _update_lbr(csv_path, fieldnames, iterations, lbr_val):
    """Fill in lbr_mbb on the already-written BR row for this milestone (rewrite
    the small CSV in place -- one row per milestone, BR + LBR in the same row)."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if int(r['iterations']) == iterations:
            r['lbr_mbb'] = lbr_val
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames)
        w.writeheader()
        w.writerows(rows)


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


# ---------------------------------------------------------------------------
# Manifest helpers (snapshot mode writes; measure mode reads)
# ---------------------------------------------------------------------------
def _read_rows(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _ensure_header(csv_path, fieldnames):
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames).writeheader()


# ---------------------------------------------------------------------------
# Snapshot mode: cross milestones and freeze the DB; never measures.
# ---------------------------------------------------------------------------
def _run_snapshot_mode(db_path, stamp, manifest, every, first, poll):
    _ensure_header(manifest, _MANIFEST_FIELDS)
    # Resume: don't re-snapshot milestones already in the manifest.
    done = {int(r['iterations']) for r in _read_rows(manifest)}
    next_at = first if first is not None else every
    while next_at in done:
        next_at += every
    print(f"Snapshotting {db_path} (run {stamp})")
    print(f"  milestone every {every:,} iters (next at {next_at:,}); "
          f"manifest -> {manifest} | snapshots -> {_SNAP_DIR}/ | Ctrl+C to stop")

    last_print = -1
    while True:
        iters = _live_iterations(db_path)
        if iters < 0:
            time.sleep(poll)
            continue
        if iters < next_at:
            if iters != last_print:
                print(f"[{time.strftime('%H:%M:%S')}] training at {iters:,} "
                      f"({100*iters/next_at:.0f}% to {next_at:,})", flush=True)
                last_print = iters
            time.sleep(poll)
            continue
        # Crossed one or more milestones: snapshot the current state once, record it
        # for every uncrossed milestone it satisfies (a big jump still advances).
        snap = _snapshot(db_path, iters, stamp)
        _append_row(manifest, _MANIFEST_FIELDS, {
            'iterations': iters, 'snapshot': os.path.basename(snap),
            'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        print(f"[{time.strftime('%H:%M:%S')}] milestone {iters:,} -> snapshot "
              f"{os.path.basename(snap)} (saved, not measured)", flush=True)
        while next_at <= iters:
            next_at += every
        time.sleep(poll)


# ---------------------------------------------------------------------------
# Measure mode: drain un-measured snapshots from the manifest into the curve.
# ---------------------------------------------------------------------------
def _run_measure_mode(stamp, manifest, curve, samples, seed, do_lbr, lbr_hands,
                      follow, poll):
    _ensure_header(curve, _CURVE_FIELDS)
    print(f"Measuring snapshots for run {stamp}")
    print(f"  manifest <- {manifest} | curve -> {curve} | "
          f"BR samples={samples}{', +LBR' if do_lbr else ''}, seed={seed}"
          f"{' | --follow' if follow else ''}")
    while True:
        measured = {int(r['iterations']) for r in _read_rows(curve)}
        pending = [r for r in _read_rows(manifest)
                   if int(r['iterations']) not in measured]
        if not pending:
            if not follow:
                print("All snapshots measured; nothing pending. Done.")
                _print_curve(curve)
                return
            time.sleep(poll)
            continue
        for r in pending:
            iters = int(r['iterations'])
            snap = os.path.join(_SNAP_DIR, r['snapshot'])
            if not os.path.exists(snap):
                print(f"  WARN snapshot missing, skipping: {snap}", flush=True)
                continue
            print(f"\n[{time.strftime('%H:%M:%S')}] measuring {iters:,} "
                  f"({r['snapshot']}) -> BR {samples} samples...", flush=True)
            t0 = time.time()
            row = _measure_br(snap, samples, seed)
            row.update({'iterations': iters, 'snapshot': r['snapshot'], 'utc': r['utc']})
            _append_row(curve, _CURVE_FIELDS, row)
            print(f"  BR done in {(time.time()-t0)/60:.1f}m: "
                  f"BR={row['exploitability_mbb']} mbb (saved)", flush=True)
            _print_curve(curve)
            if do_lbr:
                print(f"[{time.strftime('%H:%M:%S')}] running LBR "
                      f"({lbr_hands} hands)...", flush=True)
                t1 = time.time()
                lbr_val = _measure_lbr(snap, lbr_hands, seed)
                _update_lbr(curve, _CURVE_FIELDS, iters, lbr_val)
                print(f"  LBR done in {(time.time()-t1)/60:.1f}m: "
                      f"LBR={lbr_val} mbb (saved)", flush=True)
                _print_curve(curve)


def main():
    p = argparse.ArgumentParser(description="Track a running blueprint's convergence curve.")
    p.add_argument('--mode', choices=['snapshot', 'measure', 'watch'], default='snapshot',
                   help="snapshot: freeze DBs at milestones (fast, runs with training). "
                        "measure: score snapshots from the manifest (slow, run later). "
                        "watch: legacy inline snapshot+measure at each milestone.")
    p.add_argument('--db', default=None,
                   help="Blueprint DB to watch (default: the actively-training one = "
                        "most-recently-modified blueprint_*.db). Used by snapshot/watch; "
                        "for measure it only resolves the run timestamp.")
    p.add_argument('--every', type=int, default=1_000_000,
                   help="Milestone spacing in iterations (snapshot/watch).")
    p.add_argument('--first', type=int, default=None, help="First milestone (default: --every).")
    p.add_argument('--samples', type=int, default=60, help="BR board samples (modest = faster).")
    p.add_argument('--lbr', action='store_true', help="Also run LBR at each milestone (slower).")
    p.add_argument('--lbr-hands', type=int, default=1000)
    p.add_argument('--seed', type=int, default=1, help="Fixed seed so points are comparable.")
    p.add_argument('--poll', type=int, default=300, help="Seconds between DB/manifest polls.")
    p.add_argument('--follow', action='store_true',
                   help="measure mode: keep waiting for new snapshots instead of exiting "
                        "when the backlog is drained.")
    p.add_argument('--stamp', default=None,
                   help="measure mode: run timestamp to measure (default: derived from "
                        "--db or the most-recent run).")
    p.add_argument('--out', default=None,
                   help="Curve CSV path (default: analysis/training_curve/"
                        "training_curve_<run-timestamp>.csv).")
    p.add_argument('--manifest', default=None,
                   help="Manifest CSV path (default: analysis/training_curve/"
                        "snapshots_<run-timestamp>.csv).")
    args = p.parse_args()

    # Resolve the run timestamp + paths. measure mode may run with no live DB, so it
    # can take --stamp (or fall back to the most-recent run) without needing --db.
    if args.mode == 'measure' and args.stamp:
        stamp = args.stamp
    else:
        db_path = args.db or _active_run_db()
        stamp = _run_stamp(db_path)
    manifest = args.manifest or os.path.join(_CURVE_DIR, f'snapshots_{stamp}.csv')
    curve = args.out or os.path.join(_CURVE_DIR, f'training_curve_{stamp}.csv')

    if args.mode == 'snapshot':
        _run_snapshot_mode(db_path, stamp, manifest, args.every, args.first, args.poll)
        return
    if args.mode == 'measure':
        _run_measure_mode(stamp, manifest, curve, args.samples, args.seed,
                          args.lbr, args.lbr_hands, args.follow, args.poll)
        return

    # ---- legacy watch mode (inline snapshot + measure at each milestone) ----
    db_path = args.db or _active_run_db()
    out = curve
    next_at = args.first if args.first is not None else args.every
    print(f"Tracking {db_path} (run {stamp})")
    print(f"  milestone every {args.every:,} iters (first at {next_at:,}), "
          f"BR samples={args.samples}{', +LBR' if args.lbr else ''}, seed={args.seed}")
    print(f"  curve -> {out} | snapshots -> {_SNAP_DIR}/  | Ctrl+C to stop")

    fieldnames = _CURVE_FIELDS
    _ensure_header(out, fieldnames)

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
            print(f"\n[{time.strftime('%H:%M:%S')}] milestone {iters:,} -> snapshot + BR "
                  f"({args.samples} samples)...", flush=True)
            t0 = time.time()
            snap = _snapshot(db_path, iters, stamp)
            # BR first: measure, save, and print it BEFORE LBR runs, so a slow or
            # failed LBR never costs us the BR point.
            row = _measure_br(snap, args.samples, args.seed)
            row.update({'iterations': iters, 'snapshot': os.path.basename(snap),
                        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
            _append_row(out, fieldnames, row)
            print(f"  BR done in {(time.time()-t0)/60:.1f}m: "
                  f"BR={row['exploitability_mbb']} mbb (saved)", flush=True)
            _print_curve(out)

            # LBR after, updating the same row in place.
            if args.lbr:
                print(f"[{time.strftime('%H:%M:%S')}] running LBR "
                      f"({args.lbr_hands} hands)...", flush=True)
                t1 = time.time()
                lbr_val = _measure_lbr(snap, args.lbr_hands, args.seed)
                _update_lbr(out, fieldnames, iters, lbr_val)
                print(f"  LBR done in {(time.time()-t1)/60:.1f}m: "
                      f"LBR={lbr_val} mbb (saved)", flush=True)
                _print_curve(out)
            # Advance to the next uncrossed milestone (handles big jumps).
            while next_at <= iters:
                next_at += args.every
        time.sleep(args.poll)


if __name__ == '__main__':
    main()
