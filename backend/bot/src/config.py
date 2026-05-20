# backend/bot/src/config.py
"""
Central configuration / resolution helpers.

The trained blueprint is written as a timestamped SQLite file
(analysis/blueprint_<timestamp>.db). Nothing is hard-named `blueprint.db`,
so the API and the bot need a deterministic way to pick the active file.

Resolution order:
  1. ALLIN_BLUEPRINT_DB env var (explicit override) — used as-is.
  2. Among analysis/blueprint_*.db files NOT currently being written by a
     training run, the one with the highest `total_iterations` (ties broken
     by mtime). If every candidate is busy, fall back to all of them.

A SQLite database in WAL mode keeps a `<name>-wal` sidecar file only while a
connection has it open; the sidecar is checkpointed away on a clean close.
So a live `-wal` sidecar is a reliable "training is writing this right now"
signal, and such files are skipped.

Reading iterations is done over a read-only connection so an in-progress
training run is never disturbed.
"""
import os
import json
import time
import sqlite3
from pathlib import Path

# A WAL sidecar modified within this many seconds is taken as "training is
# actively writing this DB right now".
_BUSY_WINDOW_SECONDS = 300

# analysis/ lives at backend/bot/analysis — two parents up from this file
# (src/config.py -> src -> bot), then /analysis.
_DEFAULT_ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"


def _read_total_iterations(db_path):
    """Return total_iterations stored in a blueprint DB, or -1 if unreadable."""
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            row = conn.execute(
                "SELECT value FROM training_metadata WHERE key = 'total_iterations'"
            ).fetchone()
        finally:
            conn.close()
        return json.loads(row[0]) if row else 0
    except Exception:
        return -1


def _is_being_written(db_path):
    """
    Heuristic: a training run is actively writing this DB if its WAL sidecar
    exists and was modified very recently.

    A clean BlueprintDB.close() checkpoints the WAL away, so a finished run
    leaves no sidecar at all. A crashed or force-killed run can leave a STALE
    sidecar behind — the recency window treats those as idle so a perfectly
    good blueprint is not skipped forever. This is a heuristic, not a
    guarantee; set ALLIN_BLUEPRINT_DB to pin a specific file deterministically.
    """
    wal = db_path.with_name(db_path.name + "-wal")
    try:
        return (time.time() - wal.stat().st_mtime) < _BUSY_WINDOW_SECONDS
    except FileNotFoundError:
        return False


def resolve_blueprint_path(analysis_dir=None):
    """
    Return a Path to the active blueprint DB.

    Raises FileNotFoundError if no usable blueprint file exists.
    """
    env_override = os.environ.get("ALLIN_BLUEPRINT_DB")
    if env_override:
        path = Path(env_override)
        if not path.exists():
            raise FileNotFoundError(
                f"ALLIN_BLUEPRINT_DB points to a missing file: {path}")
        return path

    analysis_dir = Path(analysis_dir) if analysis_dir else _DEFAULT_ANALYSIS_DIR
    candidates = list(analysis_dir.glob("blueprint_*.db"))
    if not candidates:
        raise FileNotFoundError(
            f"No blueprint_*.db file found in {analysis_dir}. "
            f"Set ALLIN_BLUEPRINT_DB or run training first.")

    # Prefer blueprints that are NOT currently being written by training.
    # If every candidate is busy, fall back to the full set rather than fail.
    idle = [p for p in candidates if not _is_being_written(p)]
    pool = idle or candidates

    # Highest training_metadata.total_iterations wins; mtime breaks ties.
    def sort_key(p):
        return (_read_total_iterations(p), p.stat().st_mtime)

    best = max(pool, key=sort_key)
    if _read_total_iterations(best) < 0:
        raise FileNotFoundError(
            f"Found blueprint files in {analysis_dir} but none were readable.")
    return best
