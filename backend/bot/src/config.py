# backend/bot/src/config.py
"""
Central configuration / resolution helpers.

The trained blueprint is written as a timestamped SQLite file
(analysis/blueprints/blueprint_<timestamp>.db). Nothing is hard-named
`blueprint.db`, so the API and the bot need a deterministic way to pick it.

Resolution order:
  1. ALLIN_BLUEPRINT_DB env var (explicit override) — used as-is.
  2. Among analysis/blueprints/blueprint_*.db files NOT currently being written by a
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

# Blueprints live at backend/bot/analysis/blueprints — two parents up from this
# file (src/config.py -> src -> bot), then /analysis/blueprints. (The sibling
# analysis/abstractions holds committed precompute artifacts, not blueprints.)
_DEFAULT_BLUEPRINT_DIR = Path(__file__).resolve().parents[1] / "analysis" / "blueprints"


_UNREADABLE = object()


def _read_metadata(db_path, key, default):
    """training_metadata[key] of a blueprint DB (`default` when absent), or
    _UNREADABLE if the DB can't be read."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT value FROM training_metadata WHERE key = ?", (key,)
            ).fetchone()
        finally:
            conn.close()
        return json.loads(row[0]) if row else default
    except Exception:
        return _UNREADABLE


def _read_key_format(db_path):
    """training_metadata.key_format of a blueprint DB ('v2' when absent), or None
    if unreadable."""
    kf = _read_metadata(db_path, 'key_format', 'v2')
    return None if kf is _UNREADABLE else kf


def _servable_by_default(db_path):
    """Auto-resolution candidates: v2 keys on the root (v2) abstraction, under a menu
    mode the serving path accepts (an unstamped DB is 'control'). Anything else would
    be selected and then refused by BlueprintStrategy, leaving the API degraded."""
    from .abstractions.sizing import SERVING_MENU_MODES
    return (_read_key_format(db_path) == 'v2'
            and _read_metadata(db_path, 'abstraction', None) is None
            and (_read_metadata(db_path, 'menu_mode', None) or 'control') in SERVING_MENU_MODES)


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
        # A readable DB with NO total_iterations row is empty/half-written -> treat
        # as unusable (-1), not 0, so it can't be selected as "best" and then served
        # with an empty info_sets table.
        return json.loads(row[0]) if row else -1
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


def resolve_blueprint_path(blueprint_dir=None):
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

    blueprint_dir = Path(blueprint_dir) if blueprint_dir else _DEFAULT_BLUEPRINT_DIR
    # Auto-resolution serves v2-format blueprints on the root (v2) abstraction
    # only: the serving path builds v2 keys with the root buckets, so an spr5
    # (Phase 6) or v3-abstraction blueprint with more iterations must never be
    # auto-selected. Point ALLIN_BLUEPRINT_DB at one explicitly (and the readers'
    # key-format and abstraction guards decide whether it can be used).
    candidates = [p for p in blueprint_dir.glob("blueprint_*.db")
                  if _servable_by_default(p)]
    if not candidates:
        raise FileNotFoundError(
            f"No blueprint_*.db file found in {blueprint_dir}. "
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
            f"Found blueprint files in {blueprint_dir} but none were readable.")
    return best
