# backend/bot/tests/run_blueprint_trainer.py
"""
Training entrypoint for BlueprintTrainer with SQLite persistence and resume support.

Usage (from backend/bot/):

    # New run — creates a timestamped DB, e.g. blueprint_20260517_143022.db
    python -c "from tests.run_blueprint_trainer import run_training; run_training(100000)"

    # Resume from a specific DB file
    python -c "from tests.run_blueprint_trainer import run_training; run_training(50000, resume='blueprint_20260517_143022.db')"

    # Quick test
    python -c "from tests.run_blueprint_trainer import run_training; run_training(10)"

The API and bot pick the active blueprint automatically via
src.config.resolve_blueprint_path() (highest-iteration DB that is not being
written) — no manual promotion step. Set ALLIN_BLUEPRINT_DB to pin a file.
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cfr.blueprint_trainer import BlueprintTrainer
from src.storage.blueprint_db import BlueprintDB

ANALYSIS_DIR = Path(__file__).parent.parent / "analysis" / "blueprints"


def run_training(iterations, resume=None, checkpoint_every=1000,
                 seed=None, gamma=None, workers=None, merge_every=2000,
                 menu_mode='control'):
    """
    Run CFR training.

    Args:
        iterations:        Number of iterations to train.
        resume:            Filename of an existing DB to resume from
                           (e.g. 'blueprint_20260517_143022.db').
                           None = start a fresh run with a new timestamped DB.
        checkpoint_every:  Save to DB every N iterations.
        seed:              If set, seed Python's RNG so the traversal/sampling
                           trajectory is reproducible. The gamma discount does
                           NOT affect the trajectory (it only reweights the
                           average-strategy sum), so two runs with the same seed
                           and different gamma differ ONLY in the blueprint.
        gamma:             Override the trainer's strategy-sum discount exponent
                           (e.g. 0.0 for the no-discount control, 2.0 default).
                           None keeps the trainer default.
        workers:           If set (>1), use data-parallel MCCFR+ across this many
                           worker processes (block Linear-CFR discount on the
                           master; an approximation of single-thread validated by
                           exploitability, NOT bit-identical). None / 1 = the
                           reproducible single-thread path.
        merge_every:       Parallel only: iterations PER WORKER between merges.
    """
    if seed is not None:
        import random
        random.seed(seed)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    if resume:
        db_path = ANALYSIS_DIR / resume
        if not db_path.exists():
            raise FileNotFoundError(
                f"DB not found: {db_path}\n"
                f"Available DBs: {[p.name for p in ANALYSIS_DIR.glob('blueprint_*.db')]}"
            )
        print(f"Resuming from: {db_path.name}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Mark parallel-trained blueprints so they are distinguishable on disk
        # (still matches the blueprint_*.db glob used by resolve_blueprint_path
        # and the tracker, and the YYYYMMDD_HHMMSS stamp is still extractable).
        prefix = "blueprint_par" if (workers and workers > 1) else "blueprint"
        # Tag non-control menus on disk so the A/B arms are distinguishable at a
        # glance (still matches the blueprint_*.db glob). 'capped' -> _capped,
        # 'capped_no2' -> _cappedno2.
        menu_tag = "" if menu_mode == 'control' else f"_{menu_mode.replace('_', '')}"
        db_path = ANALYSIS_DIR / f"{prefix}{menu_tag}_{timestamp}.db"
        print(f"New run: {db_path.name}  (menu_mode={menu_mode})")

    db = BlueprintDB(db_path)
    trainer = BlueprintTrainer(menu_mode=menu_mode)
    if gamma is not None:
        trainer.gamma = gamma
    print(f"discount: alpha={trainer.alpha} gamma={trainer.gamma} "
          f"seed={seed if seed is not None else 'unset'}")

    mode = 'parallel' if (workers and workers > 1) else 'single'
    start_iteration = 0
    if resume:
        start_iteration = trainer.resume_from_db(db, mode=mode)
        print(f"Continuing from iteration {start_iteration}")
    # Stamp the training mode so later resumes are mode-checked. Cross-mode
    # resumes corrupt the Linear-CFR discount (single-thread vs parallel keep
    # incompatible per-info-set clocks); resume_from_db refuses a known mismatch.
    db.set_metadata('training_mode', mode)   # set_metadata json-encodes internally
    db.set_metadata('menu_mode', menu_mode)  # action-abstraction arm (control/capped)

    try:
        if workers and workers > 1:
            from src.cfr.parallel_trainer import train_blueprint_parallel
            expected_value = train_blueprint_parallel(
                trainer,
                iterations,
                db=db,
                start_iteration=start_iteration,
                checkpoint_every=checkpoint_every,
                workers=workers,
                merge_every=merge_every,
                seed=seed,
            )
        else:
            expected_value = trainer.train_blueprint(
                iterations,
                db=db,
                start_iteration=start_iteration,
                checkpoint_every=checkpoint_every,
            )
        # Final checkpoint. The parallel path self-flushes its tail inside
        # train_blueprint_parallel (its checkpoint cadence is round-based, not a
        # clean multiple of checkpoint_every), so only the single-thread path
        # needs this boundary check -- doing it for both would redundantly rewrite
        # the final iteration with an empty dirty set.
        if not (workers and workers > 1) and iterations % checkpoint_every != 0:
            trainer.checkpoint_to_db(db, start_iteration + iterations - 1)

        total = start_iteration + iterations
        print(f"\nTraining complete.")
        print(f"  Expected value:    {expected_value:.6f}")
        print(f"  Total iterations:  {total}")
        print(f"  Info sets:         {len(trainer.info_sets)}")
        print(f"  DB:                {db_path}")
        print(f"\nThe API/bot will pick this up automatically once it has the "
              f"most iterations of any blueprint in analysis/blueprints/.")
    finally:
        db.close()

    return expected_value


if __name__ == "__main__":
    # CLI flags so you never need a quoted `python -c "..."` one-liner (which is
    # easy to mangle when pasted into a shell). Example:
    #   python tests/run_blueprint_trainer.py --resume blueprint_par_X.db --workers 6
    p = argparse.ArgumentParser(description="Train / resume a blueprint.")
    p.add_argument('--iterations', type=int, default=10_000_000,
                   help="CFR iterations to run this session (default 10,000,000).")
    p.add_argument('--resume', default=None,
                   help="Filename of an existing DB in analysis/blueprints/ to "
                        "continue (e.g. blueprint_par_20260529_233056.db). "
                        "Omit to start a fresh run.")
    p.add_argument('--workers', type=int, default=None,
                   help="Parallel worker processes (>1 = parallel). Omit/1 = single-thread.")
    p.add_argument('--merge-every', type=int, default=4000,
                   help="Parallel only: iterations per worker between merges (default 4000).")
    p.add_argument('--checkpoint-every', type=int, default=50_000,
                   help="Save to the DB every N iterations (default 50,000).")
    p.add_argument('--seed', type=int, default=None,
                   help="Seed Python's RNG (single-thread reproducibility only).")
    p.add_argument('--gamma', type=float, default=None,
                   help="Override the strategy-sum discount exponent.")
    p.add_argument('--menu-mode', choices=['control', 'capped', 'capped_no2'],
                   default='control',
                   help="Action-abstraction arm: 'control' (current 4-size menu + "
                        "voluntary all-in, the A/B baseline); 'capped' (Fix-#4: 5-size "
                        "menu incl. 2.0x, voluntary all-in dropped); 'capped_no2' "
                        "(capped WITHOUT the 2.0x tier -- the clean test arm for the "
                        "2.0x tier's value). Each arm's blueprint is incompatible with "
                        "the others (different keys); resume-guarded by menu_mode.")
    args = p.parse_args()
    run_training(
        args.iterations,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        gamma=args.gamma,
        workers=args.workers,
        merge_every=args.merge_every,
        menu_mode=args.menu_mode,
    )
