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
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cfr.blueprint_trainer import BlueprintTrainer
from src.storage.blueprint_db import BlueprintDB

ANALYSIS_DIR = Path(__file__).parent.parent / "analysis"


def run_training(iterations, resume=None, checkpoint_every=1000,
                 seed=None, gamma=None):
    """
    Run CFR training.

    Args:
        iterations:        Number of iterations to train.
        resume:            Filename of an existing DB to resume from
                           (e.g. 'blueprint_20260517_143022.db').
                           None = start a fresh run with a new timestamped DB.
        checkpoint_every:  Save to DB every N iterations.
        seed:              If set, seed Python's RNG so the traversal/sampling
                           trajectory is reproducible. The DCFR gamma value does
                           NOT affect the trajectory (it only reweights the
                           average-strategy sum), so two runs with the same seed
                           and different gamma differ ONLY in the blueprint.
        gamma:             Override the trainer's DCFR strategy-sum discount
                           (e.g. 0.0 for the no-discount control, 2.0 for DCFR).
                           None keeps the trainer default.
    """
    if seed is not None:
        import random
        random.seed(seed)

    ANALYSIS_DIR.mkdir(exist_ok=True)

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
        db_path = ANALYSIS_DIR / f"blueprint_{timestamp}.db"
        print(f"New run: {db_path.name}")

    db = BlueprintDB(db_path)
    trainer = BlueprintTrainer()
    if gamma is not None:
        trainer.gamma = gamma
    print(f"DCFR: alpha={trainer.alpha} gamma={trainer.gamma} "
          f"seed={seed if seed is not None else 'unset'}")

    start_iteration = 0
    if resume:
        start_iteration = trainer.resume_from_db(db)
        print(f"Continuing from iteration {start_iteration}")

    try:
        expected_value = trainer.train_blueprint(
            iterations,
            db=db,
            start_iteration=start_iteration,
            checkpoint_every=checkpoint_every,
        )
        # Final checkpoint — only needed when the last iteration wasn't already a checkpoint boundary
        if iterations % checkpoint_every != 0:
            trainer.checkpoint_to_db(db, start_iteration + iterations - 1)

        total = start_iteration + iterations
        print(f"\nTraining complete.")
        print(f"  Expected value:    {expected_value:.6f}")
        print(f"  Total iterations:  {total}")
        print(f"  Info sets:         {len(trainer.info_sets)}")
        print(f"  DB:                {db_path}")
        print(f"\nThe API/bot will pick this up automatically once it has the "
              f"most iterations of any blueprint in analysis/.")
    finally:
        db.close()

    return expected_value


if __name__ == "__main__":
    run_training(3000000, checkpoint_every=50000)
