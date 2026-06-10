# backend/bot/scripts/run_solver_lbr.py
"""
RIGOROUS Phase-4 scoreboard: LBR exploitability of the RiverSubgameSolver,
paired against the blueprint's LBR on the SAME deals (same seed) so the delta is
low-variance. A NEGATIVE delta (solver - blueprint) means the solver is LESS
exploitable than the blueprint -> the go/no-go win.

SLOW: ~1 subgame solve per river bot decision (~1s+ each), so this is a long
offline job -- run it like training, not inline. It also competes with any
running training for CPU.

Run from backend/bot/:
    python scripts/run_solver_lbr.py --hands 1000 --max-iters 200
    python scripts/run_solver_lbr.py --hands 2000 --max-iters 200 --no-blueprint
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import resolve_blueprint_path
from src.storage.blueprint_db import BlueprintDB
from src.subgame.river_subgame_solver import RiverSubgameSolver
from src.evaluation.lbr import LBREvaluator
from src.evaluation.lbr_solver import SolverLBREvaluator


def run(hands, max_iters, time_budget, seed, also_blueprint, db_path=None,
        safe_gadget=True, gadget_anchor='auto', purify=0.01):
    import numpy as np
    path = db_path or resolve_blueprint_path()
    print(f"blueprint: {path}", flush=True)
    print(f"victim   : safe_gadget={safe_gadget} anchor={gadget_anchor} "
          f"purify={purify}  (the DEPLOYED served config)", flush=True)
    every = max(1, hands // 10)

    db = BlueprintDB(path, read_only=True)
    # Construct the solver EXACTLY as strategy_api.py serves it so the LBR number
    # reflects the deployed bot's river play (gadget + purification), not unsafe-v1.
    solver = RiverSubgameSolver(db, max_iters=max_iters, time_budget=time_budget,
                                rng=np.random.default_rng(seed),
                                safe_gadget=safe_gadget, gadget_anchor=gadget_anchor,
                                purify_threshold=purify)
    t0 = time.time()
    res = SolverLBREvaluator(db, solver, seed=seed).evaluate(
        num_hands=hands, progress_every=every, paired=True)
    solver_mbb = res['lbr_mbb']
    print(f"\nSOLVER   LBR: {solver_mbb:8.0f} mbb/hand  ({hands} hands, {time.time()-t0:.0f}s)",
          flush=True)

    if also_blueprint:
        t1 = time.time()
        rb = LBREvaluator(db, seed=seed).evaluate(
            num_hands=hands, progress_every=every, paired=True)
        bp_mbb = rb['lbr_mbb']
        print(f"BLUEPRINT LBR: {bp_mbb:8.0f} mbb/hand  (same deals, {time.time()-t1:.0f}s)",
              flush=True)
        # True paired delta: same deal i in both runs (paired=True), so hands where
        # the bot never acts on the river play identically -> delta 0, zero variance.
        # The 95% CI is driven only by the river-divergent hands -> tight.
        mbb = 1000.0 / 2.0
        d = np.array(res['per_hand']) - np.array(rb['per_hand'])      # chips, per hand
        mean = float(d.mean()) * mbb
        se = (float(d.std(ddof=1)) / len(d) ** 0.5) * mbb if len(d) > 1 else 0.0
        nz = int((np.abs(d) > 1e-9).sum())
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
        sig = "SIGNIFICANT" if hi < 0 else ("significant (worse)" if lo > 0 else "NOT significant")
        print(f"DELTA (solver - blueprint): {mean:+.0f} mbb/hand  "
              f"95% CI [{lo:+.0f}, {hi:+.0f}]  (n={hands}, {nz} river-divergent)  -> {sig}")
        print("  negative = solver LESS exploitable = GOOD")
    db.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="LBR exploitability of the river solver vs the blueprint.")
    p.add_argument('--hands', type=int, default=1000)
    p.add_argument('--max-iters', type=int, default=200)
    p.add_argument('--time-budget', type=float, default=8.0)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--db', default=None,
                   help="Blueprint/snapshot DB to score (default: resolve_blueprint_path). "
                        "Use a snapshot under analysis/blueprints/snapshots/ to score the "
                        "CURRENT training run on its own abstraction.")
    p.add_argument('--no-blueprint', action='store_true',
                   help="Skip the paired blueprint LBR baseline.")
    p.add_argument('--unsafe', action='store_true',
                   help="Score the UNSAFE-v1 solver (safe_gadget off) instead of the "
                        "deployed gadget bot -- for an unsafe-vs-gadget A/B.")
    p.add_argument('--anchor', default='auto', choices=['belief', 'blueprint', 'confidence', 'auto'],
                   help="Gadget anchor when safe_gadget is on (default: auto = deployed).")
    p.add_argument('--purify', type=float, default=0.01,
                   help="Purification threshold on the river bot (default 0.01 = deployed).")
    args = p.parse_args()
    run(args.hands, args.max_iters, args.time_budget, args.seed,
        not args.no_blueprint, db_path=args.db,
        safe_gadget=not args.unsafe, gadget_anchor=args.anchor, purify=args.purify)
