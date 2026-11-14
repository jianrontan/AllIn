# backend/bot/scripts/lbr_paired.py
"""
Paired LBR for one blueprint, saved per hand so blueprints can be compared on the
SAME deals (the per-hand difference has a far smaller standard error than either
number). Every blueprint run with the same --seed and --hands plays identical deals.

Run from backend/bot/:
    python scripts/lbr_paired.py --db analysis/blueprints/blueprint_final_v2.db --out v2.json
    python scripts/lbr_paired.py --db X.db --out x.json --hands 10000 --seed 777
"""
import argparse
import hashlib
import time
from pathlib import Path
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.evaluation.lbr import LBREvaluator  # noqa: E402
from src.storage.blueprint_db import BlueprintDB  # noqa: E402


def main():
    p = argparse.ArgumentParser(description='Paired LBR, per-hand results to JSON.')
    p.add_argument('--db', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--hands', type=int, default=10000)
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--flop-runouts', type=int, default=20)
    p.add_argument('--fractions', help='comma-separated postflop LBR sizes; preflop stays 0.5,1,1.5')
    args = p.parse_args()
    fractions = None if args.fractions is None else [float(x) for x in args.fractions.split(',')]
    source_root = Path(__file__).resolve().parents[1] / 'src'
    sources = ('evaluation/lbr.py', 'cfr/translation.py', 'game/bot_strategy.py', 'game/game_session.py')
    fingerprints = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                    for name in sources}
    started = time.perf_counter()
    db = BlueprintDB(args.db, read_only=True)
    ev = LBREvaluator(db, seed=args.seed, flop_runout_samples=args.flop_runouts, bet_fractions=fractions)
    r = ev.evaluate(num_hands=args.hands, progress_every=0, paired=True)
    with open(args.out, 'w') as f:
        json.dump({'db': args.db, 'seed': args.seed, 'hands': args.hands,
                   'postflop_fractions': list(ev.bet_fractions),
                   'anchor_enabled': ev.subminimum_anchor, 'source_sha256': fingerprints,
                   'seconds': time.perf_counter() - started, 'lbr_mbb': r['lbr_mbb'], 'per_hand': r['per_hand']}, f)
    print(args.out, r['lbr_mbb'])


if __name__ == '__main__':
    main()
