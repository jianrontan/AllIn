#!/usr/bin/env python3
"""Fit the V3.11 pooled jam read (src/exploitation/jam_read.py) from exported recaps.

Every human all-in on the flop or turn is ranked among all hands on that board with the
blueprint's own strength ranking (so the read is stamped with --db's abstraction and
refused for another one), binned by strength percentile and counted per context
(street, pot type, SPR bucket). Serve it with ALLIN_JAM_READ=<out>; score it first with
scripts/eval_facing_jams.py --jam-read lopo (leave-one-player-out) or --jam-read <out>.

Run from backend/bot (seconds):
    python scripts/fit_jam_read.py --in <hands_export.jsonl> --db <blueprint.db> --out <jam_read.json>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.blueprint_db import BlueprintDB                      # noqa: E402
from src.subgame.river_subgame_solver import RiverSubgameSolver      # noqa: E402
from src.exploitation.jam_read import JamObservations, JamRead, read_stamp, BINS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--db', required=True, help='the blueprint the read is for')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    db = BlueprintDB(args.db, read_only=True)
    strength = RiverSubgameSolver(db)._combo_strength
    recs = [json.loads(line) for line in open(args.inp, encoding='utf-8') if line.strip()]
    obs = JamObservations.from_recaps(recs, strength)
    read = JamRead.fit(obs, read_stamp(db))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(read.to_json(), f, indent=1)

    players = len({r['p'] for r in obs.rows})
    print(f"fit {read.n_jams()} human flop/turn jams from {players} players -> {args.out}")
    edges = [f"{i / BINS:.1f}" for i in range(BINS)]
    print(f"  strength-percentile bins (lower edge): {' '.join(edges)}")
    for k in sorted(read.counts, key=lambda k: (k.count('|'), k)):
        c = read.counts[k]
        if k.count('|') <= 1 and c.sum() >= 5:
            share = ' '.join(f"{x / c.sum():.2f}" for x in c)
            print(f"  {k:<14} n={int(c.sum()):>3}  {share}   top 20%: "
                  f"{float(np.sum(c[-2:]) / c.sum()):.2f}")
    db.close()


if __name__ == '__main__':
    main()
