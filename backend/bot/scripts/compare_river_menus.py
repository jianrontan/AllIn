"""Compare solver-only menus at equal aggression caps; outputs JSONL.

Each convergence gap is measured on its OWN betting tree. Smaller gaps across
menus do not establish stronger play; the final decision requires solver-on H2H.
Use --repeats >=20 on a representative host for a latency distribution. This
fixed-board local experiment does not claim production p95.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.subgame.river_tree import build_river_tree, river_menu_profile
from src.subgame.rust_river_cfr import RustRiverCFR
from src.evaluation.showdown_kernel import build_board_arrays
from src.abstractions.hand_evaluator import HandEvaluator
from src.abstractions.card_abstractions import CardAbstraction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iters',type=int,default=275)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--depths',default='3,5')
    parser.add_argument('--profiles',default='A,F,F_EXTENDED')
    args=parser.parse_args()
    if args.iters<1 or args.repeats<1:
        parser.error('iters and repeats must be positive')
    board=['HA','DK','CQ','SJ','H9']
    ba=build_board_arrays(board,HandEvaluator(),CardAbstraction(),streets=(3,))
    r0,r1=np.random.default_rng(39).random((2,ba['H']))+.1
    source=Path(__file__).resolve().parents[1]/'src/subgame/river_tree.py'
    stamp=hashlib.sha256(source.read_bytes()).hexdigest()
    for pot,stack in [(8,192),(40,180),(80,96)]:
        for depth in map(int,args.depths.split(',')):
            for name in args.profiles.split(','):
                bets,raises=river_menu_profile(name)
                samples=[]
                for _ in range(args.repeats):
                    start=time.perf_counter()
                    tree=build_river_tree(pot,(stack,stack),menu=bets,raise_menu=raises,max_aggressions=depth)
                    solver=RustRiverCFR(tree,ba)
                    solver.run(r0,r1,iters=args.iters)
                    gap=solver.exploitability(r0,r1)
                    samples.append(time.perf_counter()-start)
                print(json.dumps(dict(profile=name,pot=pot,stack=stack,depth=depth,
                                      nodes=len(tree.decision_nodes),iters=args.iters,
                                      seconds=samples,gap=gap,source_sha256=stamp)),flush=True)


if __name__=='__main__':
    main()
