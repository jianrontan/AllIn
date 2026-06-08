# backend/bot/scripts/validate_cfv.py
"""
M0 validation for the depth-limited turn-leaf value (src/subgame/cfv.py).

Step 1 (this file) -- IMPLEMENTATION CORRECTNESS of the EXACT leaf:
  The river is zero-sum, so the blueprint's river-continuation value is too. With
  ASYMMETRIC ranges (so each seat's total EV is non-trivially non-zero), the leaf
  must satisfy  E0 = sum_h R0[h]*leaf0[h]  and  E1 = sum_h R1[h]*leaf1[h]  with
  E0 + E1 ~= 0. A clean residual validates the whole pipeline at once: the 4-card
  turn -> 5-card river basis mapping, runout averaging, measure units, and the
  blueprint projection / _eval reuse. (Steps 2-3 -- bucketed-vs-exact and the
  shifted-range test -- build on this.)

Run from backend/bot/ (use a small --rivers for a fast smoke; 0 = all runouts):
    python scripts/validate_cfv.py --rivers 4
    python scripts/validate_cfv.py
"""
import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.blueprint_db import BlueprintDB, FrozenBlueprint
from src.abstractions.hand_evaluator import HandEvaluator
from src.abstractions.card_abstractions import CardAbstraction
from src.abstractions.sizing import db_menu_mode, postflop_menu_for
from src.subgame.cfv import turn_hands, turn_leaf_value_exact, FULL_DECK

_BOARD4 = ['CQ', 'SJ', 'H9', 'D5']      # a 4-card turn board (leaf basis)
_POT = 24.0
_STACKS = (88.0, 88.0)
_BROADWAY = set('TJQKA')


def _latest_capped():
    c = sorted(glob.glob('analysis/blueprints/blueprint_par_capped_*.db'))
    return c[-1] if c else None


def main():
    p = argparse.ArgumentParser(description="M0: exact turn-leaf correctness (zero-sum).")
    p.add_argument('--db', default=None)
    p.add_argument('--rivers', type=int, default=0, help="Runout sample size (0 = all).")
    args = p.parse_args()
    path = args.db or _latest_capped()
    rawdb = BlueprintDB(path, read_only=True)
    menu = postflop_menu_for(db_menu_mode(rawdb))
    db = FrozenBlueprint(rawdb)            # consistent snapshot (immune to live training)
    ev, cards = HandEvaluator(), CardAbstraction()

    hands = turn_hands(_BOARD4)
    full = {hb: 1.0 for hb in hands}
    tight = {hb: 1.0 for hb in hands if hb[0][1] in _BROADWAY and hb[1][1] in _BROADWAY}
    rivers = None
    if args.rivers > 0:
        rivers = [c for c in FULL_DECK if c not in set(_BOARD4)][:args.rivers]

    print(f"blueprint: {os.path.basename(path)} | menu={db_menu_mode(rawdb)} | "
          f"board={_BOARD4} pot={_POT} | rivers={'all' if rivers is None else len(rivers)}")
    print(f"ranges: R0=full ({len(full)} hands) vs R1=broadway ({len(tight)} hands)")

    leaf0 = turn_leaf_value_exact(_BOARD4, _POT, _STACKS, 0, full, tight, db, ev, cards,
                                  menu=menu, rivers=rivers)
    leaf1 = turn_leaf_value_exact(_BOARD4, _POT, _STACKS, 1, tight, full, db, ev, cards,
                                  menu=menu, rivers=rivers)
    rawdb.close()

    vals = list(leaf0.values()) + list(leaf1.values())
    nan = any(math.isnan(v) or math.isinf(v) for v in vals)
    e0 = sum(full[h] * leaf0.get(h, 0.0) for h in full)
    e1 = sum(tight[h] * leaf1.get(h, 0.0) for h in tight)
    resid = e0 + e1
    scale = max(abs(e0), abs(e1), 1.0)

    print(f"\n  leaf values: min {min(vals):+.2f}  max {max(vals):+.2f}  NaN/inf={nan}")
    print(f"  E0 (full vs broadway)  = {e0:+.3f}")
    print(f"  E1 (broadway vs full)  = {e1:+.3f}")
    print(f"  zero-sum residual E0+E1 = {resid:+.4f}   (|resid|/scale = {abs(resid)/scale:.2e})")

    ok = (not nan) and abs(resid) / scale < 1e-6
    print(f"\n  M0-step1 {'PASS' if ok else 'FAIL'}: exact leaf is "
          f"{'zero-sum + finite (pipeline correct)' if ok else 'NOT zero-sum/finite -- bug'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
