# backend/bot/scripts/measure_leaf_accuracy.py
"""
M0 follow-up (review HIGH #1 + MEDIUM #4): stress the bucketed reach-conditioned leaf
across TEXTURES, SPRs, and STRUCTURAL range shifts -- not just the single
rank-correlated full->broadway shift the earlier scripts used, and report the WORST
config (the solver must be safe on the bad board, not the mean board).

Why this exists: the earlier M0 conclusion ("~13-16% under shift; real shifts milder")
rested on ONE shift type that is ALIGNED with the strength partition the leaf buckets on
-> the easy case. A CFR solve concentrates the opponent's reach on hand CLASSES that can
be ORTHOGONAL to mean strength. So we test three shift shapes:
  * broadway   -- both cards TJQKA (rank-correlated; the old easy case, for comparison)
  * polarized  -- strongest frac + weakest frac by mean strength (bimodal extremes)
  * draw_heavy -- top frac by rank VARIANCE over runouts (draws: similar mean strength,
                  different trajectory -> the shift most ORTHOGONAL to the partition)

Metrics per (board, SPR, shift), at the chosen leaf resolution (default n=128, card-
removal-aware reconstruction):
  * rel-RMSE vs the exact rollout UNDER THAT SHIFT (the standard number), AND
  * STABLE rel-RMSE: same error but normalized by RMS of the NO-SHIFT exact -- removes
    the denominator inflation a narrow villain range causes (review MEDIUM #4), so the
    under-shift numbers are comparable across shifts and not flattered, AND
  * POT-NORMALIZED: per-PAIR error (measure / compatible mass -> avg payoff in chips)
    RMS as a fraction of the pot -- "is 13% rel-RMSE actually 3% or 9% of the pot?"

Reads through FrozenBlueprint (consistent snapshot vs a live-training DB). Heavy: warn
on wall-time. Run from backend/bot/:
    python scripts/measure_leaf_accuracy.py --rivers 12               # full sweep (slow)
    python scripts/measure_leaf_accuracy.py --rivers 6 --max-boards 2 --sprs 4   # smoke
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.blueprint_db import BlueprintDB, FrozenBlueprint
from src.abstractions.hand_evaluator import HandEvaluator
from src.abstractions.card_abstractions import CardAbstraction
from src.abstractions.sizing import db_menu_mode, postflop_menu_for
from src.evaluation.showdown_kernel import build_turn_board_arrays, compatible_mass
from src.subgame.cfv import (turn_hands, turn_leaf_value_exact, turn_leaf_matrix_both,
                             turn_strength, equal_freq_partition, bucketed_measure_leaf_cr,
                             FULL_DECK)

# Representative 4-card turn textures (SuitRank).
_BOARDS = [
    ['CQ', 'SJ', 'H9', 'D5'],   # rainbow, disconnected broadway (canonical)
    ['SK', 'HK', 'D7', 'C2'],   # paired, dry, rainbow
    ['H9', 'H8', 'D7', 'S2'],   # two-tone, connected (wet)
    ['SA', 'HK', 'HQ', 'D5'],   # broadway, two-tone
    ['S8', 'S5', 'S2', 'HK'],   # three-flush (flush-heavy)
    ['D6', 'C6', 'S6', 'H2'],   # trips (very paired)
]
_BROADWAY = set('TJQKA')


def _latest_capped():
    c = sorted(glob.glob('analysis/blueprints/blueprint_par_capped_*.db'))
    return c[-1] if c else None


def _rms(d):
    v = np.array(list(d.values()))
    return float(np.sqrt(np.mean(v ** 2))) if v.size else 0.0


def _rel_rmse(pred, exact, denom):
    keys = [h for h in exact if h in pred]
    p = np.array([pred[h] for h in keys])
    e = np.array([exact[h] for h in keys])
    return float(np.sqrt(np.mean((p - e) ** 2)) / (denom or 1.0))


def _pot_norm_err(pred, exact, ba, vill_reach, pot):
    """Per-PAIR error as a fraction of pot: convert each hand's MEASURE error to an
    average-payoff error (divide by that hand's compatible villain mass), RMS / pot."""
    hands = ba['hands']
    idx = {h: i for i, h in enumerate(hands)}
    rv = np.array([vill_reach.get(h, 0.0) for h in hands])
    cm = compatible_mass(ba, rv)
    errs = []
    for h in exact:
        if h in pred and h in idx and cm[idx[h]] > 1e-9:
            errs.append((pred[h] - exact[h]) / cm[idx[h]])
    return float(np.sqrt(np.mean(np.square(errs))) / pot) if errs else 0.0


def _shifts(strength, var, frac=0.25):
    """Return {name: set(hands)} for the three structural shifts."""
    hands = list(strength)
    by_str = sorted(hands, key=lambda h: strength[h])         # ascending = weak->strong
    k = max(1, int(frac * len(hands)))
    broadway = {h for h in hands if h[0][1] in _BROADWAY and h[1][1] in _BROADWAY}
    polarized = set(by_str[:k]) | set(by_str[-k:])            # weakest + strongest
    by_var = sorted(hands, key=lambda h: var[h], reverse=True)
    draw_heavy = set(by_var[:max(1, int(frac * len(hands)))])  # most rank-volatile = draws
    return {'broadway': broadway, 'polarized': polarized, 'draw_heavy': draw_heavy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=None)
    ap.add_argument('--rivers', type=int, default=12)
    ap.add_argument('--nbuckets', type=int, default=128)
    ap.add_argument('--max-boards', type=int, default=len(_BOARDS))
    ap.add_argument('--sprs', type=float, nargs='+', default=[2.0, 4.0, 6.0])
    ap.add_argument('--pot', type=float, default=24.0)
    args = ap.parse_args()
    path = args.db or _latest_capped()
    rawdb = BlueprintDB(path, read_only=True)
    menu = postflop_menu_for(db_menu_mode(rawdb))
    db = FrozenBlueprint(rawdb)
    ev, cards = HandEvaluator(), CardAbstraction()

    boards = _BOARDS[:args.max_boards]
    print(f"blueprint: {os.path.basename(path)} | menu={db_menu_mode(rawdb)} | n={args.nbuckets} "
          f"| rivers={args.rivers} | boards={len(boards)} | sprs={args.sprs}")
    print(f"metrics per (board, SPR, shift): rel-RMSE (vs shifted exact) | "
          f"STABLE rel-RMSE (vs no-shift RMS) | pot-norm (per-pair err / pot)\n")

    worst = {'coarse': (0.0, None)}
    shift_worst = {}
    rows = []
    for board in boards:
        ba_cache = {}
        rivers = [c for c in FULL_DECK if c not in set(board)]
        if args.rivers > 0:
            rivers = rivers[:args.rivers]
        strength, var = turn_strength(board, ev, cards, rivers=rivers,
                                      ba_cache=ba_cache, with_var=True)
        part = equal_freq_partition(strength, args.nbuckets)
        shifts = _shifts(strength, var)
        ba_turn = build_turn_board_arrays(board)
        hands = ba_turn['hands']
        full = {h: 1.0 for h in hands}

        for spr in args.sprs:
            behind = spr * args.pot
            if args.pot + 2 * behind > 2 * 200.0 + 1e-9:    # exceeds table chips (2x STARTING)
                continue
            stacks = (behind, behind)
            M0, M1, buckets, bidx, tb = turn_leaf_matrix_both(
                board, args.pot, stacks, db, ev, cards, menu=menu, rivers=rivers,
                partition=part, ba_cache=ba_cache)
            exact_full = turn_leaf_value_exact(board, args.pot, stacks, 0, full, full,
                                               db, ev, cards, menu=menu, rivers=rivers,
                                               ba_cache=ba_cache)
            ref = _rms(exact_full)                          # stable denominator
            pred_full = bucketed_measure_leaf_cr(M0, bidx, tb, hands, full)
            coarse = _rel_rmse(pred_full, exact_full, _rms(exact_full))
            if coarse > worst['coarse'][0]:
                worst['coarse'] = (coarse, f"{board} spr{spr:g}")
            print(f"  {str(board):28s} SPR {spr:>4g}  coarseness "
                  f"{coarse:6.1%} (stable) | pot-norm "
                  f"{_pot_norm_err(pred_full, exact_full, ba_turn, full, args.pot):6.1%}")
            for name, hset in shifts.items():
                vr = {h: 1.0 for h in hset}
                if not vr:
                    continue
                exact_s = turn_leaf_value_exact(board, args.pot, stacks, 0, full, vr,
                                                db, ev, cards, menu=menu, rivers=rivers,
                                                ba_cache=ba_cache)
                pred_s = bucketed_measure_leaf_cr(M0, bidx, tb, hands, vr)
                rr = _rel_rmse(pred_s, exact_s, _rms(exact_s))
                rr_stable = _rel_rmse(pred_s, exact_s, ref)
                pn = _pot_norm_err(pred_s, exact_s, ba_turn, vr, args.pot)
                rows.append((board, spr, name, rr, rr_stable, pn))
                w = shift_worst.get(name, (0.0, None))
                if rr_stable > w[0]:
                    shift_worst[name] = (rr_stable, f"{board} spr{spr:g}")
                print(f"      shift={name:11s} rel {rr:6.1%} | stable {rr_stable:6.1%} "
                      f"| pot-norm {pn:6.1%}  (|villain|={len(hset)})")
    rawdb.close()

    print("\n=== WORST CONFIG (the safety-relevant number) ===")
    print(f"  coarseness (no shift): {worst['coarse'][0]:.1%}  @ {worst['coarse'][1]}")
    for name in ('broadway', 'polarized', 'draw_heavy'):
        if name in shift_worst:
            print(f"  worst {name:11s} (stable rel-RMSE): "
                  f"{shift_worst[name][0]:.1%}  @ {shift_worst[name][1]}")
    overall = max((w[0] for w in shift_worst.values()), default=0.0)
    print(f"\n  OVERALL worst under-shift (stable rel-RMSE across all shifts): {overall:.1%}")
    print("  Interpretation: if the worst structural shift (esp. draw_heavy, the one "
          "orthogonal to the\n  strength partition) stays within tolerance, the leaf is "
          "robust beyond the easy broadway case.")


if __name__ == '__main__':
    main()
