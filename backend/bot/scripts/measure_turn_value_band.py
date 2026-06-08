# backend/bot/scripts/measure_turn_value_band.py
"""
M-pre (depth-limited turn solver): MEASURE-FIRST gate. Before building the turn
solver, quantify whether it can add value over the blueprint + the existing
all-in guard + the SPR gate.

Two questions:
  (1) Belief sharpness at TURN entry vs RIVER entry -- the turn solver consumes the
      range tracker's belief; if the turn-entry belief is much vaguer than river entry
      (where the solver already works), the turn solve quality degrades. Reported as
      the EFFECTIVE number of opponent hands (participation ratio 1/sum p^2; lower =
      sharper). Confidence is ~1 in self-play (the opponent IS the blueprint), so
      sharpness, not confidence, is the informative metric.
  (2) The ADDRESSABLE BAND: of turn decisions, how many would the full subtree solver
      actually fire on and add value -- i.e. NOT facing a stack-committing jam (the
      `_facing_allin_guard` already handles those), NOT high-SPR (the SOLVER_MAX_SPR
      gate skips those), NOT trivially low-SPR jam-or-fold, and with a real aggression
      option + a meaningful pot. If that band is thin, the cheap one-ply alternative
      wins and the heavy build isn't worth it.

READ-ONLY (opens the blueprint read-only). Run from backend/bot/:
    python scripts/measure_turn_value_band.py --hands 1200
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.blueprint_db import BlueprintDB
from src.cfr.poker_game import STARTING_STACK
from src.game.game_session import GameSession
from src.game.bot_strategy import BlueprintStrategy
from src.game.range_tracker import RangeTracker
from src.abstractions.card_abstractions import CardAbstraction
from src.abstractions.sizing import db_menu_mode

SPR_GATE = 6.0          # mirrors SOLVER_MAX_SPR (turn solver would skip SPR > this)
LOW_SPR = 1.5           # below this ~ jam-or-fold (blueprint/guard fine)
POT_FLOOR_BB = 8.0      # a "meaningful" pot


def _latest_capped():
    c = sorted(glob.glob('analysis/blueprints/blueprint_par_capped_*.db'))
    return c[-1] if c else None


def _eff_hands(tracker):
    w = np.asarray(tracker.w, dtype=float)
    w = w[w > 0]
    if w.size == 0:
        return 0.0
    p = w / w.sum()
    return float(1.0 / np.sum(p * p))     # participation ratio (effective # combos)


def run(db_path, n_hands, max_steps=400):
    db = BlueprintDB(db_path, read_only=True)
    menu = db_menu_mode(db)
    cards = CardAbstraction()
    strat_fn = BlueprintStrategy(db).range_model_fn()
    bp = BlueprintStrategy(db)
    sess = GameSession.new('mpre', 'p', strategy_fn=strat_fn, menu_mode=menu)

    turn_eff, river_eff = [], []
    classes = {'facing_jam': 0, 'spr_high': 0, 'spr_low': 0, 'addressable': 0}
    addressable_meaningful = 0
    turn_decisions = 0
    seen_turn = seen_river = False

    h = 0
    while h < n_hands:
        steps = 0
        seen_turn = seen_river = False
        while sess.data['status'] == 'in_hand' and steps < max_steps:
            d = sess.data
            street = d['street']
            seat = sess.current_player()
            legal = sess.legal_actions()

            # belief sharpness at first turn / first river decision of the hand
            if street in (2, 3) and d.get('opp_range') is not None:
                if street == 2 and not seen_turn:
                    seen_turn = True
                    turn_eff.append(_eff_hands(RangeTracker.from_dict(d['opp_range'], cards)))
                elif street == 3 and not seen_river:
                    seen_river = True
                    river_eff.append(_eff_hands(RangeTracker.from_dict(d['opp_range'], cards)))

            # band classification on TURN decisions
            if street == 2:
                turn_decisions += 1
                behind = d['p0_stack'] if seat == 0 else d['p1_stack']
                pot = 2 * STARTING_STACK - d['p0_stack'] - d['p1_stack']   # all committed chips
                to_call = sess._action_cost('call') if 'call' in legal else 0.0
                has_aggr = any(a.startswith(('bet_', 'raise_')) or a == 'allin' for a in legal)
                spr = behind / pot if pot > 1e-9 else 999.0
                if to_call >= behind - 1e-9:
                    classes['facing_jam'] += 1
                elif spr > SPR_GATE:
                    classes['spr_high'] += 1
                elif spr <= LOW_SPR:
                    classes['spr_low'] += 1
                else:
                    classes['addressable'] += 1
                    if has_aggr and pot / 2.0 >= POT_FLOOR_BB:   # pot in BB
                        addressable_meaningful += 1

            a = bp.decide(sess.info_set_key(seat), legal, {})
            sess.apply_action(a)
            steps += 1

        if sess.data['status'] == 'hand_over':
            sess.start_next_hand()
        h += 1
    db.close()

    def q(xs):
        if not xs:
            return (0.0, 0.0, 0.0)
        a = np.array(xs)
        return (float(np.percentile(a, 25)), float(np.median(a)), float(np.percentile(a, 75)))

    print(f"\nblueprint: {os.path.basename(db_path)} | menu={menu} | hands={n_hands}")
    print(f"\n(1) BELIEF SHARPNESS -- effective # opponent combos (lower = sharper):")
    te, ri = q(turn_eff), q(river_eff)
    print(f"  turn entry  (n={len(turn_eff)}):  median {te[1]:.0f}   [q1 {te[0]:.0f}, q3 {te[2]:.0f}]")
    print(f"  river entry (n={len(river_eff)}): median {ri[1]:.0f}   [q1 {ri[0]:.0f}, q3 {ri[2]:.0f}]")
    if te[1] and ri[1]:
        print(f"  -> turn belief is {te[1]/ri[1]:.1f}x vaguer than river (1.0 = equally sharp)")

    print(f"\n(2) ADDRESSABLE BAND -- of {turn_decisions} turn decisions:")
    for k in ('facing_jam', 'spr_high', 'spr_low', 'addressable'):
        pct = 100.0 * classes[k] / turn_decisions if turn_decisions else 0.0
        print(f"  {k:12s}: {classes[k]:6d}  ({pct:4.1f}%)")
    am = 100.0 * addressable_meaningful / turn_decisions if turn_decisions else 0.0
    print(f"  -> ADDRESSABLE w/ aggression option + pot>={POT_FLOOR_BB:.0f}BB: "
          f"{addressable_meaningful} ({am:.1f}% of turn decisions)")
    print("\nVERDICT GUIDE: if the addressable band is thin (say <~15%) and/or turn belief is "
          ">~2x vaguer than river, prefer the cheap one-ply alternative over the full subtree solver.")


def main():
    p = argparse.ArgumentParser(description="M-pre: turn-solver value-band measurement.")
    p.add_argument('--db', default=None, help="Blueprint DB (default: latest capped snapshot).")
    p.add_argument('--hands', type=int, default=1200)
    args = p.parse_args()
    path = args.db or _latest_capped()
    if not path:
        print("No capped blueprint found.")
        sys.exit(1)
    run(path, args.hands)


if __name__ == '__main__':
    main()
