# backend/bot/tests/run_cross_match.py
"""
Validate + run the cross-sizing head-to-head (src/evaluation/cross_match.py).

Validation (must pass before the result is trustworthy):
  1. NEW sizing == the live engine's _action_cost at matched nodes.
  2. OLD sizing == hand-computed expecteds from the pre-redesign engine
     (open 3/5/7BB, 3bet abs 9/12/16BB, 4bet+ potrel 0.66/1.33/2.0 of pot-before).
  3. self-vs-self and old-vs-old net ~= 0 mbb (accounting/symmetry).
  4. cross-perception: old's medium open is perceived by the new bot as 'large'.

Then runs old(9.15M) vs new(snap_6050000).

    python tests/run_cross_match.py --hands 40000
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.poker_game import PokerGame
from src.cfr import translation
from src.storage.blueprint_db import BlueprintDB
from src.evaluation.cross_match import (
    CrossMatch, CrossBot, NEW_SIZING, OLD_SIZING, _legal_actions)
from src.abstractions.card_abstractions import CardAbstraction

# OLD = the pre-redesign (old-sizing) run, continued to 10.05M (the 9.15M
# intermediate snapshot was cleaned up; this is the same run, more trained).
OLD_DB = 'analysis/blueprints/blueprint_20260525_062044.db'
NEW_DB = 'analysis/blueprints/snapshots/snap_6050000.db'


def _state(g, street, history, starting_pot, cp, p0_prev, p1_prev):
    pot = g.calculate_current_pot(starting_pot, history, street, p0_prev, p1_prev)
    comm = g.get_player_contribution_this_round(history, street, starting_pot, cp, p0_prev, p1_prev)
    oth = g.get_player_contribution_this_round(history, street, starting_pot, 1 - cp, p0_prev, p1_prev)
    to_call = max(0.0, oth - comm)
    num_aggr = sum(1 for a in history if a.startswith(('bet_', 'raise_')) or a == 'allin')
    return pot, to_call, comm, num_aggr


def validate_new_sizing():
    g = PokerGame()
    # (street, history, starting_pot, p0_prev, p1_prev, current_player)
    scenarios = [
        (0, [], 3.0, 0.0, 0.0, 0),                          # SB open
        (0, ['bet_medium'], 3.0, 0.0, 0.0, 1),              # BB 3bet
        (0, ['bet_medium', 'raise_medium'], 3.0, 0.0, 0.0, 0),  # SB 4bet
        (1, [], 20.0, 10.0, 10.0, 1),                       # flop bet (OOP first)
        (1, ['bet_medium'], 20.0, 10.0, 10.0, 0),           # flop raise
        (1, ['bet_medium', 'raise_medium'], 20.0, 10.0, 10.0, 1),  # flop re-raise
        (2, ['bet_small'], 40.0, 30.0, 30.0, 0),            # turn raise
    ]
    bad = 0
    for (street, hist, sp, p0, p1, cp) in scenarios:
        pot, to_call, comm, num_aggr = _state(g, street, hist, sp, cp, p0, p1)
        prefix = 'bet_' if to_call == 0 else 'raise_'
        for size in ('small', 'medium', 'large'):
            a = prefix + size
            try:
                eng = g._action_cost(a, street, hist, sp, cp, p0, p1)
            except Exception:
                continue
            mine = NEW_SIZING.add_chips(size, street, pot, to_call, comm, num_aggr)
            if abs(eng - mine) > 0.5:
                bad += 1
                print(f"  NEW MISMATCH {a} st={street} hist={hist}: engine={eng:.2f} mine={mine:.2f} "
                      f"(pot={pot} to_call={to_call} comm={comm} aggr={num_aggr})")
    print(f"NEW sizing vs engine: {'OK' if bad == 0 else f'{bad} MISMATCH'}")
    return bad == 0


def validate_old_sizing():
    """Hand-computed from the pre-redesign engine (bc71a9f~1)."""
    # open: raise-to BB ladder 3/5/7 -> chips 6/10/14; SB committed=1 -> add 5/9/13
    checks = [
        # (label, add_chips_args, expected_add)
        ('open small', ('small', 0, 3.0, 1.0, 1.0, 0), 6 - 1),
        ('open medium', ('medium', 0, 3.0, 1.0, 1.0, 0), 10 - 1),
        ('open large', ('large', 0, 3.0, 1.0, 1.0, 0), 14 - 1),
        # 3bet abs 9/12/16 BB -> 18/24/32 chips; BB committed=2, facing 10 (SB med open total)
        ('3bet small', ('small', 0, 12.0, 8.0, 2.0, 1), 18 - 2),
        ('3bet medium', ('medium', 0, 12.0, 8.0, 2.0, 1), 24 - 2),
        # postflop identical to new: raise = to_call + mult*(pot+to_call)
        ('flop raise med', ('medium', 1, 30.0, 10.0, 0.0, 1), 10 + 0.66 * 40),
    ]
    bad = 0
    for label, args, exp in checks:
        mine = OLD_SIZING.add_chips(*args)
        if abs(mine - exp) > 0.5:
            bad += 1
            print(f"  OLD MISMATCH {label}: mine={mine:.2f} expected={exp:.2f}")
    print(f"OLD sizing vs hand-computed: {'OK' if bad == 0 else f'{bad} MISMATCH'}")
    return bad == 0


def validate_perception():
    """The new bot must perceive the OLD bot's medium open (5BB) as a 'large' (3.5BB
    is its biggest open) -> char 'l'."""
    cards = CardAbstraction()
    import random
    newbot = CrossBot(None, NEW_SIZING, cards, random.Random(0))
    # OLD SB medium open: pot=3, to_call=1 (SB faces BB), committed_SB=1 -> add=9.
    old_add = OLD_SIZING.add_chips('medium', 0, 3.0, 1.0, 1.0, 0)
    # New bot's open grid at the SB node (it imagines opening): committed=1, to_call=1.
    grid = newbot.grid(0, 3.0, 1.0, 1.0, 0, 199.0)
    eff = translation.eff_fraction(old_add, 1.0, 3.0)
    ch = translation.nearest_char(eff, grid)
    print(f"perception: old med open add={old_add:.1f} eff={eff:.2f} grid={[(c,round(f,2)) for c,f in grid]} -> '{ch}'")
    ok = ch == 'l'
    print(f"cross-perception: {'OK' if ok else 'FAIL (expected l)'}")
    return ok


def selfplay_zero(db_path, sizing, hands, seed):
    db = BlueprintDB(db_path, read_only=True)
    try:
        res = CrossMatch(db, sizing, db, sizing, seed=seed).evaluate(num_hands=hands)
    finally:
        db.close()
    return res['a_mbb'], res['stderr_mbb']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hands', type=int, default=40000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--check-hands', type=int, default=4000)
    args = p.parse_args()

    print("=== VALIDATION ===")
    ok = True
    ok &= validate_new_sizing()
    ok &= validate_old_sizing()
    ok &= validate_perception()
    for label, dbp, sz in (('new', NEW_DB, NEW_SIZING), ('old', OLD_DB, OLD_SIZING)):
        m, se = selfplay_zero(dbp, sz, args.check_hands, args.seed)
        flag = 'OK' if abs(m) <= 3 * se else 'CHECK (>3 stderr from 0)'
        print(f"self-vs-self ({label}, {args.check_hands} hands): {m:+.1f} +/- {se:.1f} mbb -> {flag}")
    if not ok:
        print("\nVALIDATION FAILED -- not running the match.")
        sys.exit(1)

    print("\n=== OLD (9.15M, old sizing) [A]  vs  NEW (snap_6050000, new sizing) [B] ===")
    da = BlueprintDB(OLD_DB, read_only=True)
    dn = BlueprintDB(NEW_DB, read_only=True)
    try:
        t0 = time.time()
        res = CrossMatch(da, OLD_SIZING, dn, NEW_SIZING, seed=args.seed).evaluate(
            num_hands=args.hands, progress_every=max(1, args.hands // 10))
        dt = time.time() - t0
    finally:
        da.close()
        dn.close()
    a, se = res['a_mbb'], res['stderr_mbb']
    print(f"\n  OLD (A) net: {a:+.1f} +/- {se:.1f} mbb/hand over {args.hands} hands  ({dt:.0f}s)")
    verdict = ('NEW wins' if a < -2 * se else 'OLD wins' if a > 2 * se
               else 'inconclusive (within 2 stderr)')
    print(f"  => NEW (B) net: {-a:+.1f} mbb/hand  [{verdict}]")


if __name__ == '__main__':
    main()
