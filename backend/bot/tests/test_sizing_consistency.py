# backend/bot/tests/test_sizing_consistency.py
"""
Guards that every consumer of the betting-size abstraction agrees with the
single source of truth (abstractions/sizing.py): the trainer/live engine
(poker_game), the LBR harness (lbr), the PyPokerEngine path (action_abstractions),
AND the head-to-head/AIVAT match player (match). This is the anti-drift test for
the sizing centralisation (the same idea as cfr/keys.py for info-set keys).

NOTE: match.py was a FOURTH copy that originally drifted (it kept the old scheme)
because an earlier version of this test only covered three modules. Every sizing
consumer must be locked here, or its drift is structurally invisible.

Run: python tests/test_sizing_consistency.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.abstractions import sizing
from src.cfr.poker_game import PokerGame, STARTING_STACK
from src.abstractions.action_abstractions import ActionAbstraction
from src.evaluation.lbr import LBREvaluator
from src.evaluation.match import _sizing as match_sizing

_passed = _failed = 0


def check(name, cond, extra=''):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name} {extra}")


# All three trained postflop menu arms. Drift-guards must cover every served
# arm; the prod-deployed blueprint is `capped`, so a control-only test would
# leave the prod path structurally invisible to this guard. Each entry maps
# (menu_mode, sizing.<dict>) so a future arm is one line to add.
POSTFLOP_MENUS = (
    ('control',     sizing.POSTFLOP_BET_MULT),
    ('capped',      sizing.POSTFLOP_BET_MULT_CAPPED),
    ('capped_no2',  sizing.POSTFLOP_BET_MULT_CAPPED_NO2),
)


def test_engine_matches_sizing():
    # Preflop side (mode-independent): one check covers it.
    g = PokerGame()
    check('engine open == sizing.preflop_open_chips',
          g.get_preflop_bet_amounts('open', 3) == sizing.preflop_open_chips(),
          f"({g.get_preflop_bet_amounts('open', 3)})")
    pr = g.get_preflop_bet_amounts('pot_relative', 100)
    expected = {k: m * 100 for k, m in sizing.PREFLOP_RAISE_MULT.items()}
    check('engine pot_relative == sizing.PREFLOP_RAISE_MULT x pot', pr == expected,
          f"({pr})")
    # 3-bet is unified into pot_relative (no separate absolute tier).
    check('3-bet action type is pot_relative',
          g.get_preflop_action_type(['bet_medium']) == 'pot_relative')

    # Postflop side: per-menu-mode. Each arm's engine must match its dict.
    for mode, expected_menu in POSTFLOP_MENUS:
        gm = PokerGame(postflop_menu=expected_menu,
                       voluntary_allin=not sizing.is_capped_mode(mode))
        check(f'engine[{mode}] BET_MULTIPLIERS == sizing menu',
              gm.BET_MULTIPLIERS == expected_menu,
              f"({gm.BET_MULTIPLIERS} vs {expected_menu})")
        check(f'engine[{mode}] voluntary_allin matches mode',
              gm.voluntary_allin == (not sizing.is_capped_mode(mode)))


def test_action_abstraction_matches_sizing():
    # Preflop (mode-independent).
    aa = ActionAbstraction()
    gstate = {'pot_size': 3, 'current_bet': 0, 'player_contribution': 0, 'big_blind': 2}
    rstate = {'street': 'preflop', 'action_histories': {'preflop': []}}
    opens = {s: aa._calculate_target_amount(s, 'bet', gstate, rstate)
             for s in sizing.PREFLOP_OPEN_BB}
    check('action_abstraction open == sizing.preflop_open_chips',
          opens == sizing.preflop_open_chips(), f"({opens})")
    # Postflop: per-menu-mode. ActionAbstraction reads its menu from menu_mode.
    pstate = {'street': 'flop', 'action_histories': {'flop': []}}
    pg = {'pot_size': 20, 'current_bet': 0, 'player_contribution': 0, 'big_blind': 2}
    for mode, expected_menu in POSTFLOP_MENUS:
        aam = ActionAbstraction(menu_mode=mode)
        postflop = {s: aam._calculate_target_amount(s, 'bet', pg, pstate)
                    for s in expected_menu}
        expected_post = {s: m * 20 for s, m in expected_menu.items()}
        check(f'action_abstraction[{mode}] postflop == sizing menu x pot',
              postflop == expected_post, f"({postflop} vs {expected_post})")


def test_lbr_open_matches_sizing():
    lbr = LBREvaluator(_NullDB())
    open_chips = sizing.preflop_open_chips()
    for s in sizing.PREFLOP_OPEN_BB:
        # Open with nothing committed beyond the SB(1): add == raise-to - 1.
        add = lbr._bot_sizing(s, street=0, pot=3, to_call=1, bot_committed=1, num_aggr=0)
        check(f'lbr open {s} add matches engine increment',
              add == int(round(open_chips[s] - 1)), f"(add={add}, to={open_chips[s]})")


def test_match_matches_sizing():
    """The head-to-head/AIVAT match player must use the same sizes, or version
    comparisons are biased. Mirrors the lbr check (same _sizing signature)."""
    open_chips = sizing.preflop_open_chips()
    for s in sizing.PREFLOP_OPEN_BB:
        add = match_sizing(s, 0, pot=3, to_call=1, committed=1, num_aggr=0)
        check(f'match open {s} add matches engine increment',
              add == int(round(open_chips[s] - 1)), f"(add={add}, to={open_chips[s]})")
    # 3-bet (num_aggr=1) is pot-relative, unified with 4-bet+ (no absolute ladder).
    pot, to_call = 10, 4
    add1 = match_sizing('medium', 0, pot=pot, to_call=to_call, committed=4, num_aggr=1)
    expected = int(round(to_call + sizing.PREFLOP_RAISE_MULT['medium'] * (pot + to_call)))
    check('match 3-bet is pot-relative (unified)', add1 == expected,
          f"(add={add1}, expected={expected})")


class _NullDB:
    def get_average_strategy(self, key):
        return None


if __name__ == '__main__':
    test_engine_matches_sizing()
    test_action_abstraction_matches_sizing()
    test_lbr_open_matches_sizing()
    test_match_matches_sizing()
    print(f"\nResults: {_passed} passed, {_failed} failed out of {_passed + _failed}")
    sys.exit(1 if _failed else 0)
