# backend/bot/tests/test_action_abstraction_roundtrip.py
"""
Round-trip consistency between the engine's bet SIZING (poker_game.py) and the
inference-side bet CATEGORISATION (action_abstractions.categorize_bet_size):
a bet the engine produces for size X must categorise back to X. This guards the
M-A/M-B family for the paths that are actually consistent (postflop bets,
preflop opens).

NOTE on raises: `categorize_bet_size` sizes a RAISE as a raw fraction of the
pre-bet pot, which inflates raises toward 'large' (the documented M-B issue). The
*live* bot no longer relies on that path — off-grid bets/raises are mapped via
cfr/translation.py (eff_fraction = (add - to_call)/(pot+to_call)), covered by
test_custom_betting.py. So this file deliberately round-trips the consistent
paths and leaves the legacy raise-categorisation to its (filed) status.

Run: python tests/test_action_abstraction_roundtrip.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.poker_game import PokerGame
from src.abstractions.action_abstractions import ActionAbstraction

_passed = _failed = 0


def check(name, cond, extra=''):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name} {extra}")


def test_postflop_bet_roundtrip():
    g = PokerGame()
    aa = ActionAbstraction()
    sp = 20.0          # pre-bet pot on the flop
    for size in ('small', 'medium', 'large'):
        amount = g.calculate_bet_amount(f'bet_{size}', 1, sp, [], 0.0, 0.0)
        gstate = {'pot_size': sp, 'big_blind': 2, 'player_stack': 200}
        cat = aa.categorize_bet_size({'action': 'bet', 'amount': amount}, gstate, [], 'flop')
        check(f'postflop bet_{size} round-trips', cat == size,
              f'(amount={amount:.2f} -> {cat})')


def test_preflop_open_roundtrip():
    g = PokerGame()
    aa = ActionAbstraction()
    amounts = g.get_preflop_bet_amounts('open', 3.0)   # {small:6, medium:10, large:14}
    for size in ('small', 'medium', 'large'):
        gstate = {'pot_size': 3, 'big_blind': 2, 'player_stack': 200}
        cat = aa.categorize_bet_size(
            {'action': 'bet', 'amount': amounts[size]}, gstate, [], 'preflop')
        check(f'preflop open {size} round-trips', cat == size,
              f'(amount={amounts[size]} -> {cat})')


def test_allin_categorisation():
    aa = ActionAbstraction()
    gstate = {'pot_size': 20, 'big_blind': 2, 'player_stack': 100}
    cat = aa.categorize_bet_size({'action': 'bet', 'amount': 100}, gstate, [], 'flop')
    check('full-stack bet categorises as allin', cat == 'allin', f'(got {cat})')


if __name__ == '__main__':
    test_postflop_bet_roundtrip()
    test_preflop_open_roundtrip()
    test_allin_categorisation()
    print(f"\nResults: {_passed} passed, {_failed} failed out of {_passed + _failed}")
    sys.exit(1 if _failed else 0)
