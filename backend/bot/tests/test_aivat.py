# backend/bot/tests/test_aivat.py
"""
Validates the AIVAT river control variate is genuinely ZERO-MEAN (so AIVAT's
estimate is unbiased, not just lower-variance). The river support excludes BOTH
players' hole cards — the true chance support — so averaging the correction over
every possible river must give 0. (This directly checks the property a review
questioned.)

Run: python tests/test_aivat.py    (loads the card-abstraction table; ~slow)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.aivat import AIVATEstimator
from src.evaluation.lbr import BotRange, _FULL_DECK

_passed = _failed = 0


def check(name, cond, extra=''):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name} {extra}")


class _FakeDB:
    """The river CV path never queries strategy; only equity_vs_range is used."""
    def get_average_strategy(self, key):
        return None


def test_river_control_variate_is_zero_mean():
    est = AIVATEstimator(_FakeDB(), seed=0)   # __init__ also asserts 169 preflop hands
    hand_a = ('HA', 'HK')
    hand_b = ('DA', 'DQ')
    turn_board = ['S2', 'S7', 'CJ', 'D5']
    brange = BotRange(hand_a, est.cards)
    brange.reveal(turn_board)

    dead = set(turn_board) | set(hand_a) | set(hand_b)
    rivers = [c for c in _FULL_DECK if c not in dead]
    pot = 50.0
    vals = [est._river_correction(hand_a, hand_b, turn_board, r, brange, pot)
            for r in rivers]
    mean = sum(vals) / len(vals)
    # Averaging realized-minus-mean over the exact support is 0 to float precision.
    check('river CV mean ~ 0 over the full river support', abs(mean) < 1e-6,
          f'(mean={mean:.3e} over {len(rivers)} rivers)')
    # Sanity: it is not the trivial all-zero CV (it actually varies by river).
    check('river CV is non-degenerate', max(vals) - min(vals) > 1e-6,
          f'(spread={max(vals) - min(vals):.3f})')


if __name__ == '__main__':
    test_river_control_variate_is_zero_mean()
    print(f"\nResults: {_passed} passed, {_failed} failed out of {_passed + _failed}")
    sys.exit(1 if _failed else 0)
