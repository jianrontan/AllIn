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
    assert cond, f"{name} {extra}"   # fail under pytest too, not just the __main__ count


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


class _Spr5FakeDB:
    """An spr5 blueprint stand-in that records every key the range replay looks up."""
    def __init__(self):
        self.keys = []

    def get_metadata(self, key, default=None):
        return {'key_format': 'spr5', 'menu_mode': 'capped'}.get(key, default)

    def get_average_strategy(self, key):
        self.keys.append(key)
        return None


def test_range_replay_builds_spr5_keys():
    """c2's replay of B's range must look B's flop/turn actions up in B's own key
    format: for an spr5 B, a v2-format key misses every time."""
    db = _Spr5FakeDB()
    est = AIVATEstimator(db, seed=0)
    flop = ['S2', 'S7', 'CJ']
    rec = {'seat_of_A': 0, 'hand_a': ('HA', 'HK'), 'hand_b': ('DA', 'DQ'),
           'board': flop + ['D5', 'C9'], 'allin_street': None, 'folded': None,
           'result': 0.0, 'invested': [10, 10],
           'events': [
               {'type': 'street_start', 'street': 0, 'vis': [], 'pot': 3},
               {'type': 'street_start', 'street': 1, 'vis': flop, 'pot': 20},
               {'type': 'action', 'street': 1, 'seat': 1, 'vis': flop, 'pattern': '',
                'legal': ['check', 'bet_small'], 'choice': 0},
           ]}
    est._hand_variates(rec)
    flop_keys = [k for k in db.keys if '_flop_' in k]
    check('AIVAT range replay looks up flop keys', bool(flop_keys))
    # Street-start pot 20 -> SPR (200 - 10) / 20 = 9.5 -> bucket 2.
    check('AIVAT range replay builds spr5 flop keys', all('_flop_r2_' in k for k in flop_keys),
          f'({flop_keys[:2]})')
    assert flop_keys and all('_flop_r2_' in k for k in flop_keys), flop_keys[:2]


if __name__ == '__main__':
    test_river_control_variate_is_zero_mean()
    test_range_replay_builds_spr5_keys()
    print(f"\nResults: {_passed} passed, {_failed} failed out of {_passed + _failed}")
    sys.exit(1 if _failed else 0)
