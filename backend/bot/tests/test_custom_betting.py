# backend/bot/tests/test_custom_betting.py
"""
Tests for unrestricted (custom) human bet sizing + pseudo-harmonic action
translation (Phase 1a). Run: python tests/test_custom_betting.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.poker_game import PokerGame, STARTING_STACK, make_custom_action, _custom_total
from src.cfr import translation
from src.game.game_session import GameSession
from src.game.bot_strategy import BlueprintStrategy

_passed = _failed = 0


def check(name, cond, extra=''):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name} {extra}")


# ---------------------------------------------------------------- engine
def test_engine_custom_chip_math():
    g = PokerGame()
    street, sp, p0_prev, p1_prev = 1, 4.0, 2.0, 2.0
    hist = []
    bet = make_custom_action(False, 30)
    check('custom bet cost = total (contrib 0)',
          g._action_cost(bet, street, hist, sp, 1, p0_prev, p1_prev) == 30.0)
    hist = [bet]
    check('to_call vs custom bet-30 = 30',
          g.get_call_amount_from_history(street, hist, sp, p0_prev, p1_prev) == 30.0)
    rr = make_custom_action(True, 80)
    check('custom raise-to-80 cost (contrib 0) = 80',
          g._action_cost(rr, street, hist, sp, 0, p0_prev, p1_prev) == 80.0)
    hist = [bet, rr, 'call']
    pot = g.calculate_current_pot(sp, hist, street, p0_prev, p1_prev)
    check('pot after custom bet/raise/call = 4+80+80', pot == 164.0, f'(got {pot})')
    c0 = g.get_player_contribution_this_round(hist, street, sp, 0, p0_prev, p1_prev)
    c1 = g.get_player_contribution_this_round(hist, street, sp, 1, p0_prev, p1_prev)
    inv0, inv1 = p0_prev + c0, p1_prev + c1
    conserved = inv0 + inv1 + (STARTING_STACK - inv0) + (STARTING_STACK - inv1)
    check('chip conservation holds', conserved == 2 * STARTING_STACK, f'(got {conserved})')


def test_custom_bounds():
    g = PokerGame()
    # Flop, no prior bet: min = 1 BB (2 chips), max = all-in total.
    b = g.custom_bet_bounds(1, [], 4.0, 1, STARTING_STACK - 2, STARTING_STACK - 2, 2.0, 2.0)
    check('bounds present', b is not None)
    check('min bet = 2 chips', b[0] == 2.0, f'(got {b})')
    check('max = all-in total (198)', b[1] == 198.0, f'(got {b})')
    # Facing a 30 bet: min raise-to = 60, the call portion enforced by min_raise.
    b2 = g.custom_bet_bounds(1, [make_custom_action(False, 30)], 4.0, 0,
                             STARTING_STACK - 2, STARTING_STACK - 2, 2.0, 2.0)
    check('min raise-to vs bet-30 = 60', b2[0] == 60.0, f'(got {b2})')


# ---------------------------------------------------------------- translation
def test_translation_blend():
    g = translation.POSTFLOP_GRID
    check('on-grid 0.66 -> single m', translation.translate_bet(0.66, g) == [('m', 1.0)])
    t = translation.translate_bet(0.5, g)
    check('off-grid 0.5 -> blends s and m', len(t) == 2 and {c for c, _ in t} == {'s', 'm'})
    w = dict(t)
    check('0.5 weights sum to 1', abs(w['s'] + w['m'] - 1.0) < 1e-9)
    check('0.5 leans to nearer m', w['m'] > w['s'])
    grid_ai = g + [('a', 4.0)]
    t2 = translation.translate_bet(1.5, grid_ai)
    check('overbet 1.5 -> blends l and a', {c for c, _ in t2} == {'l', 'a'})


# ---------------------------------------------------------------- session + bot
class _FakeDB:
    """Toy blueprint: larger bets get more folds, so blending is observable."""
    def get_average_strategy(self, key):
        if key.endswith('a'): return {'fold': 0.9, 'call': 0.1}
        if key.endswith('l'): return {'fold': 0.7, 'call': 0.3}
        if key.endswith('m'): return {'fold': 0.3, 'call': 0.7}
        if key.endswith('s'): return {'fold': 0.1, 'call': 0.9}
        return None


def _strat_fn(key, legal):
    s = _FakeDB().get_average_strategy(key)
    n = len(legal)
    if s:
        w = np.array([max(0.0, s.get(a, 0.0)) for a in legal])
        t = w.sum()
        return w / t if t > 1e-12 else np.ones(n) / n
    return np.ones(n) / n


def test_session_custom_and_translation():
    sess = GameSession.new('s', 'p', strategy_fn=_strat_fn)
    d = sess.data
    # Human (SB) custom-opens to 6 chips (between medium=5 and large=7), so it's
    # off-grid and translation blends the bot's two bracketing responses.
    sess.apply_action(make_custom_action(False, 6))
    check('custom action recorded in history', d['history'] == ['bet_custom_6'])
    check('pattern got a grid char (not x)', d['bet_pattern'] in ('s', 'm', 'l'))
    check('pending_translation set for bot', d.get('pending_translation') is not None)
    check('stacks reflect a 6-chip open', d['p0_stack'] == STARTING_STACK - 6)

    ps = sess.bot_public_state()
    check('public_state carries translation keys', len(ps.get('translation', [])) == 2)

    bot = BlueprintStrategy(_FakeDB())
    dist = bot.explain(sess.info_set_key(sess.current_player()), sess.legal_actions(), ps)
    # The bot's fold/call mix must lie strictly between its single-key responses
    # to the two bracketing sizes (proof the blend happened).
    keys = [k for k, _ in ps['translation']]
    lo = bot._distribution(keys[0], sess.legal_actions())['fold']
    hi = bot._distribution(keys[1], sess.legal_actions())['fold']
    check('blended fold prob between brackets',
          min(lo, hi) - 1e-9 <= dist['fold'] <= max(lo, hi) + 1e-9,
          f"(fold={dist['fold']:.3f}, brackets={lo:.3f},{hi:.3f})")


def test_all_in_normalisation():
    sess = GameSession.new('s', 'p', strategy_fn=_strat_fn)
    # A custom amount >= stack must normalise to all-in.
    sess.apply_action(make_custom_action(False, 10_000))
    check('over-stack custom -> all-in', 'allin' in sess.data['history'])


if __name__ == '__main__':
    test_engine_custom_chip_math()
    test_custom_bounds()
    test_translation_blend()
    test_session_custom_and_translation()
    test_all_in_normalisation()
    print(f"\nResults: {_passed} passed, {_failed} failed out of {_passed + _failed}")
    sys.exit(1 if _failed else 0)
