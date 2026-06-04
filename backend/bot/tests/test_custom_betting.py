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
    # 1.5 is now an ON-grid point ('o' = overbet), so it must NOT blend -> single 'o'.
    # (This test predated the overbet tier being added to POSTFLOP_GRID; updated to use
    #  a fraction that is genuinely BETWEEN the top grid size and the all-in anchor.)
    grid_ai = g + [('a', 4.0)]                       # [s .33, m .66, l 1.0, o 1.5, a 4.0]
    check('on-grid 1.5 -> single o', translation.translate_bet(1.5, grid_ai) == [('o', 1.0)])
    t2 = translation.translate_bet(2.5, grid_ai)     # between o(1.5) and a(4.0)
    check('off-grid 2.5 -> blends o and a', {c for c, _ in t2} == {'o', 'a'},
          f'(got {t2})')


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


def test_substack_custom_not_snapped_to_allin():
    """BUG-016: a SUB-stack custom bet whose pot-fraction is closest to the all-in edge
    must NOT be recorded as all-in (char 'a') -- doing so makes the bot's next info-set
    key + range update wrongly think the opponent shoved while money is still behind. It
    snaps to a SIZED char instead. (An at/above-stack custom is still normalised to
    'allin' by _validate_custom -- see test_all_in_normalisation.) Would FAIL pre-fix:
    the old grid carried an 'a' edge, and a 150-chip open is far above the largest sized
    open (xlarge=5 BB) and nearest the all-in edge (200), so it snapped to 'a'."""
    sess = GameSession.new('s', 'p', strategy_fn=_strat_fn)
    # SB opens to 150 chips (75 BB): 50 chips / 25 BB remain behind -> NOT all-in.
    sess.apply_action(make_custom_action(False, 150))
    d = sess.data
    check('sub-stack near-shove recorded as a custom bet, not all-in',
          d['history'] == ['bet_custom_150'], d['history'])
    check('pattern char is sized, not all-in (a)',
          bool(d['bet_pattern']) and d['bet_pattern'][-1] != 'a', d['bet_pattern'])
    check('money is still behind (genuinely sub-stack)', d['p0_stack'] > 0, d['p0_stack'])


if __name__ == '__main__':
    test_engine_custom_chip_math()
    test_custom_bounds()
    test_translation_blend()
    test_session_custom_and_translation()
    test_all_in_normalisation()
    test_substack_custom_not_snapped_to_allin()
    print(f"\nResults: {_passed} passed, {_failed} failed out of {_passed + _failed}")
    sys.exit(1 if _failed else 0)
