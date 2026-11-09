"""v3 step V3.1: the preflop cap jam ('capped_pj' menu mode). Facing the preflop
raise cap -- a 4-bet, the 3rd aggression -- the player may also go all-in. Without
it the trained tree has no 5-bet at all (the BUG-027 root cause)."""
import os
import random
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.abstractions import sizing  # noqa: E402
from src.abstractions.sizing import game_kwargs, postflop_menu_for  # noqa: E402
from src.cfr import translation  # noqa: E402
from src.cfr.poker_game import PokerGame, STARTING_STACK  # noqa: E402

CAP = ['bet_small', 'raise_small', 'raise_small']        # open, 3-bet, 4-bet


def _game(mode):
    return PokerGame(**game_kwargs(mode))


def _legal(g, street, history, pot=3.0, p_inv=(0.0, 0.0)):
    cp = g._acting_player(len(history), street)
    return g.get_legal_actions(street, history, pot, cp,
                               STARTING_STACK, STARTING_STACK, p_inv[0], p_inv[1])


# ------------------------------------------------------------------- menu plumbing

def test_game_kwargs_and_menus():
    assert game_kwargs('control') == {}
    pj = game_kwargs('capped_pj')
    assert pj['preflop_cap_jam'] is True and pj['voluntary_allin'] is False
    assert pj['postflop_menu'] is sizing.POSTFLOP_BET_MULT_CAPPED
    assert game_kwargs('capped')['preflop_cap_jam'] is False
    assert postflop_menu_for('capped_pj') is postflop_menu_for('capped')
    assert translation.postflop_grid_for('capped_pj') == translation.postflop_grid_for('capped')
    for bad in ('cappedpj', None, 'Capped'):
        with pytest.raises(ValueError):
            postflop_menu_for(bad)                      # no silent fallback to control
    assert sizing.MENU_MODES == ('control', 'capped', 'capped_no2', 'capped_pj')


# ------------------------------------------------------------------- the engine

def test_cap_node_offers_the_jam_only_under_capped_pj():
    assert _legal(_game('capped'), 0, CAP) == ['fold', 'call']
    assert _legal(_game('control'), 0, CAP) == ['fold', 'call']
    assert _legal(_game('capped_pj'), 0, CAP) == ['fold', 'call', 'allin']


@pytest.mark.parametrize('open_, three, four', [
    (o, t, f) for o in ('bet_small', 'bet_medium', 'bet_large', 'bet_xlarge')
    for t in ('raise_small', 'raise_medium', 'raise_large')
    for f in ('raise_small', 'raise_medium', 'raise_large')])
def test_every_sized_4bet_can_be_jammed(open_, three, four):
    g = _game('capped_pj')
    h = [open_, three]
    if four not in _legal(g, 0, h):
        pytest.skip('this 4-bet clamps to all-in already')
    assert _legal(g, 0, h + [four]) == ['fold', 'call', 'allin']


def test_facing_the_jam_is_fold_or_call_and_closes_the_hand():
    g = _game('capped_pj')
    jam = CAP + ['allin']
    assert _legal(g, 0, jam) == ['fold', 'call']
    assert g.is_terminal(jam + ['fold'], 0)
    assert g.is_terminal(jam + ['call'], 0)             # all-in called -> showdown
    pot = g.calculate_current_pot(3.0, jam + ['call'], 0)
    assert pot == 2 * STARTING_STACK


def test_postflop_cap_is_unchanged():
    g = _game('capped_pj')
    assert _legal(g, 1, CAP, pot=20.0, p_inv=(10.0, 10.0)) == ['fold', 'call']


def test_live_uncapped_play_never_reaches_a_cap():
    g = PokerGame(**game_kwargs('capped_pj'), max_raises_per_street=float('inf'))
    legal = _legal(g, 0, CAP)
    assert any(a.startswith('raise_') for a in legal)    # live: re-raises stay open


def test_state_path_matches_history_path_on_random_walks():
    """The trainer's state-threaded legal actions / costs / pots must equal the
    history-based engine everywhere, including at and after the cap jam."""
    g = _game('capped_pj')
    rng = random.Random(7)
    jams = 0
    for _ in range(3000):
        g._calc_cache.clear()
        pot, p_inv, street, history = 3.0, [0.0, 0.0], 0, []
        st = g.init_node_state(0, pot)
        for _guard in range(80):
            if g.is_terminal(history, street):
                break
            stacks = [STARTING_STACK - p_inv[i] - st['c'][i] for i in (0, 1)]
            cp = g._acting_player(len(history), street)
            eng = g.get_legal_actions(street, history, pot, cp, stacks[0], stacks[1],
                                      p_inv[0], p_inv[1])
            assert eng == g.state_legal_actions(street, st, cp, stacks[cp]), history
            assert g.calculate_current_pot(pot, history, street, p_inv[0], p_inv[1]) == st['pot']
            if not eng:
                if street == 3:
                    break
                p_inv = [p_inv[i] + st['c'][i] for i in (0, 1)]
                pot, street, history = st['pot'], street + 1, []
                st = g.init_node_state(street, pot)
                continue
            if street == 0 and st['num_br'] >= 3 and 'allin' in eng:
                jams += 1
            for a in eng:
                assert g._action_cost(a, street, history, pot, cp, p_inv[0], p_inv[1]) \
                    == g.state_action_cost(a, street, st, cp, stacks[cp])
            non_fold = [a for a in eng if a != 'fold']
            a = rng.choice(non_fold if non_fold and rng.random() > 0.1 else eng)
            cost = g.state_action_cost(a, street, st, cp, stacks[cp])
            assert 0 <= cost <= stacks[cp] + 1e-9
            st = g.advance_node_state(st, a, street, cp, stacks[cp], tuple(p_inv))
            history = history + [a]
            assert abs(sum(p_inv) + sum(st['c']) - st['pot']) < 1e-9   # chips conserved
    assert jams > 100, 'walks must exercise the cap jam'


# ------------------------------------------------------------------- training

def test_trainer_learns_the_jam_node():
    from src.cfr.blueprint_trainer import BlueprintTrainer
    random.seed(3)
    t = BlueprintTrainer(menu_mode='capped_pj')
    for i in range(400):
        t._run_iteration(i)
    jam_keys = [k for k in t.info_sets if re.match(r'pf_\d+_(ip|oop)_c?[smlx][sml]{2}a$', k)]
    cap_keys = [k for k in t.info_sets if re.match(r'pf_\d+_(ip|oop)_c?[smlx][sml]{2}$', k)]
    assert jam_keys, 'no info set faces the 5-bet jam'
    assert any('allin' in t.info_sets[k].legal_actions for k in cap_keys)


def test_trainer_rejects_unknown_menu():
    from src.cfr.blueprint_trainer import BlueprintTrainer
    with pytest.raises(ValueError):
        BlueprintTrainer(menu_mode='capped_jam')


# ------------------------------------------------------------------- mirrors + guards

def test_match_and_lbr_mirror_the_cap_jam():
    from src.evaluation.match import _legal_actions
    from src.evaluation.lbr import _cap_jam_open
    assert _legal_actions(0, 30, 3, 150, 90, 'capped_pj') == ['fold', 'call', 'allin']
    assert _legal_actions(0, 30, 3, 150, 90, 'capped') == ['fold', 'call']
    assert _legal_actions(1, 30, 3, 150, 90, 'capped_pj') == ['fold', 'call']   # postflop
    assert _legal_actions(0, 30, 3, 30, 90, 'capped_pj') == ['fold', 'call']    # can't raise
    assert _cap_jam_open(True, 0, 3, 30, 150)
    assert not _cap_jam_open(False, 0, 3, 30, 150)
    assert not _cap_jam_open(True, 1, 3, 30, 150)
    assert not _cap_jam_open(True, 0, 2, 30, 150)


def test_serving_refuses_capped_pj(tmp_path):
    from src.game.bot_strategy import BlueprintStrategy
    from src.storage.blueprint_db import BlueprintDB
    db = BlueprintDB(str(tmp_path / 'bp.db'))
    db.set_metadata('menu_mode', 'capped_pj')
    with pytest.raises(ValueError, match='menu_mode'):
        BlueprintStrategy(db)
    db.set_metadata('menu_mode', 'capped')
    BlueprintStrategy(db)                                # the served arm still loads
    db.close()


def test_rust_engine_matches_the_jam():
    core = pytest.importorskip('allin_core')
    e = core.Engine('capped_pj')
    st = e.init_node_state(0, 3.0)
    stack = [STARTING_STACK - 1.0, STARTING_STACK - 2.0]
    for i, a in enumerate(CAP):
        cp = e.acting_player(i, 0)
        cost = e.action_cost(a, 0, st, cp, stack[cp])
        st = e.advance(st, a, 0, cp, stack[cp], (0.0, 0.0))
        stack[cp] -= cost
    cp = e.acting_player(3, 0)
    assert e.legal_actions(0, st, cp, stack[cp]) == ['fold', 'call', 'allin']
    assert core.Engine('capped').legal_actions(0, st, cp, stack[cp]) == ['fold', 'call']
