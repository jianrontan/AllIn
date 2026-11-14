# backend/bot/tests/test_lbr_engine_mirror.py
"""LBR models the bot with its own lightweight game state, so its view of the bot's
options must mirror the engine the blueprint was trained on: at every node of random
walks through PokerGame's state machine, LBREvaluator._menu_legal must equal
state_legal_actions, and _bot_sizing must equal state_action_cost (to LBR's whole-chip
rounding). Nodes where a size lands within a chip of the stack are skipped: there
LBR's rounding can legitimately flip the all-in clamp.

Run: python -m pytest tests/test_lbr_engine_mirror.py -q
"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.abstractions.sizing import game_kwargs  # noqa: E402
from src.cfr.poker_game import PokerGame, STARTING_STACK  # noqa: E402
from src.evaluation.lbr import LBREvaluator  # noqa: E402


class _NullDB:
    def get_average_strategy(self, key):
        return None


def _walk_nodes(game, rng, hands):
    """Yield (street, state, actor, stack, legal) at every decision of random walks."""
    for _ in range(hands):
        prev = [0.0, 0.0]
        pot = 3.0
        for street in range(4):
            st = game.init_node_state(street, pot)
            for n in range(40):
                actor = (n + (1 if street > 0 else 0)) & 1
                stack = STARTING_STACK - prev[actor] - st['c'][actor]
                legal = game.state_legal_actions(street, st, actor, stack)
                if not legal:
                    break
                yield street, st, actor, stack, legal
                aggressive = [a for a in legal if a not in ('fold', 'check', 'call')]
                if aggressive and rng.random() < 0.55:
                    action = rng.choice(aggressive)
                else:
                    action = 'call' if 'call' in legal else 'check'
                st = game.advance_node_state(st, action, street, actor, stack, prev)
            prev = [prev[0] + st['c'][0], prev[1] + st['c'][1]]
            pot = st['pot']
            if st['allin_seen']:
                break


@pytest.mark.parametrize('menu_mode', ['capped', 'capped_pj', 'control'])
def test_lbr_menu_and_sizes_match_the_engine(menu_mode):
    game = PokerGame(**game_kwargs(menu_mode))
    lbr = LBREvaluator(_NullDB(), menu_mode=menu_mode)
    rng = random.Random(7)
    checked = legal_bad = size_bad = 0
    examples = []
    for street, st, actor, stack, legal in _walk_nodes(game, rng, 400):
        to_call = max(0.0, st['bet_to'] - st['c'][actor])
        committed = st['c'][actor]
        sized = [a for a in legal if a.startswith(('bet_', 'raise_'))]
        costs = {a: game.state_action_cost(a, street, st, actor, stack) for a in sized}
        # Every menu size's engine cost, legal or not, to spot near-stack boundaries.
        menu = (['bet_small', 'bet_medium', 'bet_large', 'bet_xlarge'] if street == 0
                and st['num_br'] == 0 else
                [f'raise_{s}' for s in ('small', 'medium', 'large')] if street == 0 else
                [f'{"raise" if to_call > 0 else "bet"}_{s}' for s in game.BET_MULTIPLIERS])
        near = any(abs(game.state_action_cost(a, street, st, actor, stack) - stack) < 1.0
                   for a in menu if not (street == 0 and st['num_br'] == 0 and a.startswith('raise')))
        if near:
            continue
        checked += 1
        got = lbr._menu_legal(street, st['pot'], to_call, committed, st['num_br'], stack)
        if sorted(got) != sorted(legal):
            legal_bad += 1
            if len(examples) < 5:
                examples.append(('legal', street, st['num_br'], got, legal))
        for a, cost in costs.items():
            add = lbr._bot_sizing(a.split('_')[1], street, st['pot'], to_call, committed,
                                  st['num_br'])
            if abs(add - cost) > 0.5 + 1e-9:
                size_bad += 1
                if len(examples) < 5:
                    examples.append(('size', street, a, add, cost))
    assert checked > 500, checked
    assert legal_bad == 0 and size_bad == 0, (legal_bad, size_bad, examples)


@pytest.mark.parametrize('to_call,pot', [(0, 10), (1, 3), (2, 10), (7, 30)])
def test_lbr_offgrid_candidates_obey_minimum_increment(to_call, pot):
    import numpy as np
    from types import SimpleNamespace
    ev = LBREvaluator.__new__(LBREvaluator)
    ev.bet_fractions = (.01, .05, .1, .25, .5, 1., 2.5)
    ev.equity_vs_range = lambda *args: .5
    ev._menu_legal = lambda *args: []
    seen = []
    def aggressive(*args):
        add = args[-2]
        seen.append(add)
        return (1., 's', add, True, [('s', 1.)])
    ev._value_aggressive = aggressive
    ev._lbr_decide(0, ('HA', 'DK'), [], 0 if to_call == 1 else 1,
                   [pot / 2, pot / 2], [100, 100], [0, to_call], to_call,
                   0, '', SimpleNamespace(hands=[], w=np.array([])), 1)
    assert seen
    assert all(add >= to_call + max(to_call, 2) for add in seen)


def test_translation_grid_uses_only_live_sized_brackets():
    ev = LBREvaluator(_NullDB(), menu_mode='control')
    grid = ev._translation_grid(1, 4, 0, 0, 0, 198)
    assert 's' not in dict(grid)  # 0.33 * 4 < the two-chip minimum
    assert 'm' in dict(grid)
    assert 'a' not in dict(grid)  # sub-stack bets never snap to a shove
    assert 'a' not in dict(ev._translation_grid(0, 3, 1, 1, 0, 199))
    assert ev._translation_grid(1, 30, 0, 0, 0, 2) == []


@pytest.mark.parametrize('stored', [None, {}, {'bet_unavailable': 1.0}])
def test_single_bracket_price_execution_and_belief_match_live(stored, monkeypatch):
    import numpy as np
    from src.game.game_session import GameSession
    from src.game.bot_strategy import BlueprintStrategy
    from src.cfr.poker_game import make_custom_action
    from src.evaluation.lbr import BotRange
    class DB:
        def get_average_strategy(self, key):
            return stored
    session=GameSession.new('review','offline')
    session.apply_action(make_custom_action(False,20))
    legal=session.legal_actions()
    key=session.info_set_key(1)
    live=BlueprintStrategy(DB()).explain(key,legal,session.bot_public_state())
    assert session.data.get('pending_translation') is None
    ev=LBREvaluator(DB(),seed=1)
    ev.equity_vs_range=lambda *args: .5
    belief=BotRange(session.data['p0_cards'],ev.cards)
    price=ev._value_aggressive(0,session.data['p0_cards'],[],0,[1,2],[199,198],
                              [1,2],1,0,'',belief,1,19,False)
    assert price[4]==[('x',1.)]
    # Calling creates a 40-chip pot and costs the opener 20, so checkdown EV=0;
    # folding wins two chips. The correct .5 fold probability prices this at 1.
    assert price[0]==pytest.approx(1.)
    pending=('x','',price[4])
    base=ev.restricted_probs(key,tuple(legal))
    actual=ev._blended_probs(pending,0,'oop',session.data['p1_cards'],[],
                              [20,2],[20,2],legal,base)
    np.testing.assert_allclose(actual,[live.get(a,0.) for a in legal])
    rows=[]
    belief.observe=lambda lookup,action,street,pos,pattern,legal,vis,**kw: rows.append(lookup(key,tuple(legal)))
    monkeypatch.setattr(ev.rng,'choices',lambda *a,**kw: ['call'])
    ev._pending=pending
    ev._bot_act(1,session.data['p1_cards'],[],0,[20,2],[180,198],[20,2],18,1,'x',belief,0)
    np.testing.assert_allclose(rows[0],[live.get(a,0.) for a in legal])


def test_real_multi_bracket_missing_response_still_folds():
    import numpy as np
    from src.game.game_session import GameSession
    from src.game.bot_strategy import BlueprintStrategy
    from src.cfr.poker_game import make_custom_action
    session=GameSession.new('review','offline')
    session.apply_action(make_custom_action(False,6))
    state=session.bot_public_state()
    assert len(state['translation'])==2
    legal=session.legal_actions()
    key=session.info_set_key(1)
    live=BlueprintStrategy(_NullDB()).explain(key,legal,state)
    ev=LBREvaluator(_NullDB(),seed=1)
    translated=[(k[-1],w) for k,w in state['translation']]
    np.testing.assert_allclose(ev._response_probs(key,tuple(legal),translated),
                               [live.get(a,0.) for a in legal])
    assert live['fold']==1.
