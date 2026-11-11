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
