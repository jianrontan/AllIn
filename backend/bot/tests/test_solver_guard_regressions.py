# backend/bot/tests/test_solver_guard_regressions.py
"""Regressions from the 2026-11-11 river-solver review:

1. On the TURN, the bot's guards must read the LIVE opponent belief. The shelved turn
   solver's turn-entry snapshot used to overwrite `opp_range`, so a turn jam was judged
   against a belief that had not seen it, at its pre-jam confidence.
2. On a high-SPR RIVER the solver skips the spot; an untrained faced raise (a 4th river
   aggression, legal live) must still get the deep-raise guard, not the blueprint's
   fold/call coin flip (it folded the nuts ~46% of the time).

Needs the served v2 blueprint (analysis/blueprints/blueprint_final_v2.db); skipped
without it. Run: python -m pytest tests/test_solver_guard_regressions.py -q
"""
import collections
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.abstractions.card_abstractions import CardAbstraction  # noqa: E402
from src.abstractions.sizing import db_menu_mode  # noqa: E402
from src.cfr.keys import make_info_set_key  # noqa: E402
from src.game.game_session import GameSession  # noqa: E402
from src.game.range_tracker import RangeTracker  # noqa: E402
from src.storage.blueprint_db import BlueprintDB  # noqa: E402
from src.subgame.river_subgame_solver import RiverSubgameSolver  # noqa: E402

_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'analysis', 'blueprints', 'blueprint_final_v2.db')


@pytest.fixture(scope='module')
def bot():
    if not os.path.exists(_DB):
        pytest.skip(f'needs the v2 blueprint at {_DB}')
    db = BlueprintDB(_DB, read_only=True)
    yield RiverSubgameSolver(db, max_iters=50, check_every=10, time_budget=5.0,
                             safe_gadget=True, gadget_anchor='auto', purify_threshold=0.01)
    db.close()


def test_turn_guards_read_the_live_belief(bot):
    s = GameSession.new('t', 'p', strategy_fn=bot.range_model_fn(),
                        menu_mode=db_menu_mode(bot.db),
                        hero_strategy_fn=bot.hero_range_model_fn())
    d = s.data
    for a in ('call', 'check', 'check', 'check'):    # limp, check; flop checks through
        s.apply_action(a)
    assert d['street'] == 2
    s.apply_action('check')                          # the bot (BB) checks the turn
    legal = s.legal_actions()
    s.apply_action('allin' if 'allin' in legal else f"bet_custom_{d['p0_stack']:g}")
    assert d.get('turn_entry_opp') is not None, 'no turn-entry snapshot: test is vacuous'
    ps = s.bot_public_state()
    live = RangeTracker.from_dict(d['opp_range'], s.cards)
    snap = RangeTracker.from_dict(d['turn_entry_opp'], s.cards)
    assert snap.confidence != pytest.approx(live.confidence), 'jam did not move the belief'
    assert ps['opp_range'].confidence == pytest.approx(live.confidence)
    assert np.allclose(ps['opp_range'].w, live.w)
    assert ps['turnEntryOppRange'].confidence == pytest.approx(snap.confidence)


def test_high_spr_river_untrained_raise_does_not_fold_the_nuts(bot):
    cards = CardAbstraction()
    board = ['HA', 'DK', 'CQ', 'SJ', 'H9']
    hole = ['HT', 'S2']                              # broadway: the nuts on AKQJ9
    key = make_info_set_key(3, 'oop', cards.get_bucket(hole, None),
                            cards.get_bucket(hole, board), 'ssss')
    assert not bot.db.get_average_strategy(key), 'expected an untrained 4th-aggression key'
    opp = RangeTracker(tuple(hole), cards)
    opp.reveal(board)
    hero = RangeTracker((), cards)
    hero.reveal(board)
    legal = ['fold', 'call', 'raise_small', 'raise_medium', 'raise_large', 'allin']
    ps = {'street': 'river', 'pot': 80, 'community': board, 'history': [],
          'p0_stack': 158, 'p1_stack': 174, 'seat': 1, 'hole_cards': hole,
          'to_call': 16.0, 'opp_range': opp, 'exploit': None,
          'botSeat': 1, 'riverEntryPot': 20.0, 'riverEntryStacks': (190.0, 190.0),
          'hero_range': hero,
          'riverPath': [('bet', 4.0), ('raise', 12.0), ('raise', 20.0), ('raise', 24.0)],
          'turnDeviated': False}
    assert bot._solver_inputs(ps) is None, 'expected the solver to skip this high-SPR spot'
    acts = collections.Counter(bot.decide(key, legal, ps) for _ in range(60))
    assert acts['fold'] == 0, acts
