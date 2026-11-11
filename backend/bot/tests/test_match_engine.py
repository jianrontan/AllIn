# backend/bot/tests/test_match_engine.py
"""The head-to-head match plays each side on its own engine: every action a player
is offered must be exactly PokerGame.get_legal_actions for that history (sized
bets that exceed the stack become one all-in, sub-minimum sizes are dropped, the
capped_pj cap jam exists only for that arm), and chips are conserved.

Run: python -m pytest tests/test_match_engine.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.poker_game import PokerGame, STARTING_STACK  # noqa: E402
from src.abstractions.sizing import game_kwargs  # noqa: E402
from src.evaluation.match import HeadToHeadMatch  # noqa: E402


class _EveryAction:
    """A strategy with equal mass on every action name, so play is aggressive and
    reaches stack clamps, cap jams and short-stack nodes."""
    _ALL = ['fold', 'check', 'call', 'allin'] + [
        f'{k}_{s}' for k in ('bet', 'raise')
        for s in ('small', 'medium', 'large', 'xlarge', 'overbet', 'overbet2')]

    def get_average_strategy(self, key):
        return {a: 1.0 for a in self._ALL}


def _replay_engine_legal(rec, menu_modes):
    """Re-derive each recorded node's legal actions with the history-based engine."""
    games = {seat: PokerGame(**game_kwargs(mm)) for seat, mm in menu_modes.items()}
    history, start_pot, prev = [], None, None
    for ev in rec['events']:
        if ev['type'] == 'street_start':
            history = []
            start_pot = ev['pot']
            prev = [0.0, 0.0] if ev['street'] == 0 else list(ev['invested'])
            continue
        seat, street = ev['seat'], ev['street']
        stacks = [STARTING_STACK - x for x in ev['invested']]
        want = games[seat].get_legal_actions(street, history, start_pot, seat,
                                             stacks[0], stacks[1], prev[0], prev[1])
        assert ev['legal'] == want, (ev['pattern'], ev['legal'], want)
        history.append(ev['action'])


@pytest.mark.parametrize('mm_a,mm_b', [('capped', 'capped'), ('capped_pj', 'capped_pj'),
                                       ('capped_pj', 'capped'), ('control', 'control')])
def test_match_offers_exactly_the_engine_actions(mm_a, mm_b):
    m = HeadToHeadMatch(_EveryAction(), _EveryAction(), seed=11,
                        menu_mode_a=mm_a, menu_mode_b=mm_b)
    rec = []
    m.evaluate(num_hands=300, record=rec)
    allins = cap_jams = 0
    for r in rec:
        modes = {r['seat_of_A']: mm_a, 1 - r['seat_of_A']: mm_b}
        _replay_engine_legal(r, modes)
        assert all(0 < x <= STARTING_STACK + 1e-6 for x in r['invested'])
        a = r['invested'][r['seat_of_A']]
        b = r['invested'][1 - r['seat_of_A']]
        assert -a - 1e-9 <= r['result'] <= b + 1e-9          # A can't win/lose more than staked
        for ev in r['events']:
            if ev['type'] == 'action' and ev['action'] == 'allin':
                allins += 1
                cap_jams += ev['street'] == 0 and ev['pattern'].count('a') == 0 and \
                    sum(c in 'smlxo2' for c in ev['pattern']) == 3
    assert allins > 0
    # The cap jam is offered only to a capped_pj player.
    if 'capped_pj' not in (mm_a, mm_b):
        assert cap_jams == 0


def test_capped_pj_player_can_cap_jam_against_capped():
    m = HeadToHeadMatch(_EveryAction(), _EveryAction(), seed=5,
                        menu_mode_a='capped_pj', menu_mode_b='capped')
    rec = []
    m.evaluate(num_hands=400, record=rec)
    seen = set()
    for r in rec:
        for ev in r['events']:
            if ev['type'] == 'action' and ev['street'] == 0 and 'allin' in ev['legal'] \
                    and sum(c in 'smlx' for c in ev['pattern']) == 3:
                seen.add(ev['seat'] == r['seat_of_A'])
    assert seen == {True}             # only A (capped_pj) is ever offered the cap jam
