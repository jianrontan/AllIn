"""v3 step V3.11: the pooled read on how strong human jams are (exploitation.jam_read),
its use in the facing-jam guard, and the logged-spot evaluator's exact scoring."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'scripts'))

from src.abstractions.postflop_features import rank7  # noqa: E402
from src.exploitation import jam_read as jr  # noqa: E402
from src.exploitation.opponent_model import blueprint_model_stamp  # noqa: E402
from src.subgame.river_subgame_solver import RiverSubgameSolver  # noqa: E402

BOARD = ['HK', 'SQ', 'D7']


def made_strength(h, board):
    """A simple strength for tests: the made hand on the board (higher = stronger)."""
    return -float(rank7(list(h) + list(board)))


def _read(top, **params):
    """A read whose jams sit in the top bin (`top`=True) or the bottom bin."""
    c = np.zeros(jr.BINS)
    c[-1 if top else 0] = 200.0
    return jr.JamRead({'all': c, 'flop': c, 'flop|4bp': c}, stamp={}, **params)


# ------------------------------------------------------------------ pieces

def test_pot_type_and_context():
    assert jr.pot_type(['bet_small', 'call']) == 'srp'
    assert jr.pot_type(['call', 'check']) == 'srp'
    assert jr.pot_type(['bet_small', 'raise_small', 'call']) == '3bp'
    assert jr.pot_type(['bet_small', 'raise_small', 'raise_medium', 'allin']) == '4bp'
    assert jr.context_keys('turn', '3bp', 2) == ['all', 'turn', 'turn|3bp', 'turn|3bp|r2']


def test_percentiles_split_ties():
    u = jr.percentiles([(None, 1.0), (None, 1.0), (None, 2.0), (None, 3.0)])
    assert list(u) == [0.25, 0.25, 0.625, 0.875]
    assert jr.u_bin(1.0) == jr.BINS - 1 and jr.u_bin(0.0) == 0


def test_posterior_backs_off_to_the_coarser_rung():
    c = np.zeros(jr.BINS)
    c[-1] = 100.0
    read = jr.JamRead({'all': c}, stamp={}, alpha=10.0)
    conc, n = read.posterior(jr.context_keys('flop', '4bp', 1))
    mean = conc / conc.sum()
    assert mean[-1] > 0.85 and n == 0                # empty cells inherit 'all'
    assert conc.sum() == pytest.approx(10.0)         # ... with only alpha's confidence


def test_observations_from_a_recap():
    rec = {'playerId': 'p', 'humanSeat': 1, 'humanHole': ['Ks', 'Kc'],
           'botHole': ['Ah', 'Ad'], 'community': ['Kh', 'Qs', '7d', '2c', '3c'],
           'actionLog': [
               {'player': 0, 'street': 'preflop', 'action': 'bet_small', 'chips': 3},
               {'player': 1, 'street': 'preflop', 'action': 'raise_small', 'chips': 6},
               {'player': 0, 'street': 'preflop', 'action': 'raise_medium', 'chips': 18},
               {'player': 1, 'street': 'preflop', 'action': 'call', 'chips': 13},
               {'player': 1, 'street': 'flop', 'action': 'allin', 'chips': 176},
               {'player': 0, 'street': 'flop', 'action': 'call', 'chips': 176}]}
    obs = jr.JamObservations.from_recaps([rec], made_strength)
    assert len(obs.rows) == 1
    r = obs.rows[0]
    # A set of kings is the top of the board; the pot type is a 4-bet pot (3 raises).
    assert r['b'] == jr.BINS - 1 and r['ctx'][2] == 'flop|4bp'
    assert r['ctx'][3] == 'flop|4bp|r1'               # flop pot 43, 178 behind: SPR 4.1
    assert obs.excluding('p').rows == []


# ------------------------------------------------------------------ deciding

DRAW = ['HJ', 'DT']                                  # open-ended on K Q 7: folds or calls


def test_decide_folds_to_strong_jams_and_calls_weak_ones():
    hole = DRAW
    # pot 40, bot calls 80 to win 200: needs 0.4 (final pot 200, cost 80).
    act, dbg = _read(top=True).decide(hole, BOARD, 'flop', '4bp', 1, 200.0, 80.0,
                                      made_strength)
    assert act == 'fold', dbg
    act, dbg = _read(top=False).decide(hole, BOARD, 'flop', '4bp', 1, 200.0, 80.0,
                                       made_strength)
    assert act == 'call', dbg


def test_decide_defers_when_the_read_is_unsure():
    c = np.zeros(jr.BINS)
    c[-1] = c[0] = 1.0                                # two jams: one nut, one air
    read = jr.JamRead({'all': c}, stamp={}, alpha=1.0)
    act, dbg = read.decide(DRAW, BOARD, 'flop', '4bp', 1, 200.0, 80.0, made_strength)
    assert act is None and 0.05 < dbg['jamReadPPos'] < 0.95, dbg


def test_decide_mixes_in_the_tracker_by_the_data_weight():
    read = _read(top=True, p_max=0.5, s=0.0)          # w = 0.5
    act, dbg = read.decide(DRAW, BOARD, 'flop', '4bp', 1, 200.0, 80.0, made_strength,
                           tracker_eq=1.0)
    assert dbg['jamReadW'] == 0.5 and act == 'call', dbg   # the tracker outweighs the read


def test_read_round_trip_and_stamp(tmp_path):
    read = _read(top=True)
    read.stamp = blueprint_model_stamp(None)
    p = tmp_path / 'jam_read.json'
    p.write_text(json.dumps(read.to_json()))
    back = jr.JamRead.load(str(p), None)
    assert back.n_jams() == 200 and np.allclose(back.counts['flop'], read.counts['flop'])
    m = read.to_json()
    m['stamp'] = dict(m['stamp'], key_format='spr5')
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='stamp mismatch'):
        jr.JamRead.load(str(p), None)


# ------------------------------------------------------------------ the guard

class _Tracker:
    def __init__(self, eq, confidence):
        self._eq, self.confidence = eq, confidence

    def hero_equity(self, hole, board, n_runouts=None, rng=None):
        return self._eq


def _ps(tracker, hole=DRAW):
    return {'street': 'flop', 'community': list(BOARD), 'hole_cards': list(hole),
            'seat': 0, 'p0_stack': 80.0, 'p1_stack': 0.0, 'to_call': 80.0, 'pot': 120.0,
            'street_pot': 40.0, 'preflop_actions': ['bet_small', 'raise_small', 'raise_medium',
                                                    'call'],
            'opp_range': tracker}


def test_guard_uses_the_read_on_a_collapsed_tracker():
    """BUG-027's shape: the tracker's confidence collapsed, so without the read the guard
    defers (to the blueprint); with it, the read decides."""
    s = RiverSubgameSolver(None, guard_confidence=0.2, guard_margin=1.0)
    s._combo_strength = made_strength
    collapsed = _Tracker(0.9, confidence=0.04)
    assert s._facing_allin_guard(['fold', 'call'], _ps(collapsed)) is None
    s.jam_read = _read(top=True, p_max=1.0)
    assert s._facing_allin_guard(['fold', 'call'], _ps(collapsed)) == 'fold'
    assert s.stats['jam_read'] == 1
    assert s._tls.allin_guard_dbg['jamRead'] == 'fold'


def test_guard_falls_back_when_the_read_is_unsure():
    s = RiverSubgameSolver(None, guard_confidence=0.2, guard_margin=1.0)
    s._combo_strength = made_strength
    c = np.zeros(jr.BINS)
    c[-1] = c[0] = 1.0
    s.jam_read = jr.JamRead({'all': c}, stamp={}, alpha=1.0, p_max=1.0)
    trusted = _Tracker(0.9, confidence=1.0)
    assert s._facing_allin_guard(['fold', 'call'], _ps(trusted)) == 'call'   # tracker path
    assert s.stats['jam_read'] == 0


def test_guard_ignores_the_read_preflop_and_on_errors():
    s = RiverSubgameSolver(None, guard_confidence=0.2, guard_margin=1.0)
    s.jam_read = _read(top=True, p_max=1.0)
    ps = dict(_ps(_Tracker(0.9, confidence=1.0)), street='preflop', community=[])
    assert s._facing_allin_guard(['fold', 'call'], ps) == 'call'   # preflop: tracker path
    s._combo_strength = lambda h, b: 1 / 0                          # a broken read -> usual path
    assert s._facing_allin_guard(['fold', 'call'], _ps(_Tracker(0.9, 1.0))) == 'call'


# ------------------------------------------------------------------ the evaluator

def test_evaluator_spots_and_exact_equity():
    import eval_facing_jams as ev
    rec = {'humanSeat': 1, 'actionLog': [
        {'player': 0, 'street': 'preflop', 'action': 'bet_small', 'chips': 3},
        {'player': 1, 'street': 'preflop', 'action': 'call', 'chips': 2},
        {'player': 1, 'street': 'flop', 'action': 'allin', 'chips': 196},
        {'player': 0, 'street': 'flop', 'action': 'fold', 'chips': 0}]}
    (sp,) = ev.facing_jam_spots(rec)
    # Bot (seat 0) has 4 in; pot 8 + jam 196 = 204; calling costs 196 to a 400 pot.
    assert sp['index'] == 2 and sp['cost'] == 196 and sp['final_pot'] == 400
    assert not sp['logged_call']
    river = ['HK', 'SQ', 'D7', 'C2', 'S3']
    assert ev.exact_equity(['HA', 'DA'], ['CK', 'DK'], river, seed='x') == 0.0
    aa_kk = ev.exact_equity(['HA', 'DA'], ['CK', 'DK'], [], seed='x', n_mc=4000)
    assert 0.78 < aa_kk < 0.86
