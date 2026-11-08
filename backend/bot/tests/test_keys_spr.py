# backend/bot/tests/test_keys_spr.py
"""Phase 6 spr5 key format: spr_bucket + make_info_set_key(spr=...) + trainer stamp.

Run from backend/bot/:  python -m pytest tests/test_keys_spr.py -q
"""
import os
import random
import re
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr import keys  # noqa: E402
from src.cfr.keys import make_info_set_key, spr_bucket, SPR_EDGES  # noqa: E402
from src.cfr.poker_game import PokerGame, STARTING_STACK  # noqa: E402
from src.abstractions.sizing import postflop_menu_for  # noqa: E402


def test_starting_stack_constant_matches_engine():
    assert keys._KEY_STARTING_STACK == STARTING_STACK


def test_buckets_follow_pot_types():
    # limped pot (4) -> >40, 2BB open called (8) -> 12-40, a 3-bet pot (~20) -> 5-12,
    # a 4-bet pot (~45) -> 2-5, a huge pot (100) -> <2.
    assert spr_bucket('spr5', 1, 4.0) == 4
    assert spr_bucket('spr5', 1, 8.0) == 3
    assert spr_bucket('spr5', 2, 20.0) == 2
    assert spr_bucket('spr5', 2, 45.0) == 1
    assert spr_bucket('spr5', 1, 100.0) == 0


@pytest.mark.parametrize('edge', SPR_EDGES)
def test_edges_are_inclusive_lower_bounds(edge):
    pot = 200.0 / (edge + 0.5)            # SPR == edge exactly (up to float)
    spr = (200 - pot / 2.0) / pot
    want = sum(1 for e in SPR_EDGES if spr >= e)
    assert spr_bucket('spr5', 1, pot) == want


def test_no_bucket_where_none_belongs():
    assert spr_bucket('v2', 1, 10.0) is None
    assert spr_bucket('spr5', 0, 3.0) is None
    assert spr_bucket('spr5', 3, 10.0) is None
    for bad in (0.0, -1.0, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            spr_bucket('spr5', 1, bad)
    with pytest.raises(ValueError):
        spr_bucket('spr6', 1, 10.0)
    with pytest.raises(ValueError):
        spr_bucket('bogus', 0, 10.0)


def test_key_strings():
    assert make_info_set_key(1, 'ip', 'pf_160', 5, 'km', 3) == 'pf_9_5_ip_flop_r3_km'
    assert make_info_set_key(2, 'oop', 'pf_0', 0, '', 0) == 'pf_0_0_oop_turn_r0_'
    # spr=None keeps the v2 key exactly
    assert make_info_set_key(2, 'ip', 'pf_160', 5, 'm') == 'pf_9_5_ip_turn_m'
    with pytest.raises(ValueError):
        make_info_set_key(0, 'ip', 'pf_160', None, '', 1)
    with pytest.raises(ValueError):
        make_info_set_key(3, 'ip', 'pf_160', 5, 'k', 1)


@pytest.mark.parametrize('mode', ['control', 'capped'])
def test_street_start_spr_is_a_function_of_the_pot(mode):
    """The premise of spr_bucket: at every street start both stacks are equal, so
    effective stack / pot == (STARTING_STACK - pot/2) / pot. Random walks through
    the engine the trainer uses."""
    game = (PokerGame() if mode == 'control'
            else PokerGame(postflop_menu=postflop_menu_for(mode), voluntary_allin=False))
    rng = random.Random(7)
    starts = 0
    for _ in range(3000):
        street, history = 0, []
        p_inv = [0.0, 0.0]
        stacks = [STARTING_STACK - 1, STARTING_STACK - 2]
        st = game.init_node_state(0, 3.0)
        for _ in range(80):
            if street > 3 or game.is_terminal(history, street):
                break
            cp = game._acting_player(len(history), street)
            legal = game.state_legal_actions(street, st, cp, stacks[cp])
            if not legal:
                if street == 3:
                    break
                p_inv = [p_inv[0] + st['c'][0], p_inv[1] + st['c'][1]]
                stacks = [STARTING_STACK - p_inv[0], STARTING_STACK - p_inv[1]]
                street += 1
                history = []
                st = game.init_node_state(street, st['pot'])
                # Equal up to float rounding of the chip arithmetic (~1e-14).
                assert stacks[0] == pytest.approx(stacks[1], abs=1e-9), (stacks, history)
                if street in (1, 2) and min(stacks) > 0:
                    starts += 1
                    real = min(stacks) / st['pot']
                    assert real == pytest.approx((STARTING_STACK - st['pot'] / 2) / st['pot'],
                                                 abs=1e-9)
                    if all(abs(real - e) > 1e-9 for e in SPR_EDGES):
                        want = sum(1 for e in SPR_EDGES if real >= e)
                        assert spr_bucket('spr5', street, st['pot']) == want
                continue
            a = rng.choice(legal)
            cost = game.state_action_cost(a, street, st, cp, stacks[cp])
            st = game.advance_node_state(st, a, street, cp, stacks[cp], tuple(p_inv))
            stacks[cp] -= cost
            history.append(a)
    assert starts > 1000


def test_trainer_stamps_and_guards_key_format():
    from src.cfr.blueprint_trainer import BlueprintTrainer
    from src.storage.blueprint_db import BlueprintDB
    t = BlueprintTrainer(menu_mode='capped', key_format='spr5')
    for i in range(300):
        t._run_iteration(i)
    has_spr = re.compile(r'_(flop|turn)_r[0-4]_')
    flop_turn = [k for k in t.info_sets if '_flop_' in k or '_turn_' in k]
    assert flop_turn and all(has_spr.search(k) for k in flop_turn)
    others = [k for k in t.info_sets if k not in flop_turn]
    assert others and not any(re.search(r'_r[0-9]', k) for k in others)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'bp.db')
        db = BlueprintDB(p)
        t.checkpoint_to_db(db, 299)
        assert db.get_metadata('key_format') == 'spr5'
        db.close()
        db = BlueprintDB(p)
        with pytest.raises(ValueError, match='key-format'):
            BlueprintTrainer(menu_mode='capped').resume_from_db(db)
        assert BlueprintTrainer(menu_mode='capped', key_format='spr5').resume_from_db(db) == 300
        db.close()
    with pytest.raises(ValueError):
        BlueprintTrainer(key_format='spr4')


def test_v2_trainer_writes_no_key_format():
    from src.cfr.blueprint_trainer import BlueprintTrainer
    from src.storage.blueprint_db import BlueprintDB
    t = BlueprintTrainer(menu_mode='capped')
    for i in range(50):
        t._run_iteration(i)
    with tempfile.TemporaryDirectory() as d:
        db = BlueprintDB(os.path.join(d, 'bp.db'))
        t.checkpoint_to_db(db, 49)
        assert db.get_metadata('key_format') is None
        db.close()
