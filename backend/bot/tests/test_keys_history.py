"""spr5a history keys must agree across training, live sessions and recap replay."""
import sys
import re
from pathlib import Path
from collections import defaultdict
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cfr.keys import (last_aggressor, history_tags, make_info_set_key, spr_bucket)
from src.game.game_session import GameSession
from src.exploitation.replay import replay_hand
from src.exploitation.opponent_model import _levels


def test_history_context_and_explicit_validation():
    assert last_aggressor('ck', 0) == 'n'
    assert last_aggressor('smc', 0) == 'o'
    assert last_aggressor('ksc', 1) == 'i'
    assert history_tags('spr5a', 1, ['smc', '', '', '']) == 'o'
    assert history_tags('spr5a', 2, ['smc', 'ksc', '', '']) == 'io'
    assert history_tags('spr5a', 2, ['ck', 'kk', '', '']) == 'nn'
    assert spr_bucket('spr5a', 3, 20) == 2
    args = (2, 'oop', 'pf_100', 5, 'km')
    k = make_info_set_key(*args, spr=2, key_format='spr5a', tags='io')
    assert k.endswith('_turn_r2_tio_km')
    assert _levels(k)[1].endswith('|m')
    for tags in (None, '', 'i', 'xq'):
        with pytest.raises(ValueError):
            make_info_set_key(*args, spr=2, key_format='spr5a', tags=tags)
    with pytest.raises(ValueError):
        make_info_set_key(*args, key_format='spr5a', tags='io')
    assert make_info_set_key(3, 'ip', 'pf_100', 5, '', spr=2,
                             key_format='spr5a', tags='').endswith('_river_r2_')


def test_live_context_survives_roundtrip_and_matches_replay():
    s = GameSession.new('history', 'p', key_format='spr5a')
    # Limped pot, checked flop: both completed streets have no aggressor.
    for action in ('call', 'check', 'check', 'check', 'check', 'check'):
        s.apply_action(action)
        if s.data['street'] in (1, 2, 3):
            suffix = {1: '_flop_r4_tn_', 2: '_turn_r4_tnn_', 3: '_river_r4_'}[s.data['street']]
            assert suffix in s.info_set_key(s.current_player())
    assert s.bot_public_state()['tags'] == ''
    import copy
    data = copy.deepcopy(s.to_dict())
    data.pop('street_patterns')                         # existing in-flight session
    restored = GameSession.from_dict(data, key_format='spr5a')
    assert restored.info_set_key(restored.current_player()) == s.info_set_key(s.current_player())
    # Replay observed human decisions uses the same street-constant tags.
    rec = {'humanSeat': 0, 'humanHole': ['Ah', 'Kd'], 'community': ['2h','3d','4c','5s','6h'],
           'actionLog': s.data['action_log']}
    class Cards:
        def preflop_bucket(self, h): return 'pf_100'
        def get_bucket(self, h, b): return 5
    sink = defaultdict(lambda: defaultdict(int))
    replay_hand(rec, Cards(), sink, key_format='spr5a')
    assert any('_flop_r4_tn_' in k for k in sink)
    assert any('_turn_r4_tnn_' in k for k in sink)


def test_eval_consumers_keep_the_new_context(tmp_path):
    from tests.test_keys_spr import _Recorder, _trained_db
    from src.evaluation.match import HeadToHeadMatch
    from src.evaluation.cross_match import CrossMatch, NEW_SIZING
    from src.evaluation.lbr import LBREvaluator
    a = _Recorder(_trained_db(str(tmp_path), 'spr5a', iters=60))
    b = _Recorder(_trained_db(str(tmp_path), 'v2', iters=60))
    for match in (HeadToHeadMatch(a,b,seed=6,menu_mode_a='capped',menu_mode_b='capped'),
                  CrossMatch(a,NEW_SIZING,b,NEW_SIZING,seed=6)):
        match.evaluate(num_hands=20)
    LBREvaluator(a,seed=3,flop_runout_samples=2).evaluate(num_hands=4,progress_every=0)
    post = [k for k in a.keys if '_flop_' in k or '_turn_' in k]
    assert post
    assert all(re.search(r'_r[0-4]_t[nio]{1,2}_', k) for k in post)
    rivers = [k for k in a.keys if '_river_' in k]
    assert rivers and all('_river_r' in k for k in rivers)
    assert not any(re.search(r'_r[0-4]_', k) for k in b.keys)


def test_custom_actions_keep_the_historical_aggressor():
    from src.cfr.keys import logged_patterns
    # An unmapped custom raise still occupies its actor's turn in a legacy log.
    rec = {'humanSeat': 0, 'humanHole': ['Ah','Kd'], 'community': ['2h','3d','4c'],
           'actionLog': [
               {'street':'preflop','player':0,'action':'call','chips':1},
               {'street':'preflop','player':1,'action':'raise_custom_6','chips':4},
               {'street':'preflop','player':0,'action':'call','chips':4},
               {'street':'flop','player':1,'action':'check','chips':0},
               {'street':'flop','player':0,'action':'check','chips':0}]}
    assert history_tags('spr5a',1,logged_patterns(rec['actionLog'])) == 'o'
    class Cards:
        def preflop_bucket(self,h):return 'pf_100'
        def get_bucket(self,h,b):return 5
    sink = defaultdict(lambda: defaultdict(int))
    replay_hand(rec,Cards(),sink,key_format='spr5a')
    assert any('_flop_r3_to_k' in k for k in sink)
    # Uncapped live postflop raises do not alter the street-constant tag.
    session = GameSession.new('uncapped','p',key_format='spr5a')
    session.apply_action('call');session.apply_action('check')
    for a in ('bet_custom_2','raise_custom_4','raise_custom_6','raise_custom_8'):
        session.apply_action(a)
        assert '_flop_r4_tn_' in session.info_set_key(session.current_player())


def test_live_range_models_receive_the_same_history_context():
    import numpy as np
    seen = []
    def model(key,legal):
        seen.append(key)
        return np.full(len(legal),1/len(legal))
    session = GameSession.new('tracked-history','p',key_format='spr5a',
                              menu_mode='capped_pj',strategy_fn=model)
    for a in ('bet_small','raise_small','raise_small','call','check','check','check','check','check'):
        session.apply_action(a)
    flop = [k for k in seen if '_flop_' in k]
    turn = [k for k in seen if '_turn_' in k]
    river = [k for k in seen if '_river_' in k]
    assert flop and all(re.search(r'_flop_r[0-4]_ti_',k) for k in flop)
    assert turn and all(re.search(r'_turn_r[0-4]_tni_',k) for k in turn)
    assert river and all(re.search(r'_river_r[0-4]_',k) for k in river)


def test_river_projection_infers_spr5a_from_its_blueprint():
    import numpy as np
    from src.subgame.blueprint_projection import blueprint_strategy_on_tree
    from src.subgame.river_tree import build_river_tree
    seen = []
    class DB:
        def get_metadata(self,k,default=None): return 'spr5a' if k == 'key_format' else default
        def get_average_strategy(self,k):seen.append(k);return None
    ba = {'H':1, 'pf':['pf_100'], 'strg':{3:[5]},
          'groups':{3:[(np.array([True]),0)]}}
    tree = build_river_tree(380,(10,10))
    out = blueprint_strategy_on_tree(tree,ba,DB().get_average_strategy)
    assert out and seen and all('_river_r0_' in k for k in seen)
