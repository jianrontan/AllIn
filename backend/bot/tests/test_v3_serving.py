"""v3 step V3.6: the live serving path for spr5 / capped_pj / named-abstraction
blueprints. GameSession builds keys in the blueprint's format and buckets with its
abstraction; the preflop cap jam is played as trained; the bot, the API and
auto-resolution accept stamped v3 blueprints; v2 behaviour is unchanged.

The last section replays the BUG-027 hand through GameSession on the real v3
snapshots (A300 = v2 cards, C300 = v3 cards) and is skipped when they are absent."""
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.abstractions import abstraction_set as A  # noqa: E402
from src.cfr.keys import spr_bucket  # noqa: E402
from src.cfr.poker_game import make_custom_action, _custom_total  # noqa: E402
from src.game.game_session import GameSession, session_kwargs  # noqa: E402
from src.game.range_tracker import RangeTracker  # noqa: E402
from src.storage.blueprint_db import BlueprintDB  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RS = os.path.join(HERE, 'analysis', 'blueprints_rs')
SNAPS = {
    'A300': os.path.join(RS, 'blueprint_par_cappedpj_spr5_rs_20261110_004432_c58d21_snaps',
                         'snap_300000000.db'),
    'C300': os.path.join(RS, 'blueprint_par_cappedpj_spr5_v3_rs_20261110_013712_8d0598_snaps',
                         'snap_300000000.db'),
}


def _uniform_fn(seen=None):
    """A model strategy_fn that records the keys it is asked for."""
    def fn(key, legal):
        if seen is not None:
            seen.append(key)
        return np.ones(len(legal)) / len(legal)
    return fn


def _rig(session, p0, p1, board):
    """Fix the deal (and reset both trackers to it), as the API's explorer does."""
    d = session.data
    d['p0_cards'], d['p1_cards'], d['community'] = list(p0), list(p1), list(board)
    if session.strategy_fn is not None:
        bot = d['p1_cards'] if d['human_seat'] == 0 else d['p0_cards']
        d['opp_range'] = RangeTracker(bot, session.cards).to_dict()
        d['bot_range'] = RangeTracker((), session.cards).to_dict()


# ------------------------------------------------------------------ spr5 keys

def test_spr5_session_keys_carry_the_street_start_spr():
    seen = []
    s = GameSession.new('t', 'p', strategy_fn=_uniform_fn(seen), menu_mode='capped_pj',
                        key_format='spr5')
    _rig(s, ['HA', 'DA'], ['HK', 'DK'], ['S2', 'C7', 'D9', 'HT', 'CJ'])
    d = s.data
    d['human_seat'] = 0
    s.apply_action('bet_large')                     # SB opens 3.5BB
    s.apply_action('call')                          # BB calls -> flop, pot 14
    assert d['street'] == 1 and d['starting_pot'] == 14.0
    want = spr_bucket('spr5', 1, 14.0)              # (200-7)/14 = 13.8 -> bucket 3
    assert want == 3
    key = s.info_set_key(1)
    assert f'_flop_r{want}_' in key and key.endswith('_r3_')
    assert s.bot_public_state()['spr'] == want
    seen.clear()
    s.apply_action('check')                         # BB (the bot) checks: bot tracker observes
    assert seen and all('_flop_r3_' in k for k in seen)
    # Preflop and river keys carry no SPR token.
    assert '_r' not in GameSession.new('t', 'p', key_format='spr5').info_set_key(0)


def test_v2_session_keys_are_unchanged():
    s = GameSession.new('t', 'p', menu_mode='capped')
    _rig(s, ['HA', 'DA'], ['HK', 'DK'], ['S2', 'C7', 'D9', 'HT', 'CJ'])
    s.apply_action('bet_large')
    s.apply_action('call')
    assert s.info_set_key(1).endswith('_oop_flop_')
    assert s.bot_public_state()['spr'] is None


def test_unknown_key_format_is_refused():
    with pytest.raises(ValueError):
        GameSession.new('t', 'p', key_format='v9')


def test_sessions_share_one_card_abstraction_per_directory():
    a = GameSession.new('a', 'p')
    b = GameSession.new('b', 'p', abstraction_dir=A.ROOT)
    assert a.cards is b.cards


# --------------------------------------------------------------- preflop cap jam

def _to_cap(s):
    """Open, 3-bet, 4-bet: the 4-bet is the 3rd aggression, the trained cap."""
    for a in ('bet_small', 'raise_small', 'raise_small'):
        s.apply_action(a)


def test_bot_at_the_cap_plays_the_trained_node():
    s = GameSession.new('t', 'p', menu_mode='capped_pj')
    s.data['human_seat'] = 0                        # the bot is the BB, facing the 4-bet
    _to_cap(s)
    assert not s.is_human_turn()
    assert s.legal_actions() == ['fold', 'call', 'allin']


def test_a_human_5bet_of_any_size_is_the_trained_jam():
    for five in ('raise_small', 'allin', None):
        seen = []
        s = GameSession.new('t', 'p', strategy_fn=_uniform_fn(seen), menu_mode='capped_pj',
                            key_format='spr5')
        s.data['human_seat'] = 1                    # the human is the BB, facing the 4-bet
        _to_cap(s)
        assert s.is_human_turn()
        legal = s.legal_actions()
        assert 'allin' in legal and 'raise_small' in legal   # live play keeps 5-bet sizes
        if five is None:
            lo, hi = s.custom_bounds()
            five = make_custom_action(True, (lo + hi) / 2)
        seen.clear()
        s.apply_action(five)
        d = s.data
        assert d['bet_pattern'].endswith('a'), (five, d['bet_pattern'])
        assert d['action_log'][-1]['action'] == 'allin'
        assert seen and all(k.endswith('_oop_sss') for k in seen)   # the cap node
        # The bot now answers from the trained facing-jam key.
        assert s.info_set_key(0).endswith('_ip_sssa')


def test_capped_v2_menu_has_no_cap_node_mapping():
    s = GameSession.new('t', 'p', menu_mode='capped')
    s.data['human_seat'] = 1
    _to_cap(s)
    s.apply_action('raise_small')
    assert s.data['bet_pattern'].endswith('s')


# --------------------------------------------------------------- replay (refits)

def test_replay_builds_spr5_keys_from_the_logged_chips():
    from src.exploitation.replay import replay_hand
    from src.abstractions.card_abstractions import shared_card_abstraction
    rec = {'humanSeat': 1, 'humanHole': ['Ah', 'Ad'],
           'community': ['2s', '7c', '9d', 'Th', 'Jc'],
           'actionLog': [
               {'player': 0, 'street': 'preflop', 'action': 'bet_large', 'chips': 6.0},
               {'player': 1, 'street': 'preflop', 'action': 'call', 'chips': 5.0},
               {'player': 1, 'street': 'flop', 'action': 'check', 'chips': 0.0}]}
    ca = shared_card_abstraction()
    # Flop pot = blinds 3 + 6 + 5 = 14 -> SPR (200 - 7) / 14 = 13.8 -> bucket 3.
    for kf, tail in (('v2', '_oop_flop_'), ('spr5', '_oop_flop_r3_')):
        sink = defaultdict(Counter)
        recs = []
        replay_hand(rec, ca, sink, recs, key_format=kf)
        flop = [k for k, _ in recs if '_flop_' in k]
        assert len(flop) == 1 and flop[0].endswith(tail), (kf, flop)


def test_opponent_model_levels_parse_spr5_keys():
    from src.exploitation.opponent_model import _levels
    assert _levels('pf_3_7_ip_flop_r2_km') == _levels('pf_3_7_ip_flop_km')[:2] + \
        ['pf_3_7_ip_flop_r2_km']


def test_human_model_refuses_a_v3_blueprint(tmp_path):
    from src.exploitation.opponent_model import HumanModel
    db = BlueprintDB(str(tmp_path / 'bp.db'))
    db.set_metadata('key_format', 'spr5')
    with pytest.raises(ValueError, match='HumanModel'):
        HumanModel(str(tmp_path), db)
    db.close()


# ------------------------------------------------------------- stamps accepted

def test_blueprint_strategy_accepts_spr5_and_capped_pj(tmp_path):
    from src.game.bot_strategy import BlueprintStrategy
    db = BlueprintDB(str(tmp_path / 'bp.db'))
    db.set_metadata('key_format', 'spr5')
    db.set_metadata('menu_mode', 'capped_pj')
    BlueprintStrategy(db)
    kw = session_kwargs(db)
    assert kw['key_format'] == 'spr5' and kw['menu_mode'] == 'capped_pj'
    assert A.is_root(kw['abstraction_dir'])
    db.set_metadata('menu_mode', 'nonsense')
    with pytest.raises(ValueError):
        BlueprintStrategy(db)
    db.close()


# ------------------------------------------------------ river-solver review items

def test_custom_amounts_keep_full_precision():
    assert make_custom_action(True, 40.0) == 'raise_custom_40'
    assert make_custom_action(False, 37.5) == 'bet_custom_37.5'
    t = 100.0 / 3.0 + 1e-7
    assert _custom_total(make_custom_action(True, t)) == t


def test_untrained_nodes_project_as_passive_play():
    from src.subgame.river_subgame_solver import blueprint_to_tree_dist

    class Node:
        def __init__(self, actions, to_call):
            self.actions, self.to_call, self.pot_mid, self.sc, self.player = (
                actions, to_call, 20.0, (0.0, 10.0), 0)
    facing = Node(['fold', 'call', 'raise:30', 'allin'], 10.0)
    assert blueprint_to_tree_dist({}, facing) == {'fold': 0.0, 'call': 1.0,
                                                  'raise:30': 0.0, 'allin': 0.0}
    first = Node(['check', 'bet:10', 'allin'], 0.0)
    assert blueprint_to_tree_dist(None, first)['check'] == 1.0


class _Stub:
    def __init__(self):
        self.meta = {}

    def get_metadata(self, k, default=None):
        return self.meta.get(k, default)

    def get_average_strategy(self, key):
        return None                                  # every key untrained


def test_deep_raise_guard_needs_no_tracker():
    from src.subgame.river_subgame_solver import RiverSubgameSolver
    bot = RiverSubgameSolver(_Stub())
    ps = {'street': 'preflop', 'hole_cards': ['HA', 'DA'], 'community': [], 'seat': 1,
          'to_call': 190.0, 'pot': 392.0, 'p0_stack': 0.0, 'p1_stack': 190.0,
          'opp_range': None}
    assert bot._facing_deep_raise_guard('pf_168_oop_ssssa', ['fold', 'call'], ps) == 'call'
    ps['hole_cards'] = ['H2', 'D7']
    assert bot._facing_deep_raise_guard('pf_0_oop_ssssa', ['fold', 'call'], ps) == 'fold'


def test_zero_reach_spot_is_counted_and_played_by_the_fallback(monkeypatch):
    from src.subgame import river_subgame_solver as M
    bot = M.RiverSubgameSolver(_Stub())

    def unsolvable(**_):
        raise M.UnsolvableSpot('zero reach')
    monkeypatch.setattr(bot, 'solve_for_action', unsolvable)
    monkeypatch.setattr(bot, '_solver_inputs', lambda ps: {'hole': ps['hole_cards']})
    ps = {'street': 'river', 'hole_cards': ['HA', 'DA'],
          'community': ['S2', 'C7', 'D9', 'HT', 'CJ'], 'seat': 1, 'to_call': 50.0,
          'pot': 150.0, 'p0_stack': 50.0, 'p1_stack': 100.0, 'opp_range': None}
    action = bot.decide('pf_9_9_oop_river_kl', ['fold', 'call', 'raise_small'], ps)
    assert action in ('call', 'fold')
    assert bot.stats['zero_reach'] == 1 and bot.stats['fallback'] == 0


def test_gadget_fallback_keeps_its_reserved_budget():
    from src.subgame.river_subgame_solver import RiverSubgameSolver
    assert 0.0 < RiverSubgameSolver._GADGET_RESERVE < 1.0


# ---------------------------------------- the BUG-027 hand through GameSession, v3

# Bot = SB with J9hh. Open 3.5BB, the BB 3-bets to 8.5BB (custom 17 chips), the bot
# 4-bets (raise_medium), the BB calls. Flop Qs9c3c: the BB leads all-in.
J9, VIL, BOARD = ['HJ', 'H9'], ['CA', 'CK'], ['SQ', 'C9', 'C3', 'D2', 'H5']


@pytest.mark.parametrize('arm', sorted(SNAPS))
def test_bug027_hand_through_game_session(arm):
    path = SNAPS[arm]
    if not os.path.exists(path):
        pytest.skip(f'{arm} snapshot not present')
    from src.subgame.river_subgame_solver import RiverSubgameSolver
    db = BlueprintDB(path, read_only=True)
    try:
        kw = session_kwargs(db)
        assert kw['key_format'] == 'spr5' and kw['menu_mode'] == 'capped_pj'
        assert A.is_root(kw['abstraction_dir']) == (arm == 'A300')
        bot = RiverSubgameSolver(db, max_iters=60, check_every=20, time_budget=5.0,
                                 safe_gadget=True, gadget_anchor='auto',
                                 rng=np.random.default_rng(0))
        s = GameSession.new('t', 'p', strategy_fn=bot.range_model_fn(),
                            hero_strategy_fn=bot.hero_range_model_fn(), **kw)
        s.data['human_seat'] = 1
        _rig(s, J9, VIL, BOARD)
        trained = []

        def bot_acts(action):
            key = s.info_set_key(0)
            trained.append((key, db.get_average_strategy(key) is not None))
            s.apply_action(action)
        bot_acts('bet_large')
        s.apply_action(make_custom_action(True, 17.0))
        bot_acts('raise_medium')
        s.apply_action('call')
        d = s.data
        assert d['street'] == 1 and d['starting_pot'] == 88.0
        s.apply_action('allin')                     # the BB leads all-in
        key = s.info_set_key(0)
        assert '_flop_r0_a' in key                  # SPR 1.8 -> the jam-zone bucket
        trained.append((key, db.get_average_strategy(key) is not None))
        assert all(t for _, t in trained), trained  # every bot node is a trained key
        legal = s.legal_actions()
        action = bot.decide(key, legal, s.bot_public_state())
        assert action in ('fold', 'call')

        # A human 5-bet (sized) at the cap is answered from the trained jam node.
        s2 = GameSession.new('t2', 'p', strategy_fn=bot.range_model_fn(),
                             hero_strategy_fn=bot.hero_range_model_fn(), **kw)
        s2.data['human_seat'] = 0                   # the bot is the BB here
        _rig(s2, VIL, ['HA', 'SA'], BOARD)
        s2.apply_action('bet_small')
        s2.apply_action('raise_small')
        s2.apply_action('raise_small')              # the human 4-bets: the bot is at the cap
        assert s2.legal_actions() == ['fold', 'call', 'allin']
        s2.apply_action('allin')                    # the bot jams
        assert s2.data['bet_pattern'] == 'sssa'
        assert s2.is_human_turn() and s2.legal_actions() == ['fold', 'call']
    finally:
        db.close()
