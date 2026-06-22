#!/usr/bin/env python3
"""Phase 0 of TURN_LATENCY_PLAN: profile ONE turn solve to see where the 20-30s goes --
leaf build (build_board_arrays + the per-bucket _eval) vs CFR vs the exact GRADE (turn_leaf_value_exact)
-- cold vs warm. Rigs a GameSession at a low-SPR turn-solve spot and cProfiles the bot's decide().

  python scripts/profile_turn_solve.py
"""
import cProfile
import io
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.game.game_session as gs
from src.storage.blueprint_db import BlueprintDB
from src.abstractions.sizing import db_menu_mode
from src.game.game_session import GameSession
from src.game.cards import shuffled_deck as real_deck
from src.subgame.turn_subgame_solver import TurnSubgameSolver
from src.cfr.poker_game import make_custom_action

DB = os.environ.get('ALLIN_BLUEPRINT_DB', 'analysis/blueprints/snapshots/snap_52500000.db')
FRONT = ['DQ', 'HT', 'H2', 'S3', 'C6', 'CT', 'D2', 'CQ', 'SQ']   # bot, human, board(4 turn + river)
KEYFNS = ('build_board_arrays', 'turn_leaf_value_exact', 'turn_leaf_matrix', '_eval', 'board_winrates',
          'get_bucket', 'get_raw_hand_value', 'solve_turn_for_action', 'node_action_values',
          '_hand_action_evs', 'ExactLeafTurnCFR')

db = BlueprintDB(DB, read_only=True)
mm = db_menu_mode(db)
_rest = [c for c in real_deck() if c not in FRONT]
gs.shuffled_deck = lambda: list(FRONT) + _rest

bot = TurnSubgameSolver(db, n_buckets=24, leaf_rivers=4, max_spr_turn=8, turn_time_budget=12,
                        multivalued_leaf=False, max_iters=200, check_every=40, time_budget=10.0,
                        safe_gadget=True, gadget_anchor='auto', purify_threshold=0.01)
rf = bot.range_model_fn()


def spot():
    s = GameSession.new('x', 'p', strategy_fn=rf, hero_strategy_fn=rf,
                        menu_mode=mm, max_raises_per_street=float('inf'))
    s._deal_hand(hand_number=2, human_seat=1)              # bot = seat 0 = button
    for act in [make_custom_action(True, 6.0),             # bot raise to 3BB
                make_custom_action(True, 18.0),            # human 3-bet to 9BB
                'call',                                    # bot call
                make_custom_action(False, 20.0),           # human flop bet 10BB
                'call',                                    # bot flop call
                make_custom_action(False, 30.0)]:          # human turn bet 15BB -> bot to act
        s.apply_action(act)
    return s


def run(label):
    s = spot()
    seat = s.current_player()
    key = s.info_set_key(seat)
    legal = s.legal_actions()
    public = s.bot_public_state()
    ps_pot = public.get('turnEntryPot')
    ps_stk = public.get('turnEntryStacks')
    spr = (ps_stk[0] / ps_pot) if (ps_pot and ps_stk) else None
    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    bot.decide(key, legal, public)
    pr.disable()
    dt = time.time() - t0
    mode = (bot.last_debug or {}).get('mode')
    print(f"\n========== {label}: total {dt:.2f}s | mode={mode} | key={key} | SPR={spr} ==========")
    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf).sort_stats('cumulative')
    st.print_stats(22)
    out = buf.getvalue().splitlines()
    print('\n'.join(l for l in out if ('cumtime' in l) or any(fn in l for fn in KEYFNS)))


run('COLD (first solve)')
run('WARM (second solve -- module caches warm, but ba_cache is per-solve)')
db.close()
