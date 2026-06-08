# backend/bot/scripts/measure_turn_match.py
"""
B2 -- head-to-head: does the served stack win chips vs the blueprint?

Plays one BOT strategy (turn solver / river solver / blueprint) against a BLUEPRINT
opponent through the real GameSession (so the bot gets its live trackers + turn/river-
entry fields, exactly like serving), alternating seats each hand. Reports the bot's
chip win rate (mbb/hand).

Compare runs:
  * blueprint vs blueprint   -> sanity, should be ~0
  * river  vs blueprint      -> the CURRENTLY-deployed stack's edge
  * turn   vs blueprint      -> turn-solver stack's edge
The (turn - river) gap is the turn solver's marginal value over today's deployment.

NOTE: NOT common-random-numbers (the solver's RNG + diverging trajectories desync the
deck across runs), so this is RAW and high-variance -- a DIRECTIONAL check, not the
exploitability gate (that's the LBR run). Slow (live solves); run in the background.

Run from backend/bot/:
    python scripts/measure_turn_match.py --bot turn --hands 150 --n-buckets 20 --leaf-rivers 4 --turn-iters 80
    python scripts/measure_turn_match.py --bot river --hands 150
    python scripts/measure_turn_match.py --bot blueprint --hands 400
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import resolve_blueprint_path
from src.storage.blueprint_db import BlueprintDB, FrozenBlueprint
from src.abstractions.sizing import db_menu_mode, BIG_BLIND
from src.game.game_session import GameSession
from src.game.bot_strategy import BlueprintStrategy
from src.subgame.river_subgame_solver import RiverSubgameSolver
from src.subgame.turn_subgame_solver import TurnSubgameSolver


def _build_bot(kind, rawdb, fb, args):
    if kind == 'blueprint':
        return BlueprintStrategy(fb)
    if kind == 'river':
        s = RiverSubgameSolver(rawdb, max_iters=args.river_iters, time_budget=args.river_budget)
    elif kind == 'turn':
        s = TurnSubgameSolver(rawdb, n_buckets=args.n_buckets, leaf_rivers=args.leaf_rivers,
                              turn_max_iters=args.turn_iters, max_iters=args.river_iters,
                              time_budget=args.river_budget)
    else:
        raise ValueError(kind)
    s.db = fb                       # cache blueprint reads across hands (huge speedup)
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bot', choices=['turn', 'river', 'blueprint'], default='turn')
    p.add_argument('--db', default=None)
    p.add_argument('--hands', type=int, default=150)
    p.add_argument('--n-buckets', type=int, default=20)
    p.add_argument('--leaf-rivers', type=int, default=4)
    p.add_argument('--turn-iters', type=int, default=80)
    p.add_argument('--river-iters', type=int, default=150)
    p.add_argument('--river-budget', type=float, default=12.0)
    p.add_argument('--max-steps', type=int, default=400)
    args = p.parse_args()
    path = args.db or resolve_blueprint_path()
    rawdb = BlueprintDB(path, read_only=True)
    fb = FrozenBlueprint(rawdb)
    menu = db_menu_mode(rawdb)
    bot = _build_bot(args.bot, rawdb, fb, args)
    human = BlueprintStrategy(fb)
    strat_fn = BlueprintStrategy(fb).range_model_fn()

    print(f"bot={args.bot} vs blueprint | {os.path.basename(path)} | menu={menu} | "
          f"hands={args.hands} | turn(n={args.n_buckets},rivers={args.leaf_rivers},"
          f"it={args.turn_iters})", flush=True)

    sess = GameSession.new('m2', 'p', strategy_fn=strat_fn, menu_mode=menu)
    deltas = []
    t0 = time.time()
    for h in range(args.hands):
        steps = 0
        while sess.data['status'] == 'in_hand' and steps < args.max_steps:
            seat = sess.current_player()
            legal = sess.legal_actions()
            key = sess.info_set_key(seat)
            bot_seat = 1 - sess.data['human_seat']
            if seat == bot_seat:
                a = bot.decide(key, legal, sess.bot_public_state())
            else:
                a = human.decide(key, legal, {})
            sess.apply_action(a)
            steps += 1
        if sess.data['status'] == 'hand_over':
            deltas.append(-float(sess.data['result']['humanDelta']))   # bot's chips
        if h < args.hands - 1:
            sess.start_next_hand()
        if (h + 1) % max(1, args.hands // 10) == 0:
            n = len(deltas)
            mbb = (sum(deltas) / n / BIG_BLIND * 1000.0) if n else 0.0
            print(f"  hand {h + 1:5d}: bot {mbb:+8.1f} mbb/hand  ({time.time() - t0:.0f}s)",
                  flush=True)
    rawdb.close()

    n = len(deltas)
    mean_chips = sum(deltas) / n if n else 0.0
    var = sum((d - mean_chips) ** 2 for d in deltas) / n if n else 0.0
    mbb = mean_chips / BIG_BLIND * 1000.0
    stderr = (math.sqrt(var / n) / BIG_BLIND * 1000.0) if n else 0.0
    print(f"\n  RESULT bot={args.bot}: {mbb:+.1f} +/- {stderr:.1f} mbb/hand "
          f"over {n} hands ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
