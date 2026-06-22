#!/usr/bin/env python3
"""Phase 0/1 measurement for TURN_LATENCY_PLAN: wall-clock latency of turn solves.

Two modes:
  --fresh K   solve K DISTINCT random turn spots (each board new -> no cross-board cache benefit).
              This is the realistic COLD floor the Phase-1 cache must beat. (default)
  --repeat K  solve the SAME spot K times (warm-cache benefit, once a cross-solve cache exists).

Reports per-solve wall time (no cProfile inflation) + the solver's own leaf/CFR/grade split, and
mean/p50/p90. Run before AND after building the Phase-1 cache to quantify the recovery.

  python scripts/bench_turn_solve.py --fresh 12
  python scripts/bench_turn_solve.py --repeat 6
"""
import argparse
import os
import random
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


def make_spot(bot, rf, mm, seed):
    """Deal a random hand and drive a line that reaches a LOW-SPR turn solve (3-bet pot, flop call,
    turn bet faced). Returns the session at the bot's turn decision, or None if the line didn't land
    on a turn-solve node (folded/all-in/SPR-gated) -- caller retries with a new seed."""
    rnd = random.Random(seed)
    deck = real_deck()
    rnd.shuffle(deck)
    gs.shuffled_deck = lambda d=deck: list(d)
    s = GameSession.new('x', 'p', strategy_fn=rf, hero_strategy_fn=rf,
                        menu_mode=mm, max_raises_per_street=float('inf'))
    s._deal_hand(hand_number=2, human_seat=1)             # bot = seat 0 = button
    try:
        for act in [make_custom_action(True, 6.0), make_custom_action(True, 18.0), 'call',
                    make_custom_action(False, 20.0), 'call', make_custom_action(False, 30.0)]:
            if s.data['status'] != 'in_hand' or not _is_actor(s, act):
                return None
            s.apply_action(act)
        if s.data['status'] != 'in_hand' or s.data['street'] != 2:
            return None
        return s
    except Exception:
        return None


def _is_actor(s, act):
    return True   # the scripted line is legal by construction for this stack/blind setup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fresh', type=int, default=0, help="K distinct random boards (cold floor)")
    ap.add_argument('--repeat', type=int, default=0, help="K solves of the SAME spot (warm)")
    ap.add_argument('--buckets', type=int, default=24)
    ap.add_argument('--rivers', type=int, default=4)
    args = ap.parse_args()
    k = args.fresh or args.repeat or 12
    mode = 'repeat' if args.repeat else 'fresh'

    db = BlueprintDB(DB, read_only=True)
    mm = db_menu_mode(db)
    bot = TurnSubgameSolver(db, n_buckets=args.buckets, leaf_rivers=args.rivers, max_spr_turn=8,
                            turn_time_budget=99, multivalued_leaf=False, max_iters=200, check_every=40,
                            time_budget=10.0, safe_gadget=True, gadget_anchor='auto', purify_threshold=0.01)
    rf = bot.range_model_fn()

    times, solved = [], 0
    seed, base_spot = 1000, None
    while solved < k:
        s = base_spot if (mode == 'repeat' and base_spot is not None) else make_spot(bot, rf, mm, seed)
        seed += 1
        if s is None:
            if seed > 1000 + 60 * k:
                break
            continue
        if mode == 'repeat':
            base_spot = s
        seat = s.current_player()
        key = s.info_set_key(seat)
        legal = s.legal_actions()
        public = s.bot_public_state()
        t0 = time.time()
        bot.decide(key, legal, public)
        dt = time.time() - t0
        dbg = bot.last_debug or {}
        if dbg.get('mode') != 'turn_solver':           # gated / fell back -> not a real solve sample
            continue
        times.append(dt)
        solved += 1
        print(f"  [{solved:2}/{k}] {dt:6.2f}s  key={key}", flush=True)

    db.close()
    if not times:
        print("no turn solves landed -- adjust the line/SPR gate")
        return
    times.sort()
    n = len(times)
    p = lambda q: times[min(n - 1, int(q * n))]
    print(f"\n{mode.upper()} n={n} buckets={args.buckets} rivers={args.rivers}: "
          f"mean {sum(times)/n:.2f}s  p50 {p(0.5):.2f}s  p90 {p(0.9):.2f}s  "
          f"min {times[0]:.2f}s  max {times[-1]:.2f}s")


if __name__ == '__main__':
    main()
