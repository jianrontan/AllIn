# backend/bot/scripts/measure_serving_memory.py
"""
Resident memory of the serving path for one blueprint (v3 plan V3.6: a C-arm blueprint
loads a second set of card tables, so measure what each candidate costs to serve).

Builds the served bot the way strategy_api does (RiverSubgameSolver with the safe
gadget, the range-model callbacks) and a GameSession bound with session_kwargs, then
plays `--hands` hands against a random-legal human through advance_bot_turns, so the
card tables, the river board cache and the solver all warm up as they do live. Reports
RSS after the imports, after the bot is built, and after the hands, plus the peak.

Run one blueprint per process (from backend/bot/), e.g.
    python scripts/measure_serving_memory.py --db analysis/blueprints/blueprint_final_v2.db
"""
import argparse
import json
import os
import random
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def _peak_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == 'darwin' else peak / 1024   # bytes vs KiB


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--db', required=True)
    ap.add_argument('--hands', type=int, default=40)
    ap.add_argument('--max-iters', type=int, default=60,
                    help='river solve iterations (memory is dominated by tables, not iters)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    from src.storage.blueprint_db import BlueprintDB
    from src.subgame.river_subgame_solver import RiverSubgameSolver
    from src.game.game_session import GameSession, advance_bot_turns, session_kwargs
    import numpy as np
    out = {'db': os.path.basename(args.db), 'rss_imports_mb': round(_rss_mb(), 1)}

    db = BlueprintDB(args.db, read_only=True)
    kw = session_kwargs(db)
    bot = RiverSubgameSolver(db, max_iters=args.max_iters, check_every=20, time_budget=6.0,
                             safe_gadget=True, gadget_anchor='auto', purify_threshold=0.01,
                             rng=np.random.default_rng(args.seed))
    range_fn, hero_fn = bot.range_model_fn(), bot.hero_range_model_fn()
    out.update({'key_format': kw['key_format'], 'menu_mode': kw['menu_mode'],
                'abstraction': os.path.basename(kw['abstraction_dir']),
                'rss_bot_built_mb': round(_rss_mb(), 1)})

    rng = random.Random(args.seed)
    random.seed(args.seed)
    t0 = time.time()
    data = None
    for h in range(args.hands):
        s = (GameSession.new('m', 'p', strategy_fn=range_fn, hero_strategy_fn=hero_fn, **kw)
             if data is None else
             GameSession.from_dict(data, strategy_fn=range_fn, hero_strategy_fn=hero_fn, **kw))
        if data is not None:
            s.start_next_hand()
        while s.data['status'] == 'in_hand':
            if s.is_human_turn():
                # A sticky caller: reaches the river with real pots, so river solves run.
                legal = s.legal_actions()
                passive = [a for a in ('call', 'check') if a in legal]
                if passive and rng.random() < 0.7:
                    s.apply_action(passive[0])
                else:
                    s.apply_action(rng.choice([a for a in legal if a != 'fold'] or legal))
            else:
                advance_bot_turns(s, bot)
        data = s.to_dict()
    out.update({'hands': args.hands, 'seconds': round(time.time() - t0, 1),
                'rss_after_hands_mb': round(_rss_mb(), 1), 'peak_mb': round(_peak_mb(), 1),
                'river_solves': bot.stats.get('solved', 0)})
    print(json.dumps(out))


if __name__ == '__main__':
    main()
