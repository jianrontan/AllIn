# backend/bot/tests/run_maniac_live.py
"""
LIVE max-presser sanity check -- drives the ACTUAL served bot through the same
uncapped GameSession path the website uses, not the capped match engine.

Why a separate harness from run_maniac.py: run_maniac.py reuses match.py, whose
engine CAPS aggression at 3/street (matching training). A human on the live site
plays through GameSession with max_raises_per_street=inf, so they can 5-bet /
6-bet+. Those beyond-cap nodes (e.g. pf_29_ip_slll) are NEVER in the blueprint,
so the bot hits BlueprintStrategy's passive fallback (uniform call/fold) and can
FOLD THE NUTS. The capped harness can't reach that node, so it can't see the leak.

Two maniac STYLES (both never fold):
  * maxbet -- always press the biggest NON-all-in size; when no sized raise keeps
    chips behind, just CALL. Produces deep NON-all-in raises (exercises the
    deep-raise guard, fix #1).
  * jam    -- escalate with max non-all-in raises while there is room, then SHOVE
    (all-in) once committed instead of calling. Produces beyond-cap FACED-ALL-INS
    for the bot (exercises the untrained-jam half of the guard, fix #2).

`--cripple2` reinstates the pre-#2 behavior (the deep guard defers on faced
all-ins) so a jam-style A/B shows fix #2's marginal effect.

Reports the BOT's win rate (BB/hand; >0 = bot beats the maniac, as a sound
strategy should CRUSH one) and quantifies the premium-fold leak + which guard
fired.

Usage (from backend/bot/):
    python tests/run_maniac_live.py --style maxbet --hands 20000
    python tests/run_maniac_live.py --style jam --hands 5000
    python tests/run_maniac_live.py --style jam --hands 5000 --cripple2   # #1-only A/B
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import resolve_blueprint_path
from src.storage.blueprint_db import BlueprintDB
from src.abstractions.sizing import db_menu_mode
from src.game.game_session import GameSession, advance_bot_turns
from src.subgame.river_subgame_solver import RiverSubgameSolver
from src.cfr.poker_game import make_custom_action

PREMIUM_BUCKET = 27          # top-3 fine preflop buckets (~QQ+/AK) on the 30-bucket scheme


def _bot_stack(ps):
    seat = ps.get('seat')
    return float(ps['p0_stack'] if seat == 0 else ps['p1_stack'])


class RecordingSolver(RiverSubgameSolver):
    """The real served bot, plus a per-decision log so we can attribute the leak.
    `cripple_guard2=True` simulates the pre-#2 bot: the deep guard defers on a faced
    all-in (to_call >= bot_stack) instead of deciding it on equity."""

    def __init__(self, *a, cripple_guard2=False, **k):
        super().__init__(*a, **k)
        self.log = []
        self.cripple_guard2 = cripple_guard2
        self.deep_allin = 0      # deep-guard fires that were FACED ALL-INS (the #2 path)
        self.deep_nonallin = 0   # deep-guard fires with money behind (the #1 path)
        self.value_jams = 0      # C1: deep-guard upgraded a monster's call to an all-in
        self.blend_fires = 0     # #4: decisions where the bot consumed an off-grid translation
        self.trans_folds = 0     # #4: folds while a translation was active (any -- mostly legit)
        self.blend_untrained_bracket = 0   # #4: a bracket KEY was untrained (the leak precondition)
        self.blend_untrained_fold = 0      # #4: folded WHILE a bracket was untrained (the actual leak)
        self.firstact_untrained_checks = 0   # #5: first-to-act check at an untrained key

    def _facing_deep_raise_guard(self, key, legal, ps):
        if self.cripple_guard2:
            to_call = float(ps.get('to_call') or 0.0)
            if to_call >= _bot_stack(ps) - 1e-6:
                return None      # pre-#2: punt the faced-all-in to the blueprint
        return super()._facing_deep_raise_guard(key, legal, ps)

    def decide(self, key, legal, public):
        trans = (public or {}).get('translation')
        action = super().decide(key, legal, public)
        hit = self.db.get_average_strategy(key) is not None
        mode = (self.last_debug or {}).get('mode')
        street = (public or {}).get('street')
        if mode == 'deep_raise_guard':
            to_call = float((public or {}).get('to_call') or 0.0)
            if to_call >= _bot_stack(public) - 1e-6:
                self.deep_allin += 1
            else:
                self.deep_nonallin += 1
            if action == 'allin':
                self.value_jams += 1     # C1 fired (deep guard upgraded a call to a jam)
        if trans:
            self.blend_fires += 1
            untrained_bracket = any(self.db.get_average_strategy(bk) is None for bk, _ in trans)
            if untrained_bracket:
                self.blend_untrained_bracket += 1
            if action == 'fold':
                self.trans_folds += 1
                if untrained_bracket:
                    self.blend_untrained_fold += 1     # the ACTUAL #4 leak signal
        # #5: first-to-act (no bet faced) at an untrained key -> blind check (misses value).
        if 'call' not in legal and action == 'check' and not hit:
            self.firstact_untrained_checks += 1
        bucket = None
        if street == 'preflop' and key.startswith('pf_'):
            try:
                bucket = int(key.split('_')[1])
            except (ValueError, IndexError):
                bucket = None
        self.log.append({'street': street, 'key': key, 'action': action,
                         'db_hit': hit, 'mode': mode, 'bucket': bucket,
                         'hole': tuple((public or {}).get('hole_cards') or ())})
        return action


def human_action(session, style):
    """maxbet: biggest NON-all-in sized button, else call. jam: same, but SHOVE when
    committed. overbet: a CUSTOM off-grid total between two sized tiers (the ONLY thing
    that triggers the bot's action-translation/blend, for testing #4), else call.
    Never folds."""
    legal = session.legal_actions()
    d = session.data
    stack = d['p0_stack'] if d['human_seat'] == 0 else d['p1_stack']
    best, best_cost = None, -1.0
    for a in legal:
        if a.startswith(('bet_', 'raise_')):
            cost = session._action_cost(a)
            if cost < stack and cost > best_cost:    # strictly < stack => leaves chips => not all-in
                best, best_cost = a, cost
    if style == 'overbet' and best is not None:
        bounds = session.custom_bounds()             # (min_total, max_total) raise-to chips
        if bounds is not None and bounds[1] > bounds[0]:
            lo, hi = bounds
            # max_total = contribution + stack, so contribution = hi - stack. A sized
            # tier's raise-to total = contribution + its add. Target the MIDPOINT of the
            # two largest sized tiers -> strictly off-grid AND sub-stack, so translate_bet
            # returns 2 brackets and pending_translation is set (the #4 trigger).
            costs = sorted(session._action_cost(a) for a in legal
                           if a.startswith(('bet_', 'raise_')) and session._action_cost(a) < stack)
            if len(costs) >= 2:
                total = (hi - stack) + 0.5 * (costs[-1] + costs[-2])
                total = min(max(total, lo + 1e-6), hi - 1e-6)
                return make_custom_action('call' in legal, total)
    if best is not None:
        return best                                  # room for a non-all-in max raise -> escalate
    if style == 'jam' and 'allin' in legal:
        return 'allin'                               # committed -> shove (creates a faced all-in)
    if 'call' in legal:
        return 'call'
    if 'check' in legal:
        return 'check'
    return 'allin' if 'allin' in legal else 'fold'   # degenerate node only


def play_one_hand(session, bot, style):
    guard = 0
    while session.data['status'] == 'in_hand':
        advance_bot_turns(session, bot)
        if session.data['status'] != 'in_hand':
            break
        if session.is_human_turn():
            session.apply_action(human_action(session, style))
        guard += 1
        if guard > 400:
            break


def run(path, hands, seed, style, cripple2, value_jam=False):
    db = BlueprintDB(path, read_only=True)
    menu_mode = db_menu_mode(db)
    bot = RecordingSolver(db, max_iters=120, check_every=40, time_budget=2.0,
                          cripple_guard2=cripple2, value_jam=value_jam)
    range_fn = bot.range_model_fn()

    import random
    random.seed(seed)

    session = GameSession.new('maniac-live', 'maniac', strategy_fn=range_fn,
                              menu_mode=menu_mode, max_raises_per_street=float('inf'))
    results = []
    prev_net = 0.0
    premium_folds = []       # (hand#, hole, key, db_hit)
    t0 = time.time()
    for h in range(hands):
        if h > 0:
            session.start_next_hand()
        bot.log.clear()
        play_one_hand(session, bot, style)
        net = session.data['human_net']
        results.append(-(net - prev_net) / 2.0)
        prev_net = net
        for e in bot.log:
            if (e['action'] == 'fold' and e['street'] == 'preflop'
                    and e['bucket'] is not None and e['bucket'] >= PREMIUM_BUCKET):
                premium_folds.append((h, e['hole'], e['key'], e['db_hit']))
        if (h + 1) % 1000 == 0:
            print(f"  {h + 1}/{hands}  bot={sum(results) / len(results):+.3f} BB/hand"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    db.close()
    return results, premium_folds, bot.stats, bot


def main():
    p = argparse.ArgumentParser(description="Live (uncapped) max-presser vs the served bot.")
    p.add_argument('--db', default=None)
    p.add_argument('--hands', type=int, default=20000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--style', choices=['maxbet', 'jam', 'overbet'], default='maxbet')
    p.add_argument('--cripple2', action='store_true',
                   help="Simulate the pre-#2 bot (deep guard defers on faced all-ins).")
    p.add_argument('--value-jam', dest='value_jam', action='store_true',
                   help="C1: value-jam monsters at beyond-cap nodes (default off).")
    args = p.parse_args()

    path = args.db or resolve_blueprint_path()
    print(f"Blueprint : {path}")
    print(f"Hands     : {args.hands}  (seed {args.seed})   style={args.style}"
          f"{'  [CRIPPLE #2]' if args.cripple2 else ''}"
          f"{'  [VALUE-JAM]' if args.value_jam else ''}")
    print("(bot BB/hand > 0 = bot beats the maniac, as a sound strategy should)\n")

    results, premium_folds, stats, bot = run(
        path, args.hands, args.seed, args.style, args.cripple2, args.value_jam)
    n = len(results)
    mean = sum(results) / n
    se = math.sqrt(sum((r - mean) ** 2 for r in results) / n / n)
    print(f"\nBot win rate : {mean * 1000:+.1f} +/- {se * 1000:.1f} mbb/hand"
          f"  ({mean:+.3f} BB/hand)")
    print(f"Premium preflop FOLDS (bucket >= {PREMIUM_BUCKET}): "
          f"{len(premium_folds)} / {n}  ({100.0 * len(premium_folds) / n:.2f}%)")
    beyond_cap = sum(1 for _, _, _, hit in premium_folds if not hit)
    print(f"  of which at an UNTRAINED beyond-cap key: {beyond_cap}")
    for hno, hole, key, hit in premium_folds[:8]:
        print(f"    hand {hno}: folded {hole}  key={key}  db_hit={hit}")
    print(f"\nDeep-raise guard fires: faced-all-in (#2 path)={bot.deep_allin}  "
          f"money-behind (#1 path)={bot.deep_nonallin}  C1 value-jams={bot.value_jams}")
    print(f"#4 translation: blend consumed={bot.blend_fires}  folds-under-translation={bot.trans_folds}")
    print(f"   untrained-bracket present={bot.blend_untrained_bracket}  "
          f"fold-WITH-untrained-bracket (the leak)={bot.blend_untrained_fold}")
    print(f"#5 first-to-act untrained checks={bot.firstact_untrained_checks}")
    print(f"Bot decision modes: {dict(stats)}")


if __name__ == '__main__':
    main()
