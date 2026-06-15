# backend/bot/tests/compare_gadget_policies.py
"""
Compare the three river-solver SERVING policies head-to-head on the live (uncapped)
GameSession path, against a maniac opponent:

  * off  -- safe_gadget=False                       (unsafe-v1, today's served bot)
  * A    -- safe_gadget=True, gadget_anchor='confidence'  (confidence gate only)
  * B    -- safe_gadget=True, gadget_anchor='auto'        (confidence pre-filter + self-check)

It reports, per policy:
  * bot win rate (BB/hand +/- se) -- the EV axis (>0 = bot beats the maniac).
  * river-solve count + the anchor-decision breakdown (how often each policy exploited
    vs clamped to the safe gadget) -- so you can see WHETHER the policies even diverged.
  * per-river-solve latency p50/p99 (ms) + total wall time -- the latency axis.

WHY a maniac opponent: the gadget only changes RIVER-solve decisions, and the policies
only diverge when the river belief is UNTRUSTED (off-model play). A maniac plays off-model
-> decays confidence -> exactly the regime where 'off' may over-exploit a wrong river
belief and bleed chips while A/B clamp. The 'maxbet' style (deep NON-all-in betting) reaches
the river most often, so it is the default; pure jam/shove styles end preflop and rarely
exercise the solver (watch the river-solve count -- a low count = weak EV signal, lean on
the exploitability battery in test_safe_river_gadget.py instead).

Common random numbers (same seed) across policies for variance reduction; hands still
diverge because the bot acts differently, so treat the BB/hand gap as indicative, not
paired. For the worst-case SAFETY axis see tests/test_safe_river_gadget.py (exact BR).

Usage (from backend/bot/):
    python tests/compare_gadget_policies.py --hands 4000 --style maxbet
    python tests/compare_gadget_policies.py --hands 2000 --style maxbet --policies off,B
"""
import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for run_maniac_live

import numpy as np

from src.config import resolve_blueprint_path
from src.storage.blueprint_db import BlueprintDB
from src.abstractions.sizing import db_menu_mode
from src.cfr.poker_game import STARTING_STACK
from src.game.game_session import GameSession
from src.subgame.river_subgame_solver import RiverSubgameSolver
from run_maniac_live import human_action, play_one_hand


def _play_and_record(session, bot, style, bot_seat):
    """Play one hand and build an AIVAT record (aivat.AIVATEstimator format). We wrap
    session.apply_action -- which BOTH the human loop and advance_bot_turns route every
    action through -- to log each action's PRE-street (the all-in runout adds no actions,
    so the last logged street IS the all-in street) and the folder seat. allin_street is
    set only for an all-in SHOWDOWN (both fully invested, no fold); the AIVAT c3 all-in CV
    keys off it. events=[] -> c2 (river-runout) is skipped: vs an off-model maniac the
    reconstructed opponent range would be wrong anyway, while c1 (preflop equity) + c3
    (the dominant preflop-stackoff variance) are valid and high-value here."""
    d = session.data
    log = {'streets': [], 'folder': None}
    orig = session.apply_action

    def wrapped(action, solved_hero_probs=None):
        # advance_bot_turns (1a continual re-solving) calls apply_action with the
        # solved_hero_probs kwarg; the wrapper must accept and forward it or the
        # bot loop crashes mid-hand.
        if action == 'fold':
            log['folder'] = session.current_player()
        log['streets'].append(d['street'])
        return orig(action, solved_hero_probs=solved_hero_probs)

    session.apply_action = wrapped
    try:
        play_one_hand(session, bot, style)
    finally:
        session.apply_action = orig

    res = d.get('result') or {}
    # TOTAL committed per seat = pre-final-street invested + this street's contribution
    # (the final street's bets live in `history`, not yet folded into p*_invested), via
    # the engine's own accounting so it includes blinds and matches the final pot. This
    # is what AIVAT's c3 needs (pot = sum(invested), inv_a = invested[seat]); the raw
    # p*_invested understates it and missed every all-in.
    g = session.game
    st = min(d['street'], 3)
    p0_tot = d['p0_invested'] + g.get_player_contribution_this_round(
        d['history'], st, d['starting_pot'], 0, d['p0_invested'], d['p1_invested'])
    p1_tot = d['p1_invested'] + g.get_player_contribution_this_round(
        d['history'], st, d['starting_pot'], 1, d['p0_invested'], d['p1_invested'])
    showdown = res.get('reason') == 'showdown'
    both_allin = (abs(p0_tot - STARTING_STACK) < 1e-6 and abs(p1_tot - STARTING_STACK) < 1e-6)
    allin_street = log['streets'][-1] if (showdown and both_allin and log['streets']) else None
    a_cards = d['p0_cards'] if bot_seat == 0 else d['p1_cards']
    b_cards = d['p1_cards'] if bot_seat == 0 else d['p0_cards']
    return {
        'seat_of_A': bot_seat,
        'hand_a': list(a_cards),
        'hand_b': list(b_cards),
        'board': list(d['community']),
        'events': [],                                  # skip c2 (wrong range vs a maniac)
        'allin_street': allin_street,
        'folded': log['folder'],
        'invested': [p0_tot, p1_tot],
        # 'result' is filled in by the caller (needs the cross-hand human_net delta).
    }

_POLICIES = {
    'off': dict(safe_gadget=False),
    'A':   dict(safe_gadget=True, gadget_anchor='confidence'),
    'B':   dict(safe_gadget=True, gadget_anchor='auto'),
}


class _PolicyBot(RiverSubgameSolver):
    """The served bot + per-river-solve timing and anchor-decision tally."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.river_latencies = []          # seconds per river-SOLVER decision
        self.anchor_counts = {}            # anchor label -> count

    def decide(self, key, legal, public):
        t0 = time.perf_counter()
        action = super().decide(key, legal, public)
        dbg = self.last_debug or {}
        if dbg.get('mode') == 'river_solver':
            self.river_latencies.append(time.perf_counter() - t0)
            anc = dbg.get('anchor') or 'unsafe'
            self.anchor_counts[anc] = self.anchor_counts.get(anc, 0) + 1
        return action


def run_policy(path, hands, seed, style, config, aivat=False):
    db = BlueprintDB(path, read_only=True)
    menu_mode = db_menu_mode(db)
    bot = _PolicyBot(db, max_iters=120, check_every=40, time_budget=2.0, **config)
    range_fn = bot.range_model_fn()
    random.seed(seed)                      # common random numbers across policies
    session = GameSession.new('cmp', 'maniac', strategy_fn=range_fn,
                              menu_mode=menu_mode, max_raises_per_street=float('inf'))
    results = []
    records = []
    prev_net = 0.0
    bot_seat = 1 - session.data['human_seat']
    t0 = time.time()
    for h in range(hands):
        if h > 0:
            session.start_next_hand()
            bot_seat = 1 - session.data['human_seat']
        if aivat:
            rec = _play_and_record(session, bot, style, bot_seat)
        else:
            play_one_hand(session, bot, style)
        net = session.data['human_net']
        bot_chips = -(net - prev_net)               # bot net chips this hand
        results.append(bot_chips / 2.0)             # BB
        prev_net = net
        if aivat:
            rec['result'] = bot_chips               # AIVAT result is in chips
            records.append(rec)
    wall = time.time() - t0
    db.close()
    return results, bot, wall, records


def _fmt_latency(lat):
    if not lat:
        return "      n/a (no river solves)"
    ms = np.array(lat) * 1000.0
    return (f"p50={np.percentile(ms, 50):6.1f}ms  p99={np.percentile(ms, 99):7.1f}ms  "
            f"n={len(ms)}")


def main():
    p = argparse.ArgumentParser(description="Compare river gadget serving policies (off/A/B).")
    p.add_argument('--db', default=None)
    p.add_argument('--hands', type=int, default=4000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--style', choices=['maxbet', 'jam', 'overbet', 'shove', 'widejam'],
                   default='maxbet')
    p.add_argument('--policies', default='off,A,B',
                   help="comma list from off,A,B (default all three)")
    p.add_argument('--aivat', action='store_true',
                   help="Report AIVAT-corrected BB/hand (c1 preflop-equity + c3 all-in-EV "
                        "control variates) alongside raw -- cuts the preflop-stackoff variance.")
    args = p.parse_args()

    path = args.db or resolve_blueprint_path()
    policies = [p.strip() for p in args.policies.split(',') if p.strip()]
    print(f"Blueprint : {path}")
    print(f"Hands     : {args.hands}  (seed {args.seed})  style={args.style}  "
          f"policies={policies}")
    print("(bot BB/hand > 0 = bot beats the maniac; river-solve count gauges the EV signal)\n")

    estimator = aivat_db = None
    if args.aivat:
        from src.evaluation.aivat import AIVATEstimator
        aivat_db = BlueprintDB(path, read_only=True)
        estimator = AIVATEstimator(aivat_db, seed=args.seed)

    rows = []
    for name in policies:
        if name not in _POLICIES:
            print(f"  (skip unknown policy {name!r})")
            continue
        results, bot, wall, records = run_policy(path, args.hands, args.seed, args.style,
                                                 _POLICIES[name], aivat=args.aivat)
        n = len(results)
        mean = sum(results) / n
        se = math.sqrt(sum((r - mean) ** 2 for r in results) / n / n)
        rows.append((name, mean, se, bot, wall))
        print(f"[{name:>3}] raw {mean * 1000:+8.1f} +/- {se * 1000:5.1f} mbb/hand   "
              f"river-solves={len(bot.river_latencies):5d}   wall={wall:5.0f}s")
        if estimator is not None:
            est = estimator.estimate(records)
            print(f"      AIVAT {est['aivat_mbb']:+8.1f} +/- {est['aivat_stderr_mbb']:5.1f} "
                  f"mbb/hand  (var -{est['var_reduction'] * 100:.0f}%; raw se "
                  f"{est['raw_stderr_mbb']:.1f})")
        print(f"      anchors: {bot.anchor_counts or '{}'}")
        print(f"      river latency: {_fmt_latency(bot.river_latencies)}\n")
    if aivat_db is not None:
        aivat_db.close()

    if len(rows) > 1:
        base = rows[0]
        print(f"Deltas vs [{base[0]}] (BB/hand; positive = policy wins more chips):")
        for name, mean, se, _bot, _wall in rows[1:]:
            d = mean - base[1]
            ds = math.sqrt(se ** 2 + base[2] ** 2)
            print(f"  {name:>3} - {base[0]}: {d * 1000:+8.1f} +/- {ds * 1000:5.1f} mbb/hand")
    print("\nSafety (worst-case exploitability) is measured separately: "
          "python tests/test_safe_river_gadget.py")


if __name__ == '__main__':
    main()
