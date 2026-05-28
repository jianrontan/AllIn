# backend/bot/tests/run_maniac.py
"""
Exploitability sanity check vs a NAIVE maniac, not a best-response.

The user reports the bot bleeds chips when a human just spams all-in /
bet-medium / raise-medium and never folds. A sound strategy should *profit*
hugely against such a player (call wide with strong hands, fold trash, let the
maniac stack off light). If the blueprint loses, that is a real, exploitable
leak -- the kind LBR/BR can miss because they end hands before showdown.

This reuses match.py's engine loop but swaps player B for a scripted maniac:
  - never folds (calls when it cannot raise);
  - when it can be aggressive, plays per --profile:
        jam     -> always all-in
        medium  -> always bet/raise medium
        mixed   -> 50/50 medium vs all-in   (closest to the reported human line)

Result is the BLUEPRINT's win rate in mbb/hand (>0 means the bot beats the
maniac, as it should). Raw (high-variance) -- use many hands.

Usage (from backend/bot/):
    python tests/run_maniac.py --db analysis/blueprints/snapshots/snap_7500000.db --hands 40000
    python tests/run_maniac.py --profile jam --hands 40000
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import resolve_blueprint_path
from src.storage.blueprint_db import BlueprintDB
from src.evaluation.match import (
    HeadToHeadMatch, BlueprintPlayer, _legal_actions, _sizing, _FULL_DECK)
from src.cfr.keys import action_char


class ManiacPlayer:
    """Never folds. Aggresses (per profile) whenever legal, else calls/checks."""

    def __init__(self, rng, profile='mixed'):
        self.rng = rng
        self.profile = profile

    def act(self, seat, hand, vis, street, pot, committed, stack, to_call, num_aggr, pattern):
        legal = _legal_actions(to_call, num_aggr, stack, pot)
        can_aggr = ('allin' in legal)  # implies sized bets/raises are legal too

        if can_aggr:
            if self.profile == 'jam':
                action = 'allin'
            elif self.profile == 'medium':
                action = 'raise_medium' if to_call > 0 else 'bet_medium'
            else:  # mixed
                med = 'raise_medium' if to_call > 0 else 'bet_medium'
                action = self.rng.choice([med, 'allin'])
        else:
            action = 'call' if to_call > 0 else 'check'  # never fold

        if action == 'check':
            char, add, aggr = 'k', 0, False
        elif action == 'call':
            char, add, aggr = 'c', to_call, False
        elif action == 'allin':
            char, add, aggr = 'a', stack, True
        else:
            size = action.split('_')[1]
            add = max(_sizing(size, street, pot, to_call, committed, num_aggr), to_call + 1)
            if add >= stack:
                char, add, aggr = 'a', stack, True
            else:
                char, add, aggr = action_char(action), add, True
        return char, add, aggr, {}


def run(blueprint, profile, hands, seed):
    match = HeadToHeadMatch(blueprint, blueprint, seed=seed)
    # Seat A = blueprint (the player we measure), seat B = maniac.
    match.pb = ManiacPlayer(match.rng, profile=profile)

    total = 0.0
    sq = 0.0
    for i in range(hands):
        c = match.rng.sample(_FULL_DECK, 9)
        r = match.play_hand(i % 2, (c[0], c[1]), (c[2], c[3]), c[4:9])
        total += r
        sq += r * r
    mean_chips = total / hands
    var_chips = max(0.0, sq / hands - mean_chips ** 2)
    mbb = mean_chips * 1000.0 / 2.0
    stderr = math.sqrt(var_chips / hands) * 1000.0 / 2.0
    return mbb, stderr


def main():
    p = argparse.ArgumentParser(description="Blueprint vs naive maniac.")
    p.add_argument('--db', default=None, help="Blueprint DB (default: active resolve).")
    p.add_argument('--profile', default='all',
                   choices=['jam', 'medium', 'mixed', 'all'])
    p.add_argument('--hands', type=int, default=40000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    path = args.db or resolve_blueprint_path()
    print(f"Blueprint : {path}")
    print(f"Hands     : {args.hands}  (seed {args.seed})  [blueprint's perspective]")
    print("(>0 = blueprint beats the maniac, as a sound strategy should)\n")

    bp = BlueprintDB(path, read_only=True)
    try:
        profiles = ['jam', 'medium', 'mixed'] if args.profile == 'all' else [args.profile]
        for prof in profiles:
            t0 = time.time()
            mbb, se = run(bp, prof, args.hands, args.seed)
            print(f"  {prof:7s}: {mbb:+9.1f}  +/- {se:6.1f} mbb/hand"
                  f"   ({time.time() - t0:.1f}s)")
    finally:
        bp.close()


if __name__ == '__main__':
    main()
