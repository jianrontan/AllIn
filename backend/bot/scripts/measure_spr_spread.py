# backend/bot/scripts/measure_spr_spread.py
"""
MEASUREMENT A -- SPR spread per postflop info-set key.

THE QUESTION. A postflop info-set key (coarse_strength_pos_street_pattern) does NOT
encode SPR (stack-to-pot ratio). So one key can be reached at many different real
SPRs -- a flop check-call line at 200bb deep and the same line 3-bet-pot-shallow map
to the SAME key and share ONE trained strategy. If those SPR regimes want
incompatible play (e.g. one wants to pot-bet, one wants to jam), the blueprint is
forced to average them -> a structural leak that no amount of training fixes (the
M1 limitation behind the over-jamming diagnosis).

WHAT THIS MEASURES. Replay self-play under the served blueprint (both players sample
their average strategy). At every POSTFLOP decision node, record the key, the REAL
SPR at that node, the street, and a visit count. Then per key summarise the SPR
distribution and flag keys whose SPR spread crosses "regime" boundaries -- the
decision mass living in such keys is the size of the prize for SPR bucketing.

This trains NOTHING and writes NOTHING to the blueprint -- it's a read-only diagnostic
on the EXISTING blueprint. Its output chooses K (how many SPR buckets) and which
streets, and can kill the SPR-bucket candidate configs (#3/#4) before any retrain if
the spread turns out to be small. See docs/ROADMAP.md "Candidate abstraction configs".

SPR DEFINITION. SPR = min(remaining stack over the two players) / pot, measured AT the
decision node (before the action), in chips. min-stack because the effective stack is
what's actually playable. We bucket SPR into regimes with solver-conventional cutpoints
(deep / mid / low / jam-zone); the cutpoints are reported so they can be tuned.

Run from backend/bot/:
    python scripts/measure_spr_spread.py --hands 20000
    python scripts/measure_spr_spread.py --hands 50000 --db analysis/blueprints/blueprint_par_20260529_233056.db
"""
import argparse
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import resolve_blueprint_path
from src.storage.blueprint_db import BlueprintDB
from src.cfr.poker_game import PokerGame, STARTING_STACK
from src.cfr.keys import make_info_set_key, action_char
from src.abstractions.card_abstractions import CardAbstraction

# SPR regime cutpoints (chips: min-remaining-stack / pot). Boundaries chosen at
# the SPRs where the abstract menu's character changes: above ~6 every pot-fraction
# bet is well below stack (rich menu); 3-6 large/overbet start brushing the stack;
# 1-3 medium+ collapse toward jam; below 1 nearly everything IS a jam. Tunable.
_SPR_EDGES = [1.0, 3.0, 6.0]
_SPR_LABELS = ['jam(<1)', 'low(1-3)', 'mid(3-6)', 'deep(>6)']


def _spr_regime(spr):
    """Index 0..3 of the SPR regime, by _SPR_EDGES."""
    for i, e in enumerate(_SPR_EDGES):
        if spr < e:
            return i
    return len(_SPR_EDGES)


def _street_name(street):
    return ['preflop', 'flop', 'turn', 'river'][street]


class SelfPlaySampler:
    """Walks one hand at a time, sampling BOTH players from the blueprint's average
    strategy, recording every postflop decision node's (key, SPR, street). Mirrors
    the trainer's node-state + key construction (poker_game + keys) so the keys and
    chip math are exactly what training/inference use."""

    def __init__(self, db):
        self.db = db
        # Build the engine under the SAME action abstraction the blueprint was
        # trained on (control vs capped) -- else a capped blueprint is sampled
        # through a control engine that re-introduces the voluntary all-in it never
        # trained, skewing the very SPR fingerprint this script measures.
        from src.abstractions.sizing import db_menu_mode, postflop_menu_for, is_capped_mode
        mm = db_menu_mode(db)
        if is_capped_mode(mm):
            self.game = PokerGame(postflop_menu=postflop_menu_for(mm),
                                  voluntary_allin=False)
        else:
            self.game = PokerGame()
        self.cards = CardAbstraction()
        self.deck = [s + r for r in
                     ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
                     for s in ['H', 'D', 'C', 'S']]
        # key -> {regime_idx: weighted_visits}, and key -> street
        self.key_spr = defaultdict(lambda: np.zeros(len(_SPR_LABELS)))
        self.key_street = {}
        self.n_decisions = 0
        self.n_postflop_decisions = 0

    def _strategy(self, key, legal):
        """Blueprint average strategy restricted to `legal` (uniform if untrained)."""
        stored = self.db.get_average_strategy(key)
        if stored:
            w = np.array([max(0.0, stored.get(a, 0.0)) for a in legal])
            t = w.sum()
            if t > 1e-12:
                return w / t
        return np.ones(len(legal)) / len(legal)

    def play_hand(self, rng):
        shuffled = self.deck[:]
        rng.shuffle(shuffled)
        p0_cards, p1_cards = shuffled[0:2], shuffled[2:4]
        community = shuffled[4:9]
        pf = {0: self.cards.get_bucket(p0_cards, None),
              1: self.cards.get_bucket(p1_cards, None)}
        self._walk(p0_cards, p1_cards, community, pf, rng)

    def _walk(self, p0_cards, p1_cards, community, pf, rng):
        """Iterative street-by-street walk: deal, then sample actions until the
        street closes, advancing to the next street, until the hand ends. Records
        each postflop decision node. (No regret/value math -- pure forward sampling.)"""
        street = 0
        history = []
        starting_pot = 3.0
        p0_inv, p1_inv = 0.0, 0.0
        p0_stack = STARTING_STACK - 1
        p1_stack = STARTING_STACK - 2
        st = self.game.init_node_state(street, starting_pot)
        bet_pattern = ''
        depth = 0

        while True:
            depth += 1
            if depth > 60 or street > 3:
                return
            if self.game.is_terminal(history, street):
                return
            cp = self.game._acting_player(len(history), street)
            stack_cp = p0_stack if cp == 0 else p1_stack
            legal = self.game.state_legal_actions(street, st, cp, stack_cp)
            if not legal:
                # Street closed -> advance, resetting per-street pattern + node state.
                if street < 3:
                    p0_this, p1_this = st['c'][0], st['c'][1]
                    p0_inv += p0_this
                    p1_inv += p1_this
                    p0_stack = STARTING_STACK - p0_inv
                    p1_stack = STARTING_STACK - p1_inv
                    street += 1
                    history = []
                    bet_pattern = ''
                    st = self.game.init_node_state(street, st['pot'])
                    continue
                return

            # -- build the key for this decision node --
            position = 'ip' if cp == 0 else 'oop'
            if street > 0:
                board = community[:street + 2]
                strength = self.cards.get_bucket(
                    p0_cards if cp == 0 else p1_cards, board)
            else:
                strength = None
            key = make_info_set_key(street, position, pf[cp], strength, bet_pattern)

            self.n_decisions += 1
            if street > 0:
                # -- REAL SPR at this node (chips): min remaining stack / pot --
                eff_stack = min(p0_stack, p1_stack)
                pot = st['pot']
                spr = (eff_stack / pot) if pot > 1e-9 else 0.0
                self.key_spr[key][_spr_regime(spr)] += 1.0
                self.key_street[key] = _street_name(street)
                self.n_postflop_decisions += 1

            # -- sample an action from the blueprint and advance --
            strat = self._strategy(key, legal)
            action = rng.choices(legal, weights=strat)[0]
            cost = self.game.state_action_cost(action, street, st, cp, stack_cp)
            st = self.game.advance_node_state(
                st, action, street, cp, stack_cp, (p0_inv, p1_inv))
            if cp == 0:
                p0_stack -= cost
            else:
                p1_stack -= cost
            history = history + [action]
            bet_pattern = bet_pattern + action_char(action)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hands', type=int, default=20000)
    ap.add_argument('--db', default=None, help="blueprint path (default: resolve active)")
    ap.add_argument('--seed', type=int, default=20260601)
    ap.add_argument('--min-visits', type=int, default=30,
                    help="ignore keys seen fewer than this many times (noise floor)")
    args = ap.parse_args()

    db_path = args.db or str(resolve_blueprint_path())
    print(f"Measurement A -- SPR spread per postflop key")
    print(f"  blueprint : {db_path}")
    print(f"  hands     : {args.hands:,}")
    print(f"  SPR edges : {_SPR_EDGES}  -> regimes {_SPR_LABELS}\n")

    db = BlueprintDB(db_path, read_only=True)
    sampler = SelfPlaySampler(db)
    rng = random.Random(args.seed)
    for i in range(args.hands):
        sampler.play_hand(rng)
        if (i + 1) % 5000 == 0:
            print(f"  ...{i + 1:,} hands  ({sampler.n_postflop_decisions:,} postflop decisions)")
    db.close()

    print(f"\nTotal decisions: {sampler.n_decisions:,} "
          f"({sampler.n_postflop_decisions:,} postflop)\n")

    # -- per-key analysis --
    # For each key: total visits, the SPR-regime distribution, the dominant regime's
    # share, the entropy of the regime distribution (how spread), and the fraction of
    # visits OUTSIDE the dominant regime ("crossover mass" = the averaging penalty).
    rows = []
    by_street_mass = defaultdict(float)
    by_street_cross = defaultdict(float)
    for key, counts in sampler.key_spr.items():
        total = counts.sum()
        if total < args.min_visits:
            continue
        share = counts / total
        dom = int(share.argmax())
        cross = float(1.0 - share[dom])               # mass not in the dominant regime
        # span = how many regimes hold >=10% of the mass (a coarse "incompatible
        # regimes blended" count: 1 = clean, >=2 = the key straddles regimes).
        span = int((share >= 0.10).sum())
        street = sampler.key_street[key]
        rows.append((key, street, total, share, dom, cross, span))
        by_street_mass[street] += total
        by_street_cross[street] += cross * total

    if not rows:
        print("No postflop keys cleared the visit floor -- raise --hands.")
        return

    total_mass = sum(r[2] for r in rows)
    # Decision mass in keys that STRADDLE regimes (span >= 2).
    straddle_mass = sum(r[2] for r in rows if r[6] >= 2)
    # Mass-weighted mean crossover (avg fraction of a key's visits outside its own
    # dominant SPR regime) -- the headline "how much is the key forced to average".
    wmean_cross = sum(r[5] * r[2] for r in rows) / total_mass

    # THE decision-driver: total decision mass in the COLLAPSE-PRONE SPR regimes
    # (jam<1 and low 1-3), summed over ALL analysed keys regardless of per-key
    # dominance. SPR card-buckets only earn their keep if a real slice of decisions
    # actually happen at low SPR AND those decisions live in keys that ALSO see deep
    # SPR (so the key is forced to blend). Two numbers:
    #   - low/jam mass overall: how often we even ARE at low SPR postflop.
    #   - low/jam mass that sits in a key whose mass is MOSTLY (>=60%) deep/mid:
    #     the low-SPR decisions being drowned out by a deep-dominated average -- the
    #     exact over-jamming-leak prize (a shove blended into a pot-bet key).
    regime_mass = np.zeros(len(_SPR_LABELS))
    drowned_low = 0.0
    for key, street, total, share, dom, cross, span in rows:
        regime_mass += share * total
        low_jam = share[0] + share[1]                 # jam(<1) + low(1-3)
        deep_mid = share[2] + share[3]
        if deep_mid >= 0.60 and low_jam > 0.0:
            drowned_low += low_jam * total            # low-SPR visits in a deep-dom key

    print("=" * 70)
    print("HEADLINE")
    print(f"  postflop keys analysed (>= {args.min_visits} visits) : {len(rows):,}")
    print(f"  decision mass in straddle keys (>=2 regimes >=10%)   : "
          f"{straddle_mass / total_mass:6.1%}")
    print(f"  mass-weighted crossover (avg visits off dom. regime) : {wmean_cross:6.1%}")
    print(f"  overall SPR-regime mass split  "
          f"[{' '.join(l.split('(')[0] for l in _SPR_LABELS)}] : "
          f"[{' '.join(f'{x / total_mass:.1%}' for x in regime_mass)}]")
    print(f"  low/jam decisions DROWNED in a deep-dominated key    : "
          f"{drowned_low / total_mass:6.1%}  <- the SPR-bucket prize")
    print("  -> high drowned-low = SPR bucketing targets the over-jam leak; low =")
    print("     the over-jam is better fixed by Fix #4 (menu cap), not SPR buckets.")

    print("\nBY STREET (where does the spread live?)")
    print(f"  {'street':<8}{'keys':>8}{'mass':>12}{'wtd crossover':>16}")
    for s in ('flop', 'turn', 'river'):
        ks = [r for r in rows if r[1] == s]
        if not ks:
            continue
        m = by_street_mass[s]
        wc = by_street_cross[s] / m if m else 0.0
        print(f"  {s:<8}{len(ks):>8}{m:>12,.0f}{wc:>15.1%}")

    print("\nWORST 15 KEYS BY STRADDLE (most incompatible SPR mix, by mass)")
    print(f"  {'key':<34}{'street':<7}{'visits':>8}  regime shares "
          f"[{' '.join(l.split('(')[0] for l in _SPR_LABELS)}]")
    worst = sorted([r for r in rows if r[6] >= 2], key=lambda r: -r[2])[:15]
    for key, street, total, share, dom, cross, span in worst:
        shares = ' '.join(f"{x:4.2f}" for x in share)
        print(f"  {key:<34}{street:<7}{int(total):>8}  [{shares}]")

    print("\nNote: tune _SPR_EDGES if the regimes don't match the menu's collapse "
          "points; this run used", _SPR_EDGES)
    print("=" * 70)


if __name__ == '__main__':
    main()
