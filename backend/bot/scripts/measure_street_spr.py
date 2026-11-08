# backend/bot/scripts/measure_street_spr.py
"""
Phase 6.1 -- measure before adding SPR buckets (docs/PHASE6_ABSTRACTION_PLAN.md).

Self-play under a blueprint (both seats sample its average strategy; same engine,
menu mode and keys as training -- reuses measure_spr_spread.SelfPlaySampler) and
records, for every FLOP and TURN decision, the stack-to-pot ratio at the START of
that street (the quantity an SPR bucket would key on). Read-only; trains nothing.

Three outputs:
  1. Street-start SPR distribution, weighted by decisions AND by pot size at the
     decision (chips at stake) -- the 2026-06 Measurement A counted decisions only,
     which under-weights big pots.
  2. Reachable key counts: distinct flop/turn keys seen, and distinct
     (key, SPR bucket) pairs under candidate bucketings -> the real key multiplier.
  3. Facing-all-in probe (the BUG-027 fingerprint): at every postflop decision
     facing an all-in, the blueprint's call probability and the REALIZED value of
     calling (exact showdown vs the shover's actual cards and runout), by preflop
     pot type x hand-strength tercile. A key cannot see pot type, so v2 calls at the
     same rate in a 4-bet pot as in a single-raised pot; if calling loses chips in
     4-bet pots but not in SRPs, that loss is what SPR buckets can recover.

Run from backend/bot/:
    python scripts/measure_street_spr.py --hands 40000 --workers 10
"""
import argparse
import os
import random
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from measure_spr_spread import SelfPlaySampler  # noqa: E402
from src.storage.blueprint_db import BlueprintDB  # noqa: E402
from src.cfr.poker_game import STARTING_STACK  # noqa: E402
from src.cfr.keys import make_info_set_key, action_char, spr_bucket, db_key_format  # noqa: E402
from src.abstractions.hand_evaluator import HandEvaluator  # noqa: E402

# Fine SPR bins for the distribution (upper edges; last bin is open).
EDGES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
LABELS = ['<1', '1-2', '2-3', '3-4', '4-6', '6-8', '8-12', '12-16', '16-24', '24-32', '32-48', '>48']
# Candidate bucketings (upper edges).
CANDIDATES = {
    'K4 <2/2-4/4-8/>8': [2, 4, 8],
    'K5 <2/2-4/4-8/8-16/>16': [2, 4, 8, 16],
    'K3 <3/3-8/>8': [3, 8],
    # aligned to the preflop pot types seen in v2 self-play (limped >48, SRP 8-32,
    # 3-bet mostly 4-12, 4-bet+ mostly <4)
    'P5 <2/2-5/5-12/12-40/>40': [2, 5, 12, 40],
    'P4 <4/4-12/12-40/>40': [4, 12, 40],
}
POT_TYPES = ['limped', 'SRP', '3bet', '4bet+']
STRENGTH_K = {1: 30, 2: 24, 3: 10}      # served v2 per-street strength buckets
AGGR = set('smlxo2a')                    # pattern chars that bet/raise/jam


def _bin(x, edges):
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


def _pot_type(preflop_pattern):
    n = sum(1 for ch in preflop_pattern if ch in AGGR)
    return min(n, 3)


class StreetSprSampler(SelfPlaySampler):
    def __init__(self, db):
        super().__init__(db)
        # Build keys in the blueprint's own format (v2 or spr5) so an spr5 blueprint's
        # flop/turn lookups hit.
        self.key_format = db_key_format(db)
        self.ev = HandEvaluator()
        self.hands = 0
        self.dec = {s: np.zeros(len(LABELS)) for s in (1, 2)}         # decisions per SPR bin
        self.potw = {s: np.zeros(len(LABELS)) for s in (1, 2)}        # pot-weighted
        self.ptype_bin = {s: np.zeros((4, len(LABELS))) for s in (1, 2)}
        self.flop_hands = np.zeros(len(LABELS))                       # hands reaching the flop
        self.flop_final_pot = np.zeros(len(LABELS))                   # ... weighted by final pot
        self.keys = {s: set() for s in (1, 2)}
        self.keys_b = {name: {s: set() for s in (1, 2)} for name in CANDIDATES}
        # facing-all-in probe: (street, pot_type, tercile) -> [n, sum_p, sum_p_ev, sum_ev]
        self.jam = defaultdict(lambda: np.zeros(4))
        self.look = np.zeros(4)                    # blueprint lookups per street
        self.miss = np.zeros(4)                    # ... that found no trained key

    def play_hand(self, rng):
        self.hands += 1
        super().play_hand(rng)

    def _walk(self, p0_cards, p1_cards, community, pf, rng):
        street = 0
        history = []
        p0_inv, p1_inv = 0.0, 0.0
        p0_stack = STARTING_STACK - 1
        p1_stack = STARTING_STACK - 2
        st = self.game.init_node_state(street, 3.0)
        bet_pattern = ''
        preflop_pattern = ''
        spr0 = None
        street_pot = 3.0
        ptype = None
        flop_bin = None
        depth = 0
        hole = {0: p0_cards, 1: p1_cards}

        def finish():
            if flop_bin is not None:
                self.flop_hands[flop_bin] += 1
                self.flop_final_pot[flop_bin] += st['pot']

        while True:
            depth += 1
            if depth > 60 or street > 3 or self.game.is_terminal(history, street):
                return finish()
            cp = self.game._acting_player(len(history), street)
            stack_cp = p0_stack if cp == 0 else p1_stack
            legal = self.game.state_legal_actions(street, st, cp, stack_cp)
            if not legal:
                if street < 3:
                    p0_inv += st['c'][0]
                    p1_inv += st['c'][1]
                    p0_stack = STARTING_STACK - p0_inv
                    p1_stack = STARTING_STACK - p1_inv
                    if street == 0:
                        preflop_pattern = bet_pattern
                        ptype = _pot_type(preflop_pattern)
                    street += 1
                    history = []
                    bet_pattern = ''
                    st = self.game.init_node_state(street, st['pot'])
                    street_pot = st['pot']
                    spr0 = min(p0_stack, p1_stack) / st['pot']
                    if street == 1:
                        flop_bin = _bin(spr0, EDGES)
                    continue
                return finish()

            position = 'ip' if cp == 0 else 'oop'
            strength = None
            if street > 0:
                strength = self.cards.get_bucket(hole[cp], community[:street + 2])
            key = make_info_set_key(street, position, pf[cp], strength, bet_pattern,
                                    spr_bucket(self.key_format, street, street_pot))
            self.look[street] += 1
            if not self.db.get_average_strategy(key):
                self.miss[street] += 1
            strat = self._strategy(key, legal)

            if street in (1, 2):
                b = _bin(spr0, EDGES)
                self.dec[street][b] += 1
                self.potw[street][b] += st['pot']
                self.ptype_bin[street][ptype, b] += 1
                self.keys[street].add(key)
                for name, edges in CANDIDATES.items():
                    self.keys_b[name][street].add((key, _bin(spr0, edges)))

            # facing-all-in probe
            opp_stack = p1_stack if cp == 0 else p0_stack
            to_call = st['bet_to'] - st['c'][cp]
            if street > 0 and to_call > 1e-9 and opp_stack <= 1e-9 and 'call' in legal:
                call_amt = min(to_call, stack_cp)
                contested = st['pot'] - (to_call - call_amt) + call_amt
                mine = self.ev.get_raw_hand_value(hole[cp], community)
                theirs = self.ev.get_raw_hand_value(hole[1 - cp], community)
                share = 1.0 if mine < theirs else (0.5 if mine == theirs else 0.0)
                ev_call = share * contested - call_amt
                p = float(strat[legal.index('call')])
                terc = min(2, strength * 3 // STRENGTH_K[street])
                self.jam[(street, ptype, terc)] += np.array([1.0, p, p * ev_call, ev_call])

            action = rng.choices(legal, weights=strat)[0]
            cost = self.game.state_action_cost(action, street, st, cp, stack_cp)
            st = self.game.advance_node_state(st, action, street, cp, stack_cp, (p0_inv, p1_inv))
            if cp == 0:
                p0_stack -= cost
            else:
                p1_stack -= cost
            history = history + [action]
            bet_pattern = bet_pattern + action_char(action)

    def result(self):
        return {
            'hands': self.hands, 'dec': self.dec, 'potw': self.potw,
            'ptype_bin': self.ptype_bin, 'flop_hands': self.flop_hands,
            'flop_final_pot': self.flop_final_pot, 'keys': self.keys,
            'keys_b': self.keys_b, 'jam': dict(self.jam),
            'look': self.look, 'miss': self.miss, 'key_format': self.key_format,
        }


def _work(args):
    db_path, hands, seed = args
    db = BlueprintDB(db_path, read_only=True)
    s = StreetSprSampler(db)
    rng = random.Random(seed)
    for _ in range(hands):
        s.play_hand(rng)
    db.close()
    return s.result()


def _merge(parts):
    out = parts[0]
    for r in parts[1:]:
        out['hands'] += r['hands']
        for s in (1, 2):
            out['dec'][s] += r['dec'][s]
            out['potw'][s] += r['potw'][s]
            out['ptype_bin'][s] += r['ptype_bin'][s]
            out['keys'][s] |= r['keys'][s]
            for name in CANDIDATES:
                out['keys_b'][name][s] |= r['keys_b'][name][s]
        out['look'] += r['look']
        out['miss'] += r['miss']
        out['flop_hands'] += r['flop_hands']
        out['flop_final_pot'] += r['flop_final_pot']
        for k, v in r['jam'].items():
            out['jam'][k] = out['jam'].get(k, np.zeros(4)) + v
    return out


def _pct_row(v):
    t = v.sum()
    return ' '.join(f'{(x / t if t else 0):6.1%}' for x in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='analysis/blueprints/blueprint_final_v2.db')
    ap.add_argument('--hands', type=int, default=40000)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--seed', type=int, default=20261109)
    a = ap.parse_args()

    per = [a.hands // a.workers + (1 if i < a.hands % a.workers else 0) for i in range(a.workers)]
    with Pool(a.workers) as pool:
        parts = pool.map(_work, [(a.db, n, a.seed + 7919 * i) for i, n in enumerate(per) if n])
    r = _merge(parts)
    H = r['hands']

    print(f'Phase 6.1 street-start SPR measurement -- {a.db}, {H:,} self-play hands\n')
    print(f"   key format {r['key_format']}; blueprint lookups found: "
          + ', '.join(f"{['preflop', 'flop', 'turn', 'river'][s]} "
                      f"{1 - r['miss'][s] / max(r['look'][s], 1):.2%}" for s in range(4)) + '\n')
    print('1. STREET-START SPR (eff. stack / pot when the street begins)')
    print(f"   {'bin':<10}" + ''.join(f'{lbl:>7}' for lbl in LABELS))
    for s, name in ((1, 'flop'), (2, 'turn')):
        print(f'   {name + " dec":<10}' + ''.join(f'{x:>7}' for x in _pct_row(r['dec'][s]).split()))
        print(f'   {name + " pot":<10}' + ''.join(f'{x:>7}' for x in _pct_row(r['potw'][s]).split()))
    print(f"   {'flop hand':<10}" + ''.join(f'{x:>7}' for x in _pct_row(r['flop_hands']).split()))
    print(f"   {'final pot':<10}" + ''.join(f'{x:>7}' for x in _pct_row(r['flop_final_pot']).split()))
    print('   (dec = share of decisions; pot = weighted by pot at the decision; flop hand = hands\n'
          '    reaching the flop by flop SPR; final pot = those hands weighted by final pot size)')

    print('\n   Flop decisions by preflop pot type x flop SPR bin (row shares):')
    for t in range(4):
        row = r['ptype_bin'][1][t]
        if row.sum():
            print(f'   {POT_TYPES[t]:<8} n={int(row.sum()):>7,}  ' + _pct_row(row))

    print('\n2. REACHABLE KEYS (distinct keys seen in self-play)')
    for s, name in ((1, 'flop'), (2, 'turn')):
        base = len(r['keys'][s])
        cand = '  '.join(f"{c.split()[0]}: {len(r['keys_b'][c][s]):,} ({len(r['keys_b'][c][s]) / base:.2f}x)"
                         for c in CANDIDATES)
        print(f'   {name}: {base:,} keys -> {cand}')

    print('\n3. FACING AN ALL-IN (postflop): blueprint call % and realized value of calling')
    print('   EV(call) = exact showdown value of calling vs folding, chips; mbb/hand = '
          'p(call) x EV(call) summed over all hands / hands played')
    print(f"   {'street':<7}{'pot type':<9}{'strength':<9}{'n':>7}{'call %':>8}{'EV(call)':>10}{'mbb/hand':>10}")
    terc = ['weak', 'medium', 'strong']
    for key in sorted(r['jam']):
        n, sp, spev, sev = r['jam'][key]
        if n < 30:
            continue
        street, t, k = key
        print(f"   {['', 'flop', 'turn', 'river'][street]:<7}{POT_TYPES[t]:<9}{terc[k]:<9}{int(n):>7,}"
              f"{sp / n:>8.0%}{sev / n:>10.1f}{spev / H * 500:>10.1f}")
    print('   (mbb/hand: chips -> mbb at BB = 2 chips; negative = calling there loses money)')


if __name__ == '__main__':
    main()
