# backend/bot/src/evaluation/aivat.py
"""
AIVAT-style variance reduction for head-to-head match results.

THE IDEA (control variates)
---------------------------
The raw per-hand result is dominated by luck: which hole cards you were dealt,
and which board cards came. We subtract baselines that are correlated with that
luck but have KNOWN expected value, so the corrected estimator has the same mean
(unbiased) but far lower variance. Each control variate here has true mean ZERO,
so for ANY scaling beta the estimator mean is unchanged -- beta is then chosen by
regression purely to minimise variance.

Three control variates (the biggest luck sources):
  c1  preflop-hand-equity luck: equity of A's exact starting hand vs a random
      hand (the precomputed preflop table), demeaned by its true average. Removes
      the "I was dealt aces / I was dealt 72o" swing.
  c2  river-runout luck: at the river chance node (for hands that reach river
      betting), pot * (equity(realized river) - mean equity over all rivers),
      where equity is A's exact hand vs B's RECONSTRUCTED range on the turn board.
      Its conditional mean is exactly 0 by construction, so its overall mean is 0.
  c3  all-in-runout luck (the dominant term in practice): when betting locks
      all-in before the river, both hole cards are known, so we subtract
      realized - (exact equity of A vs B over remaining boards) * pot. This is
      the classic all-in EV adjustment and removes the biggest variance bucket.

AIVAT_value_i = raw_i - X_i . beta,  X = [c1 - E[c1],  c2,  c3],  beta = argmin var.
Unbiased because E[X]=0; lower variance because X correlates with raw.

This is AIVAT's chance-correction core with on-the-fly equity baselines. (Action
control variates and a learned value function would tighten it further; this
already removes the dominant card luck.) Consumes the trajectories recorded by
match.HeadToHeadMatch(..., record=[...]).
"""
from itertools import combinations

import numpy as np

from .lbr import LBREvaluator, BotRange, _FULL_DECK
from ..abstractions.card_abstractions import _PREFLOP_EQUITY


def _combos(handstr):
    if len(handstr) == 2:      # pair
        return 6
    return 4 if handstr[2] == 's' else 12


class AIVATEstimator:
    def __init__(self, strat_b, seed=None):
        # Reuse LBR's equity_vs_range, restricted_probs, _raw_strategy, cards.
        self.lbr = LBREvaluator(strat_b, seed=seed)
        self.cards = self.lbr.cards
        # Guard: the preflop control variate's true mean assumes all 169
        # canonical starting hands are present; a missing key would make
        # _preflop_eq fall back to 0.5 and bias the CV.
        assert len(_PREFLOP_EQUITY) == 169, (
            f"_PREFLOP_EQUITY has {len(_PREFLOP_EQUITY)} hands, expected 169")
        # True mean of the preflop-equity control variate (combo-weighted).
        tot = sum(_PREFLOP_EQUITY[k] * _combos(k) for k in _PREFLOP_EQUITY)
        n = sum(_combos(k) for k in _PREFLOP_EQUITY)
        self.Ec1 = tot / n

    # ------------------------------------------------------------------
    def _preflop_eq(self, hand):
        return _PREFLOP_EQUITY.get(self.cards.cards_to_string(list(hand)), 0.5)

    def _allin_equity(self, hand_a, hand_b, known_board, max_samples=600):
        """A's exact equity vs B's KNOWN hand over runouts from `known_board`.
        Exact on flop (990 turn+river) and turn (46 rivers); sampled preflop."""
        ev = self.lbr.evaluator
        dead = set(hand_a) | set(hand_b) | set(known_board)
        deck = [c for c in _FULL_DECK if c not in dead]
        need = 5 - len(known_board)
        if need <= 0:
            runouts = [()]
        elif need <= 2:
            runouts = list(combinations(deck, need))
        else:
            runouts = [tuple(self.lbr.rng.sample(deck, need)) for _ in range(max_samples)]
        win = tie = 0
        for ro in runouts:
            full = known_board + list(ro)
            ra = ev.get_raw_hand_value(list(hand_a), full)
            rb = ev.get_raw_hand_value(list(hand_b), full)
            if ra < rb:
                win += 1
            elif ra == rb:
                tie += 1
        n = len(runouts)
        return (win + 0.5 * tie) / n if n else 0.5

    def _river_correction(self, hand_a, hand_b, turn_board, river_card, brange, pot):
        """pot * (equity at realized river - mean equity over all possible rivers).
        The river support excludes BOTH players' hole cards (the true chance
        support), so the control variate's conditional mean is exactly zero."""
        dead = set(turn_board) | set(hand_a) | set(hand_b)
        deck = [c for c in _FULL_DECK if c not in dead]
        eqs = []
        realized = None
        for r in deck:
            eq = self.lbr.equity_vs_range(hand_a, turn_board + [r], brange.hands, brange.w)
            eqs.append(eq)
            if r == river_card:
                realized = eq
        if realized is None:                 # river collided with dead set (shouldn't)
            return 0.0
        return pot * (realized - sum(eqs) / len(eqs))

    def _hand_variates(self, rec):
        """Return (c1, c2, c3) control variates for one recorded hand.
        c1 = preflop-hand-equity luck, c2 = river-runout luck,
        c3 = all-in-runout luck (realized - expected over remaining boards)."""
        b_seat = 1 - rec['seat_of_A']
        pos_b = 'ip' if b_seat == 0 else 'oop'
        hand_a = rec['hand_a']
        board = rec['board']

        # c2 (river-runout luck) needs B's range on the TURN board. Two sources:
        #   * 'river_range' (LIVE GameSession path): a snapshot of the bot's OWN on-model
        #     RangeTracker belief about B at river entry -- duck-types as brange (.hands/.w).
        #     More accurate than replaying LBR's BotRange (which can drift from the deployed
        #     tracker; BUG-008), so prefer it when present.
        #   * 'events' (capped match.py path): reconstruct B's range by replaying the
        #     recorded street-starts + B's actions through LBR's BotRange.
        # Either way the control variate's conditional mean is exactly 0 (realized minus
        # the same-range mean over rivers), so c2 stays UNBIASED for any fixed range.
        c2 = 0.0
        rr = rec.get('river_range')
        if rr is not None:
            c2 = self._river_correction(hand_a, rec['hand_b'], rr['turn_board'],
                                        rr['river_card'], rr['range'], rr['pot'])
        elif rec.get('events'):
            brange = BotRange(hand_a, self.cards)   # B's hands exclude A's cards
            for ev in rec['events']:
                if ev['type'] == 'street_start':
                    st = ev['street']
                    if st == 3:
                        # River chance node: B's range is current (revealed through the
                        # turn, updated through turn actions) and the river is not yet
                        # dealt. Compute the correction, then stop.
                        c2 = self._river_correction(
                            hand_a, rec['hand_b'], board[:4], board[4], brange, ev['pot'])
                        break
                    if st > 0:
                        brange.reveal(ev['vis'])
                elif ev['type'] == 'action' and ev['seat'] == b_seat:
                    action = ev['legal'][ev['choice']]
                    brange.observe(self.lbr.restricted_probs, action, ev['street'],
                                   pos_b, ev['pattern'], ev['legal'], ev['vis'])

        c1 = self._preflop_eq(hand_a)

        # c3: all-in runout adjustment. When betting locked all-in before the
        # river, both hole cards are known -> exact equity over remaining boards.
        c3 = 0.0
        ast = rec.get('allin_street')
        if ast is not None and rec['folded'] is None:
            known_board = [] if ast == 0 else board[:2 + ast]
            if len(known_board) < 5:
                eq = self._allin_equity(hand_a, rec['hand_b'], known_board)
                pot = sum(rec['invested'])
                inv_a = rec['invested'][rec['seat_of_A']]
                c3 = rec['result'] - (eq * pot - inv_a)   # realized - all-in EV
        return c1, c2, c3

    # ------------------------------------------------------------------
    def estimate(self, records, progress_every=0):
        raw = np.array([r['result'] for r in records], dtype=float)
        c1 = np.empty(len(records))
        c2 = np.empty(len(records))
        c3 = np.empty(len(records))
        for i, rec in enumerate(records):
            c1[i], c2[i], c3[i] = self._hand_variates(rec)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  aivat hand {i + 1}/{len(records)}", flush=True)

        # Control-variate columns with TRUE mean zero.
        X = np.column_stack([c1 - self.Ec1, c2, c3])
        beta, *_ = np.linalg.lstsq(X, raw, rcond=None)   # variance-minimising scaling
        aivat = raw - X @ beta

        to_mbb = 1000.0 / 2.0
        n = len(raw)
        return {
            'num_hands': n,
            'raw_mbb': raw.mean() * to_mbb,
            'raw_stderr_mbb': raw.std(ddof=1) / np.sqrt(n) * to_mbb,
            'aivat_mbb': aivat.mean() * to_mbb,
            'aivat_stderr_mbb': aivat.std(ddof=1) / np.sqrt(n) * to_mbb,
            'var_reduction': 1.0 - aivat.var() / raw.var(),
            'beta': beta.tolist(),
        }
