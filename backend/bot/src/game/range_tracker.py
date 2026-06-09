# backend/bot/src/game/range_tracker.py
"""
Hand-level Bayesian opponent range tracker (Phase 3).

WHAT IT IS
----------
A belief over the OPPONENT's two hole cards, from the perspective of a player
who knows their own cards. It starts uniform over every two-card combo the
opponent could hold (the 50 cards not in the hero's hand) and is refined as the
hand plays out:

  * reveal(board)  -- zero combos that collide with dealt board cards (the
                      opponent cannot hold a card that is on the board), and
  * observe(...)   -- the Bayesian update: when the opponent is seen taking
                      action `a`, each candidate hand's weight is multiplied by
                      the blueprint probability of `a` for that hand's info set.

This is the mirror of evaluation/lbr.py's BotRange (which tracks the *bot's*
range from an exploiter's view); here the live bot tracks the *opponent's*
range, which is what a subgame solver consumes. The two SHOULD eventually share
this class -- BotRange predates it and is left untouched for now so the
validated LBR harness is not disturbed (migrate later, like keys.py).

CONFIDENCE
----------
A scalar in [0, 1], starting at 1.0, that measures how well the opponent's
observed actions match the model (the blueprint) we use to update the belief.
On each observed action we compare its surprise (-log p_a, where p_a is the
range-averaged blueprint probability of the action) to the model's own entropy
H of that node's action distribution:

    confidence *= exp( -max(0, (-log p_a) - H) )

Rationale: a Nash-ish blueprint is MIXED, so even perfectly on-model play won't
pick the single most-likely action -- penalising by "did they pick the modal
action" would wrongly punish correct mixing. Comparing surprise to entropy is
the right yardstick: on-model play has surprise ~= H (no decay); an action the
model deemed near-impossible (a human deviating, or an off-tree bet that maps to
a rarely-played abstract action) has surprise >> H, so confidence collapses.
A consumer falls back to equity/blueprint play once confidence drops below a
threshold instead of trusting a stale belief.

The model strategy is supplied as a callback `strategy_fn(key, legal_tuple) ->
np.ndarray over legal` so this module depends on neither the DB nor Flask, and
is fully unit-testable. State is JSON-serialisable via to_dict / from_dict.
"""
import random
from itertools import combinations

import numpy as np

from ..cfr.keys import make_info_set_key
from ..abstractions.postflop_features import rank7

_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
_SUITS = ['H', 'D', 'C', 'S']
_FULL_DECK = [s + r for r in _RANKS for s in _SUITS]

# Confidence multiplier for an OFF-MENU opponent action (one the abstract model has
# no column for -- a custom all-in, a far-off-grid bet). Such an action is maximally
# off-model: the model assigns it ~0 probability, so the confidence factor
# exp(-(surprise - H)) -> 0 in the limit. We apply a strong finite collapse (well below
# any consumer's trust threshold, e.g. the guards' guard_confidence=0.2) rather than a
# hard zero, so repeated weirdness still compounds. See BUG-022.
_OFF_MENU_CONFIDENCE_DECAY = 0.1


class RangeTracker:
    def __init__(self, hero_hole, cards_abstraction):
        """hero_hole: the knowing player's two SuitRank cards (e.g. ('HA','DK')),
        removed from the opponent's possible combos. cards_abstraction: a
        CardAbstraction used to bucket opponent hands the same way training did."""
        self.cards = cards_abstraction
        self.hero_hole = tuple(hero_hole)
        dead = set(self.hero_hole)
        self.hands = [h for h in combinations(_FULL_DECK, 2)
                      if h[0] not in dead and h[1] not in dead]
        self.w = np.ones(len(self.hands))
        self.confidence = 1.0
        self._bucket_cache = {}   # (street, board_tuple) -> (pf_list, strength_list)

    # -- card removal --------------------------------------------------
    def reveal(self, board):
        """Zero the weight of every opponent hand that collides with the board."""
        bset = set(board)
        for i, h in enumerate(self.hands):
            if h[0] in bset or h[1] in bset:
                self.w[i] = 0.0

    # -- bucketing (one blueprint key per distinct bucket, like BotRange) ---
    def _buckets(self, street, board):
        ck = (street, tuple(board))
        cached = self._bucket_cache.get(ck)
        if cached is not None:
            return cached
        pf = [None] * len(self.hands)
        strength = [None] * len(self.hands)
        slice_board = None if street == 0 else board[:2 + street]  # flop=3,turn=4,river=5
        for i, h in enumerate(self.hands):
            if self.w[i] <= 0.0:
                continue
            hl = list(h)
            pf[i] = self.cards.get_bucket(hl, None)
            if street > 0:
                strength[i] = self.cards.get_bucket(hl, slice_board)
        self._bucket_cache[ck] = (pf, strength)
        return pf, strength

    def _action_matrix(self, strategy_fn, street, position, bet_pattern, legal, board):
        """[n_hands, n_legal] model probabilities per opponent hand at this node.
        Hands sharing an info-set key share a row, so the model is queried once
        per distinct bucket key."""
        n = len(self.hands)
        legal_t = tuple(legal)
        mat = np.zeros((n, len(legal_t)))
        pf, strength = self._buckets(street, board)
        seen = {}
        for i in range(n):
            if self.w[i] <= 0.0:
                continue
            gid = pf[i] if street == 0 else (pf[i], strength[i])
            row = seen.get(gid)
            if row is None:
                if street == 0:
                    key = make_info_set_key(0, position, pf[i], None, bet_pattern)
                else:
                    key = make_info_set_key(street, position, pf[i], strength[i], bet_pattern)
                row = strategy_fn(key, legal_t)
                seen[gid] = row
            mat[i] = row
        return mat

    # -- Bayesian update + confidence ---------------------------------
    def observe(self, strategy_fn, action, street, position, bet_pattern, legal, board):
        """Condition the belief on the opponent having taken `action` here, and
        update confidence by how well that action matched the model."""
        legal = list(legal)
        if action not in legal:
            # Off-MENU action: an emergent/custom all-in (or far-off-grid bet) that the
            # node's ABSTRACT legal set doesn't list. E.g. a capped/deep-stack river jam:
            # with voluntary_allin=False, 'allin' enters legal only via stack-clamp, so a
            # player who instead makes a voluntary custom raise-to-stack has it normalized
            # to 'allin' by apply_action -- which is then NOT in `legal`. The opponent
            # model has no column for it, so we can't do the Bayesian reweight; keep the
            # prior range rather than crash (this previously raised ValueError and 500'd
            # the live hand). Conditioning on such actions would need the model to expose
            # an all-in probability at these nodes (future work).
            #
            # BUT the action is MAXIMALLY off-model, so we MUST still collapse confidence:
            # leaving it untouched is what let the bot hold a UNIFORM, never-updated belief
            # at 100% confidence and trust "opponent jams any-two", calling off 100 BB with
            # T8o vs a real (strong) jam range (BUG-022). Drop confidence below the guards'
            # trust threshold so consumers fall back to a no-read default.
            self.confidence *= _OFF_MENU_CONFIDENCE_DECAY
            return
        ai = legal.index(action)
        mat = self._action_matrix(strategy_fn, street, position, bet_pattern, legal, board)

        wt = self.w.sum()
        if wt > 1e-12:
            avg = (self.w[:, None] * mat).sum(axis=0) / wt   # range-averaged dist over legal
            avg_sum = avg.sum()
            if avg_sum > 1e-12:
                avg = avg / avg_sum
                p_a = float(avg[ai])
                entropy = float(-np.sum(avg * np.log(avg + 1e-12)))
                surprise = -np.log(p_a + 1e-12)
                # Cap excess-surprise so a single legal-but-near-zero-probability
                # action can't collapse confidence in one shot. Without this, a
                # ~0-prob legal action gives surprise ~= 27.6 (from the 1e-12
                # floor) while entropy is bounded by log(|legal|) <= ~1.4 — one
                # observation multiplies confidence by exp(-26) ~= 5e-12. Cap the
                # per-observation decay at exp(-ln(10)) = 0.1 (one off-model
                # action drops confidence by 10x at most; recurrence still
                # collapses, but no single action can blind the bot).
                excess = min(max(0.0, surprise - entropy), float(np.log(10.0)))
                self.confidence *= float(np.exp(-excess))

        new_w = self.w * mat[:, ai]
        s = new_w.sum()
        if s > 1e-12:
            self.w = new_w / s          # renormalise to keep weights from underflowing
        # else: the observed action has ~zero model-probability across EVERY live
        # hand (the opponent is off-model). Applying this update would zero the
        # belief permanently and blind the bot (hero_equity -> 0.5 forever). Keep
        # the prior range instead; the confidence drop computed above already
        # records that the action was unexpected, and the consumer falls back to
        # blueprint/equity when confidence is low.

    # -- accessors -----------------------------------------------------
    def weighted_hands(self):
        """[(hand, weight), ...] for hands with positive weight (unnormalised)."""
        return [(h, float(w)) for h, w in zip(self.hands, self.w) if w > 0.0]

    def normalized_weights(self):
        """Weights as a probability distribution (sums to 1), or all-zero if dead."""
        s = self.w.sum()
        return (self.w / s) if s > 1e-12 else self.w.copy()

    def top_hands(self, k=10):
        """The k most-likely opponent hands as [(hand, prob), ...], descending."""
        probs = self.normalized_weights()
        order = np.argsort(probs)[::-1][:k]
        return [(self.hands[i], float(probs[i])) for i in order if probs[i] > 0.0]

    # -- equity of a known hand vs the believed range -----------------
    def hero_equity(self, hero_hand, board, n_runouts=120, rng=None):
        """P(hero wins) + 0.5 P(tie) for `hero_hand` against the current weighted
        belief, integrated over board runouts. River -> exact; turn -> all rivers;
        flop/preflop -> Monte Carlo over `n_runouts` completions. Card removal
        excludes opponent hands colliding with the hero, the board, or the runout.
        Returns 0.5 if the belief has no live mass (degenerate)."""
        rng = rng or random.Random()
        hero = list(hero_hand)
        dead = set(hero) | set(board)
        undealt = [c for c in _FULL_DECK if c not in dead]
        need = 5 - len(board)
        if need <= 0:
            runouts = [()]
        elif need == 1:
            runouts = [(c,) for c in undealt]
        else:
            runouts = [tuple(rng.sample(undealt, need)) for _ in range(n_runouts)]

        live = [(h, w) for h, w in zip(self.hands, self.w)
                if w > 0.0 and h[0] not in dead and h[1] not in dead]
        if not live:
            return 0.5

        total_eq = 0.0
        total_ro = 0
        for ro in runouts:
            roset = set(ro)
            full = board + list(ro)
            hr = rank7(hero + full)
            eq = wt = 0.0
            for h, w in live:
                if h[0] in roset or h[1] in roset:
                    continue                       # runout makes this hand impossible
                orr = rank7([h[0], h[1]] + full)
                if hr < orr:                       # lower rank = stronger -> hero wins
                    eq += w
                elif hr == orr:
                    eq += 0.5 * w
                wt += w
            if wt > 0.0:
                total_eq += eq / wt
                total_ro += 1
        return total_eq / total_ro if total_ro else 0.5

    # -- serialisation (JSON-friendly for GameSession state) ----------
    def to_dict(self):
        """Compact, JSON-serialisable state. Hands are rebuilt from hero_hole."""
        return {
            'hero_hole': list(self.hero_hole),
            'w': self.w.tolist(),
            'confidence': self.confidence,
        }

    @classmethod
    def from_dict(cls, d, cards_abstraction):
        t = cls(d['hero_hole'], cards_abstraction)
        w = np.asarray(d['w'], dtype=float)
        if len(w) != len(t.hands):
            raise ValueError(
                f"serialised weight vector length {len(w)} != hand count {len(t.hands)}")
        t.w = w
        t.confidence = float(d['confidence'])
        return t
