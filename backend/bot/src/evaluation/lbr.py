# backend/bot/src/evaluation/lbr.py
"""
Local Best Response (LBR) -- Lisy & Bowling 2017.

WHAT IT MEASURES
----------------
A LOWER bound on exploitability that the in-abstraction best response
(best_response.py) cannot see. Our BR attacks the blueprint only inside its own
bucket/bet-size abstraction. A real exploiter is not so constrained: it can bet
off-tree sizes the blueprint never trained on and play its EXACT hand. LBR is a
cheap, greedy such exploiter. The value LBR wins (mbb/hand) is a lower bound on
true exploitability; the gap above the in-abstraction BR number is a direct
read on how leaky the abstraction is.

HOW LBR PLAYS (the "Local" in LBR)
----------------------------------
LBR plays real hands against the blueprint. It does NOT solve a full game tree.
At each of its own decisions it considers a fixed menu of actions (fold, check,
call, and several bet sizes INCLUDING off-tree ones) and ranks them by a cheap
estimate: "take this action, then assume the rest of the hand is checked down to
showdown" (the wprollout). It then plays the argmax. Because it never looks ahead
at its own future betting, it is cheap -- and because it sometimes leaves money
on the table, the winnings are a lower bound, never an over-estimate.

To pick actions it needs a belief over the blueprint's hand: the bot's RANGE,
tracked by Bayesian updates from the blueprint's action probabilities (range
tracker, added in a later stage). Equity of LBR's exact hand vs that range,
integrated over board runouts, is the core primitive built here.

This module is assembled in stages:
  1. equity_vs_range  -- LBR exact hand vs the bot's weighted range (THIS FILE).
  2. range tracker     -- Bayesian range updates + blueprint key lookup.
  3. betting/decision  -- off-tree action menu + wprollout action ranking.
  4. main loop / CLI   -- Monte Carlo over deals, report mbb/hand.
"""
import random
from itertools import combinations

import numpy as np

from ..cfr.poker_game import PokerGame, STARTING_STACK
from ..cfr.keys import make_info_set_key, action_char
from ..cfr import translation
from ..abstractions.sizing import (
    preflop_open_chips, PREFLOP_RAISE_MULT, POSTFLOP_BET_MULT, SIZE_CHAR)
from ..abstractions.card_abstractions import CardAbstraction
from ..abstractions.action_abstractions import ActionAbstraction
from ..abstractions.hand_evaluator import HandEvaluator

SB, BB = 1, 2
MAX_AGGR_PER_STREET = 3   # 1 bet + 2 raises, matching PokerGame.max_raises_per_street
# LBR's bet menu as fractions of (pot + amount-to-call). 0.5/1.5 are OFF-TREE
# (the blueprint trains 0.33/0.66/1.0); 1.0 overlaps. Plus an explicit all-in.
LBR_BET_FRACTIONS = [0.5, 1.0, 1.5]

_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
_SUITS = ['H', 'D', 'C', 'S']
_FULL_DECK = [s + r for r in _RANKS for s in _SUITS]


class LBREvaluator:
    def __init__(self, blueprint_db, seed=None, flop_runout_samples=120):
        self.db = blueprint_db
        self.game = PokerGame()
        self.cards = CardAbstraction()
        self.actions = ActionAbstraction()
        self.evaluator = HandEvaluator()
        self.rng = random.Random(seed)
        self._base_seed = seed
        self.flop_runout_samples = flop_runout_samples

        # Memoized blueprint lookups (same pattern as best_response.py).
        self._strategy_cache = {}     # key -> {action: prob} or None
        self._restricted_cache = {}   # (key, legal_tuple) -> np.array over legal

    # ------------------------------------------------------------------
    # Blueprint strategy access (memoized) -- how the bot responds.
    # ------------------------------------------------------------------

    def _raw_strategy(self, key):
        cached = self._strategy_cache.get(key, 0)
        if cached == 0:
            cached = self.db.get_average_strategy(key)
            self._strategy_cache[key] = cached
        return cached

    def restricted_probs(self, key, legal):
        """Blueprint probabilities over `legal`, renormalised. Uniform if unknown."""
        cache_key = (key, legal)
        cached = self._restricted_cache.get(cache_key)
        if cached is not None:
            return cached
        stored = self._raw_strategy(key)
        n = len(legal)
        if stored:
            w = np.array([max(0.0, stored.get(a, 0.0)) for a in legal])
            total = w.sum()
            probs = w / total if total > 1e-12 else np.ones(n) / n
        else:
            probs = np.ones(n) / n
        self._restricted_cache[cache_key] = probs
        return probs

    # ------------------------------------------------------------------
    # Stage 1: equity of LBR's exact hand vs the bot's weighted range
    # ------------------------------------------------------------------

    def equity_vs_range(self, lbr_hand, board, range_hands, range_w):
        """
        LBR's pot-share equity (P(win) + 0.5*P(tie)) holding `lbr_hand` against a
        weighted opponent range, integrated over board runouts.

        board       : list of 3/4/5 dealt community cards (SuitRank, e.g. 'HA').
        range_hands : list of (cardA, cardB) tuples the bot could hold.
        range_w     : sequence of non-negative weights, aligned with range_hands.

        Card removal: any bot hand sharing a card with the dealt board or with
        lbr_hand contributes nothing; for each runout, bot hands that collide with
        the freshly dealt runout cards are also excluded (they were impossible).

        Runouts: river -> exact (no cards left); turn -> enumerate the 46 possible
        rivers; flop -> Monte Carlo over `flop_runout_samples` two-card runouts.
        Returns 0.5 when the range has no mass (degenerate, treat as a coinflip).
        """
        ev = self.evaluator
        dead = set(board) | set(lbr_hand)
        undealt = [c for c in _FULL_DECK if c not in dead]
        need = 5 - len(board)

        if need <= 0:
            runouts = [()]
        elif need == 1:
            runouts = [(c,) for c in undealt]            # enumerate every river
        else:
            # flop (need=2) or preflop (need=5): Monte Carlo over completions.
            runouts = [tuple(self.rng.sample(undealt, need))
                       for _ in range(self.flop_runout_samples)]

        # Pre-filter range hands that are structurally impossible (share a card
        # with the dealt board or LBR's hand). Keep weights aligned.
        hands, weights = [], []
        for h, w in zip(range_hands, range_w):
            if w <= 0.0 or h[0] in dead or h[1] in dead:
                continue
            hands.append(h)
            weights.append(w)
        if not hands:
            return 0.5

        total_eq = 0.0
        total_runouts = 0
        for ro in runouts:
            ro_set = set(ro)
            full = board + list(ro)
            lbr_raw = ev.get_raw_hand_value(list(lbr_hand), full)
            eq = 0.0
            wt = 0.0
            for h, w in zip(hands, weights):
                if h[0] in ro_set or h[1] in ro_set:
                    continue  # runout makes this bot hand impossible
                h_raw = ev.get_raw_hand_value(list(h), full)
                if lbr_raw < h_raw:        # lower raw = stronger -> LBR wins
                    eq += w
                elif lbr_raw == h_raw:
                    eq += 0.5 * w
                wt += w
            if wt > 0.0:
                total_eq += eq / wt
                total_runouts += 1

        return total_eq / total_runouts if total_runouts else 0.5

    # ------------------------------------------------------------------
    # Stage 3-4: full-hand simulation, off-tree decision, main loop
    # ------------------------------------------------------------------

    @staticmethod
    def _pos(seat):
        return 'ip' if seat == 0 else 'oop'

    @staticmethod
    def _visible_board(board, street):
        return [] if street == 0 else board[:2 + street]

    def _lbr_decide(self, lbr_seat, lbr_hand, vis, street, invested, stack,
                    committed, to_call, num_aggr, pattern, botrange, bot_seat):
        """
        Pick LBR's action by ranking each candidate's value under the wprollout
        (assume checkdown to showdown after this action). Returns (char, add, aggr).
        """
        pot = sum(invested)
        lbr_inv = invested[lbr_seat]
        cands = []  # (value, char, add_chips, aggressive)

        if to_call == 0:
            eq = self.equity_vs_range(lbr_hand, vis, botrange.hands, botrange.w)
            cands.append((eq * pot - lbr_inv, 'k', 0, False))
        else:
            cands.append((-lbr_inv, 'f', 0, False))            # fold
            eq = self.equity_vs_range(lbr_hand, vis, botrange.hands, botrange.w)
            cands.append((eq * (pot + to_call) - (lbr_inv + to_call), 'c', to_call, False))

        if num_aggr < MAX_AGGR_PER_STREET and stack[lbr_seat] > to_call:
            seen_adds = set()
            for frac in LBR_BET_FRACTIONS:
                total_add = int(round(to_call + frac * (pot + to_call)))
                if total_add <= to_call or total_add >= stack[lbr_seat]:
                    continue                                   # degenerate or all-in
                if total_add in seen_adds:
                    continue
                seen_adds.add(total_add)
                cands.append(self._value_aggressive(
                    lbr_seat, lbr_hand, vis, street, invested, stack, committed,
                    to_call, num_aggr, pattern, botrange, bot_seat, total_add, False))
            cands.append(self._value_aggressive(
                lbr_seat, lbr_hand, vis, street, invested, stack, committed,
                to_call, num_aggr, pattern, botrange, bot_seat, stack[lbr_seat], True))

        best = max(cands, key=lambda c: c[0])
        return best[1], best[2], best[3]

    def _value_aggressive(self, lbr_seat, lbr_hand, vis, street, invested, stack,
                          committed, to_call, num_aggr, pattern, botrange, bot_seat,
                          total_add, is_allin):
        """Value of LBR betting/raising `total_add` chips: P(fold)*win + P(call)*showdown."""
        pot = sum(invested)
        lbr_inv = invested[lbr_seat]
        new_committed_lbr = committed[lbr_seat] + total_add
        is_raise = to_call > 0

        if is_allin:
            char = 'a'
            translated = [('a', 1.0)]
        elif street == 0:
            # Preflop: the DEPLOYED bot pseudo-harmonic-translates EVERY street, so
            # mirror that here too (was snap-to-nearest -> a preflop strawman victim
            # the off-tree exploiter could over-beat). Build the node's preflop grid
            # via the shared helper (same definition the live API uses), then blend
            # the bracketing sizes -- on-grid / single-bracket reduces to one char.
            grid = translation.preflop_grid(
                num_aggr, committed[lbr_seat], to_call, pot,
                preflop_open_chips(), PREFLOP_RAISE_MULT, SIZE_CHAR)
            allin_frac = translation.eff_fraction(stack[lbr_seat], to_call, pot)
            if allin_frac > (grid[-1][1] if grid else 0.0):
                grid = grid + [('a', allin_frac)]
            eff_frac = translation.eff_fraction(total_add, to_call, pot)
            translated = translation.translate_bet(eff_frac, grid)
            char = translation.nearest_char(eff_frac, grid)
        else:
            # Postflop: the DEPLOYED bot translates an off-grid bet pseudo-
            # harmonically onto its two bracketing grid sizes (cfr/translation.py,
            # bot_strategy.py). Mirror that here so LBR measures the translating
            # bot, not a snap-to-nearest victim that LBR could over-exploit.
            eff_frac = translation.eff_fraction(total_add, to_call, pot)
            allin_frac = translation.eff_fraction(stack[lbr_seat], to_call, pot)
            grid = list(translation.POSTFLOP_GRID)
            if allin_frac > 1.0:
                grid.append(('a', allin_frac))
            translated = translation.translate_bet(eff_frac, grid)
            char = translation.nearest_char(eff_frac, grid)

        # The bot faces LBR's bet: how often does the blueprint fold (per hand)?
        # Blend the bracketing sizes' per-hand fold probabilities (a single
        # bracket -- on-grid or preflop -- reduces to the exact lookup).
        pfold = None
        for c, w in translated:
            # missing=1.0: an untrained bracket routes to fold, mirroring the
            # deployed bot (translation.blend missing_action='fold'). Without this
            # the victim is modelled as CALLING untrained off-grid lines, biasing
            # LBR to under-value exactly the off-tree bets it exists to find.
            p = botrange.per_hand_action_prob(
                self._raw_strategy, 'fold', street, self._pos(bot_seat),
                pattern + c, vis, missing=1.0)
            pfold = w * p if pfold is None else pfold + w * p
        w = botrange.w
        wt = w.sum()
        p_fold = float((w * pfold).sum()) / wt if wt > 1e-12 else 0.0
        cont_w = w * (1.0 - pfold)

        v_fold = float(invested[bot_seat])                     # LBR wins bot's chips
        bot_call_add = min(new_committed_lbr - committed[bot_seat], stack[bot_seat])
        pot_after = pot + total_add + bot_call_add
        eq = self.equity_vs_range(lbr_hand, vis, botrange.hands, cont_w)
        v_cont = eq * pot_after - (lbr_inv + total_add)
        value = p_fold * v_fold + (1.0 - p_fold) * v_cont
        return (value, char, total_add, True)

    def _bot_sizing(self, size, street, pot, to_call, bot_committed, num_aggr):
        """Chips the bot ADDS for an abstract bet/raise size (training sizing).
        Sizes come from abstractions/sizing.py (single source of truth)."""
        if street == 0:
            if num_aggr == 0:                          # open: absolute BB ladder
                to_amt = preflop_open_chips()[size]
                return int(round(to_amt - bot_committed))
            # 3-bet / 4-bet+: pot-relative (unified, matches the engine).
            mult = PREFLOP_RAISE_MULT[size]
            return int(round(to_call + mult * (pot + to_call)))
        mult = POSTFLOP_BET_MULT[size]
        if to_call > 0:
            return int(round(to_call + mult * (pot + to_call)))
        return int(round(mult * pot))

    def _bot_act(self, bot_seat, bot_hand, vis, street, invested, stack, committed,
                 to_call, num_aggr, pattern, botrange, lbr_seat):
        """Bot samples its blueprint; updates LBR's belief. Returns (char, add, aggr)."""
        pot = sum(invested)
        can_aggr = num_aggr < MAX_AGGR_PER_STREET and stack[bot_seat] > max(0, to_call)
        # Engine-matching action names (opens are bet_*, not raise_*, so the
        # blueprint lookup hits the stored open keys): preflop open -> 4-size bet_*
        # ladder; preflop 3-bet/4-bet -> raise_* (3); postflop -> bet_/raise_ incl
        # overbet. Voluntary all-in when a shove is a genuine raise.
        if to_call > 0:
            legal = ['fold', 'call']
            if street == 0 and num_aggr == 0:
                sized = ['bet_small', 'bet_medium', 'bet_large', 'bet_xlarge']
            elif street == 0:
                sized = ['raise_small', 'raise_medium', 'raise_large']
            else:
                sized = ['raise_small', 'raise_medium', 'raise_large', 'raise_overbet']
        else:
            legal = ['check']
            sized = (['bet_small', 'bet_medium', 'bet_large', 'bet_xlarge'] if street == 0
                     else ['bet_small', 'bet_medium', 'bet_large', 'bet_overbet'])
        if can_aggr:
            legal += sized + ['allin']

        pos = self._pos(bot_seat)
        if street == 0:
            key = make_info_set_key(
                0, pos, self.cards.get_bucket(list(bot_hand), None), None, pattern)
        else:
            key = make_info_set_key(
                street, pos, self.cards.get_bucket(list(bot_hand), None),
                self.cards.get_bucket(list(bot_hand), vis), pattern)

        probs = self.restricted_probs(key, tuple(legal))
        action = self.rng.choices(legal, weights=probs)[0]

        # Bayesian belief update on the observed action (decision-time context).
        botrange.observe(self.restricted_probs, action, street, pos, pattern, legal, vis)

        if action == 'fold':
            return 'f', 0, False
        if action == 'check':
            return 'k', 0, False
        if action == 'call':
            return 'c', to_call, False
        if action == 'allin':
            return 'a', stack[bot_seat], True
        size = action.split('_')[1]
        add = self._bot_sizing(size, street, pot, to_call, committed[bot_seat], num_aggr)
        add = max(add, to_call + 1)                # ensure a real raise
        if add >= stack[bot_seat]:
            return 'a', stack[bot_seat], True       # collapses to all-in
        return action_char(action), add, True

    def play_hand(self, lbr_seat, lbr_hand, bot_hand, board):
        """Play one full hand: LBR (off-tree exploiter) vs the blueprint bot.
        Returns LBR's net chip result for the hand."""
        bot_seat = 1 - lbr_seat
        invested = [SB, BB]
        stack = [STARTING_STACK - SB, STARTING_STACK - BB]
        botrange = BotRange(lbr_hand, self.cards)
        folded = None

        street = 0
        while street <= 3:
            vis = self._visible_board(board, street)
            if street > 0:
                botrange.reveal(vis)
            committed = [SB, BB] if street == 0 else [0, 0]
            num_aggr = 0
            pattern = ''
            actor = 0 if street == 0 else 1
            need = {0, 1}
            guard = 0
            while need and folded is None:
                guard += 1
                if guard > 16:
                    break
                other = 1 - actor
                if stack[actor] <= 0:
                    need.discard(actor)
                    actor = other
                    continue
                to_call = max(0, committed[other] - committed[actor])
                if actor == lbr_seat:
                    char, add, aggr = self._lbr_decide(
                        lbr_seat, lbr_hand, vis, street, invested, stack, committed,
                        to_call, num_aggr, pattern, botrange, bot_seat)
                else:
                    char, add, aggr = self._bot_act(
                        bot_seat, bot_hand, vis, street, invested, stack, committed,
                        to_call, num_aggr, pattern, botrange, lbr_seat)
                if char == 'f':
                    folded = actor
                    break
                add = min(add, stack[actor])
                invested[actor] += add
                committed[actor] += add
                stack[actor] -= add
                pattern += char
                if aggr:
                    num_aggr += 1
                    need = {other}
                else:
                    need.discard(actor)
                actor = other

            if folded is not None:
                break
            if min(stack) <= 0:        # someone is all-in and matched -> deal to showdown
                break
            street += 1

        return self._resolve(lbr_seat, lbr_hand, bot_hand, board, invested, folded)

    def _resolve(self, lbr_seat, lbr_hand, bot_hand, board, invested, folded):
        pot = sum(invested)
        if folded is not None:
            return (pot - invested[lbr_seat]) if folded != lbr_seat else (-invested[lbr_seat])
        lbr_raw = self.evaluator.get_raw_hand_value(list(lbr_hand), board)
        bot_raw = self.evaluator.get_raw_hand_value(list(bot_hand), board)
        if lbr_raw < bot_raw:
            return pot - invested[lbr_seat]
        if lbr_raw > bot_raw:
            return -invested[lbr_seat]
        return pot / 2.0 - invested[lbr_seat]

    def evaluate(self, num_hands=2000, progress_every=200, paired=False):
        """
        Monte Carlo: play num_hands of LBR vs the victim (alternating seats).
        Returns LBR's win rate in mbb/hand -- a LOWER bound on exploitability,
        plus the per-hand chip results.

        paired=True re-seeds the per-hand RNG to (seed, hand_index) so the DEAL
        and the pre-river play are deterministic per hand and INDEPENDENT of how
        many RNG draws the victim makes. This makes two evaluators (e.g. blueprint
        vs blueprint+solver) play IDENTICAL deals + identical pre-river lines, so
        the per-hand difference isolates the river change (a true paired
        comparison) -- without it, a victim that draws the RNG differently
        desynchronises the deal stream.
        """
        total = 0.0
        per_hand = []
        for i in range(num_hands):
            if paired:
                # str seed -> deterministic & reproducible across processes (a tuple
                # is not a valid Random seed; hash() of a tuple is per-process salted).
                self.rng = random.Random(f"{self._base_seed}|{i}")
            c = self.rng.sample(_FULL_DECK, 9)
            lbr_hand = (c[0], c[1])
            bot_hand = (c[2], c[3])
            board = c[4:9]
            r = self.play_hand(i % 2, lbr_hand, bot_hand, board)
            per_hand.append(r)
            total += r
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  hand {i + 1}/{num_hands}", flush=True)
        avg_chips = total / num_hands
        return {'lbr_mbb': avg_chips * 1000.0 / 2.0, 'num_hands': num_hands,
                'per_hand': per_hand}


class BotRange:
    """
    The exploiter's belief over the blueprint bot's two hole cards.

    Starts uniform over every two-card combo the bot could hold (all cards minus
    LBR's own two). As the hand plays out:
      * reveal(board)  zeros combos that collide with newly dealt board cards
                       (the bot cannot hold a card that is on the board), and
      * observe(...)   does the Bayesian update: when the bot is seen taking
                       action `a`, each hand's weight is multiplied by the
                       blueprint probability of `a` for that hand's info set.

    Hands are bucketed exactly as training/inference key them (make_info_set_key
    over CardAbstraction buckets), so the belief uses the same strategy the bot
    actually plays. Equity queries read the (unnormalised) weights directly.
    """

    def __init__(self, lbr_hand, cards):
        self.cards = cards
        dead = set(lbr_hand)
        self.hands = [h for h in combinations(_FULL_DECK, 2)
                      if h[0] not in dead and h[1] not in dead]
        self.w = np.ones(len(self.hands))
        self._bucket_cache = {}   # (street, board_tuple) -> (pf_list, strength_list)

    # -- card removal --------------------------------------------------
    def reveal(self, board):
        """Zero the weight of every bot hand that collides with the board."""
        bset = set(board)
        for i, h in enumerate(self.hands):
            if h[0] in bset or h[1] in bset:
                self.w[i] = 0.0

    # -- bucketing -----------------------------------------------------
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

    # -- strategy matrix ----------------------------------------------
    def action_probs(self, lookup, street, position, bet_pattern, legal, board):
        """
        [n_hands, n_legal] blueprint probabilities per bot hand at this node.
        `lookup` is LBREvaluator.restricted_probs. Hands sharing an info-set key
        share a row, so we query the blueprint once per distinct bucket key.
        """
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
                row = lookup(key, legal_t)
                seen[gid] = row
            mat[i] = row
        return mat

    # -- Bayesian update ----------------------------------------------
    def observe(self, lookup, action, street, position, bet_pattern, legal, board):
        """Condition the belief on the bot having taken `action` at this node."""
        legal = list(legal)
        ai = legal.index(action)
        mat = self.action_probs(lookup, street, position, bet_pattern, legal, board)
        new_w = self.w * mat[:, ai]
        s = new_w.sum()
        if s > 1e-12:
            self.w = new_w / s    # renormalise to keep weights from underflowing
        # else: the action has ~zero model-probability across every live hand.
        # Applying it would zero the belief permanently; keep the prior instead.
        # Matches RangeTracker.observe (the live sibling) so the two never
        # diverge on off-model handling. In practice unreachable here (the bot
        # samples its action FROM this blueprint, so the action always has
        # positive mass for some live hand) -- the guard is defensive parity.

    # -- per-hand probability of one action (from the raw average strategy) ---
    def per_hand_action_prob(self, raw_lookup, action, street, position, pattern,
                             board, missing=0.0):
        """
        Array of P(bot plays `action`) per bot hand, read from the UNRESTRICTED
        average strategy (so e.g. fold probability is not inflated by dropping the
        bot's raise mass). raw_lookup(key) -> {action: prob} or None.

        `missing` is the per-hand value to use when the bracket KEY IS UNTRAINED
        (raw_lookup returns None/{}). This distinguishes "trained, but this action
        has 0 mass" (-> 0.0, correct) from "this bracket was never trained". The
        deployed bot routes an untrained translation bracket's weight to FOLD
        (translation.blend(missing_action='fold') via bot_strategy._blend_lookup),
        so the victim model here must do the same: pass missing=1.0 when querying
        'fold' so an off-grid bet landing on an untrained bracket is modelled as a
        fold, not a call. Defaulting missing=0.0 would model the bot calling there
        -> LBR under-values off-grid bets (the exact off-tree regime LBR measures).
        """
        n = len(self.hands)
        out = np.zeros(n)
        pf, strength = self._buckets(street, board)
        seen = {}
        for i in range(n):
            if self.w[i] <= 0.0:
                continue
            gid = pf[i] if street == 0 else (pf[i], strength[i])
            p = seen.get(gid, -1.0)
            if p < 0.0:
                if street == 0:
                    key = make_info_set_key(0, position, pf[i], None, pattern)
                else:
                    key = make_info_set_key(street, position, pf[i], strength[i], pattern)
                strat = raw_lookup(key)
                p = float(strat.get(action, 0.0)) if strat else missing
                seen[gid] = p
            out[i] = p
        return out
