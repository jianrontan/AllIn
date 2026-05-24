# backend/bot/src/evaluation/best_response.py
"""
Best-response exploitability of a blueprint, via Monte Carlo board sampling.

WHAT IT MEASURES
----------------
Exploitability = how much a perfect counter-strategy beats the blueprint by.
For a 2-player zero-sum game it is BR_0(sigma_1) + BR_1(sigma_0): seat each side
in turn as the "hero" who best-responds while the other ("villain") plays the
blueprint. If the blueprint were an exact Nash equilibrium of the real game,
exploitability would be 0. A high number means training has not converged (or
the abstraction is leaky). This is the convergence scoreboard for the gamma
experiment: run it before and after a change and watch the number drop.

HOW IT WORKS (vectorized, reach-weighted public-tree walk)
----------------------------------------------------------
Poker is hidden-information, so "best response" is NOT "pick the best action
knowing villain's cards". The hero does not see villain's hand. We walk the
PUBLIC betting tree ONCE per board, carrying:

  * villain reach   rv[v]   -- a vector over all H hands villain could hold,
                              = prior * product of villain's blueprint action
                              probabilities along the path so far.
  * hero value      a vector over all H hands the hero could hold, returned by
                    the recursion: the best-response value for each hero hand.

At each node:
  * villain decision node : split rv across actions by villain's blueprint
                            strategy (per villain hand) and SUM the children's
                            hero-value vectors. Summing integrates over villain.
  * hero decision node    : the hero best-responds with its EXACT hand, so we
                            take an ELEMENTWISE MAX over actions (same villain
                            reach down every branch).
  * terminal              : hero_value[h] = sum over villain hands v
                            (compatible with h) of rv[v] * utility(h, v).

Because every hero hand is evaluated in a single walk (vector payload instead
of one scalar per sampled hero hand), one board sample integrates all 1081
hero hands x 990 compatible villain hands at once. That is the whole point of
the rewrite: orders of magnitude lower variance per board, and one tree walk
per board instead of one per (board, hero-hand).

CARD REMOVAL
------------
Hero and villain cannot share a card. We let villain's reach vector range over
ALL H hands (overlap and all) as it flows down the tree -- that is linear and
harmless -- and subtract the blocked combinations at the TERMINALS, where it
actually matters, using the standard O(H) "per-card running sums" algorithm.
Every card sits in exactly 46 of the H = C(47,2) = 1081 hands, so every hero
hand is blocked by exactly 46 + 46 - 1 = 91 villain hands, leaving 990
compatible. That count is constant across hero hands, so the per-hero
normalization is just a single division by 990 at the end.

This is a FULL-GAME best response (hero uses exact cards), i.e. real
exploitability, not merely exploitability inside the abstraction.
"""
import random
import numpy as np
from itertools import combinations

from ..cfr.poker_game import PokerGame, STARTING_STACK
from ..cfr.keys import action_char, make_info_set_key
from ..abstractions.card_abstractions import CardAbstraction
from ..abstractions.hand_evaluator import HandEvaluator

_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
_SUITS = ['H', 'D', 'C', 'S']
_FULL_DECK = [s + r for r in _RANKS for s in _SUITS]
_CARD_ID = {c: i for i, c in enumerate(_FULL_DECK)}
_NUM_CARDS = len(_FULL_DECK)  # 52

# Every card appears in 46 of the C(47,2)=1081 hands; each hero hand is blocked
# by 46 + 46 - 1 = 91 villain hands, leaving C(45,2) = 990 compatible.
_COMPATIBLE = 990


class BestResponseEvaluator:
    def __init__(self, blueprint_db, seed=None):
        self.db = blueprint_db
        self.game = PokerGame()
        self.cards = CardAbstraction()
        self.evaluator = HandEvaluator()
        self.rng = random.Random(seed)

        # Memoize blueprint lookups: info_set_key -> {action: prob} (or None).
        self._strategy_cache = {}
        # Memoize restricted+renormalised probs: (key, legal_tuple) -> np.array.
        self._restricted_cache = {}

    # ------------------------------------------------------------------
    # Blueprint strategy access (memoized)
    # ------------------------------------------------------------------

    def _raw_strategy(self, key):
        cached = self._strategy_cache.get(key, 0)
        if cached == 0:
            cached = self.db.get_average_strategy(key)
            self._strategy_cache[key] = cached
        return cached

    def _restricted_probs(self, key, legal):
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
    # Per-board precompute (independent of which seat is hero)
    # ------------------------------------------------------------------

    def _board_arrays(self, board):
        """
        Precompute, for one board, everything that does not depend on the
        betting line or which seat is hero:

          hands : list of (cardA, cardB) for all H hands not using a board card
          raw   : showdown rank per hand  (lower = stronger)
          c1,c2 : integer card ids per hand (for card-removal at terminals)
          g, G  : dense strength-group id per hand (0 = strongest) and group count
          pf    : preflop bucket per hand   (object array, for villain keys)
          strg  : {street: bucket-per-hand} for flop/turn/river villain keys
        """
        board_set = set(board)
        pool = [c for c in _FULL_DECK if c not in board_set]
        hands = list(combinations(pool, 2))
        H = len(hands)

        raw = np.empty(H)
        c1 = np.empty(H, dtype=np.int64)
        c2 = np.empty(H, dtype=np.int64)
        pf = [None] * H
        s1 = [None] * H
        s2 = [None] * H
        s3 = [None] * H
        for i, (a, b) in enumerate(hands):
            hl = [a, b]
            raw[i] = self.evaluator.get_raw_hand_value(hl, board)
            c1[i] = _CARD_ID[a]
            c2[i] = _CARD_ID[b]
            pf[i] = self.cards.get_bucket(hl, None)
            s1[i] = self.cards.get_bucket(hl, board[:3])
            s2[i] = self.cards.get_bucket(hl, board[:4])
            s3[i] = self.cards.get_bucket(hl, board[:5])

        # Dense strength groups: ascending raw -> group 0 is the strongest.
        uniq = np.unique(raw)
        g = np.searchsorted(uniq, raw)
        G = len(uniq)

        pf = np.array(pf, dtype=object)
        strg = {1: np.array(s1, dtype=object),
                2: np.array(s2, dtype=object),
                3: np.array(s3, dtype=object)}

        # Villain hands sharing the same blueprint key share a strategy row.
        # The grouping (by preflop bucket, or (preflop, strength) postflop) is
        # independent of position/pattern/legal-set, so precompute the masks and
        # a representative hand per group ONCE -- reused at every villain node and
        # for both seats. groups[street] = list of (mask, rep_idx).
        def build_groups(labels):
            out = []
            for lab in set(labels.tolist()):
                mask = labels == lab
                out.append((mask, int(np.argmax(mask))))
            return out

        groups = {0: build_groups(pf)}
        for s in (1, 2, 3):
            labels = np.array([f"{pf[i]}|{strg[s][i]}" for i in range(H)], dtype=object)
            groups[s] = build_groups(labels)

        return {
            'hands': hands, 'H': H, 'raw': raw, 'c1': c1, 'c2': c2,
            'g': g, 'G': G, 'gNC': g.astype(np.int64) * _NUM_CARDS,
            'pf': pf, 'strg': strg, 'groups': groups,
        }

    # ------------------------------------------------------------------
    # Best-response value vector for one (board, hero_seat)
    # ------------------------------------------------------------------

    def _showdown_measure(self, ba, rv, final_pot, hero_total):
        """
        Vectorized showdown value (measure, hero perspective) per hero hand, with
        card removal. Lower raw = stronger; hero wins vs WEAKER villains (villain
        raw > hero raw). O(H) via per-card running sums. Extracted so it can be
        validated against a brute-force oracle.
        """
        c1 = ba['c1']
        c2 = ba['c2']
        g = ba['g']
        G = ba['G']

        total = float(rv.sum())
        cardTot = (np.bincount(c1, weights=rv, minlength=_NUM_CARDS) +
                   np.bincount(c2, weights=rv, minlength=_NUM_CARDS))
        compatM = total - (cardTot[c1] + cardTot[c2] - rv)

        groupSum = np.bincount(g, weights=rv, minlength=G)
        cum = np.cumsum(groupSum)
        strongerGroupCum = cum - groupSum            # reach of strictly stronger groups
        gNC = ba['gNC']
        flat = (np.bincount(gNC + c1, weights=rv, minlength=G * _NUM_CARDS) +
                np.bincount(gNC + c2, weights=rv, minlength=G * _NUM_CARDS))
        gc = flat.reshape(G, _NUM_CARDS)
        gcum = np.cumsum(gc, axis=0)
        strongerCardCum = gcum - gc                  # reach of stronger groups, per card

        sg = strongerGroupCum[g]                     # stronger total (no removal)
        grp = groupSum[g]                            # tie-group total
        wk = total - sg - grp                        # weaker total (no removal)

        scc1 = strongerCardCum[g, c1]
        scc2 = strongerCardCum[g, c2]
        gcc1 = gc[g, c1]
        gcc2 = gc[g, c2]
        wcc1 = cardTot[c1] - scc1 - gcc1
        wcc2 = cardTot[c2] - scc2 - gcc2

        loseM = sg - scc1 - scc2                     # stronger villains, compatible
        tieM = grp - gcc1 - gcc2 + rv                # tied villains, compatible (drop self)
        winM = wk - wcc1 - wcc2                      # weaker villains, compatible

        # payoff = winnings - hero_total; winnings = final_pot (win),
        # final_pot/2 (tie), 0 (lose). compatM == winM + tieM + loseM.
        return final_pot * winM + (final_pot / 2.0) * tieM - hero_total * compatM

    def _board_value(self, hero_seat, board, ba):
        """
        Return a length-H vector: best-response value (chips, hero perspective)
        for each hero hand on this board, integrating over villain's blueprint
        range. Values are MEASURES (villain reach starts at 1 per hand); divide
        by _COMPATIBLE to get per-hand expectations.
        """
        villain_seat = 1 - hero_seat
        villain_pos = 'ip' if villain_seat == 0 else 'oop'

        H = ba['H']
        raw = ba['raw']
        c1 = ba['c1']
        c2 = ba['c2']
        g = ba['g']
        G = ba['G']
        pf = ba['pf']
        strg = ba['strg']

        def villain_probs_matrix(street, pattern, legal):
            """[H, n_legal] blueprint probs per villain hand, grouped by key."""
            mat = np.empty((H, len(legal)))
            for mask, rep in ba['groups'][street]:
                if street == 0:
                    key = make_info_set_key(0, villain_pos, pf[rep], None, pattern)
                else:
                    key = make_info_set_key(
                        street, villain_pos, pf[rep], strg[street][rep], pattern)
                mat[mask] = self._restricted_probs(key, legal)
            return mat

        def terminal_value(history, street, starting_pot, p0_inv, p1_inv, rv):
            final_pot = self.game.calculate_current_pot(
                starting_pot, history, street, p0_inv, p1_inv)
            p0_this = self.game.get_player_contribution_this_round(
                history, street, starting_pot, 0, p0_inv, p1_inv)
            p0_total = p0_inv + p0_this
            hero_total = p0_total if hero_seat == 0 else (final_pot - p0_total)

            if 'fold' in history:
                total = float(rv.sum())
                cardTot = (np.bincount(c1, weights=rv, minlength=_NUM_CARDS) +
                           np.bincount(c2, weights=rv, minlength=_NUM_CARDS))
                compatM = total - (cardTot[c1] + cardTot[c2] - rv)
                folder_idx = next(i for i, a in enumerate(history) if a == 'fold')
                folder = self.game._acting_player(folder_idx, street)
                val = (-hero_total) if folder == hero_seat else (final_pot - hero_total)
                return val * compatM

            # Showdown.
            return self._showdown_measure(ba, rv, final_pot, hero_total)

        def walk(history, street, starting_pot, p0_inv, p1_inv,
                 p0_stack, p1_stack, pattern, rv, depth):
            if depth > 50 or street > 3:
                return terminal_value(history, street, starting_pot, p0_inv, p1_inv, rv)
            if self.game.is_terminal(history, street):
                return terminal_value(history, street, starting_pot, p0_inv, p1_inv, rv)

            current_player = self.game._acting_player(len(history), street)
            legal = self.game.get_legal_actions(
                street, history, starting_pot, current_player,
                p0_stack, p1_stack, p0_inv, p1_inv)

            if not legal:
                if street < 3:
                    p0_this = self.game.get_player_contribution_this_round(
                        history, street, starting_pot, 0, p0_inv, p1_inv)
                    p1_this = self.game.get_player_contribution_this_round(
                        history, street, starting_pot, 1, p0_inv, p1_inv)
                    new_pot = self.game.calculate_current_pot(
                        starting_pot, history, street, p0_inv, p1_inv)
                    new_p0_inv = p0_inv + p0_this
                    new_p1_inv = p1_inv + p1_this
                    return walk([], street + 1, new_pot, new_p0_inv, new_p1_inv,
                                STARTING_STACK - new_p0_inv, STARTING_STACK - new_p1_inv,
                                '', rv, depth + 1)
                return terminal_value(history, street, starting_pot, p0_inv, p1_inv, rv)

            legal_t = tuple(legal)

            def child(action, child_rv):
                cost = self.game._action_cost(
                    action, street, history, starting_pot, current_player, p0_inv, p1_inv)
                nh = history + [action]
                npat = pattern + action_char(action)
                if current_player == 0:
                    return walk(nh, street, starting_pot, p0_inv, p1_inv,
                                p0_stack - cost, p1_stack, npat, child_rv, depth + 1)
                return walk(nh, street, starting_pot, p0_inv, p1_inv,
                            p0_stack, p1_stack - cost, npat, child_rv, depth + 1)

            if current_player == hero_seat:
                # Hero best-responds with its exact hand: same villain reach down
                # every branch, take the elementwise max over actions.
                vals = [child(a, rv) for a in legal]
                return np.maximum.reduce(vals)

            # Villain node: split reach by blueprint strategy, sum children.
            probs = villain_probs_matrix(street, pattern, legal_t)
            total = np.zeros(H)
            for ai, a in enumerate(legal):
                total = total + child(a, rv * probs[:, ai])
            return total

        rv0 = np.ones(H)  # unit reach per villain hand; normalise by _COMPATIBLE later
        return walk([], 0, 3, 0.0, 0.0,
                    STARTING_STACK - 1, STARTING_STACK - 2, '', rv0, 0)

    # ------------------------------------------------------------------
    # Top-level estimate
    # ------------------------------------------------------------------

    def evaluate(self, num_samples=100, progress_every=10):
        """
        Return a dict with BR values per seat and total exploitability, all in
        milli-big-blinds per hand (mbb/hand). BB = 2 chips. Each board sample
        integrates all hero hands and the full villain range, so far fewer
        samples are needed than the old per-hero-hand estimator.
        """
        br_chips = {0: 0.0, 1: 0.0}

        for s in range(num_samples):
            board = self.rng.sample(_FULL_DECK, 5)
            ba = self._board_arrays(board)
            for hero_seat in (0, 1):
                vec = self._board_value(hero_seat, board, ba)
                # mean over hero hands of (measure / compatible villain count)
                br_chips[hero_seat] += float(vec.mean()) / _COMPATIBLE

            if progress_every and (s + 1) % progress_every == 0:
                print(f"  sample {s + 1}/{num_samples}", flush=True)

        br0 = br_chips[0] / num_samples
        br1 = br_chips[1] / num_samples
        # chips -> mbb: divide by BB(2), times 1000.
        to_mbb = 1000.0 / 2.0
        return {
            'br_seat0_mbb': br0 * to_mbb,
            'br_seat1_mbb': br1 * to_mbb,
            'exploitability_mbb': (br0 + br1) * to_mbb,
            'num_samples': num_samples,
        }
