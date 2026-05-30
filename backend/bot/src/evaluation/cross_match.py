# backend/bot/src/evaluation/cross_match.py
"""
Faithful CROSS-SIZING head-to-head: two blueprints trained on DIFFERENT bet-size
grids play each other, each using ITS OWN sizing and pseudo-harmonic action
translation to interpret the opponent's off-grid bets.

Why this is needed (and why match.py is not enough): match.py sizes BOTH players
with the current sizing.py and shares one betting pattern, so it cannot run the
old-sizing bot (whose opens were ~2x bigger). Here each player carries:
  * its own SIZING spec (target raise-to total -> additional chips, == the engine
    formula `target - committed`, validated against each native engine), and
  * its own per-street PATTERN, built by perceiving the opponent's bet through its
    own grid via translation (nearest_char for the stored pattern; translate_bet +
    blend for the immediate decision facing an off-grid bet) -- exactly mirroring
    the live BlueprintStrategy / GameSession translation path.

Objective chip state (invested / committed / stack / pot) is shared and truthful;
only the abstraction (sizing + pattern perception) is per-player. Validated by:
tests/run_cross_match.py (sizing-vs-native-engine, self-vs-self ~= 0, new-vs-new
== match.py within noise, and a cross-perception unit check).
"""
import random

import numpy as np

from ..cfr.keys import make_info_set_key, action_char
from ..cfr import translation
from ..cfr.poker_game import STARTING_STACK
from ..abstractions.card_abstractions import CardAbstraction
from ..abstractions.hand_evaluator import HandEvaluator

SB, BB = 1, 2
MAX_AGGR_PER_STREET = 3
_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
_SUITS = ['H', 'D', 'C', 'S']
_FULL_DECK = [s + r for r in _RANKS for s in _SUITS]


# ---------------------------------------------------------------------------
# Sizing specs -- return the raise-TO street total (chips). additional = total -
# committed, matching the engine's calculate_raise_amount/calculate_bet_amount.
# ---------------------------------------------------------------------------
class Sizing:
    """A bet-size scheme. `target_total(size, street, pot, to_call, committed,
    num_aggr)` is the raise-to street-total chips for an abstract size."""

    def __init__(self, name, open_to_bb, three_bet, four_bet, postflop_mult):
        self.name = name
        self.open_to_bb = open_to_bb          # {size: raise-to total in BB}  (preflop open)
        self.three_bet = three_bet            # ('abs', {size: BB}) or ('potrel', {size: mult})
        self.four_bet = four_bet              # same forms (preflop 4bet+)
        self.postflop_mult = postflop_mult    # {size: pot fraction}

    def _potrel_3bet_target(self, mode_vals, pot, to_call, committed):
        mode, vals = mode_vals
        if mode == 'abs':
            return vals[None] if False else None  # unreachable; handled by caller
        return None

    def target_total(self, size, street, pot, to_call, committed, num_aggr):
        if street == 0:
            if num_aggr == 0:                                  # open
                return self.open_to_bb[size] * BB
            spec = self.three_bet if num_aggr == 1 else self.four_bet
            mode, vals = spec
            if mode == 'abs':                                  # absolute raise-to (BB)
                return vals[size] * BB
            # pot-relative. Two engine conventions exist:
            #   new 3bet/4bet : target = mult*(pot+to_call) + to_call   (pot AFTER call)
            #   old 4bet      : target = mult*pot          + to_call   (pot BEFORE call)
            base = (pot + to_call) if vals.get('_after_call', True) else pot
            return vals[size] * base + to_call
        # postflop raise-TO total == engine: mult*(pot+to_call) + to_call (raise),
        # mult*pot (bet). NOT committed-inclusive -- add = target - committed.
        mult = self.postflop_mult[size]
        if to_call > 0:
            return to_call + mult * (pot + to_call)
        return mult * pot

    def add_chips(self, size, street, pot, to_call, committed, num_aggr):
        """Additional chips for an abstract sized action (= target - committed),
        floored to a legal min-raise (> to_call)."""
        total = self.target_total(size, street, pot, to_call, committed, num_aggr)
        return max(total - committed, to_call + 1.0)


def _potrel(vals, after_call=True):
    d = dict(vals)
    d['_after_call'] = after_call
    return ('potrel', d)


# NEW sizing (the 2026-05-29 redesign): open 2/2.5/3.5/5 BB (4th = xlarge);
# 3bet AND 4bet+ unified pot-relative 0.66/1.0/1.5 of pot-after-call; postflop
# 0.33/0.66/1.0/1.5x pot (4th = overbet); voluntary all-in everywhere.
NEW_SIZING = Sizing(
    'new',
    open_to_bb={'small': 2.0, 'medium': 2.5, 'large': 3.5, 'xlarge': 5.0},
    three_bet=_potrel({'small': 0.66, 'medium': 1.0, 'large': 1.5}, after_call=True),
    four_bet=_potrel({'small': 0.66, 'medium': 1.0, 'large': 1.5}, after_call=True),
    postflop_mult={'small': 0.33, 'medium': 0.66, 'large': 1.0, 'overbet': 1.5})

# OLD sizing (blueprint_20260525_062044_9150000it): open 3/5/7 BB; 3bet absolute
# 9/12/16 BB; 4bet+ pot-relative 0.66/1.33/2.0 of pot BEFORE call.
OLD_SIZING = Sizing(
    'old',
    open_to_bb={'small': 3.0, 'medium': 5.0, 'large': 7.0},
    three_bet=('abs', {'small': 9.0, 'medium': 12.0, 'large': 16.0}),
    four_bet=_potrel({'small': 0.66, 'medium': 1.33, 'large': 2.0}, after_call=False),
    postflop_mult={'small': 0.33, 'medium': 0.66, 'large': 1.0})


def _legal_actions(street, to_call, num_aggr, stack, open_sizes, postflop_sizes):
    """Legal abstract actions for a bot, using its OWN size sets and engine action
    NAMES (opens are bet_*, not raise_*; 3-bet/4-bet are raise_* with 3 sizes).
    `open_sizes`/`postflop_sizes` are the acting bot's Sizing size-name lists (NEW
    has 4 each incl. xlarge/overbet; OLD has 3). Voluntary all-in always available."""
    can_aggr = num_aggr < MAX_AGGR_PER_STREET and stack > max(0.0, to_call)
    if to_call > 0:
        legal = ['fold', 'call']
        if street == 0 and num_aggr == 0:
            sized = [f'bet_{s}' for s in open_sizes]
        elif street == 0:
            sized = ['raise_small', 'raise_medium', 'raise_large']
        else:
            sized = [f'raise_{s}' for s in postflop_sizes]
    else:
        legal = ['check']
        sized = ([f'bet_{s}' for s in open_sizes] if street == 0
                 else [f'bet_{s}' for s in postflop_sizes])
    if can_aggr:
        legal += sized + ['allin']
    return legal


class CrossBot:
    """A blueprint player with its own Sizing and its own translated view of the
    betting. Holds no per-hand state across hands (patterns live in the match)."""

    def __init__(self, strategy_source, sizing, cards, rng):
        self.src = strategy_source
        self.sizing = sizing
        self.cards = cards
        self.rng = rng
        self._raw_cache = {}
        self._restricted_cache = {}

    def _raw(self, key):
        c = self._raw_cache.get(key, 0)
        if c == 0:
            c = self.src.get_average_strategy(key)
            self._raw_cache[key] = c
        return c

    def _restricted(self, key, legal):
        ck = (key, legal)
        cached = self._restricted_cache.get(ck)
        if cached is not None:
            return cached
        stored = self._raw(key)
        n = len(legal)
        if stored:
            w = np.array([max(0.0, stored.get(a, 0.0)) for a in legal])
            t = w.sum()
            probs = w / t if t > 1e-12 else np.ones(n) / n
        else:
            probs = np.ones(n) / n
        self._restricted_cache[ck] = probs
        return probs

    def grid(self, street, pot, to_call, committed, num_aggr, stack):
        """This bot's bet-size grid at a node as sorted [(char, eff_frac)], on the
        eff_fraction axis (bet / pot-after-call) -- mirrors GameSession._node_grid.
        Used to perceive the OPPONENT's bet (built in the bettor's node context)."""
        g = {}
        # This bot's own size set + engine action-name kind so the pattern char is
        # right. Opens MUST use bet_ (the 4th open xlarge only has a bet_ form;
        # action_char('raise_xlarge') is now an error); 3-bet/4-bet & postflop-facing
        # use raise_ (bet_X / raise_X share a char for the others).
        if street == 0:
            if num_aggr == 0:
                size_names, kind = list(self.sizing.open_to_bb), 'bet_'
            else:
                size_names, kind = ['small', 'medium', 'large'], 'raise_'
        else:
            size_names = list(self.sizing.postflop_mult)
            kind = 'raise_' if to_call > 0 else 'bet_'
        for size in size_names:
            add = self.sizing.add_chips(size, street, pot, to_call, committed, num_aggr)
            if add >= stack:
                continue                              # collapses to all-in
            g[action_char(kind + size)] = translation.eff_fraction(add, to_call, pot)
        g['a'] = translation.eff_fraction(stack, to_call, pot)
        return sorted(g.items(), key=lambda cf: cf[1])

    def key_for(self, seat, hand, vis, street, pattern):
        pos = 'ip' if seat == 0 else 'oop'
        pre = self.cards.get_bucket(list(hand), None)
        if street == 0:
            return make_info_set_key(0, pos, pre, None, pattern)
        return make_info_set_key(street, pos, pre, self.cards.get_bucket(list(hand), vis), pattern)

    def blend_dist(self, seat, hand, vis, street, base, brackets):
        """Pseudo-harmonic blend of the blueprint response across the two
        bracketing sizes (reuses translate_bet weights). Returns {action: prob} or
        None if no bracketing key has stored mass (caller falls back to nearest)."""
        merged = {}
        for ch, w in brackets:
            if w <= 0:
                continue
            stored = self._raw(self.key_for(seat, hand, vis, street, base + ch))
            if stored:
                for a, p in stored.items():
                    merged[a] = merged.get(a, 0.0) + w * max(0.0, p)
            else:
                # Untrained bracket (e.g. a >3.5BB open's all-in-open overflow):
                # fold that weight rather than drop it (== translation.blend).
                merged['fold'] = merged.get('fold', 0.0) + w
        s = sum(merged.values())
        return {a: p / s for a, p in merged.items()} if s > 1e-12 else None


def _sample_action(bot, dist_or_key, legal, rng):
    """Sample from a blended dist (dict) or a single key (str), restricted to legal."""
    legal_t = tuple(legal)
    if isinstance(dist_or_key, dict):
        w = np.array([max(0.0, dist_or_key.get(a, 0.0)) for a in legal])
        t = w.sum()
        probs = w / t if t > 1e-12 else np.ones(len(legal)) / len(legal)
    else:
        probs = bot._restricted(dist_or_key, legal_t)
    return legal[rng.choices(range(len(legal)), weights=probs)[0]]


class CrossMatch:
    """Play hands between two CrossBots (A and B). A's perspective; seat alternates."""

    def __init__(self, strat_a, sizing_a, strat_b, sizing_b, seed=None):
        self.cards = CardAbstraction()
        self.evaluator = HandEvaluator()
        self.rng = random.Random(seed)
        self.pa = CrossBot(strat_a, sizing_a, self.cards, self.rng)
        self.pb = CrossBot(strat_b, sizing_b, self.cards, self.rng)

    def play_hand(self, seat_of_A, hand_a, hand_b, board, record=None):
        players = {seat_of_A: self.pa, 1 - seat_of_A: self.pb}
        hands = {seat_of_A: hand_a, 1 - seat_of_A: hand_b}
        invested = [float(SB), float(BB)]
        stack = [STARTING_STACK - SB, STARTING_STACK - BB]
        folded = None
        events = [] if record is not None else None

        street = 0
        while street <= 3:
            vis = [] if street == 0 else board[:2 + street]
            committed = [float(SB), float(BB)] if street == 0 else [0.0, 0.0]
            num_aggr = 0
            pattern = ['', '']                 # per-seat perceived pattern
            pending = [None, None]             # per-seat pending translation (faces an off-grid bet)
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
                to_call = max(0.0, committed[other] - committed[actor])
                pot = sum(invested)
                bot = players[actor]
                legal = _legal_actions(
                    street, to_call, num_aggr, stack[actor],
                    list(bot.sizing.open_to_bb), list(bot.sizing.postflop_mult))

                # Decide: blend the bracketing keys if facing an off-grid bet,
                # else a single key from this bot's own perceived pattern.
                pend = pending[actor]
                dist = None
                if pend and len(pend['brackets']) > 1:
                    dist = bot.blend_dist(actor, hands[actor], vis, street,
                                          pend['base'], pend['brackets'])
                target = dist if dist is not None else bot.key_for(
                    actor, hands[actor], vis, street, pattern[actor])
                chosen = _sample_action(bot, target, legal, self.rng)

                # Resolve the chosen action -> chips + own char.
                if chosen == 'fold':
                    own_char, add, aggr = 'f', 0.0, False
                elif chosen == 'check':
                    own_char, add, aggr = 'k', 0.0, False
                elif chosen == 'call':
                    own_char, add, aggr = 'c', to_call, False
                elif chosen == 'allin':
                    own_char, add, aggr = 'a', stack[actor], True
                else:
                    size = chosen.split('_')[1]
                    add = bot.sizing.add_chips(size, street, pot, to_call, committed[actor], num_aggr)
                    if add >= stack[actor]:
                        own_char, add, aggr = 'a', stack[actor], True
                    else:
                        own_char, aggr = action_char(chosen), True

                if events is not None:
                    events.append({'seat': actor, 'street': street, 'char': own_char,
                                   'add': add, 'aggr': aggr, 'chosen': chosen,
                                   'pot': pot, 'to_call': to_call})

                if own_char == 'f':
                    folded = actor
                    break

                # Opponent perceives this action through ITS grid (built in the
                # actor's node context), for its pattern char + (if aggressive,
                # off-grid) pending blend.
                if aggr:
                    opp = players[other]
                    ogrid = opp.grid(street, pot, to_call, committed[actor],
                                     num_aggr, stack[other])
                    eff = translation.eff_fraction(add, to_call, pot)
                    nchar = translation.nearest_char(eff, ogrid)
                    brackets = translation.translate_bet(eff, ogrid)
                    base_other = pattern[other]
                    pattern[other] += nchar
                    pending[other] = {'base': base_other, 'brackets': brackets}
                else:
                    # check/call/fold perceived identically; no translation.
                    pattern[other] += own_char
                    pending[other] = None

                # Actor records its OWN action under its own char; its pending clears.
                pattern[actor] += own_char
                pending[actor] = None

                add = min(add, stack[actor])
                invested[actor] += add
                committed[actor] += add
                stack[actor] -= add
                if aggr:
                    num_aggr += 1
                    need = {other}
                else:
                    need.discard(actor)
                actor = other

            if folded is not None:
                break
            if min(stack) <= 0:
                break
            street += 1

        return self._resolve(seat_of_A, hand_a, hand_b, board, invested, folded)

    def _resolve(self, seat_of_A, hand_a, hand_b, board, invested, folded):
        pot = sum(invested)
        a = seat_of_A
        if folded is not None:
            return (pot - invested[a]) if folded != a else (-invested[a])
        ra = self.evaluator.get_raw_hand_value(list(hand_a), board)
        rb = self.evaluator.get_raw_hand_value(list(hand_b), board)
        if ra < rb:
            return pot - invested[a]
        if ra > rb:
            return -invested[a]
        return pot / 2.0 - invested[a]

    def evaluate(self, num_hands=20000, progress_every=0):
        total = 0.0
        sq = 0.0
        for i in range(num_hands):
            c = self.rng.sample(_FULL_DECK, 9)
            r = self.play_hand(i % 2, (c[0], c[1]), (c[2], c[3]), c[4:9])
            total += r
            sq += r * r
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  hand {i + 1}/{num_hands}", flush=True)
        avg = total / num_hands
        var = max(0.0, sq / num_hands - avg * avg)
        to_mbb = 1000.0 / 2.0
        stderr = (var / num_hands) ** 0.5 * to_mbb
        return {'a_mbb': avg * to_mbb, 'stderr_mbb': stderr, 'num_hands': num_hands}
