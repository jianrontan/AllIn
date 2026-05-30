# backend/bot/src/evaluation/match.py
"""
Head-to-head match runner: play real hands between two KNOWN strategies and
report player A's win rate in mbb/hand.

This is the substrate AIVAT (aivat.py) corrects for variance. On its own it is
also the "does version B actually beat version A?" tool -- but raw, so it is
high-variance and needs many hands. When `record=True`, play_hand emits a full
trajectory (public states, chance events with their remaining-deck support, and
each acting strategy's action distribution + sampled action) that AIVAT consumes
to subtract the luck.

A "strategy" here is any object exposing get_average_strategy(key) -> {action:
prob} or None (i.e. a BlueprintDB, or a fake for testing). Both players act in
the blueprint's abstract action set; the engine tracks real chips.
"""
import random

import numpy as np

from ..cfr.keys import make_info_set_key, action_char
from ..cfr.poker_game import STARTING_STACK
from ..abstractions.sizing import preflop_open_chips, PREFLOP_RAISE_MULT, POSTFLOP_BET_MULT
from ..abstractions.card_abstractions import CardAbstraction
from ..abstractions.hand_evaluator import HandEvaluator

SB, BB = 1, 2
MAX_AGGR_PER_STREET = 3
_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
_SUITS = ['H', 'D', 'C', 'S']
_FULL_DECK = [s + r for r in _RANKS for s in _SUITS]


def _legal_actions(street, to_call, num_aggr, stack, pot):
    """Mirror poker_game.get_legal_actions using the SAME engine action NAMES so a
    blueprint lookup hits the stored keys (the DB stores opens as bet_*, NOT raise_*):
      * preflop OPEN (street 0, num_aggr==0) -> bet_* on the 4-size BB ladder (incl
        xlarge);  preflop 3-bet/4-bet (num_aggr>=1) -> raise_* (3 pot-relative sizes);
      * postflop first-in -> bet_* incl overbet;  facing a bet -> raise_* incl overbet.
    Voluntary all-in is always available when a shove is a genuine raise."""
    can_aggr = num_aggr < MAX_AGGR_PER_STREET and stack > max(0, to_call)
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
    return legal


def _sizing(size, street, pot, to_call, committed, num_aggr):
    """Chips ADDED for an abstract bet/raise size. Sizes come from
    abstractions/sizing.py (single source of truth); mirrors lbr._bot_sizing."""
    if street == 0:
        if num_aggr == 0:                          # open: absolute BB ladder
            to_amt = preflop_open_chips()[size]
            return int(round(to_amt - committed))
        # 3-bet / 4-bet+: pot-relative (unified, matches the engine).
        mult = PREFLOP_RAISE_MULT[size]
        return int(round(to_call + mult * (pot + to_call)))
    mult = POSTFLOP_BET_MULT[size]
    if to_call > 0:
        return int(round(to_call + mult * (pot + to_call)))
    return int(round(mult * pot))


class BlueprintPlayer:
    """Samples its abstract action from a blueprint at the bucketed info-set key."""

    def __init__(self, strategy_source, cards, rng):
        self.src = strategy_source
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

    def act(self, seat, hand, vis, street, pot, committed, stack, to_call, num_aggr, pattern):
        pos = 'ip' if seat == 0 else 'oop'
        if street == 0:
            key = make_info_set_key(0, pos, self.cards.get_bucket(list(hand), None), None, pattern)
        else:
            key = make_info_set_key(
                street, pos, self.cards.get_bucket(list(hand), None),
                self.cards.get_bucket(list(hand), vis), pattern)
        legal = _legal_actions(street, to_call, num_aggr, stack, pot)
        probs = self._restricted(key, tuple(legal))
        idx = self.rng.choices(range(len(legal)), weights=probs)[0]
        action = legal[idx]

        if action == 'fold':
            char, add, aggr = 'f', 0, False
        elif action == 'check':
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

        info = {'seat': seat, 'key': key, 'legal': legal, 'probs': probs, 'choice': idx}
        return char, add, aggr, info


class HeadToHeadMatch:
    """
    Play hands between strategy A and strategy B. Player A's perspective.
    `seat_of_A` alternates each hand to cancel positional asymmetry.
    """

    def __init__(self, strat_a, strat_b, seed=None):
        self.cards = CardAbstraction()
        self.evaluator = HandEvaluator()
        self.rng = random.Random(seed)
        self.pa = BlueprintPlayer(strat_a, self.cards, self.rng)
        self.pb = BlueprintPlayer(strat_b, self.cards, self.rng)

    def play_hand(self, seat_of_A, hand_a, hand_b, board, record=None):
        """Returns A's net chips. If `record` is a list, append a trajectory dict."""
        players = {seat_of_A: self.pa, 1 - seat_of_A: self.pb}
        hands = {seat_of_A: hand_a, 1 - seat_of_A: hand_b}
        invested = [SB, BB]
        stack = [STARTING_STACK - SB, STARTING_STACK - BB]
        folded = None
        allin_street = None      # street at which betting locked all-in (board cards known)
        events = [] if record is not None else None

        street = 0
        while street <= 3:
            vis = [] if street == 0 else board[:2 + street]
            if events is not None:
                events.append({'type': 'street_start', 'street': street,
                               'vis': list(vis), 'pot': sum(invested),
                               'invested': list(invested)})
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
                pot = sum(invested)
                char, add, aggr, info = players[actor].act(
                    actor, hands[actor], vis, street, pot, committed[actor],
                    stack[actor], to_call, num_aggr, pattern)
                if events is not None:
                    events.append({
                        'type': 'action', 'street': street, 'vis': list(vis),
                        'pot': pot, 'invested': list(invested), 'pattern': pattern,
                        **info, 'char': char, 'add': add, 'aggr': aggr})
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
            if min(stack) <= 0:
                allin_street = street     # all-in locked; board cards beyond this are pure chance
                break
            street += 1

        result = self._resolve(seat_of_A, hand_a, hand_b, board, invested, folded)
        if record is not None:
            record.append({
                'seat_of_A': seat_of_A, 'hand_a': hand_a, 'hand_b': hand_b,
                'board': list(board), 'events': events, 'result': result,
                'invested': list(invested), 'folded': folded,
                'allin_street': allin_street})
        return result

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

    def evaluate(self, num_hands=5000, record=None, progress_every=0):
        total = 0.0
        for i in range(num_hands):
            c = self.rng.sample(_FULL_DECK, 9)
            r = self.play_hand(i % 2, (c[0], c[1]), (c[2], c[3]), c[4:9], record=record)
            total += r
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  hand {i + 1}/{num_hands}", flush=True)
        avg = total / num_hands
        return {'raw_mbb': avg * 1000.0 / 2.0, 'num_hands': num_hands}
