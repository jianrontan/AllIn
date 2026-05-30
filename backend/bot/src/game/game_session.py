# backend/bot/src/game/game_session.py
"""
GameSession — one heads-up hand against the bot, played under the abstracted
rules the blueprint was trained on.

Design notes
------------
* All persistent state lives in `self.data`, a plain JSON-serialisable dict.
  PokerGame / CardAbstraction are stateless helpers rebuilt on construction,
  so a session round-trips cleanly through any SessionStore.
* Player frame matches training: player 0 = SB/button (acts first preflop,
  "ip"), player 1 = BB (acts first postflop, "oop"). `human_seat` records
  which of those the human is this hand; it alternates each hand.
* Each hand starts both players at STARTING_STACK (the blueprint assumes ~200
  effective). Cross-hand profit/loss is tracked separately in `human_net`.
* Cards are stored in engine format (SuitRank); conversion to display format
  happens only in public_view().
"""
import copy
import math

from ..cfr.poker_game import (
    PokerGame, STARTING_STACK, _is_custom, _custom_total, make_custom_action)
from ..cfr.keys import action_char, make_info_set_key
from ..cfr import translation
from ..abstractions.card_abstractions import CardAbstraction
from ..abstractions.hand_evaluator import HandEvaluator, RANK_MAP
from .cards import shuffled_deck, to_display_list
from .range_tracker import RangeTracker

_STREET_NAMES = ['preflop', 'flop', 'turn', 'river']
_BOARD_COUNT = [0, 3, 4, 5]              # community cards visible per street
BIG_BLIND = 2

_RANK_NAMES = {
    '2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five', '6': 'Six', '7': 'Seven',
    '8': 'Eight', '9': 'Nine', 'T': 'Ten', 'J': 'Jack', 'Q': 'Queen',
    'K': 'King', 'A': 'Ace',
}
_RANK_PLURAL = {
    '2': 'Twos', '3': 'Threes', '4': 'Fours', '5': 'Fives', '6': 'Sixes',
    '7': 'Sevens', '8': 'Eights', '9': 'Nines', 'T': 'Tens', 'J': 'Jacks',
    'Q': 'Queens', 'K': 'Kings', 'A': 'Aces',
}
_HAND_TYPE_LABEL = {
    'high_card': 'High card', 'pair': 'Pair', 'two_pair': 'Two pair',
    'three_of_kind': 'Three of a kind', 'straight': 'Straight', 'flush': 'Flush',
    'full_house': 'Full house', 'four_of_kind': 'Four of a kind',
    'straight_flush': 'Straight flush',
}


def _read_group_label(hand, relevant):
    """Poker-shorthand label for a hole-card combo (two engine SuitRank cards,
    e.g. ('HA','CK')), collapsing strategically-equivalent suits. `relevant` is
    the set of flush-relevant board suits ('H'/'D'/'C'/'S'). A card's suit is
    shown only if it's flush-relevant; suited/offsuit is always preserved.

      rainbow board:  AhAc/AhAs/... -> 'AA'   AhKh -> 'AKs'   AhKc -> 'AKo'
      two-heart board: Ah-anything-A -> 'AhA' (holds the heart ace, a blocker),
                       AhKh -> 'AhKh' (the flush draw), AhKc -> 'AhK'.
    """
    (s1, r1), (s2, r2) = sorted(((c[0], c[1]) for c in hand),
                                key=lambda sr: -RANK_MAP[sr[1]])

    def disp(rank, suit):
        return rank + suit.lower() if suit in relevant else rank

    if r1 == r2:                                   # pair
        toks = sorted([disp(r1, s1), disp(r2, s2)], key=len, reverse=True)
        return ''.join(toks)                       # 'AA' or 'AhA' or 'AhAs'
    if s1 == s2:                                   # suited
        return f"{disp(r1, s1)}{disp(r2, s2)}" if s1 in relevant else f"{r1}{r2}s"
    d1, d2 = disp(r1, s1), disp(r2, s2)            # offsuit
    if len(d1) == 1 and len(d2) == 1:              # no flush-relevant suit shown
        return f"{r1}{r2}o"
    return f"{d1}{d2}"                              # which exact flush card(s) held


class GameError(Exception):
    """Raised on an illegal request (bad action, wrong turn, ...)."""


class GameSession:
    def __init__(self, data, strategy_fn=None):
        self.data = data
        self.game = PokerGame()
        self.cards = CardAbstraction()
        self.evaluator = HandEvaluator()
        # Opponent model for the hand-level range tracker (Phase 3). A callback
        # strategy_fn(key, legal)->probs; the live API injects the blueprint's
        # average strategy. None disables tracking (the tracker is never built),
        # so DB-free callers (tests, the strategy explorer) are unaffected. Never
        # serialised -- it's a rebuilt helper like self.game / self.cards.
        self.strategy_fn = strategy_fn

    # ------------------------------------------------------------------
    # Construction / serialisation
    # ------------------------------------------------------------------

    @classmethod
    def new(cls, session_id, player_id, strategy_fn=None):
        session = cls({
            'session_id': session_id,
            'player_id': player_id,
            'human_net': 0.0,
        }, strategy_fn=strategy_fn)
        session._deal_hand(hand_number=1, human_seat=0)
        return session

    @classmethod
    def from_dict(cls, data, strategy_fn=None):
        # Deep-copy so the live session never aliases the stored dict's nested
        # lists/dicts (history, action_log, community, opp_range, ...). Without
        # this, in-place mutations would leak into the store before put() and a
        # mid-apply failure could corrupt the persisted state.
        return cls(copy.deepcopy(data), strategy_fn=strategy_fn)

    def to_dict(self):
        return self.data

    def start_next_hand(self):
        """Deal the next hand; the button (SB) alternates."""
        if self.data['status'] != 'hand_over':
            raise GameError("Current hand is not over.")
        self._deal_hand(
            hand_number=self.data['hand_number'] + 1,
            human_seat=1 - self.data['human_seat'])

    def _deal_hand(self, hand_number, human_seat):
        deck = shuffled_deck()
        d = self.data
        d['hand_number'] = hand_number
        d['human_seat'] = human_seat
        d['p0_cards'] = deck[0:2]
        d['p1_cards'] = deck[2:4]
        d['community'] = deck[4:9]
        d['street'] = 0
        d['history'] = []
        d['bet_pattern'] = ''
        d['starting_pot'] = 3.0                  # SB(1) + BB(2)
        d['p0_invested'] = 0.0
        d['p1_invested'] = 0.0
        d['p0_stack'] = float(STARTING_STACK - 1)
        d['p1_stack'] = float(STARTING_STACK - 2)
        d['status'] = 'in_hand'
        d['action_log'] = []
        d['result'] = None
        d['revealed_board'] = 0
        # Hand-level belief over the HUMAN's hole cards, from the bot's seat
        # (the bot knows its own cards, so they're removed from the combos).
        # Only built when an opponent model is available; None disables tracking.
        if self.strategy_fn is not None:
            bot_seat = 1 - human_seat
            bot_cards = d['p0_cards'] if bot_seat == 0 else d['p1_cards']
            d['opp_range'] = RangeTracker(bot_cards, self.cards).to_dict()
            # The bot's OWN blueprint-reach range, for the Phase-4 river solver
            # (hero range). hero_hole=() spans all hands -- the solver does not
            # condition on the human's cards; pairwise removal is left to the
            # showdown kernel. Built by observing the BOT's actions (below).
            d['bot_range'] = RangeTracker((), self.cards).to_dict()
        else:
            d['opp_range'] = None
            d['bot_range'] = None
        # River-entry snapshots of both beliefs (frozen before river betting) that
        # the subgame solver consumes; filled when the river is dealt.
        d['river_entry_opp'] = None
        d['river_entry_bot'] = None
        self._settle()

    # ------------------------------------------------------------------
    # Opponent range tracking (Phase 3)
    # ------------------------------------------------------------------

    def _load_range(self):
        rt = self.data.get('opp_range')
        return RangeTracker.from_dict(rt, self.cards) if rt is not None else None

    def _load_bot_range(self):
        rt = self.data.get('bot_range')
        return RangeTracker.from_dict(rt, self.cards) if rt is not None else None

    def _opp_position(self):
        """The human's position string (the opponent the tracker models)."""
        return 'ip' if self.data['human_seat'] == 0 else 'oop'

    def _bot_position(self):
        """The bot's position string (opposite of the human)."""
        return 'oop' if self.data['human_seat'] == 0 else 'ip'

    def opponent_read(self, k=8):
        """Bot's current belief about the human's hand: confidence + top-k hand
        GROUPS. Strategically-equivalent combos are merged given the board: suits
        are hidden unless flush-relevant, so e.g. all 6 pocket-aces collapse to
        'AA' on a rainbow board, but on a two-heart board the heart aces split out
        ('AhA'), and a heart flush draw shows 'AhKh'. None when tracking is off."""
        tracker = self._load_range()
        if tracker is None:
            return None
        street = min(self.data['street'], 3)
        board = self.data['community'][:_BOARD_COUNT[street]]
        # A suit is flush-relevant when a hole card of it can still be part of a
        # 5-card flush: >=2 on the board pre-river (draw), >=3 on the river (made).
        thr = max(2, len(board) - 2)
        bcount = {}
        for c in board:
            bcount[c[0]] = bcount.get(c[0], 0) + 1
        relevant = {s for s, n in bcount.items() if n >= thr}

        agg = {}
        for h, p in zip(tracker.hands, tracker.normalized_weights()):
            if p <= 0.0:
                continue
            label = _read_group_label(h, relevant)
            agg[label] = agg.get(label, 0.0) + float(p)
        top = sorted(agg.items(), key=lambda kv: -kv[1])[:k]
        return {
            'confidence': round(tracker.confidence, 4),
            'topHands': [{'label': lab, 'prob': round(pr, 4)} for lab, pr in top],
        }

    # ------------------------------------------------------------------
    # Turn / action queries
    # ------------------------------------------------------------------

    def current_player(self):
        return self.game._acting_player(
            len(self.data['history']), self.data['street'])

    def is_human_turn(self):
        return (self.data['status'] == 'in_hand'
                and self.current_player() == self.data['human_seat'])

    def legal_actions(self):
        if self.data['status'] != 'in_hand':
            return []
        d = self.data
        return self.game.get_legal_actions(
            d['street'], d['history'], d['starting_pot'], self.current_player(),
            d['p0_stack'], d['p1_stack'], d['p0_invested'], d['p1_invested'])

    def _action_cost(self, action):
        d = self.data
        return self.game._action_cost(
            action, d['street'], d['history'], d['starting_pot'],
            self.current_player(), d['p0_invested'], d['p1_invested'])

    def _current_pot(self):
        d = self.data
        return self.game.calculate_current_pot(
            d['starting_pot'], d['history'], d['street'],
            d['p0_invested'], d['p1_invested'])

    def _node_grid(self, legal):
        """The trained bet-size grid available at the current node, as a sorted
        [(char, frac), ...] on the eff_fraction axis (bet / pot-after-call).
        Built from the engine's real sizes so it is correct preflop (absolute
        ladders) and postflop (pot multipliers). Translation maps an off-grid
        custom bet onto these chars."""
        pot = self._current_pot()
        to_call = self._action_cost('call') if 'call' in legal else 0.0
        grid = {}
        for a in legal:
            if a.startswith(('bet_', 'raise_')):
                frac = translation.eff_fraction(self._action_cost(a), to_call, pot)
                grid[action_char(a)] = frac
            elif a == 'allin':
                frac = translation.eff_fraction(self._action_cost('allin'), to_call, pot)
                grid['a'] = frac
        return sorted(grid.items(), key=lambda cf: cf[1])

    def _river_path_specs(self):
        """The realized river actions as RiverSubgameSolver navigation specs:
        a plain label ('check'/'call'/'fold'/'allin') or ('bet'|'raise', chips)
        for a sized action (chips = the additional cost when it was taken). Used
        to walk the solver's tree to the bot's current decision node."""
        d = self.data
        if d['street'] != 3:
            return []
        specs, hist = [], d['history']
        for i, a in enumerate(hist):
            if a in ('check', 'call', 'fold', 'allin'):
                specs.append(a)
            else:
                cost = self.game._action_cost(
                    a, 3, hist[:i], d['starting_pot'],
                    self.game._acting_player(i, 3), d['p0_invested'], d['p1_invested'])
                specs.append(('raise' if a.startswith('raise') else 'bet', cost))
        return specs

    def custom_bounds(self):
        """Min/max legal raise-to TOTAL (CHIPS) for a custom bet/raise by the
        player to act, or None if no custom bet/raise is legal here. The API/UI
        validate a human custom amount against this."""
        if self.data['status'] != 'in_hand':
            return None
        d = self.data
        return self.game.custom_bet_bounds(
            d['street'], d['history'], d['starting_pot'], self.current_player(),
            d['p0_stack'], d['p1_stack'], d['p0_invested'], d['p1_invested'])

    # ------------------------------------------------------------------
    # Applying actions
    # ------------------------------------------------------------------

    def apply_action(self, action):
        """Apply one action by whoever is currently to act.

        `action` is a grid action (check/call/fold/bet_*/raise_*/allin) or a
        custom action 'bet_custom_<total>' / 'raise_custom_<total>' carrying an
        unrestricted raise-to chip total. A custom bet is mapped onto the trained
        grid (pseudo-harmonic translation): its pattern char is the nearest grid
        size, and a blended response for the bot is stashed in pending_translation."""
        if self.data['status'] != 'in_hand':
            raise GameError("Hand is over.")
        d = self.data
        legal = self.legal_actions()
        player = self.current_player()
        custom = _is_custom(action)

        # -- legality --
        snapped_action = action
        translated = None
        base_pattern = d['bet_pattern']
        if custom:
            action = self._validate_custom(action, legal)
            if action == 'allin':
                # A custom raise-to that meets/exceeds the stack normalizes to an
                # all-in. Its char is 'a' directly -- do NOT snap via nearest_char,
                # which (when the grid omits an 'a' edge) would mis-record the shove
                # as the nearest SIZED char ('l'/'o') and corrupt the info-set key
                # and the range-tracker observe.
                char = action_char('allin')
                snapped_action = 'allin'
            else:
                grid = self._node_grid(legal)
                pot, to_call = self._current_pot(), (self._action_cost('call') if 'call' in legal else 0.0)
                eff = translation.eff_fraction(self._action_cost(action), to_call, pot)
                translated = translation.translate_bet(eff, grid)
                char = translation.nearest_char(eff, grid)
                snapped_action = self._grid_action_for_char(char, legal)
        elif action not in legal:
            raise GameError(f"Illegal action {action!r}; legal: {legal}")
        else:
            char = action_char(action)

        cost = self._action_cost(action)

        # Range-track the HUMAN's action (the opponent, from the bot's view).
        # Must read bet_pattern/board BEFORE this action is appended. A custom
        # bet is observed as its nearest grid action (the tracker models the
        # opponent as the blueprint, which only knows grid sizes; a far-off-grid
        # bet then looks improbable and correctly erodes confidence).
        if (player == d['human_seat'] and self.strategy_fn is not None
                and d.get('opp_range') is not None):
            tracker = self._load_range()
            street = d['street']
            board = d['community'][:_BOARD_COUNT[min(street, 3)]]
            tracker.observe(self.strategy_fn, snapped_action, street, self._opp_position(),
                            d['bet_pattern'], legal, board)
            d['opp_range'] = tracker.to_dict()

        # Symmetrically, build the BOT's own blueprint-reach range (the hero range
        # the river solver consumes) by observing the bot's actions. Frozen at
        # river entry via river_entry_bot, so observing the bot's river actions
        # here is harmless (the solver uses the snapshot, not the live tracker).
        if (player != d['human_seat'] and self.strategy_fn is not None
                and d.get('bot_range') is not None):
            bt = self._load_bot_range()
            street = d['street']
            board = d['community'][:_BOARD_COUNT[min(street, 3)]]
            bt.observe(self.strategy_fn, snapped_action, street, self._bot_position(),
                       d['bet_pattern'], legal, board)
            d['bot_range'] = bt.to_dict()

        d['history'].append(action)
        d['bet_pattern'] += char
        if player == 0:
            d['p0_stack'] -= cost
        else:
            d['p1_stack'] -= cost
        d['action_log'].append({
            'player': player,
            'street': _STREET_NAMES[min(d['street'], 3)],
            'action': snapped_action if custom else action,
            'chips': round(cost, 2),
        })

        # An off-grid human bet leaves a blended response for the bot to consume
        # on its immediate next decision; any other action clears it.
        if custom and translated is not None and len(translated) > 1:
            d['pending_translation'] = {'base_pattern': base_pattern,
                                        'weights': [[c, w] for c, w in translated]}
        else:
            d.pop('pending_translation', None)

        self._settle()

    def _validate_custom(self, action, legal):
        """Validate a custom bet/raise against the node's bounds; normalise an
        at/above-stack request to 'allin'. Returns the action to apply."""
        bounds = self.custom_bounds()
        if bounds is None:
            raise GameError("No custom bet/raise is legal here.")
        lo, hi = bounds
        total = _custom_total(action)
        is_raise = action.startswith('raise_')
        if total >= hi:                       # whole stack -> all-in
            # A full-stack shove is always a legal raise when aggression is open
            # (bounds is not None), even when the engine's legal list omits
            # 'allin' because every sized bet was still affordable.
            return 'allin'
        if total < lo - 1e-9:
            raise GameError(
                f"Custom amount {total:g} below minimum {lo:g} (chips).")
        return make_custom_action(is_raise, total)

    @staticmethod
    def _grid_action_for_char(char, legal):
        """The legal grid action whose pattern char is `char` (e.g. 'l' ->
        whichever of bet_large / raise_large is legal here). Used to summarise a
        custom bet as a real grid action for the range tracker."""
        if char == 'a' and 'allin' in legal:
            return 'allin'
        for a in legal:
            if a.startswith(('bet_', 'raise_')) and action_char(a) == char:
                return a
        return 'allin' if 'allin' in legal else legal[0]

    def _settle(self):
        """Resolve terminal nodes / advance streets until a player must act."""
        while True:
            d = self.data
            if self.game.is_terminal(d['history'], d['street']):
                self._resolve()
                return
            if self.legal_actions():
                return                       # waiting for a player
            self._advance_street()           # round complete, deal next street

    def _advance_street(self):
        d = self.data
        street, hist, pot = d['street'], d['history'], d['starting_pot']
        p0_inv, p1_inv = d['p0_invested'], d['p1_invested']

        current_pot = self.game.calculate_current_pot(pot, hist, street, p0_inv, p1_inv)
        p0_this = self.game.get_player_contribution_this_round(
            hist, street, pot, 0, p0_inv, p1_inv)
        p1_this = self.game.get_player_contribution_this_round(
            hist, street, pot, 1, p0_inv, p1_inv)

        d['p0_stack'] = STARTING_STACK - (p0_inv + p0_this)
        d['p1_stack'] = STARTING_STACK - (p1_inv + p1_this)
        d['street'] = street + 1
        d['starting_pot'] = current_pot
        d['p0_invested'] = p0_inv + p0_this
        d['p1_invested'] = p1_inv + p1_this
        d['history'] = []
        d['bet_pattern'] = ''

        # Newly-dealt board cards are now impossible for either player to hold.
        # reveal() is model-free, so it runs regardless of strategy_fn.
        board_now = d['community'][:_BOARD_COUNT[min(d['street'], 3)]]
        if d.get('opp_range') is not None:
            tracker = self._load_range()
            tracker.reveal(board_now)
            d['opp_range'] = tracker.to_dict()
        if d.get('bot_range') is not None:
            bt = self._load_bot_range()
            bt.reveal(board_now)
            d['bot_range'] = bt.to_dict()

        # Entering the river: snapshot both beliefs (post card-removal, before any
        # river betting) as the subgame solver's frozen inputs -- this is what
        # avoids re-modelling river actions the live trackers will keep absorbing.
        if d['street'] == 3:
            d['river_entry_opp'] = d.get('opp_range')
            d['river_entry_bot'] = d.get('bot_range')

    def _resolve(self):
        d = self.data
        street = min(d['street'], 3)
        util_p0 = self.game.get_utility(
            d['p0_cards'], d['p1_cards'], d['community'], d['history'],
            street, d['starting_pot'], d['p0_invested'], d['p1_invested'])

        human_delta = util_p0 if d['human_seat'] == 0 else -util_p0
        d['human_net'] += human_delta
        final_pot = self.game.calculate_current_pot(
            d['starting_pot'], d['history'], street,
            d['p0_invested'], d['p1_invested'])

        folded = 'fold' in d['history']
        # All-in runs the board out; otherwise the board stops at this street.
        d['revealed_board'] = 5 if 'allin' in d['history'] else _BOARD_COUNT[street]

        if human_delta > 1e-9:
            winner = 'you'
        elif human_delta < -1e-9:
            winner = 'bot'
        else:
            winner = 'split'

        d['result'] = {
            'humanDelta': round(human_delta, 2),
            'winner': winner,
            'reason': 'fold' if folded else 'showdown',
            'finalPot': round(final_pot, 2),
        }
        d['status'] = 'hand_over'

    # ------------------------------------------------------------------
    # Info-set keys (for the bot, and for the optional "show blueprint" UI)
    # ------------------------------------------------------------------

    def info_set_key(self, player, pattern=None):
        """Blueprint info-set key for `player` at the current node. `pattern`
        overrides the live bet_pattern (used to build the bracketing keys for
        action translation)."""
        d = self.data
        cards = d['p0_cards'] if player == 0 else d['p1_cards']
        position = 'ip' if player == 0 else 'oop'
        street = d['street']
        pattern = d['bet_pattern'] if pattern is None else pattern

        # Built through keys.py (single source of truth) so the live play path
        # can never drift from the trainer / evaluation key format.
        if street == 0:
            return make_info_set_key(0, position, self.cards.get_bucket(cards, None),
                                     None, pattern)
        starting = self.cards.get_bucket(cards, None)
        board = d['community'][:_BOARD_COUNT[street]]
        strength = self.cards.get_bucket(cards, board)
        return make_info_set_key(street, position, starting, strength, pattern)

    def bot_public_state(self):
        """State handed to BotStrategy.decide() for the player to act.

        Richer than the bucketed key: includes the acting player's own
        `hole_cards` (its own cards, never the opponent's), the live `opp_range`
        tracker, and `to_call` so a range-aware/solver strategy has everything it
        needs. The plain BlueprintStrategy ignores the extras."""
        d = self.data
        street = min(d['street'], 3)
        actor = self.current_player()
        legal = self.legal_actions()
        to_call = self._action_cost('call') if 'call' in legal else 0.0
        state = {
            'street': _STREET_NAMES[street],
            'pot': self.game.calculate_current_pot(
                d['starting_pot'], d['history'], d['street'],
                d['p0_invested'], d['p1_invested']),
            'community': d['community'][:_BOARD_COUNT[street]],
            'history': list(d['history']),
            'p0_stack': d['p0_stack'],
            'p1_stack': d['p1_stack'],
            'hole_cards': d['p0_cards'] if actor == 0 else d['p1_cards'],
            'to_call': to_call,
            'opp_range': self._load_range(),
        }
        # On the RIVER, hand the subgame solver its inputs: the river-entry
        # snapshots of both ranges (frozen before river betting), the river-entry
        # pot + (equal) behind stacks, the bot's seat, and the realized river path.
        # opp_range is overridden with the snapshot so the solver does not see the
        # live tracker's river updates (which would double-model river actions).
        # Absent these keys (off-river, or no model), the solver falls back.
        if street == 3 and d.get('river_entry_bot') is not None:
            state['botSeat'] = actor
            state['riverEntryPot'] = d['starting_pot']
            state['riverEntryStacks'] = (STARTING_STACK - d['p0_invested'],
                                         STARTING_STACK - d['p1_invested'])
            state['hero_range'] = RangeTracker.from_dict(d['river_entry_bot'], self.cards)
            if d.get('river_entry_opp') is not None:
                state['opp_range'] = RangeTracker.from_dict(d['river_entry_opp'], self.cards)
            state['riverPath'] = self._river_path_specs()

        # If the opponent just made an off-grid bet, hand the bot the bracketing
        # blueprint keys + pseudo-harmonic weights so it blends the responses to
        # the two adjacent grid sizes instead of snapping to one (action
        # translation). The plain BlueprintStrategy ignores this; the
        # ConfidenceAwareStrategy / a solver consume it.
        pend = d.get('pending_translation')
        if pend:
            state['translation'] = [
                (self.info_set_key(actor, pattern=pend['base_pattern'] + c), w)
                for c, w in pend['weights']]
        return state

    def describe_hand(self, cards, board):
        """
        Plain-English label for the made hand `cards` hold given `board`
        (both in engine SuitRank format). Preflop (no board) describes the
        two hole cards directly; phevaluator needs at least 5 cards.
        """
        if not board:
            r0, r1 = cards[0][1], cards[1][1]
            if r0 == r1:
                return f"Pair of {_RANK_PLURAL[r0]}"
            hi, lo = sorted((r0, r1), key=lambda r: RANK_MAP[r], reverse=True)
            suited = ' suited' if cards[0][0] == cards[1][0] else ''
            return f"{_RANK_NAMES[hi]}-{_RANK_NAMES[lo]}{suited} high"
        hand_type, _ = self.evaluator.evaluate_hand_strength(cards, board)
        return _HAND_TYPE_LABEL[hand_type]

    # ------------------------------------------------------------------
    # Redacted view for the frontend
    # ------------------------------------------------------------------

    def public_view(self):
        d = self.data
        human, bot = d['human_seat'], 1 - d['human_seat']
        hand_over = d['status'] == 'hand_over'
        street = min(d['street'], 3)

        board_n = d['revealed_board'] if hand_over else _BOARD_COUNT[street]
        human_cards = d['p0_cards'] if human == 0 else d['p1_cards']
        bot_cards = d['p0_cards'] if bot == 0 else d['p1_cards']
        human_stack = d['p0_stack'] if human == 0 else d['p1_stack']
        bot_stack = d['p0_stack'] if bot == 0 else d['p1_stack']

        # `pot` is the chips already gathered in the middle; `*_bet` is what
        # each player currently has wagered in front of them this street
        # (PokerNow-style). pot + both bets = total chips in play.
        if hand_over:
            pot = d['result']['finalPot']
            your_bet = bot_bet = 0.0
        else:
            grand_total = self.game.calculate_current_pot(
                d['starting_pot'], d['history'], d['street'],
                d['p0_invested'], d['p1_invested'])
            p0_bet = self.game.get_player_contribution_this_round(
                d['history'], d['street'], d['starting_pot'], 0,
                d['p0_invested'], d['p1_invested'])
            p1_bet = self.game.get_player_contribution_this_round(
                d['history'], d['street'], d['starting_pot'], 1,
                d['p0_invested'], d['p1_invested'])
            pot = grand_total - p0_bet - p1_bet
            your_bet = p0_bet if human == 0 else p1_bet
            bot_bet = p0_bet if bot == 0 else p1_bet

        to_act = None
        legal = []
        custom_bounds = None
        if d['status'] == 'in_hand':
            to_act = 'you' if self.is_human_turn() else 'bot'
            if to_act == 'you':
                cb = self.custom_bounds()
                if cb is not None:
                    # Min/max raise-to TOTAL, in BB, for the custom-amount box.
                    # Round the min UP and the max DOWN to the cent so the shown
                    # range stays strictly inside the engine's exact (unrounded)
                    # bounds -- otherwise a value the UI thinks is valid can fall a
                    # hair below the real min-raise and get rejected with a 400.
                    custom_bounds = {'minBb': math.ceil(cb[0] / BIG_BLIND * 100) / 100,
                                     'maxBb': math.floor(cb[1] / BIG_BLIND * 100) / 100}
                for action in self.legal_actions():
                    cost = self.game._action_cost(
                        action, d['street'], d['history'], d['starting_pot'],
                        self.current_player(), d['p0_invested'], d['p1_invested'])
                    legal.append({
                        'action': action,
                        'chips': round(cost, 2),
                        'bb': round(cost / BIG_BLIND, 2),
                    })

        action_log = [{
            'seat': 'you' if e['player'] == human else 'bot',
            'street': e['street'],
            'action': e['action'],
            'chips': e['chips'],
        } for e in d['action_log']]

        return {
            'sessionId': d['session_id'],
            'handNumber': d['hand_number'],
            'status': d['status'],
            'street': _STREET_NAMES[street],
            'yourSeat': 'button' if human == 0 else 'bigblind',
            'yourCards': to_display_list(human_cards),
            'yourHand': self.describe_hand(human_cards, d['community'][:board_n]),
            'botCards': to_display_list(bot_cards) if hand_over else None,
            # Bot's made-hand label, only once its cards are revealed at showdown.
            'botHand': (self.describe_hand(bot_cards, d['community'][:board_n])
                        if hand_over else None),
            'community': to_display_list(d['community'][:board_n]),
            'pot': round(pot, 2),
            # Grand total in the pot INCLUDING chips wagered this street (what the
            # players are contesting right now); `pot` excludes the live bets.
            'totalPot': round(d['result']['finalPot'] if hand_over
                              else grand_total, 2),
            'yourBet': round(your_bet, 2),
            'botBet': round(bot_bet, 2),
            'yourStack': round(human_stack, 2),
            'botStack': round(bot_stack, 2),
            'humanNet': round(d['human_net'], 2),
            'toAct': to_act,
            'legalActions': legal,
            # Min/max for the unrestricted custom-bet box (BB), or None when no
            # custom bet/raise is legal at this node.
            'customBounds': custom_bounds,
            'actionLog': action_log,
            'result': d['result'],
            # The bot's belief about the human's hand (confidence + top hands).
            # Safe to show: it's a guess about the human's OWN cards and never
            # reveals the bot's cards. None when tracking is disabled.
            'botRead': None if hand_over else self.opponent_read(6),
        }


def advance_bot_turns(session, bot_strategy):
    """Apply bot actions until it is the human's turn or the hand ends."""
    guard = 0
    while (session.data['status'] == 'in_hand'
           and not session.is_human_turn()):
        bot = session.current_player()
        action = bot_strategy.decide(
            session.info_set_key(bot),
            session.legal_actions(),
            session.bot_public_state())
        try:
            session.apply_action(action)
        except GameError:
            # A borderline solver custom size the engine rejected at the margin
            # must never crash a hand -- fall back to a safe legal action.
            legal = session.legal_actions()
            if not legal:
                break                       # nothing legal -> let _settle/guard end it
            safe = next((a for a in ('check', 'call', 'fold') if a in legal), legal[0])
            try:
                session.apply_action(safe)
            except GameError:
                break                       # even the fallback is illegal -> bail safely
        guard += 1
        if guard > 64:
            raise GameError("Bot turn loop did not terminate.")
