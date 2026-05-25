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
from ..cfr.poker_game import PokerGame, STARTING_STACK
from ..abstractions.card_abstractions import CardAbstraction
from ..abstractions.hand_evaluator import HandEvaluator, RANK_MAP
from .cards import shuffled_deck, to_display_list
from .range_tracker import RangeTracker

_STREET_NAMES = ['preflop', 'flop', 'turn', 'river']
_BOARD_COUNT = [0, 3, 4, 5]              # community cards visible per street
_ACTION_CHARS = {
    'check': 'k', 'call': 'c', 'fold': 'f',
    'bet_small': 's', 'bet_medium': 'm', 'bet_large': 'l',
    'raise_small': 's', 'raise_medium': 'm', 'raise_large': 'l',
    'allin': 'a',
}
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


class GameError(Exception):
    """Raised on an illegal request (bad action, wrong turn, ...)."""


def _action_char(action):
    return _ACTION_CHARS.get(action, 'x')


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
        return cls(dict(data), strategy_fn=strategy_fn)

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
        else:
            d['opp_range'] = None
        self._settle()

    # ------------------------------------------------------------------
    # Opponent range tracking (Phase 3)
    # ------------------------------------------------------------------

    def _load_range(self):
        rt = self.data.get('opp_range')
        return RangeTracker.from_dict(rt, self.cards) if rt is not None else None

    def _opp_position(self):
        """The human's position string (the opponent the tracker models)."""
        return 'ip' if self.data['human_seat'] == 0 else 'oop'

    def opponent_read(self, k=8):
        """Bot's current belief about the human's hand: confidence + top-k hands.
        None when tracking is disabled (no opponent model)."""
        tracker = self._load_range()
        if tracker is None:
            return None
        return {
            'confidence': round(tracker.confidence, 4),
            'topHands': [{'cards': to_display_list(list(h)), 'prob': round(p, 4)}
                         for h, p in tracker.top_hands(k)],
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

    # ------------------------------------------------------------------
    # Applying actions
    # ------------------------------------------------------------------

    def apply_action(self, action):
        """Apply one action by whoever is currently to act."""
        if self.data['status'] != 'in_hand':
            raise GameError("Hand is over.")
        legal = self.legal_actions()
        if action not in legal:
            raise GameError(
                f"Illegal action {action!r}; legal: {legal}")

        d = self.data
        player = self.current_player()
        cost = self._action_cost(action)

        # Range-track the HUMAN's action (the opponent, from the bot's view).
        # Must read bet_pattern/board BEFORE this action is appended below, so
        # the lookup context matches how the blueprint keyed the decision.
        if (player == d['human_seat'] and self.strategy_fn is not None
                and d.get('opp_range') is not None):
            tracker = self._load_range()
            street = d['street']
            board = d['community'][:_BOARD_COUNT[min(street, 3)]]
            tracker.observe(self.strategy_fn, action, street, self._opp_position(),
                            d['bet_pattern'], legal, board)
            d['opp_range'] = tracker.to_dict()

        d['history'].append(action)
        d['bet_pattern'] += _action_char(action)
        if player == 0:
            d['p0_stack'] -= cost
        else:
            d['p1_stack'] -= cost
        d['action_log'].append({
            'player': player,
            'street': _STREET_NAMES[min(d['street'], 3)],
            'action': action,
            'chips': round(cost, 2),
        })
        self._settle()

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

        # Newly-dealt board cards are now impossible for the opponent to hold.
        # reveal() is model-free, so it runs regardless of strategy_fn.
        if d.get('opp_range') is not None:
            tracker = self._load_range()
            tracker.reveal(d['community'][:_BOARD_COUNT[min(d['street'], 3)]])
            d['opp_range'] = tracker.to_dict()

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

    def info_set_key(self, player):
        """Blueprint info-set key for `player` at the current node."""
        d = self.data
        cards = d['p0_cards'] if player == 0 else d['p1_cards']
        position = 'ip' if player == 0 else 'oop'
        street = d['street']
        pattern = d['bet_pattern']

        if street == 0:
            bucket = self.cards.get_bucket(cards, None)
            return f"{bucket}_{position}_{pattern}"
        starting = self.cards.get_bucket(cards, None)
        board = d['community'][:_BOARD_COUNT[street]]
        strength = self.cards.get_bucket(cards, board)
        return f"{starting}_{strength}_{position}_{_STREET_NAMES[street]}_{pattern}"

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
        return {
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
        if d['status'] == 'in_hand':
            to_act = 'you' if self.is_human_turn() else 'bot'
            if to_act == 'you':
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
            'community': to_display_list(d['community'][:board_n]),
            'pot': round(pot, 2),
            'yourBet': round(your_bet, 2),
            'botBet': round(bot_bet, 2),
            'yourStack': round(human_stack, 2),
            'botStack': round(bot_stack, 2),
            'humanNet': round(d['human_net'], 2),
            'toAct': to_act,
            'legalActions': legal,
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
        session.apply_action(action)
        guard += 1
        if guard > 64:
            raise GameError("Bot turn loop did not terminate.")
