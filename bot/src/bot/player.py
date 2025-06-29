from pypokerengine.players import BasePokerPlayer
from src.bot.game_adapter import GameAdapter
from src.cfr.information_set import InformationSet
import random


class Player(BasePokerPlayer):
    """
    PyPokerEngine format round information:
    round_state = {
        'street': 'preflop',  (Current betting round)
        'pot': {'main': {'amount': 25}},  (Pot information)
        'community_card': [],  (Community cards, empty preflop)
        'seats': [  (Player information)
            {
                'uuid': 'player_1_uuid',
                'name': 'Player1',
                'stack': 98,
                'state': 'participating'  (or 'folded', 'allin')
            },
            {
                'uuid': 'player_2_uuid', 
                'name': 'Player2',
                'stack': 97,
                'state': 'participating'
            }
        ],
        'action_histories': {  (All previous actions taken)
            'preflop': [
                {'action': 'SMALLBLIND', 'amount': 1, 'uuid': 'player_1_uuid'},
                {'action': 'BIGBLIND', 'amount': 2, 'uuid': 'player_2_uuid'},
                {'action': 'CALL', 'amount': 2, 'uuid': 'player_1_uuid'}
            ]
        }
    }
    """

    def __init__(self):
        self.info_sets = {}
        self.game_adapter = GameAdapter()  # Bridges pypokerengine to my CFR format
        self.my_uuid = None  # Set during first round
        self.load_trained_strategy()

    def load_trained_strategy(self):
        """Load trained blueprint strategy from file"""
        try:
            with open("strategies/blueprint_strategy.pkl", 'rb') as f:
                import pickle
                self.info_sets = pickle.load(f)
            print(
                f"Loaded blueprint with {len(self.info_sets)} information sets")
        except FileNotFoundError:
            print("No blueprint strategy found, using random play")
            # Keep empty info_sets for random play

    def declare_action(self, valid_actions, hole_card, round_state):
        """
        Main decision method - replaces CFR decision logic
        This is where CFR strategy gets applied
        """
        # Convert PyPokerEngine data to info set format
        info_set_key = self.game_adapter.create_info_set_key(
            hole_card, round_state)

        # Get or create information set
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = InformationSet()
        info_set = self.info_sets[info_set_key]

        # Convert PyPokerEngine actions to CFR format
        cfr_actions = self.game_adapter.action_abstractions.pypoker_to_cfr_actions(
            valid_actions, self.extract_game_state(round_state)
        )

        # Get strategy
        # For now, use uniform strategy - later this will be the trained CFR strategy
        strategy = info_set.get_average_strategy(cfr_actions)

        # Select action based on strategy
        selected_cfr_action = self.select_action_from_strategy(
            cfr_actions, strategy)

        # Convert back to PyPokerEngine format
        action, amount = self.game_adapter.action_abstractions.cfr_to_pypoker_action(
            selected_cfr_action, self.extract_game_state(round_state)
        )

        return action, amount

    def select_action_from_strategy(self, actions, strategy):
        """Select action based on strategy probabilities"""
        return random.choices(actions, weights=strategy)[0]

    def extract_game_state(self, round_state):
        """Extract relevant game state info for abstractions"""
        pot_size = round_state.get('pot', {}).get('main', {}).get('amount', 0)
        big_blind = self.extract_big_blind(round_state)
        current_bet = self.extract_current_bet(round_state)

        if self.my_uuid:
            player_stack = self.extract_player_stack(round_state, self.my_uuid)
            player_contribution = self.extract_player_contribution(
                round_state, self.my_uuid)
        else:
            player_stack = 100
            player_contribution = 0

        return {
            'pot_size': pot_size,
            'player_stack': player_stack,
            'current_bet': current_bet,
            'player_contribution': player_contribution,
            'big_blind': big_blind
        }

    def extract_big_blind(self, round_state):
        """Extract big blind directly from action history"""
        preflop_actions = round_state.get(
            'action_histories', {}).get('preflop', [])
        for action in preflop_actions:
            if action.get('action', '').upper() == 'BIGBLIND':
                return action.get('amount', 2)
        return 2  # Default

    def extract_current_bet(self, round_state):
        """Extract current bet - the last bet/raise amount, not max"""
        current_street = round_state.get(
            'street', 'preflop')
        actions = round_state.get(
            'action_histories', {}).get(current_street, [])

        current_bet = 0
        for action in actions:
            action_type = action.get('action', '').upper()
            if action_type in ['BET', 'RAISE', 'BIGBLIND']:
                current_bet = action.get('amount', 0)

        return current_bet

    def extract_player_stack(self, round_state, player_uuid):
        """Extract specific player's stack"""
        seats = round_state.get('seats', [])
        for seat in seats:
            if seat.get('uuid') == player_uuid:
                return seat.get('stack', 0)
        return 0

    def extract_player_contribution(self, round_state, player_uuid):
        """Extract how much specific player has contributed this street"""
        current_street = round_state.get('street', 'preflop')
        actions = round_state.get(
            'action_histories', {}).get(current_street, [])

        total_contribution = 0
        for action in actions:
            if action.get('uuid') == player_uuid:
                action_type = action.get('action', '').upper()
                if action_type in ['BET', 'RAISE', 'CALL', 'BIGBLIND', 'SMALLBLIND']:
                    if action_type == 'CALL':
                        total_contribution += action.get('amount', 0)
                    else:
                        total_contribution = action.get('amount', 0)

        return total_contribution

    def receive_round_start_message(self, round_count, hole_card, seats):
        if self.my_uuid is None:
            # Match by name (set when registering the player)
            for seat in seats:
                if seat.get('name') == 'CFR_Bot':  # The name used in config.register_player()
                    self.my_uuid = seat.get('uuid')
                    break

    def receive_game_start_message(self, game_info):
        """Setup game parameters"""
        self.game_config = game_info

    def receive_street_start_message(self, street, round_state):
        """Handle street transitions"""
        self.current_street = street

    def receive_game_update_message(self, action, round_state):
        """Track actions for history building"""
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        """Learn from results"""
        pass
