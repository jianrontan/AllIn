# backend/bot/src/bot/game_adapter.py
from ..abstractions.card_abstractions import CardAbstraction
from ..abstractions.action_abstractions import ActionAbstraction


class GameAdapter:
    def __init__(self):
        self.card_abstractions = CardAbstraction()
        self.action_abstractions = ActionAbstraction()

    def create_info_set_key(self, hole_card, round_state):
        """
        Leduc: f"{card}_{history}" or f"{card}_{community_card}_{history}"
        """

        betting_pattern = self.extract_betting_history(round_state)

        if round_state.get('street') == 'preflop':
            # Preflop: just starting hand
            card_bucket = self.card_abstractions.get_bucket(hole_card, None)
            return f"{card_bucket}_{betting_pattern}"
        else:
            # Postflop: starting hand + current strength + street
            starting_hand = self.card_abstractions.get_bucket(
                hole_card, None)  # Preflop bucket
            current_strength = self.card_abstractions.get_bucket(
                # Postflop bucket
                hole_card, round_state.get('community_card'))
            street = round_state.get('street')

            return f"{starting_hand}_{current_strength}_{street}_{betting_pattern}"

    def extract_betting_history(self, round_state):
        """
        Extract simplified betting history with proper game_state conversion
        """
        current_street = round_state.get('street', 'preflop')
        action_history = round_state.get(
            'action_histories', {}).get(current_street, [])

        # Convert round_state to game_state format that ActionAbstraction expects
        game_state = self.convert_round_state_to_game_state(round_state)

        return self.action_abstractions.abstract_action_history(action_history, game_state)

    def convert_round_state_to_game_state(self, round_state):
        """Convert nested round_state to flattened game_state format"""

        # Extract pot size
        pot_size = round_state.get('pot', {}).get('main', {}).get('amount', 0)
        if pot_size == 0:
            pot_size = 3  # Default starting pot for training

        # Extract player stack (use first seat as default)
        seats = round_state.get('seats', [{'stack': 100}])
        player_stack = seats[0].get('stack', 100) if seats else 100

        # Extract current bet from action history
        current_street = round_state.get('street', 'preflop')
        action_history = round_state.get(
            'action_histories', {}).get(current_street, [])
        current_bet = 0
        player_contribution = 0

        for action in action_history:
            if action.get('action') in ['bet', 'raise']:
                current_bet = action.get('amount', 0)
            elif action.get('action') == 'call':
                player_contribution += action.get('amount', 0)

        # Extract big blind
        big_blind = 2  # Default
        preflop_actions = round_state.get(
            'action_histories', {}).get('preflop', [])
        for action in preflop_actions:
            if action.get('action', '').upper() == 'BIGBLIND':
                big_blind = action.get('amount', 2)
                break

        return {
            'pot_size': pot_size,
            'player_stack': player_stack,
            'current_bet': current_bet,
            'player_contribution': player_contribution,
            'big_blind': big_blind
        }
