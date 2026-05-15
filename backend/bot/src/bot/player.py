# backend/bot/src/bot/player.py
from pypokerengine.players import BasePokerPlayer
from src.bot.game_adapter import GameAdapter
from src.cfr.information_set import InformationSet
import random
import json
from pathlib import Path


class Player(BasePokerPlayer):
    def __init__(self):
        self.info_sets = {}
        self.game_adapter = GameAdapter()
        self.my_uuid = None

        self.total_training_iterations = 0

        self.load_trained_strategy()

    def load_trained_strategy(self):
        """Load trained blueprint strategy from unified JSON file with comprehensive statistics"""
        try:
            current_dir = Path(__file__).parent
            blueprint_path = current_dir / ".." / ".." / "analysis" / "blueprint.json"
            file_path = blueprint_path.resolve()

            with open(file_path, 'r') as f:
                blueprint_data = json.load(f)

            training_metadata = blueprint_data.get('training_metadata', {})
            self.total_training_iterations = training_metadata.get(
                'iterations', 100)

            normalized_strategies = blueprint_data.get(
                'normalized_strategies', {})

            for info_set_key, strategy_data in normalized_strategies.items():
                info_set = InformationSet()
                info_set.legal_actions = strategy_data.get('legal_actions', [])
                info_set.cumulative_regrets = strategy_data.get('regrets', {})

                avg_strategy = strategy_data.get('average_strategy', {})
                info_set.cumulative_strategy = {}

                for action, prob in avg_strategy.items():
                    info_set.cumulative_strategy[action] = prob * \
                        self.total_training_iterations

                visit_metadata = strategy_data.get('visit_metadata', {})
                info_set.visit_count = visit_metadata.get('visit_count', 1)
                info_set.last_visited_iteration = visit_metadata.get(
                    'last_visited_iteration', 0)

                self.info_sets[info_set_key] = info_set

            print(
                f"✅ Loaded blueprint with {len(self.info_sets)} information sets")
            print(f"   Training iterations: {self.total_training_iterations}")
            print(
                f"   Expected value: {training_metadata.get('expected_value', 0.0):.6f}")

        except Exception as e:
            print(f"❌ Error loading blueprint: {e}")
            self.total_training_iterations = 100

    def declare_action(self, valid_actions, hole_card, round_state):
        """Simple blueprint action selection with debug output"""
        action, amount = self._get_blueprint_action(
            valid_actions, hole_card, round_state)

        # Debug output
        print(f"[CFR_Bot] Hand: {hole_card}, Action: {action}:{amount}")

        return action, amount

    def _get_blueprint_action(self, valid_actions, hole_card, round_state):
        """Get action using trained blueprint strategy"""
        # Create info set key
        info_set_key = self.game_adapter.create_info_set_key(
            hole_card, round_state)

        # Extract game state
        game_state = self.extract_game_state(round_state)

        # Check if we have this info set
        if info_set_key in self.info_sets:
            info_set = self.info_sets[info_set_key]
            print(f"[CFR_Bot] Found info set: {info_set_key}")
        else:
            # Unknown situation - create new info set (will use uniform strategy)
            info_set = InformationSet()
            self.info_sets[info_set_key] = info_set
            print(
                f"[CFR_Bot] Unknown info set: {info_set_key}, using uniform strategy")

        # Convert PyPokerEngine actions to CFR format
        cfr_actions = self.game_adapter.action_abstractions.pypoker_to_cfr_actions(
            valid_actions, game_state
        )

        print(f"[CFR_Bot] Available CFR actions: {cfr_actions}")

        # Get strategy from blueprint
        strategy = info_set.get_average_strategy(cfr_actions)

        print(f"[CFR_Bot] Strategy: {dict(zip(cfr_actions, strategy))}")

        # Select action based on strategy (random weighted choice)
        # Convert numpy array to list for random.choices
        strategy_list = strategy.tolist() if hasattr(
            strategy, 'tolist') else list(strategy)
        selected_cfr_action = random.choices(
            cfr_actions, weights=strategy_list)[0]

        print(f"[CFR_Bot] Selected CFR action: {selected_cfr_action}")

        # Convert back to PyPokerEngine format
        action, amount = self.game_adapter.action_abstractions.cfr_to_pypoker_action(
            selected_cfr_action, valid_actions, round_state, game_state
        )

        return action, amount

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
        return 2

    def extract_current_bet(self, round_state):
        """Extract current bet - the total amount a player needs to have contributed"""
        current_street = round_state.get('street', 'preflop')
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
                        # For bets/raises, this IS the total contribution
                        total_contribution = action.get('amount', 0)

        return total_contribution

    def receive_round_start_message(self, round_count, hole_card, seats):
        if self.my_uuid is None:
            for seat in seats:
                if seat.get('name') == 'CFR_Bot':
                    self.my_uuid = seat.get('uuid')
                    break

    def receive_game_start_message(self, game_info):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        """Track actions for history building"""
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        """Learn from results"""
        pass
