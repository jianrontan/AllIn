# backend/bot/src/bot/player.py
from pypokerengine.players import BasePokerPlayer
from src.bot.game_adapter import GameAdapter
from src.cfr.information_set import InformationSet
import random
import json
from pathlib import Path
import os
import sys


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

        # Foundation attributes for confidence detection
        self.total_training_iterations = 0
        self.blueprint_metadata = {}
        self.training_stats = {}

        self.load_trained_strategy()

        self.confidence_detector = None
        self.off_tree_detector = None
        self.subgame_detector = None
        self._initialize_confidence_detection()

    def _initialize_confidence_detection(self):
        """Initialize confidence detection systems"""
        # Always initialize to None first
        self.confidence_detector = None
        self.off_tree_detector = None
        self.subgame_detector = None
        
        try:
            # Try to import and create real detectors
            from ..subgame.player_blueprint_adapter import PlayerBlueprintAdapter
            from ..subgame.confidence_detector import ConfidenceDetector
            from ..subgame.off_tree_detector import OffTreeDetector
            from ..subgame.subgame_detector import SubgameDetector
            
            adapter = PlayerBlueprintAdapter(self)
            self.confidence_detector = ConfidenceDetector(adapter)
            self.off_tree_detector = OffTreeDetector()
            self.subgame_detector = SubgameDetector(adapter)
            
            print("✅ Confidence detection systems initialized")
            
        except Exception as e:
            print(f"❌ Confidence detection failed, using fallbacks: {e}")
            # Always ensure fallback detectors are created
            self._initialize_fallback_detectors()

    def _initialize_fallback_detectors(self):
        """Initialize fallback detectors - guaranteed to work"""
        try:
            class FallbackOffTreeDetector:
                def is_off_tree_situation(self, game_history, pot_sizes_history=None):
                    return False, ["Confidence detection not available"]
            
            class FallbackSubgameDetector:
                def should_trigger_subgame_solving(self, info_set_key, game_history, total_iterations, game_state=None):
                    return False, ["Confidence detection not available"]
            
            self.confidence_detector = None
            self.off_tree_detector = FallbackOffTreeDetector()
            self.subgame_detector = FallbackSubgameDetector()
            
            print("⚠️ Using fallback detectors - confidence detection disabled")
            
        except Exception as e:
            print(f"❌ Even fallback detector creation failed: {e}")
            # Create the most basic possible fallback
            class BasicFallback:
                def is_off_tree_situation(self, *args, **kwargs):
                    return False, ["Detector unavailable"]
                def should_trigger_subgame_solving(self, *args, **kwargs):
                    return False, ["Detector unavailable"]
            
            fallback = BasicFallback()
            self.off_tree_detector = fallback
            self.subgame_detector = fallback

    def load_trained_strategy(self):
        """Load trained blueprint strategy from unified JSON file with comprehensive statistics"""
        try:
            # Load from the renamed blueprint file
            current_dir = Path(__file__).parent
            blueprint_path = current_dir / ".." / ".." / "analysis" / "blueprint.json"

            # Resolve to absolute path
            file_path = blueprint_path.resolve()

            with open(file_path, 'r') as f:
                blueprint_data = json.load(f)

            # Extract training metadata from unified format
            training_metadata = blueprint_data.get('training_metadata', {})
            self.total_training_iterations = training_metadata.get(
                'iterations', 100)
            self.blueprint_metadata = training_metadata

            # Store additional training statistics for future use
            self.training_stats = {
                'expected_value': training_metadata.get('expected_value', 0.0),
                'training_duration': training_metadata.get('training_duration_seconds', 0.0),
                'total_info_sets': training_metadata.get('total_info_sets', 0),
                'visit_statistics': blueprint_data.get('visit_statistics', {}),
                'strategy_analysis': blueprint_data.get('strategy_analysis', {}),
                'convergence_metrics': blueprint_data.get('convergence_metrics', {})
            }

            # Convert unified JSON data to InformationSet objects
            normalized_strategies = blueprint_data.get(
                'normalized_strategies', {})

            for info_set_key, strategy_data in normalized_strategies.items():
                info_set = InformationSet()

                # Set legal actions
                info_set.legal_actions = strategy_data.get('legal_actions', [])

                # Set regrets
                info_set.cumulative_regrets = strategy_data.get('regrets', {})

                # Convert average strategy to cumulative strategy format
                avg_strategy = strategy_data.get('average_strategy', {})
                info_set.cumulative_strategy = {}

                # Convert average strategy probabilities to cumulative format
                # Multiply by training iterations to simulate cumulative updates
                for action, prob in avg_strategy.items():
                    info_set.cumulative_strategy[action] = prob * \
                        self.total_training_iterations

                # Extract visit tracking from unified format
                visit_metadata = strategy_data.get('visit_metadata', {})
                info_set.visit_count = visit_metadata.get('visit_count', 1)
                info_set.last_visited_iteration = visit_metadata.get(
                    'last_visited_iteration', 0)

                self.info_sets[info_set_key] = info_set

            print(
                f"✅ Loaded blueprint with {len(self.info_sets)} information sets")
            print(f"   Training iterations: {self.total_training_iterations}")
            print(
                f"   Expected value: {self.training_stats['expected_value']:.6f}")
            print(
                f"   Training duration: {self.training_stats['training_duration']:.1f}s")
            print(
                f"   Average visit frequency: {self.training_stats['visit_statistics'].get('average_visits_per_infoset', 0):.2f}")

        except FileNotFoundError:
            print(f"❌ Blueprint strategy file not found at {file_path}")
            print("   Using random play fallback")
            self.total_training_iterations = 100  # Default fallback

        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON file: {e}")
            print("   Using random play fallback")
            self.total_training_iterations = 100

        except Exception as e:
            print(f"❌ Unexpected error loading blueprint: {e}")
            print("   Using random play fallback")
            self.total_training_iterations = 100

    # NEED TO CHECK IF CONFIDENCE LOW THEN ACTIVATE SUBGAME SOLVING
    def declare_action(self, valid_actions, hole_card, round_state):
        """Enhanced action selection with confidence detection"""

        # Extract info set key and game data
        info_set_key = self.game_adapter.create_info_set_key(
            hole_card, round_state)
        game_history = self._extract_simplified_game_history(round_state)

        # Check confidence if detector available
        if self.subgame_detector:
            should_solve_subgame, reasons = self.subgame_detector.should_trigger_subgame_solving(
                info_set_key, game_history, self.total_training_iterations)

            if should_solve_subgame:
                print(
                    f"🎯 Subgame solving triggered for {info_set_key}: {reasons}")
                # TODO: Implement actual subgame solving
                # For now, fall back to blueprint

        # Use blueprint strategy
        return self._get_blueprint_action(valid_actions, hole_card, round_state)

    def _get_blueprint_action(self, valid_actions, hole_card, round_state):
        """Get action using trained blueprint strategy"""
        # Convert PyPokerEngine data to info set format
        info_set_key = self.game_adapter.create_info_set_key(
            hole_card, round_state)

        # Use trained strategy if available, otherwise create new info set
        if info_set_key in self.info_sets:
            info_set = self.info_sets[info_set_key]
        else:
            # Fallback: create new info set for unknown situations
            info_set = InformationSet()
            self.info_sets[info_set_key] = info_set

        # Convert PyPokerEngine actions to CFR format
        cfr_actions = self.game_adapter.action_abstractions.pypoker_to_cfr_actions(
            valid_actions, self.extract_game_state(round_state)
        )

        # Get strategy from trained blueprint
        strategy = info_set.get_average_strategy(cfr_actions)

        # Select action based on strategy
        selected_cfr_action = self.select_action_from_strategy(
            cfr_actions, strategy)

        # Convert back to PyPokerEngine format
        action, amount = self.game_adapter.action_abstractions.cfr_to_pypoker_action(
            selected_cfr_action, self.extract_game_state(round_state)
        )

        return action, amount

    def _extract_simplified_game_history(self, round_state):
        """Convert PyPokerEngine actions to simplified CFR format"""
        current_street = round_state.get('street', 'preflop')
        action_history = round_state.get(
            'action_histories', {}).get(current_street, [])

        simplified_history = []
        for action in action_history:
            action_type = action.get('action', '').upper()
            amount = action.get('amount', 0)

            if action_type == 'FOLD':
                simplified_history.append('fold')
            elif action_type == 'CALL':
                simplified_history.append('call')
            elif action_type == 'CHECK':
                simplified_history.append('check')
            elif action_type in ['BET', 'RAISE']:
                # Categorize based on amount relative to pot
                pot_size = round_state.get('pot', {}).get(
                    'main', {}).get('amount', 0)
                ratio = amount / pot_size if pot_size > 0 else 0

                if ratio <= 0.4:
                    category = 'small'
                elif ratio <= 0.8:
                    category = 'medium'
                else:
                    category = 'large'

                simplified_history.append(f"{action_type.lower()}_{category}")
            # Skip blind actions as they're not part of decision history

        return simplified_history

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

    def extract_actual_bet_amounts(self, round_state):
        """Extract actual bet amounts from PyPokerEngine for off-tree detection"""
        current_street = round_state.get('street', 'preflop')
        action_history = round_state.get(
            'action_histories', {}).get(current_street, [])

        bet_amounts = []
        pot_sizes = []

        # Track pot size evolution for ratio calculations
        current_pot = round_state.get('pot', {}).get(
            'main', {}).get('amount', 0)

        for action in action_history:
            action_type = action.get('action', '').upper()
            amount = action.get('amount', 0)

            if action_type in ['BET', 'RAISE']:
                bet_amounts.append({
                    'action': action_type.lower(),
                    'amount': amount,
                    'pot_size_at_time': current_pot,
                    'pot_ratio': amount / current_pot if current_pot > 0 else 0
                })
                pot_sizes.append(current_pot)
                current_pot += amount

        return bet_amounts, pot_sizes

    def get_confidence_info(self, info_set_key):
        """Get confidence information for an information set (for future subgame solving)"""
        if info_set_key not in self.info_sets:
            return {
                'exists': False,
                'visit_count': 0,
                'confidence': 'unknown'
            }

        info_set = self.info_sets[info_set_key]

        # Calculate visit frequency
        visit_frequency = info_set.visit_count / \
            max(1, self.total_training_iterations)

        # Calculate strategy entropy for confidence assessment
        try:
            if info_set.legal_actions:
                strategy = info_set.get_average_strategy(
                    info_set.legal_actions)
                import numpy as np

                # Ensure we have a proper float array
                strategy_array = np.array(strategy, dtype=np.float64)

                # Ensure valid probabilities
                strategy_array = np.clip(strategy_array, 1e-10, 1.0)

                # Calculate entropy with explicit float conversion
                log_values = np.log2(strategy_array)
                entropy_sum = np.sum(strategy_array * log_values)
                entropy = -float(entropy_sum)  # Explicit float conversion

                max_entropy = np.log2(len(strategy))
                normalized_entropy = float(
                    entropy / max_entropy) if max_entropy > 0 else 0.0

                # Determine confidence level
                if visit_frequency >= 0.01 and normalized_entropy < 0.3:
                    confidence = 'high'
                elif visit_frequency >= 0.005 and normalized_entropy < 0.7:
                    confidence = 'medium'
                else:
                    confidence = 'low'
            else:
                confidence = 'unknown'
                normalized_entropy = 1.0
        except Exception as e:
            print(
                f"Warning: Error calculating confidence for {info_set_key}: {e}")
            confidence = 'unknown'
            normalized_entropy = 1.0

        return {
            'exists': True,
            'visit_count': info_set.visit_count,
            'visit_frequency': visit_frequency,
            'strategy_entropy': normalized_entropy,
            'confidence': confidence,
            'last_visited_iteration': info_set.last_visited_iteration
        }

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
