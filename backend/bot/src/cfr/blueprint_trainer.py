# backend/bot/src/cfr/blueprint_trainer.py
import random
import pickle
from pathlib import Path
from .poker_game import PokerGame
from .information_set import InformationSet
from ..bot.game_adapter import GameAdapter
import time
import os
import json


class BlueprintTrainer:
    """
    Blueprint CFR Trainer - focused on CFR algorithm like my Leduc Trainer
    Uses PokerGame for training simulation (separate from PyPokerEngine)
    """

    def __init__(self):
        self.info_sets = {}  # Like Leduc trainer
        self.game = PokerGame()  # My own game logic for training
        self.game_adapter = GameAdapter()  # For creating info set keys

        # Create deck for dealing
        self.deck = self.create_deck()
        self.BET_MULTIPLIERS = {'small': 0.33, 'medium': 0.66, 'large': 1.00}

    def create_deck(self):
        """Create standard 52-card deck"""
        suits = ['H', 'D', 'C', 'S']
        ranks = ['2', '3', '4', '5', '6', '7',
                 '8', '9', 'T', 'J', 'Q', 'K', 'A']
        return [rank + suit for rank in ranks for suit in suits]

    def deal_random_hand(self):
        """Deal random cards for training iteration"""
        shuffled_deck = self.deck.copy()
        random.shuffle(shuffled_deck)

        p0_cards = shuffled_deck[0:2]
        p1_cards = shuffled_deck[2:4]
        community_cards = shuffled_deck[4:9]

        return p0_cards, p1_cards, community_cards

    def cfr(self, p0_cards, p1_cards, community_cards, history, p0_reach, p1_reach, street, depth=0, iteration=0, accumulated_pot=None):
        """
        Core CFR algorithm - like the Leduc cfr method
        """
        # Infinite recursion catching
        if depth > 50:
            print(
                f"WARNING: Max depth reached at street {street}, history {history}")
            return 0

        if street > 3:
            return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                         history, 3, accumulated_pot)

        # Check if terminal
        if self.game.is_terminal(history, street):
            return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                         history, street, accumulated_pot)

        if accumulated_pot is None:
            accumulated_pot = 3

        # Get legal actions using my game logic
        legal_actions = self.game.get_legal_actions(
            street, history, accumulated_pot)
        if not legal_actions:  # Round complete
            if street < 3:  # Only advance if not already at river
                return self.cfr(p0_cards, p1_cards, community_cards, [],
                                p0_reach, p1_reach, street + 1, depth + 1, iteration, accumulated_pot)
            else:
                # Already at river, game should be terminal
                return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                             history, street, accumulated_pot)

        # Determine current player
        current_player = len(history) % 2
        player_cards = p0_cards if current_player == 0 else p1_cards

        # Create info set key using my GameAdapter
        round_state = self.create_round_state_for_info_set(
            community_cards, history, street, accumulated_pot)
        info_set_key = self.game_adapter.create_info_set_key(
            player_cards, round_state)

        # Get or create information set
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = InformationSet()
        info_set = self.info_sets[info_set_key]

        # Get strategy from information set
        reach_prob = p0_reach if current_player == 0 else p1_reach
        strategy = info_set.get_strategy(legal_actions, reach_prob)
        if info_set.last_visited_iteration != iteration:
            info_set.visit_count += 1
            info_set.last_visited_iteration = iteration

        # Calculate utilities for each action
        action_utilities = {}
        node_utility = 0

        for i, action in enumerate(legal_actions):
            next_history = history + [action]

            new_accumulated_pot = self.calculate_pot_after_action(
                action, street, accumulated_pot, history)

            if current_player == 0:
                action_utilities[action] = -self.cfr(
                    p0_cards, p1_cards, community_cards, next_history,
                    p0_reach * strategy[i], p1_reach, street, depth + 1, iteration, new_accumulated_pot)
            else:
                action_utilities[action] = -self.cfr(
                    p0_cards, p1_cards, community_cards, next_history,
                    p0_reach, p1_reach * strategy[i], street, depth + 1, iteration, new_accumulated_pot)

            node_utility += strategy[i] * action_utilities[action]

        # Update regrets for all actions
        for i, action in enumerate(legal_actions):
            regret = action_utilities[action] - node_utility

            if action not in info_set.cumulative_regrets:
                info_set.cumulative_regrets[action] = 0

            if current_player == 0:
                info_set.cumulative_regrets[action] = max(
                    0, info_set.cumulative_regrets.get(action, 0) + p1_reach * regret)
            else:
                info_set.cumulative_regrets[action] = max(
                    0, info_set.cumulative_regrets.get(action, 0) + p0_reach * regret)

        return node_utility

    def calculate_pot_after_action(self, action, street, current_accumulated_pot, history):
        """Calculate what the pot size will be after taking this action"""

        if action in ['check', 'fold']:
            return current_accumulated_pot  # No change to pot

        elif action == 'call':
            # Add the call amount to the pot
            call_amount = self.game.get_call_amount_from_history(
                street, history, current_accumulated_pot)
            return current_accumulated_pot + call_amount

        elif action.startswith('bet_'):
            current_player = len(history) % 2

            if street == 0:  # Preflop - use BB-based amounts
                action_type = self.game.get_preflop_action_type(history)
                bet_amounts = self.game.get_preflop_bet_amounts(
                    action_type, current_accumulated_pot)
                size = action.split('_')[1]
                target_amount = bet_amounts[size]  # Total commitment

                # Subtract existing contribution (same logic as raises)
                existing_contribution = self.game.get_player_contribution_this_round(
                    history, street, current_accumulated_pot, current_player)
                additional_amount = target_amount - existing_contribution
            # Postflop - use pot-relative (no existing contribution issue)
            else:
                size = action.split('_')[1]
                additional_amount = self.game.BET_MULTIPLIERS[size] * \
                    current_accumulated_pot

            return current_accumulated_pot + additional_amount

        elif action.startswith('raise_'):
            current_player = len(history) % 2
            contribution = self.game.get_player_contribution_this_round(
                history, street, current_accumulated_pot, current_player)

            if street == 0:  # Preflop - use BB-based amounts
                action_type = self.game.get_preflop_action_type(history)
                if action_type != 'pot_relative':  # BB-multiple phase
                    bet_amounts = self.game.get_preflop_bet_amounts(
                        action_type, current_accumulated_pot)
                    size = action.split('_')[1]
                    target_amount = bet_amounts[size]
                else:  # Switched to pot-relative
                    size = action.split('_')[1]
                    target_amount = self.game.BET_MULTIPLIERS[size] * \
                        current_accumulated_pot
            else:  # Postflop - use pot-relative
                size = action.split('_')[1]
                target_amount = self.game.BET_MULTIPLIERS[size] * \
                    current_accumulated_pot

            raise_amount = target_amount - contribution
            return current_accumulated_pot + raise_amount

        else:
            return current_accumulated_pot

    def create_round_state_for_info_set(self, community_cards, history, street, accumulated_pot):
        """Simplified version that passes CFR history directly"""

        street_names = ['preflop', 'flop', 'turn', 'river']
        community_for_street = community_cards[:self.game.get_community_cards_count(
            street)]

        return {
            'street': street_names[street],
            'community_card': community_for_street,
            'cfr_history': history,
            'pot': {'main': {'amount': accumulated_pot}},
            'accumulated_pot': accumulated_pot
        }

    def train_blueprint(self, iterations):
        """Main training loop with better progress tracking"""
        print(
            f"Starting blueprint CFR training for {iterations} iterations...")

        expected_value = 0
        for i in range(iterations):
            # Deal random cards
            p0_cards, p1_cards, community_cards = self.deal_random_hand()

            print(f"Starting iteration {i + 1}...")

            # Run CFR iteration
            util = self.cfr(p0_cards, p1_cards,
                            community_cards, [], 1.0, 1.0, 0, 0, i, 3)
            expected_value += util

            print(f"Completed iteration {i + 1}, utility: {util}")

            # More frequent progress reporting
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{iterations} iterations")
                print(f"Expected Value: {expected_value / (i + 1)}")
                print(f"Info sets created: {len(self.info_sets)}")

        print("Training completed!")
        return expected_value / iterations

    def export_blueprint_with_visit_stats(self, filename):
        """Export blueprint strategies with visit frequency statistics"""

        blueprint_data = {
            'metadata': {
                'total_info_sets': len(self.info_sets),
                'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'visit_stats_included': True
            },
            'strategies': {},
            'visit_statistics': {
                'most_visited': [],
                'least_visited': [],
                'total_visits': 0,
                'average_visits_per_infoset': 0
            }
        }

        visit_counts = []
        total_visits = 0

        for info_set_key, info_set in self.info_sets.items():
            try:
                legal_actions = info_set.legal_actions
                if not legal_actions:
                    continue

                avg_strategy = info_set.get_average_strategy(legal_actions)
                regrets = {action: info_set.cumulative_regrets.get(action, 0)
                           for action in legal_actions}

                # Calculate visit frequency properly
                visit_frequency = (info_set.visit_count / max(1, info_set.last_visited_iteration + 1)
                                   if info_set.last_visited_iteration >= 0 else 0)

                # Include visit statistics
                blueprint_data['strategies'][info_set_key] = {
                    'average_strategy': {action: float(prob) for action, prob in
                                         zip(legal_actions, avg_strategy)},
                    'regrets': regrets,
                    'visit_count': info_set.visit_count,
                    'last_visited_iteration': info_set.last_visited_iteration,
                    'visit_frequency': visit_frequency
                }

                visit_counts.append((info_set_key, info_set.visit_count))
                total_visits += info_set.visit_count

            except Exception as e:
                print(f"Error processing info set {info_set_key}: {e}")
                continue

        # Calculate visit statistics
        if visit_counts:
            visit_counts.sort(key=lambda x: x[1], reverse=True)

            blueprint_data['visit_statistics'] = {
                'most_visited': visit_counts[:10],
                'least_visited': visit_counts[-10:],
                'total_visits': total_visits,
                'average_visits_per_infoset': total_visits / len(visit_counts) if visit_counts else 0,
                'max_visits': visit_counts[0][1] if visit_counts else 0,
                'min_visits': visit_counts[-1][1] if visit_counts else 0
            }

        # Save to file
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(blueprint_data, f, indent=2)

        print(f"Blueprint with visit statistics saved to: {filename}")
        return blueprint_data
