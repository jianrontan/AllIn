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

try:
    from ..cython_extensions.cfr_fast import (
        update_regrets_cfr_plus_fast, calculate_node_utility_fast,
        batch_action_utilities_fast, cfr_reach_update_fast,
        terminal_utility_fast
    )
    from ..cython_extensions.blueprint_optimisation_fast import (
        create_round_state_fast, fast_info_set_lookup,
        update_visit_statistics_fast
    )
    CFR_CYTHON_AVAILABLE = True
    print("✅ CFR Cython extensions loaded")
except ImportError as e:
    CFR_CYTHON_AVAILABLE = False
    print(f"⚠️ CFR Cython extensions not available: {e}")


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

    def cfr(self, p0_cards, p1_cards, community_cards, history, p0_reach, p1_reach, street, depth=0, iteration=0):
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
                                         history, 3)

        # Check if terminal
        if self.game.is_terminal(history, street):
            return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                         history, street)

        # Get legal actions using my game logic
        legal_actions = self.game.get_legal_actions(history)
        if not legal_actions:  # Round complete
            if street < 3:  # Only advance if not already at river
                return self.cfr(p0_cards, p1_cards, community_cards, [],
                                p0_reach, p1_reach, street + 1, depth + 1, iteration)
            else:
                # Already at river, game should be terminal
                return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                             history, street)

        # Determine current player
        current_player = len(history) % 2
        player_cards = p0_cards if current_player == 0 else p1_cards

        # Create info set key - use Cython for round state creation
        if CFR_CYTHON_AVAILABLE:
            round_state = create_round_state_fast(
                community_cards, history, street)
        else:
            round_state = self.create_round_state_for_info_set(
                community_cards, history, street)

        info_set_key = self.game_adapter.create_info_set_key(
            player_cards, round_state)

        # Get or create information set - use Cython lookup
        if CFR_CYTHON_AVAILABLE:
            info_set, created_new = fast_info_set_lookup(
                self.info_sets, info_set_key)
        else:
            if info_set_key not in self.info_sets:
                self.info_sets[info_set_key] = InformationSet()
            info_set = self.info_sets[info_set_key]

        # Get strategy from information set
        reach_prob = p0_reach if current_player == 0 else p1_reach
        strategy = info_set.get_strategy(legal_actions, reach_prob)

        # Update visit statistics - use Cython
        if CFR_CYTHON_AVAILABLE:
            update_visit_statistics_fast(info_set, iteration)
        else:
            if info_set.last_visited_iteration != iteration:
                info_set.visit_count += 1
                info_set.last_visited_iteration = iteration

        # Calculate utilities for each action
        action_utilities = {}

        for i, action in enumerate(legal_actions):
            next_history = history + [action]

            if current_player == 0:
                new_p0_reach = p0_reach * \
                    strategy[i] if CFR_CYTHON_AVAILABLE else p0_reach * \
                    strategy[i]
                action_utilities[action] = -self.cfr(
                    p0_cards, p1_cards, community_cards, next_history,
                    new_p0_reach, p1_reach, street, depth + 1, iteration)
            else:
                new_p1_reach = p1_reach * \
                    strategy[i] if CFR_CYTHON_AVAILABLE else p1_reach * \
                    strategy[i]
                action_utilities[action] = -self.cfr(
                    p0_cards, p1_cards, community_cards, next_history,
                    p0_reach, new_p1_reach, street, depth + 1, iteration)

        # Calculate node utility - use Cython
        if CFR_CYTHON_AVAILABLE:
            node_utility = calculate_node_utility_fast(
                legal_actions, strategy, action_utilities)
        else:
            node_utility = sum(strategy[i] * action_utilities[action]
                               for i, action in enumerate(legal_actions))

        # Update regrets - use Cython CFR+
        reach_probability = p1_reach if current_player == 0 else p0_reach

        if CFR_CYTHON_AVAILABLE:
            update_regrets_cfr_plus_fast(
                info_set.cumulative_regrets, legal_actions,
                action_utilities, node_utility, reach_probability)
        else:
            # Fallback to Python implementation
            for action in legal_actions:
                regret = action_utilities[action] - node_utility
                if action not in info_set.cumulative_regrets:
                    info_set.cumulative_regrets[action] = 0

                current_regret = info_set.cumulative_regrets[action]
                new_regret = current_regret + reach_probability * regret
                info_set.cumulative_regrets[action] = max(0.0, new_regret)

        return node_utility

    def create_round_state_for_info_set(self, community_cards, history, street):
        """Create round_state with REAL pot-relative amounts"""

        street_names = ['preflop', 'flop', 'turn', 'river']
        community_for_street = community_cards[:self.game.get_community_cards_count(
            street)]

        # Calculate REAL game state
        current_pot_size = self.game.calculate_current_pot_size(history)
        p0_stack = self.game.calculate_player_stack_after_history(0, history)
        p1_stack = self.game.calculate_player_stack_after_history(1, history)

        # Calculate current bet and contributions
        current_bet = 0
        p0_contribution = 0
        p1_contribution = 0
        current_player = 0

        for action in history:
            if action.startswith('bet_') or action.startswith('raise_'):
                bet_amount = self.game.calculate_bet_amount_for_action(
                    action, current_pot_size)
                current_bet = bet_amount
                if current_player == 0:
                    p0_contribution += bet_amount
                else:
                    p1_contribution += bet_amount
            elif action == 'call':
                if current_player == 0:
                    p0_contribution += current_bet
                else:
                    p1_contribution += current_bet

            # Advance player
            if action in ['check', 'bet_small', 'bet_medium', 'bet_large',
                          'raise_small', 'raise_medium', 'raise_large', 'call', 'fold']:
                current_player = 1 - current_player

        # Convert actions with consistent pot calculation
        converted_actions = []
        for i, action in enumerate(history):
            if action.startswith('bet_') or action.startswith('raise_'):
                # Use pot size at the time of this specific action
                partial_history = history[:i]
                pot_at_action_time = self.game.calculate_current_pot_size(
                    partial_history, 3)
                bet_amount = self.game.calculate_bet_amount_for_action(
                    action, pot_at_action_time)

                action_type = 'bet' if action.startswith('bet_') else 'raise'
                converted_actions.append(
                    {'action': action_type, 'amount': bet_amount})
            elif action in ['check', 'call', 'fold']:
                converted_actions.append({'action': action, 'amount': 0})

        return {
            'street': street_names[street],
            'community_card': community_for_street,
            'action_histories': {
                street_names[street]: converted_actions
            },
            'pot': {'main': {'amount': current_pot_size}},
            'seats': [
                {'stack': p0_stack, 'uuid': 'player_0'},
                {'stack': p1_stack, 'uuid': 'player_1'}
            ]
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
                            community_cards, [], 1.0, 1.0, 0, 0, i)
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
