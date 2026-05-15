# backend/bot/src/cfr/blueprint_trainer.py
import random
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

        # DCFR hyperparameters (Brown & Sandholm 2019)
        self.alpha = 1.5  # Regret decay: discounts early noisy regret accumulation
        self.beta = 0.0   # Strategy decay: 0 = standard reach-weighted sum (recommended)

    def create_deck(self):
        """Create standard 52-card deck"""
        suits = ['H', 'D', 'C', 'S']
        ranks = ['2', '3', '4', '5', '6', '7',
                 '8', '9', 'T', 'J', 'Q', 'K', 'A']
        return [suit + rank for rank in ranks for suit in suits]

    def deal_random_hand(self):
        """Deal random cards for training iteration"""
        shuffled_deck = self.deck.copy()
        random.shuffle(shuffled_deck)

        p0_cards = shuffled_deck[0:2]
        p1_cards = shuffled_deck[2:4]
        community_cards = shuffled_deck[4:9]

        return p0_cards, p1_cards, community_cards

    def cfr(self, p0_cards, p1_cards, community_cards, history, p0_reach, p1_reach, street, updating_player, depth=0, iteration=0, starting_pot=None):
        """
        Monte Carlo CFR+ with External Sampling
        - Updating player: explores all actions
        - Opponent: samples single action based on strategy
        """

        # Depth limiting to prevent infinite recursion
        if depth > 50:
            print(
                f"WARNING: Max depth reached at street {street}, history {history}")
            return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                         history, min(street, 3), starting_pot)

        # Initialize accumulated pot if not provided
        if starting_pot is None:
            starting_pot = 3  # Starting pot: SB(1) + BB(2)

        # Check for terminal states
        if street > 3:
            return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                         history, 3, starting_pot)

        if self.game.is_terminal(history, street):
            return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                         history, street, starting_pot)

        # Determine current player
        current_player = len(history) % 2
        player_cards = p0_cards if current_player == 0 else p1_cards

        # Get legal actions for current situation
        legal_actions = self.game.get_legal_actions(
            street, history, starting_pot, current_player)

        # If no legal actions, advance to next street or terminal
        if not legal_actions:
            current_pot = self.game.calculate_current_pot(
                starting_pot, history, street)
            if street < 3:
                # Advance to next street with empty history
                return self.cfr(p0_cards, p1_cards, community_cards, [],
                                p0_reach, p1_reach, street + 1, updating_player,
                                depth + 1, iteration, current_pot)
            else:
                # Terminal - evaluate final outcome (use starting pot as get_utility just calculates)
                return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                             history, street, starting_pot)

        # Create information set
        round_state = self.create_round_state_for_info_set(
            community_cards, history, street, starting_pot)
        info_set_key = self.game_adapter.create_info_set_key(
            player_cards, round_state)

        # Get or create information set
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = InformationSet()

        info_set = self.info_sets[info_set_key]

        # Update visit tracking
        if info_set.last_visited_iteration != iteration:
            info_set.visit_count += 1
            info_set.last_visited_iteration = iteration

        # Get current strategy
        reach_prob = p0_reach if current_player == 0 else p1_reach
        strategy = info_set.get_strategy(legal_actions, reach_prob, iteration, self.beta)

        if current_player == updating_player:
            # UPDATING PLAYER: Explore all actions
            action_utilities = {}
            node_utility = 0

            for i, action in enumerate(legal_actions):
                next_history = history + [action]

                if current_player == 0:
                    action_utilities[action] = -self.cfr(
                        p0_cards, p1_cards, community_cards, next_history,
                        p0_reach *
                        strategy[i], p1_reach, street, updating_player,
                        depth + 1, iteration, starting_pot)
                else:
                    action_utilities[action] = -self.cfr(
                        p0_cards, p1_cards, community_cards, next_history,
                        p0_reach, p1_reach *
                        strategy[i], street, updating_player,
                        depth + 1, iteration, starting_pot)

                node_utility += strategy[i] * action_utilities[action]

            # Update regrets (CFR+ floor + DCFR temporal decay)
            # Decay factor: early iterations shrink toward 0, later iterations approach 1.
            t = iteration + 1
            regret_decay = ((t - 1) / t) ** self.alpha if t > 1 else 0.0
            opponent_reach = p1_reach if current_player == 0 else p0_reach

            for i, action in enumerate(legal_actions):
                regret = action_utilities[action] - node_utility
                prior = info_set.cumulative_regrets.get(action, 0)
                info_set.cumulative_regrets[action] = max(
                    0, regret_decay * prior + opponent_reach * regret)

            return node_utility

        else:
            # OPPONENT: Sample single action based on strategy
            sampled_action = random.choices(legal_actions, weights=strategy)[0]
            sampled_prob = strategy[legal_actions.index(sampled_action)]
            next_history = history + [sampled_action]

            if current_player == 0:
                return -self.cfr(
                    p0_cards, p1_cards, community_cards, next_history,
                    p0_reach * sampled_prob, p1_reach, street, updating_player,
                    depth + 1, iteration, starting_pot)
            else:
                return -self.cfr(
                    p0_cards, p1_cards, community_cards, next_history,
                    p0_reach, p1_reach * sampled_prob, street, updating_player,
                    depth + 1, iteration, starting_pot)

    def create_round_state_for_info_set(self, community_cards, history, street, starting_pot):
        """Simplified version that passes CFR history directly"""

        street_names = ['preflop', 'flop', 'turn', 'river']
        community_for_street = community_cards[:self.game.get_community_cards_count(
            street)]
        current_pot = self.game.calculate_current_pot(
            starting_pot, history, street)

        return {
            'street': street_names[street],
            'community_card': community_for_street,
            'cfr_history': history,
            'pot': {'main': {'amount': current_pot}},
            'accumulated_pot': current_pot
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

            updating_player = i % 2

            # Run CFR iteration
            util = self.cfr(p0_cards, p1_cards,
                            community_cards, [], 1.0, 1.0, 0, updating_player, 0, i, 3)
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
