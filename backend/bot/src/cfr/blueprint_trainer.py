import random
import pickle
from pathlib import Path
from .poker_game import PokerGame
from .information_set import InformationSet
from ..bot.game_adapter import GameAdapter


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
                            community_cards, [], 1.0, 1.0, 0)
            expected_value += util

            print(f"Completed iteration {i + 1}, utility: {util}")

            # More frequent progress reporting
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{iterations} iterations")
                print(f"Expected Value: {expected_value / (i + 1)}")
                print(f"Info sets created: {len(self.info_sets)}")

        print("Training completed!")
        return expected_value / iterations

    def cfr(self, p0_cards, p1_cards, community_cards, history, p0_reach, p1_reach, street, depth=0):
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

        # Determine current player
        current_player = len(history) % 2
        player_cards = p0_cards if current_player == 0 else p1_cards

        # Create info set key using my GameAdapter
        round_state = self.create_round_state_for_info_set(
            community_cards, history, street)
        info_set_key = self.game_adapter.create_info_set_key(
            player_cards, round_state)

        # Get or create information set
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = InformationSet()
        info_set = self.info_sets[info_set_key]

        # Get legal actions using my game logic
        legal_actions = self.game.get_legal_actions(history)
        if not legal_actions:  # Round complete
            if street < 3:  # Only advance if not already at river
                return self.cfr(p0_cards, p1_cards, community_cards, [],
                                p0_reach, p1_reach, street + 1, depth + 1)
            else:
                # Already at river, game should be terminal
                return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                             history, street)

        # Get strategy from information set
        reach_prob = p0_reach if current_player == 0 else p1_reach
        strategy = info_set.get_strategy(legal_actions, reach_prob)

        # Calculate utilities for each action
        action_utilities = {}
        node_utility = 0

        for i, action in enumerate(legal_actions):
            next_history = history + [action]

            if current_player == 0:
                action_utilities[action] = -self.cfr(
                    p0_cards, p1_cards, community_cards, next_history,
                    p0_reach * strategy[i], p1_reach, street, depth + 1)
            else:
                action_utilities[action] = -self.cfr(
                    p0_cards, p1_cards, community_cards, next_history,
                    p0_reach, p1_reach * strategy[i], street, depth + 1)

            node_utility += strategy[i] * action_utilities[action]

        # Update regrets for all actions
        for i, action in enumerate(legal_actions):
            regret = action_utilities[action] - node_utility

            if action not in info_set.cumulative_regrets:
                info_set.cumulative_regrets[action] = 0

            if current_player == 0:
                info_set.cumulative_regrets[action] += p1_reach * regret
            else:
                info_set.cumulative_regrets[action] += p0_reach * regret

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
