import numpy as np


class InformationSet:
    """
    Core logic stays the same as Leduc, just handles more action types
    """

    def __init__(self):
        self.cumulative_regrets = {}
        self.cumulative_strategy = {}
        self.legal_actions = []
        # self.last_updated = {}  # Track each action's last update
        # self.prune_until = {}  # Track until which iteration to prune each action

    def get_strategy(self, legal_actions, reach_probability):
        """
        legal_actions from PokerGame (training) or ActionAbstraction (gameplay)
        """
        if not self.legal_actions:
            self.legal_actions = legal_actions.copy()
        regrets = np.array([self.cumulative_regrets.get(action, 0)
                           for action in legal_actions])
        strategy = np.maximum(regrets, 0)
        total = np.sum(strategy)

        if total > 0:
            strategy = strategy / total
        else:
            strategy = np.ones(len(legal_actions)) / len(legal_actions)

        for i, action in enumerate(legal_actions):
            if action not in self.cumulative_strategy:
                self.cumulative_strategy[action] = 0
            self.cumulative_strategy[action] += reach_probability * strategy[i]

        return strategy

    def get_average_strategy(self, legal_actions):
        """Identical to Leduc implementation"""
        actions_to_use = legal_actions or self.legal_actions
        if not actions_to_use:
            raise ValueError("No legal actions available for this info set")
        cumulative_strat = np.array([self.cumulative_strategy.get(action, 0)
                                     for action in legal_actions])
        total = np.sum(cumulative_strat)
        if total > 0:
            return cumulative_strat / total
        else:
            return np.ones(len(legal_actions)) / len(legal_actions)

    # def get_strategy_with_pruning(self, legal_actions, reach_probability, iteration=0, pruning_threshold=25):
    #     """
    #     Core strategy calculation with integrated regret-based pruning
    #     """
    #     if not self.legal_actions:
    #         self.legal_actions = legal_actions.copy()

    #     # Calculate base strategy from regrets
    #     regrets = np.array([self.cumulative_regrets.get(action, 0)
    #                        for action in legal_actions])
    #     strategy = np.maximum(regrets, 0)

    #     # Apply regret-based pruning by modifying probabilities
    #     if iteration > 50:  # Conservative start - only prune after some learning
    #         strategy = self._apply_integrated_pruning(
    #             strategy, legal_actions, iteration, pruning_threshold)

    #     # Normalize strategy
    #     total = np.sum(strategy)
    #     if total > 0:
    #         strategy = strategy / total
    #     else:
    #         strategy = np.ones(len(legal_actions)) / len(legal_actions)

    #     # Update cumulative strategy
    #     for i, action in enumerate(legal_actions):
    #         if action not in self.cumulative_strategy:
    #             self.cumulative_strategy[action] = 0
    #         self.cumulative_strategy[action] += reach_probability * strategy[i]

    #     return strategy

    # def _apply_integrated_pruning(self, strategy, legal_actions, iteration, threshold):
    #     """
    #     Apply pruning by setting probabilities to near-zero for negative regret actions
    #     """
    #     pruned_strategy = strategy.copy()
    #     pruned_count = 0

    #     for i, action in enumerate(legal_actions):
    #         regret = self.cumulative_regrets.get(action, 0)

    #         # Prune actions with consistently negative regret
    #         if regret < -threshold:
    #             # Near-zero probability (not exactly zero)
    #             pruned_strategy[i] = 1e-8
    #             pruned_count += 1

    #     # Safety: Never prune all actions
    #     if pruned_count >= len(legal_actions):
    #         return strategy  # Return original strategy

    #     # Safety: Keep at least 2 actions with meaningful probability
    #     non_zero_count = np.sum(pruned_strategy > 1e-6)
    #     if non_zero_count < 2:
    #         # Keep the two actions with highest regret
    #         sorted_indices = np.argsort(
    #             [self.cumulative_regrets.get(action, 0) for action in legal_actions])
    #         for i in sorted_indices[-2:]:  # Keep top 2
    #             pruned_strategy[i] = max(pruned_strategy[i], strategy[i])

    #     return pruned_strategy
