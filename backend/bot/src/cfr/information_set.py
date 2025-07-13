import numpy as np


class InformationSet:
    """
    Core logic stays the same as Leduc, just handles more action types
    """

    def __init__(self):
        self.cumulative_regrets = {}
        self.cumulative_strategy = {}
        self.legal_actions = []

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

    def get_strategy_with_adaptive_pruning(self, legal_actions, reach_probability, iteration=0):
        """Strategy calculation with adaptive pruning thresholds"""

        # Get adaptive threshold based on iteration
        pruning_threshold = self.get_adaptive_threshold(iteration)

        # Start with base strategy
        base_strategy = self.get_strategy(legal_actions, reach_probability)

        # Apply pruning if past the learning phase
        if iteration >= 500:  # No pruning for first 500 iterations
            pruned_strategy = self._apply_conservative_pruning(
                base_strategy, legal_actions, iteration, pruning_threshold)
            return pruned_strategy
        else:
            return base_strategy

    def get_adaptive_threshold(self, iteration):
        """Gradually reduce pruning threshold as training progresses"""
        if iteration < 500:
            return float('inf')  # No pruning
        elif iteration < 1000:
            return 100  # Very conservative
        elif iteration < 2000:
            return 75   # Moderately conservative
        elif iteration < 5000:
            return 50   # Standard
        else:
            return 25   # Aggressive (original target)

    def _apply_conservative_pruning(self, strategy, legal_actions, iteration, threshold):
        """Apply conservative pruning by modifying strategy probabilities"""
        pruned_strategy = strategy.copy()
        pruned_count = 0

        for i, action in enumerate(legal_actions):
            regret = self.cumulative_regrets.get(action, 0)

            # Only prune if regret is very negative
            if regret < -threshold:
                # Higher minimum probability than before
                pruned_strategy[i] = 0.1
                pruned_count += 1

        # Safety: Never leave fewer than 2 viable actions
        viable_actions = sum(1 for p in pruned_strategy if p > 0.05)
        if viable_actions < 2:
            return strategy  # Return original if too aggressive

        # Renormalize probabilities
        total = np.sum(pruned_strategy)
        if total > 0:
            pruned_strategy = pruned_strategy / total
        else:
            return strategy

        return pruned_strategy
