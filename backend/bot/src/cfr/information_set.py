# backend/bot/src/cfr/information_set.py
import numpy as np


class InformationSet:
    """
    Core logic stays the same as Leduc, just handles more action types
    """

    def __init__(self):
        self.cumulative_regrets = {}
        self.cumulative_strategy = {}
        self.legal_actions = []
        self.visit_count = 0             # Track how often this infoset was visited
        self.last_visited_iteration = 0  # Track recency

    def get_strategy(self, legal_actions, reach_probability):
        """CFR+ implementation - prevents negative regrets"""
        if not self.legal_actions:
            self.legal_actions = legal_actions.copy()
        else:
            # Maintain consistent ordering by using a canonical action order
            all_actions = set(self.legal_actions) | set(legal_actions)
            canonical_order = ['fold', 'call', 'check', 'bet_small', 'bet_medium',
                               'bet_large', 'raise_small', 'raise_medium', 'raise_large']
            self.legal_actions = [
                action for action in canonical_order if action in all_actions]

        # CFR+ key difference: max with 0 before storing regrets
        regrets = np.array([max(0, self.cumulative_regrets.get(action, 0))
                            for action in legal_actions])

        total = np.sum(regrets)

        if total > 0:
            strategy = regrets / total
        else:
            strategy = np.ones(len(legal_actions)) / len(legal_actions)

        # Accumulate strategy (existing logic works fine)
        for i, action in enumerate(legal_actions):
            if action not in self.cumulative_strategy:
                self.cumulative_strategy[action] = 0.0
            self.cumulative_strategy[action] += reach_probability * strategy[i]

        return strategy

    def get_average_strategy(self, legal_actions):
        """Direct regret-to-strategy conversion - simple math with CFR+"""

        # Since CFR+ guarantees regrets >= 0, just use them directly
        regrets = np.array([max(0, self.cumulative_regrets.get(action, 0))
                            for action in legal_actions])

        total = np.sum(regrets)
        if total > 1e-12:  # Very small threshold
            return regrets / total
        else:
            # Only fall back to uniform if ALL regrets are exactly zero
            return np.ones(len(legal_actions)) / len(legal_actions)
