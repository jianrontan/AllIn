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
