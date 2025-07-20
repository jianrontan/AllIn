import numpy as np

try:
    from ..cython_extensions.information_set_fast import (
        get_strategy_fast, get_average_strategy_fast, update_cumulative_regrets_fast
    )
    CYTHON_AVAILABLE = True
    print("✅ Cython extensions loaded with absolute path")
except ImportError:
    CYTHON_AVAILABLE = False
    print("Cython extensions not available, using Python fallback")


class InformationSet:
    def __init__(self):
        self.cumulative_regrets = {}
        self.cumulative_strategy = {}
        self.legal_actions = []
        self.visit_count = 0
        self.last_visited_iteration = 0

    def get_strategy(self, legal_actions, reach_probability):
        """Strategy calculation with Cython acceleration"""
        if not self.legal_actions:
            self.legal_actions = legal_actions.copy()

        if CYTHON_AVAILABLE:
            strategy = get_strategy_fast(
                self.cumulative_regrets, legal_actions, reach_probability)
        else:
            strategy = self._get_strategy_python(
                legal_actions, reach_probability)

        # Accumulate strategy for average calculation
        for i, action in enumerate(legal_actions):
            if action not in self.cumulative_strategy:
                self.cumulative_strategy[action] = 0.0
            self.cumulative_strategy[action] += reach_probability * strategy[i]

        return strategy

    def get_average_strategy(self, legal_actions):
        """Average strategy calculation with direct regret conversion"""
        if CYTHON_AVAILABLE:
            return get_average_strategy_fast(self.cumulative_regrets, legal_actions)
        else:
            return self._get_average_strategy_python(legal_actions)

    def _get_strategy_python(self, legal_actions, reach_probability):
        """Python fallback implementation"""
        regrets = np.array([max(0, self.cumulative_regrets.get(action, 0))
                           for action in legal_actions])
        total = np.sum(regrets)

        if total > 1e-12:
            return regrets / total
        else:
            return np.ones(len(legal_actions)) / len(legal_actions)

    def _get_average_strategy_python(self, legal_actions):
        """Python fallback for average strategy"""
        regrets = np.array([max(0, self.cumulative_regrets.get(action, 0))
                           for action in legal_actions])
        total = np.sum(regrets)

        if total > 1e-12:
            return regrets / total
        else:
            return np.ones(len(legal_actions)) / len(legal_actions)
