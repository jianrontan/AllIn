# backend/bot/src/cfr/information_set.py
import numpy as np


class InformationSet:
    """
    Stores cumulative regrets and the cumulative (average) strategy for one
    information set. Used by external-sampling MCCFR.

    Two distinct operations, deliberately kept separate:
      - get_strategy():        pure regret-matching, no side effects on the
                               cumulative strategy. Called at every node.
      - accumulate_strategy(): adds the current strategy into the running
                               average. Called ONLY at opponent nodes, where
                               sampling supplies the correct reach weighting.
    """

    def __init__(self):
        self.cumulative_regrets = {}
        self.cumulative_strategy = {}
        self.legal_actions = []
        self.visit_count = 0             # regret-update count (DCFR alpha/beta clock)
        self.last_visited_iteration = -1  # -1 so iteration 0 increments visit_count
        self.strategy_visit_count = 0     # avg-strategy update count (DCFR gamma clock)
        self.last_strategy_iteration = -1  # -1 so iteration 0 increments the gamma clock

    def get_strategy(self, legal_actions):
        """
        Current strategy via CFR+ regret matching. Pure: does NOT mutate the
        cumulative strategy. Negative regrets are floored at 0 (CFR+).
        """
        if not self.legal_actions:
            self.legal_actions = legal_actions.copy()
        # `legal_actions` records the FIRST action set seen, for reference only.
        # get_strategy always operates on the list passed for THIS visit, and the
        # cumulative dicts are keyed by action name — so a key whose action set
        # varies across visits (a postflop key spanning different pots) still
        # merges correctly. Read the average strategy over cumulative_strategy's
        # own keys, never over this stored list.

        regrets = np.array([max(0.0, self.cumulative_regrets.get(a, 0.0))
                            for a in legal_actions])
        total = regrets.sum()
        if total > 0:
            return regrets / total
        return np.ones(len(legal_actions)) / len(legal_actions)

    def accumulate_strategy(self, legal_actions, strategy):
        """
        Add the current strategy into the cumulative (average) strategy.

        In external-sampling MCCFR this is called only at opponent nodes. The
        opponent's actions are sampled, so reaching this node already happens
        with probability proportional to the player's own reach — therefore the
        contribution is added unweighted.
        """
        if not self.legal_actions:
            self.legal_actions = legal_actions.copy()
        for i, a in enumerate(legal_actions):
            self.cumulative_strategy[a] = self.cumulative_strategy.get(a, 0.0) + strategy[i]

    def get_average_strategy(self, legal_actions):
        """Normalised average strategy over all accumulated iterations."""
        total = sum(self.cumulative_strategy.get(a, 0.0) for a in legal_actions)
        if total > 1e-12:
            return np.array([self.cumulative_strategy.get(a, 0.0) / total
                             for a in legal_actions])
        return np.ones(len(legal_actions)) / len(legal_actions)
