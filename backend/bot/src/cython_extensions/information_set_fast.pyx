# backend/bot/src/cython_extensions/information_set_fast.pyx
from typing import Dict, List
import numpy as np
cimport numpy as cnp
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def get_strategy_fast(dict cumulative_regrets, list legal_actions, double reach_probability):
    """Cythonized version of get_strategy() - 3-5x speedup expected"""
    cdef int n_actions = len(legal_actions)
    cdef cnp.ndarray[double, ndim = 1] regrets = np.zeros(n_actions, dtype=np.float64)
    cdef cnp.ndarray[double, ndim = 1] strategy = np.zeros(n_actions, dtype=np.float64)

    cdef double total = 0.0
    cdef int i
    cdef str action
    cdef double regret_value

    # Fill regrets array with CFR+ (max with 0)
    for i in range(n_actions):
        action = legal_actions[i]
        regret_value = cumulative_regrets.get(action, 0.0)
        regrets[i] = regret_value if regret_value > 0.0 else 0.0
        total += regrets[i]

    # Calculate strategy probabilities
    if total > 1e-12:
        for i in range(n_actions):
            strategy[i] = regrets[i] / total
    else:
        # Uniform fallback
        for i in range(n_actions):
            strategy[i] = 1.0 / n_actions

    return strategy


@cython.boundscheck(False)
@cython.wraparound(False)
def get_average_strategy_fast(dict cumulative_strategy, list legal_actions):
    """Cythonized version of get_average_strategy() - direct regret conversion"""
    cdef int n_actions = len(legal_actions)
    cdef cnp.ndarray[double, ndim = 1] regrets = np.zeros(n_actions, dtype=np.float64)
    cdef cnp.ndarray[double, ndim = 1] strategy = np.zeros(n_actions, dtype=np.float64)

    cdef double total = 0.0
    cdef int i
    cdef str action
    cdef double regret_value

    # Use regrets directly (CFR+ guarantees non-negative)
    for i in range(n_actions):
        action = legal_actions[i]
        regret_value = cumulative_strategy.get(action, 0.0)
        regrets[i] = regret_value if regret_value > 0.0 else 0.0
        total += regrets[i]

    # Calculate final strategy
    if total > 1e-12:
        for i in range(n_actions):
            strategy[i] = regrets[i] / total
    else:
        for i in range(n_actions):
            strategy[i] = 1.0 / n_actions

    return strategy


@cython.boundscheck(False)
def update_cumulative_regrets_fast(dict cumulative_regrets, list legal_actions,
                                   dict action_utilities, double node_utility,
                                   double reach_probability):
    """Fast regret updates with CFR+"""
    cdef str action
    cdef double regret, current_regret, new_regret
    cdef int i

    for i in range(len(legal_actions)):
        action = legal_actions[i]
        regret = action_utilities[action] - node_utility

        current_regret = cumulative_regrets.get(action, 0.0)
        new_regret = current_regret + reach_probability * regret

        # CFR+ constraint: max with 0
        cumulative_regrets[action] = new_regret if new_regret > 0.0 else 0.0
