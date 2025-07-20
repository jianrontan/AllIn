# backend/bot/src/cython_extensions/cfr_fast.pyx
import numpy as np
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def update_regrets_cfr_plus_fast(dict cumulative_regrets, list legal_actions,
                                 dict action_utilities, double node_utility, 
                                 double reach_probability):
    """Fast CFR+ regret updates"""
    cdef str action
    cdef double regret, current_regret, new_regret
    cdef int i
    
    for i in range(len(legal_actions)):
        action = legal_actions[i]
        regret = action_utilities[action] - node_utility
        
        current_regret = cumulative_regrets.get(action, 0.0)
        new_regret = current_regret + reach_probability * regret
        
        # CFR+ constraint: max with 0
        cumulative_regrets[action] = max(0.0, new_regret)

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_node_utility_fast(list legal_actions, cnp.ndarray[double, ndim=1] strategy, 
                               dict action_utilities):
    """Fast node utility calculation"""
    cdef double node_utility = 0.0
    cdef int i
    cdef str action
    
    for i in range(len(legal_actions)):
        action = legal_actions[i]
        node_utility += strategy[i] * action_utilities[action]
    
    return node_utility

@cython.boundscheck(False)
def create_round_state_fast(list community_cards, list history, int street, 
                           double starting_pot=3.0, double starting_stack=100.0):
    """Fast round state creation"""
    cdef dict round_state = {}
    cdef list street_names = ['preflop', 'flop', 'turn', 'river']
    cdef double current_pot = starting_pot
    cdef double p0_stack = starting_stack - 1.0
    cdef double p1_stack = starting_stack - 2.0
    cdef int community_count
    
    if street == 0:
        community_count = 0
    elif street == 1:
        community_count = 3
    elif street == 2:
        community_count = 4
    else:
        community_count = 5
    
    # Simple pot calculation
    for action in history:
        if action.startswith('bet_') or action.startswith('raise_'):
            current_pot += starting_pot * 0.5
    
    round_state = {
        'street': street_names[street],
        'community_card': community_cards[:community_count],
        'pot': {'main': {'amount': current_pot}},
        'seats': [
            {'stack': p0_stack, 'uuid': 'player_0'},
            {'stack': p1_stack, 'uuid': 'player_1'}
        ]
    }
    
    return round_state

@cython.boundscheck(False)
def update_visit_statistics_fast(object info_set, int iteration):
    """Fast visit tracking updates"""
    if info_set.last_visited_iteration != iteration:
        info_set.visit_count += 1
        info_set.last_visited_iteration = iteration
