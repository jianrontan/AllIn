# backend/bot/src/cython_extensions/blueprint_optimisation_fast.pyx
import numpy as np
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
def create_round_state_fast(list community_cards, list history, int street, 
                           double starting_pot=3.0, double starting_stack=100.0):
    """Fast round state creation for info set keys"""
    cdef dict round_state = {}
    cdef list street_names = ['preflop', 'flop', 'turn', 'river']
    cdef int community_count
    cdef double current_pot = starting_pot
    cdef double p0_stack = starting_stack - 1.0  # Small blind
    cdef double p1_stack = starting_stack - 2.0  # Big blind
    
    # Fast community card slicing
    if street == 0:
        community_count = 0
    elif street == 1:
        community_count = 3
    elif street == 2:
        community_count = 4
    else:
        community_count = 5
    
    # Basic pot calculation (simplified for speed)
    current_pot += len([h for h in history if h.startswith('bet_') or h.startswith('raise_')]) * starting_pot * 0.5
    
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
def fast_info_set_lookup(dict info_sets, str info_set_key):
    """Fast information set retrieval with creation if needed"""
    if info_set_key in info_sets:
        return info_sets[info_set_key], False  # existing, created_new
    else:
        from ..cfr.information_set import InformationSet
        new_info_set = InformationSet()
        info_sets[info_set_key] = new_info_set
        return new_info_set, True

@cython.boundscheck(False)
@cython.wraparound(False)
def update_visit_statistics_fast(object info_set, int iteration):
    """Fast visit tracking updates"""
    if info_set.last_visited_iteration != iteration:
        info_set.visit_count += 1
        info_set.last_visited_iteration = iteration
