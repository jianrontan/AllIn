# backend/bot/src/cython_extensions/game_logic_fast.pyx
import numpy as np
cimport numpy as cnp
cimport cython
from typing import List, Dict

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_current_pot_size_fast(list history, double starting_pot=3.0, double starting_stack=100.0):
    """Fast pot size calculation - major bottleneck"""
    cdef double current_pot = starting_pot
    cdef double accumulated_bets = 0.0
    cdef double player_contributions[2]
    cdef int current_player = 0
    cdef str action
    cdef double bet_amount
    
    # Initialize blinds
    player_contributions[0] = 1.0  # Small blind
    player_contributions[1] = 2.0  # Big blind
    
    for action in history:
        if action == 'bet_small':
            bet_amount = 0.33 * current_pot
            accumulated_bets += bet_amount
            player_contributions[current_player] += bet_amount
        elif action == 'bet_medium':
            bet_amount = 0.66 * current_pot
            accumulated_bets += bet_amount
            player_contributions[current_player] += bet_amount
        elif action == 'bet_large':
            bet_amount = 1.0 * current_pot
            accumulated_bets += bet_amount
            player_contributions[current_player] += bet_amount
        elif action == 'raise_small':
            bet_amount = 0.33 * (current_pot + accumulated_bets)
            accumulated_bets += bet_amount
            player_contributions[current_player] += bet_amount
        elif action == 'raise_medium':
            bet_amount = 0.66 * (current_pot + accumulated_bets)
            accumulated_bets += bet_amount
            player_contributions[current_player] += bet_amount
        elif action == 'raise_large':
            bet_amount = 1.0 * (current_pot + accumulated_bets)
            accumulated_bets += bet_amount
            player_contributions[current_player] += bet_amount
        elif action == 'call':
            # Need to calculate proper call amount based on last bet
            call_amount = get_last_bet_amount_from_history_fast(history[:len(history)])
            player_stack_at_call = starting_stack - player_contributions[current_player]
            actual_call_amount = min(call_amount, player_stack_at_call)
            accumulated_bets += actual_call_amount
            player_contributions[current_player] += actual_call_amount
        
        # Switch players for betting actions
        if action in ['check', 'bet_small', 'bet_medium', 'bet_large',
                     'raise_small', 'raise_medium', 'raise_large', 'call', 'fold']:
            current_player = 1 - current_player
    
    return current_pot + accumulated_bets

@cython.boundscheck(False)
def is_round_complete_fast(list history):
    """Fast round completion check"""
    cdef int history_len = len(history)
    cdef str last_action, second_last_action
    
    if history_len == 0:
        return False
    
    # Check for fold
    for action in history:
        if action == 'fold':
            return True
    
    # Check for double check
    if history_len >= 2:
        last_action = history[history_len - 1]
        second_last_action = history[history_len - 2]
        if last_action == 'check' and second_last_action == 'check':
            return True
        
        # Check for bet/raise followed by call
        if last_action == 'call' and (second_last_action.startswith('bet_') or 
                                     second_last_action.startswith('raise_')):
            return True
    
    return False

@cython.boundscheck(False)
def count_bet_actions_fast(list history):
    """Fast betting action counting for caps"""
    cdef int count = 0
    cdef str action
    
    for action in history:
        if action.startswith('bet_') or action.startswith('raise_'):
            count += 1
    
    return count

@cython.boundscheck(False)
def get_last_bet_amount_fast(list history, double current_pot):
    """Fast last bet amount calculation"""
    cdef str action
    cdef int i
    
    # Scan backwards for last bet/raise
    for i in range(len(history) - 1, -1, -1):
        action = history[i]
        if action == 'bet_small' or action == 'raise_small':
            return 0.33 * current_pot
        elif action == 'bet_medium' or action == 'raise_medium':
            return 0.66 * current_pot
        elif action == 'bet_large' or action == 'raise_large':
            return 1.0 * current_pot
    
    return 0.0

@cython.boundscheck(False)
def calculate_player_stack_fast(int player, list history, double starting_stack=100.0):
    """Fast player stack calculation"""
    cdef double contribution = 1.0 if player == 0 else 2.0  # Blinds
    cdef int current_player = 0
    cdef double current_pot = 3.0
    cdef str action
    cdef double bet_amount
    
    for action in history:
        if current_player == player:
            if action == 'bet_small':
                contribution += 0.33 * current_pot
            elif action == 'bet_medium':
                contribution += 0.66 * current_pot
            elif action == 'bet_large':
                contribution += 1.0 * current_pot
            elif action.startswith('raise_'):
                if 'small' in action:
                    contribution += 0.33 * current_pot
                elif 'medium' in action:
                    contribution += 0.66 * current_pot
                elif 'large' in action:
                    contribution += 1.0 * current_pot
        
        # Switch players
        if action in ['check', 'bet_small', 'bet_medium', 'bet_large',
                     'raise_small', 'raise_medium', 'raise_large', 'call', 'fold']:
            current_player = 1 - current_player
    
    return starting_stack - contribution

@cython.boundscheck(False)
def get_last_bet_amount_from_history_fast(list history):
    """Get the amount of the last bet/raise for call calculations"""
    cdef str action
    cdef int i
    
    if not history:
        return 0.0
    
    # Find the last bet/raise action
    for i in range(len(history) - 1, -1, -1):
        action = history[i]
        if action.startswith('bet_') or action.startswith('raise_'):
            # Calculate pot size at the time of that bet (simplified)
            if action == 'bet_small' or action == 'raise_small':
                return 0.33 * 3.0  # Simplified pot calculation
            elif action == 'bet_medium' or action == 'raise_medium':
                return 0.66 * 3.0
            elif action == 'bet_large' or action == 'raise_large':
                return 1.0 * 3.0
    
    return 0.0