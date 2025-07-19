# backend/bot/src/cfr/poker_game.py
from ..abstractions.hand_evaluator import HandEvaluator


class PokerGame:
    """
    Simplified poker game for CFR training - like LeducPoker class
    This is separate from PyPokerEngine gameplay
    """

    def __init__(self):
        # Handle simplified game rules for training
        self.streets = ['preflop', 'flop', 'turn', 'river']
        self.max_raises_per_street = 3  # Limit re-raising to prevent infinite loops
        self.hand_evaluator = HandEvaluator()  # For hand evaluation

    def get_legal_actions(self, history):
        """
        Stack-aware simplified legal actions
        Fixed to properly handle stack sizes and call amounts
        """

        # Check round completion first before generating actions
        if self.is_round_complete(history):
            return []

        # Check if someone folded
        if 'fold' in history:
            return []

        # Check betting cap (industry standard: 1 bet + 3 raises = 4 total)
        bet_and_raise_count = sum(1 for action in history
                                  if action.startswith(('bet_', 'raise_')))

        if bet_and_raise_count >= 4:  # Betting cap
            call_amount = self.get_last_bet_amount_from_history(history)
            player_stack = self.calculate_player_stack_after_history(
                len(history) % 2, history)

            # NEED TO FIX EVENTUALLY: ALL_IN
            if player_stack >= call_amount:
                return ['fold', 'call']
            else:
                return ['fold']

        # Calculate current player's stack and pot
        current_player = len(history) % 2
        player_stack = self.calculate_player_stack_after_history(
            current_player, history)
        current_pot = self.calculate_current_pot_size(history)

        # If player has no chips, they can only fold or check (if no bet to call)
        if player_stack <= 0:
            current_bet_amount = self.get_current_bet_amount(history)
            return ['fold'] if current_bet_amount > 0 else ['check']

        # Count raises in current history
        raise_count = sum(
            1 for action in history if action.startswith('raise_'))

        if not history:  # First action
            actions = ['check']
            # Only add bet sizes player can afford
            if player_stack >= 0.33 * current_pot:
                actions.append('bet_small')
            if player_stack >= 0.66 * current_pot:
                actions.append('bet_medium')
            if player_stack >= 1.0 * current_pot:
                actions.append('bet_large')
            return actions

        if len(history) >= 2 and history[-2:] == ['check', 'check']:
            return []  # Round complete

        if history[-1] == 'call':
            return []  # Round complete

        raise_count = sum(
            1 for action in history if action.startswith('raise_'))

        last_action = history[-1]

        if last_action == 'check':
            actions = ['check']
            # Only add bet sizes player can afford
            if player_stack >= 0.33 * current_pot:
                actions.append('bet_small')
            if player_stack >= 0.66 * current_pot:
                actions.append('bet_medium')
            if player_stack >= 1.0 * current_pot:
                actions.append('bet_large')
            return actions

        elif last_action.startswith('bet_'):
            # Calculate required call amount
            call_amount = self.get_last_bet_amount_from_history(history)

            actions = []
            # Can always fold
            actions.append('fold')

            # Can only call if have enough chips
            if player_stack >= call_amount:
                actions.append('call')

            # Can only raise if no raises yet AND have enough chips
            if raise_count < 3:
                remaining_after_call = player_stack - call_amount
                potential_raises = [
                    ('raise_small', 0.33 * current_pot),
                    ('raise_medium', 0.66 * current_pot),
                    ('raise_large', 1.0 * current_pot)
                ]

                for raise_action, raise_amount in potential_raises:
                    # Check if player can afford it
                    if remaining_after_call >= raise_amount:
                        # Check if it meets minimum raise requirement
                        if self.validate_raise_meets_minimum(raise_action, history, current_pot):
                            actions.append(raise_action)
            return actions

        elif last_action.startswith('raise_'):
            # Calculate required call amount
            call_amount = self.get_last_bet_amount_from_history(history)

            actions = ['fold']
            # Can only call if have enough chips
            if player_stack >= call_amount:
                actions.append('call')

            # Can re-raise if under betting cap AND have enough chips
            if raise_count < 3:  # Allow up to 3 total raises
                remaining_after_call = player_stack - call_amount

                potential_raises = [
                    ('raise_small', 0.33 * current_pot),
                    ('raise_medium', 0.66 * current_pot),
                    ('raise_large', 1.0 * current_pot)
                ]

                for raise_action, raise_amount in potential_raises:
                    if remaining_after_call >= raise_amount:
                        if self.validate_raise_meets_minimum(raise_action, history, current_pot):
                            actions.append(raise_action)
            return actions

        elif last_action == 'call':
            return []  # Round complete

        else:
            return ['fold', 'call'] if player_stack >= self.get_last_bet_amount_from_history(history) else ['fold']

    def get_minimum_raise_amount(self, history):
        """
        Calculate minimum raise amount based on last bet/raise size
        """
        if not history:
            return 2  # Big blind as minimum

        # Find the last bet or raise
        last_bet_amount = self.get_last_bet_amount_from_history(history)

        if last_bet_amount == 0:
            return 2  # No bets yet, big blind minimum

        # Find what the last bet/raise was raising from
        previous_bet = 0
        last_bet_index = -1

        # Find the index of the last bet/raise
        for i in range(len(history) - 1, -1, -1):
            if history[i].startswith(('bet_', 'raise_')):
                last_bet_index = i
                break

        if last_bet_index == -1:
            return 2  # No bets found, use big blind

        # Determine what this bet/raise was raising from
        if history[last_bet_index].startswith('bet_'):
            # This is a bet - find what it was betting after
            if last_bet_index == 0:
                # Very first action was a bet (preflop open-raise)
                previous_bet = 2  # Raising from big blind
            else:
                # Look backwards to see what this bet was responding to
                for i in range(last_bet_index - 1, -1, -1):
                    if history[i].startswith(('bet_', 'raise_')):
                        # Betting after a previous bet/raise
                        pot_at_time = self.calculate_current_pot_size(
                            history[:i])
                        previous_bet = self.calculate_bet_amount_for_action(
                            history[i], pot_at_time)
                        break
                    elif history[i] == 'check':
                        # Betting after check(s) means raising from 0
                        previous_bet = 0
                        break
                    elif history[i] == 'call':
                        # Betting after a call - need to find what was called
                        continue  # Keep looking backwards
                else:
                    # Default to 0 if no previous bet found
                    previous_bet = 0
        else:
            # This is a raise - find the bet it was raising
            for i in range(last_bet_index - 1, -1, -1):
                if history[i].startswith(('bet_', 'raise_')):
                    pot_at_time = self.calculate_current_pot_size(history[:i])
                    previous_bet = self.calculate_bet_amount_for_action(
                        history[i], pot_at_time)
                    break
            else:
                previous_bet = 2  # Fallback to big blind

        # Minimum raise = size of last raise
        raise_size = last_bet_amount - previous_bet
        return max(2, raise_size)  # At least big blind

    def validate_raise_meets_minimum(self, raise_action, history, current_pot):
        """Check if a raise action meets minimum raise requirement"""
        proposed_amount = self.calculate_bet_amount_for_action(
            raise_action, current_pot)
        last_bet = self.get_last_bet_amount_from_history(history)
        min_raise = self.get_minimum_raise_amount(history)

        total_required = last_bet + min_raise
        return proposed_amount >= total_required

    def is_anyone_all_in(self, history):
        """Check if anyone went all-in this betting round"""
        return 'all_in' in history

    def get_current_bet_amount(self, history):
        """Get the current bet amount that needs to be called"""
        current_bet = 0
        current_player = 0

        for i, action in enumerate(history):
            if action.startswith('bet_') or action.startswith('raise_'):
                current_pot = self.calculate_current_pot_size(history[:i])
                current_bet = self.calculate_bet_amount_for_action(
                    action, current_pot)
            elif action == 'all_in':
                # Calculate actual all-in amount based on player's stack at that time
                player_stack_at_time = self.calculate_player_stack_after_history(
                    current_player, history[:i])
                current_bet = player_stack_at_time

            # Track which player made each action
            if action in ['check', 'bet_small', 'bet_medium', 'bet_large',
                          'raise_small', 'raise_medium', 'raise_large',
                          'all_in', 'call', 'fold']:
                current_player = 1 - current_player

        return current_bet

    def count_raises_in_history(self, history):
        """Count total number of raises (including initial bets) in current history"""
        raise_count = 0
        for action in history:
            if action.startswith('bet_') or action.startswith('raise_'):
                raise_count += 1
        return raise_count

    def is_round_complete(self, history):
        """Check if betting round is complete - like Leduc round_complete"""
        if not history:
            return False

        if 'fold' in history:
            return True

        # Both players checked
        if len(history) >= 2 and history[-2:] == ['check', 'check']:
            return True

        # Bet/raise followed by call
        if len(history) >= 2:
            last_two = history[-2:]
            # If last action is call, and previous was bet or raise
            if (last_two[-1] == 'call' and
                    (last_two[-2].startswith('bet_') or last_two[-2].startswith('raise_'))):
                return True

        # Handle longer sequences: bet-raise-call, bet-raise-raise-call, etc.
        if len(history) >= 3 and history[-1] == 'call':
            # Look backwards to find the betting sequence
            return self.is_betting_sequence_complete(history)

        return False

    def is_betting_sequence_complete(self, history):
        """
        Check if a complex betting sequence is complete
        Examples: bet-raise-call, bet-raise-raise-call, bet-raise-raise-raise-call
        """
        if not history or history[-1] != 'call':
            return False

        # Work backwards from the call to see if there was a bet/raise before it
        for i in range(len(history) - 2, -1, -1):
            action = history[i]
            if action.startswith('bet_') or action.startswith('raise_'):
                return True
            elif action in ['check', 'fold']:
                return False
            # Continue looking backwards through raises/calls

        return False

    def is_terminal(self, history, street):
        """Check if game is completely over"""
        if 'fold' in history:
            return True

        # If both players went all-in, game is terminal
        all_in_count = history.count('all_in')
        if all_in_count >= 2:
            return True

        # If one player went all-in and other called, game is terminal
        if 'all_in' in history and 'call' in history:
            all_in_index = history.index('all_in')
            # Check if there's a call after the all-in
            for i in range(all_in_index + 1, len(history)):
                if history[i] == 'call':
                    return True

        # Reached river and betting complete
        if street == 3 and self.is_round_complete(history):
            return True

        return False

    def get_utility(self, p0_cards, p1_cards, community_cards, history, street):
        """Calculate payoff - like Leduc get_utility"""
        # If someone folded, folder loses
        if 'fold' in history:
            folder_index = next(i for i, action in enumerate(
                history) if action == 'fold')
            folder_player = folder_index % 2
            return -1 if folder_player == 0 else 1

        # Showdown - compare hands using hand evaluator
        community_for_eval = community_cards[:self.get_community_cards_count(
            street)]

        p0_strength = self.hand_evaluator.evaluate_hand_strength(
            p0_cards, community_for_eval)[1]
        p1_strength = self.hand_evaluator.evaluate_hand_strength(
            p1_cards, community_for_eval)[1]

        if p0_strength > p1_strength:
            return 1  # I win
        elif p1_strength > p0_strength:
            return -1  # I lose
        else:
            return 0  # Tie

    def get_community_cards_count(self, street):
        """Get number of community cards for current street, bounds checking"""
        if street < 0:
            return 0
        elif street >= 3:  # River or beyond
            return 5
        else:
            return [0, 3, 4, 5][street]

    def print_game_state(self, history, street):
        """Debug helper - print current game state"""
        print(f"Street: {self.streets[street]}")
        print(f"History: {history}")
        print(f"Round complete: {self.is_round_complete(history)}")
        print(f"Terminal: {self.is_terminal(history, street)}")
        if history:
            # Can use simple also
            print(f"Legal actions: {self.get_legal_actions(history)}")
        print("---")

    def calculate_current_pot_size(self, history, starting_pot=3, starting_stack=100):
        """Calculate actual pot size based on betting history including all_in"""
        current_pot = starting_pot
        accumulated_bets = 0
        # Track how much each player has contributed
        player_contributions = {0: 0.0, 1: 0.0}
        current_player = 0

        # Add initial blinds to contributions
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
            elif action.startswith('raise_'):
                if 'small' in action:
                    raise_amount = 0.33 * (current_pot + accumulated_bets)
                elif 'medium' in action:
                    raise_amount = 0.66 * (current_pot + accumulated_bets)
                elif 'large' in action:
                    raise_amount = 1.0 * (current_pot + accumulated_bets)
                else:
                    # Default for unknown raises
                    raise_amount = 1.0 * (current_pot + accumulated_bets)
                accumulated_bets += raise_amount
                player_contributions[current_player] += raise_amount
            elif action == 'call':
                # Calculate proper call amount based on what was actually called
                call_amount = self.get_last_bet_amount_from_history(
                    history[:history.index(action)])

                # But cap it to player's available stack at that time
                player_stack_at_call = starting_stack - \
                    player_contributions[current_player]
                actual_call_amount = min(call_amount, player_stack_at_call)

                accumulated_bets += actual_call_amount
                player_contributions[current_player] += actual_call_amount
            elif action == 'all_in':
                remaining_stack = starting_stack - \
                    player_contributions[current_player]
                accumulated_bets += remaining_stack
                player_contributions[current_player] += remaining_stack

            # Switch to next player (for all betting actions)
            if action in ['check', 'bet_small', 'bet_medium', 'bet_large',
                          'raise_small', 'raise_medium', 'raise_large',
                          'all_in', 'call', 'fold']:
                current_player = 1 - current_player

        return current_pot + accumulated_bets

    def get_last_bet_amount_from_history(self, history):
        """Get the amount of the last bet/raise"""
        if not history:
            return 0

        # Find the last bet/raise action
        last_bet_action = None
        last_bet_index = -1

        for i, action in enumerate(reversed(history)):
            if action.startswith('bet_') or action.startswith('raise_'):
                last_bet_action = action
                last_bet_index = len(history) - 1 - i
                break

        if not last_bet_action:
            return 0

        # Calculate pot size at the time of that bet
        pot_at_bet_time = self.calculate_current_pot_size(
            history[:last_bet_index], 3, 100)

        # Return the actual bet amount using pot at that time
        return self.calculate_bet_amount_for_action(last_bet_action, pot_at_bet_time)

    def calculate_player_stack_after_history(self, player, history, starting_stack=100):
        """Calculate player's remaining stack after contributing to pot"""
        contribution = 0
        current_player = 0
        current_pot = 3

        # Add blinds
        if player == 0:
            contribution += 1  # Small blind
        else:
            contribution += 2  # Big blind

        for action in history:
            if current_player == player:
                if action == 'bet_small':
                    contribution += 0.33 * current_pot
                    current_pot += 0.33 * current_pot
                elif action == 'bet_medium':
                    contribution += 0.66 * current_pot
                    current_pot += 0.66 * current_pot
                elif action == 'bet_large':
                    contribution += 1.0 * current_pot
                    current_pot += 1.0 * current_pot
                elif action.startswith('raise_'):
                    if 'small' in action:
                        raise_amount = 0.33 * current_pot
                    elif 'medium' in action:
                        raise_amount = 0.66 * current_pot
                    elif 'large' in action:
                        raise_amount = 1.0 * current_pot
                    contribution += raise_amount
                    current_pot += raise_amount
                elif action == 'all_in':
                    all_in_amount = starting_stack - contribution
                    contribution += all_in_amount
                    current_pot += all_in_amount
                elif action == 'call':
                    call_amount = self.get_last_bet_amount_from_history(
                        history[:history.index(action)])
                    contribution += call_amount
                    current_pot += call_amount

            # Advance to next player
            if action in ['check', 'bet_small', 'bet_medium', 'bet_large',
                          'raise_small', 'raise_medium', 'raise_large',
                          'all_in', 'call', 'fold']:
                current_player = 1 - current_player

        return starting_stack - contribution

    def calculate_bet_amount_for_action(self, action, pot_size):
        """Convert action to actual bet amount"""
        if action == 'bet_small' or action == 'raise_small':
            return 0.33 * pot_size
        elif action == 'bet_medium' or action == 'raise_medium':
            return 0.66 * pot_size
        elif action == 'bet_large' or action == 'raise_large':
            return 1.0 * pot_size
        else:
            return 0
