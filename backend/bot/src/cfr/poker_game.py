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
        self.BET_MULTIPLIERS = {'small': 0.33, 'medium': 0.66, 'large': 1.00}

    def get_legal_actions(self, street, history, current_pot):
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

        if bet_and_raise_count >= 4:
            return ['fold', 'call']

        # Count raises in current history
        raise_count = sum(
            1 for action in history if action.startswith('raise_'))

        if not history and street == 0:
            actions = ['call', 'fold', 'bet_large']
            return actions

        if not history and street > 0:  # First action after preflop
            actions = ['check']
            # Only add bet sizes player can afford
            if 0.33 * current_pot >= 2:
                actions.append('bet_small')
            if 0.66 * current_pot >= 2:
                actions.append('bet_medium')
            if 1.0 * current_pot >= 2:
                actions.append('bet_large')
            # if current_pot >= 2:
            #     print("current_pot: ", current_pot)
            return actions

        if len(history) >= 2 and history[-2:] == ['check', 'check']:
            return []  # Round complete

        if history[-1] == 'call':
            return []  # Round complete

        last_action = history[-1]

        if last_action == 'check':
            actions = ['check']
            if 0.33 * current_pot >= 2:
                actions.append('bet_small')
            if 0.66 * current_pot >= 2:
                actions.append('bet_medium')
            if 1.0 * current_pot >= 2:
                actions.append('bet_large')
            # if current_pot >= 2:
            #     print("current_pot: ", current_pot)
            return actions

        elif last_action.startswith('bet_'):
            # Can always fold and call
            actions = ['fold', 'call']
            potential_raises = [
                ('raise_small', 0.33 * current_pot),
                ('raise_medium', 0.66 * current_pot),
                ('raise_large', 1.0 * current_pot)
            ]
            bet_amount = self.get_call_amount_from_history(
                street, history, current_pot)

            for raise_action, raise_amount in potential_raises:
                if raise_amount >= 2 and raise_amount >= 2 * bet_amount:
                    actions.append(raise_action)
            return actions

        elif last_action.startswith('raise_'):
            # Can always fold and call
            actions = ['fold', 'call']

            # Can raise if less than 3 raises
            if raise_count < 3:
                potential_raises = [
                    ('raise_small', 0.33 * current_pot),
                    ('raise_medium', 0.66 * current_pot),
                    ('raise_large', 1.0 * current_pot)
                ]
                min_raise = self.get_min_raise(history, current_pot)
                for raise_action, raise_amount in potential_raises:
                    if raise_amount >= 2 and raise_amount >= min_raise:
                        actions.append(raise_action)
            return actions

        elif last_action == 'call':
            return []  # Round complete

        else:
            return ['fold', 'call']

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

        # Reached river and betting complete
        if street == 3 and self.is_round_complete(history):
            return True

        return False

    def get_utility(self, p0_cards, p1_cards, community_cards, history, street, final_pot):
        """Calculate payoff using accurate player contribution tracking"""

        # Calculate actual contributions for each player
        p0_betting_contribution = self.get_player_contribution_this_round(
            history, final_pot, 0)
        p1_betting_contribution = self.get_player_contribution_this_round(
            history, final_pot, 1)

        # Add blind contributions only for preflop
        if street == 0:  # Preflop
            p0_total_contribution = p0_betting_contribution + 1  # Small blind
            p1_total_contribution = p1_betting_contribution + 2  # Big blind
        else:  # Postflop streets
            p0_total_contribution = p0_betting_contribution
            p1_total_contribution = p1_betting_contribution

        if 'fold' in history:
            # Find who folded
            folder_index = next(i for i, action in enumerate(
                history) if action == 'fold')
            folder_player = folder_index % 2

            if folder_player == 0:  # P0 folded, P1 wins
                return -p0_total_contribution  # P0 loses their contribution
            else:  # P1 folded, P0 wins
                return final_pot - p0_total_contribution  # P0's net gain

        else:  # Showdown - no fold in history
            # Get community cards for hand evaluation
            community_for_eval = community_cards[:self.get_community_cards_count(
                street)]

            # Evaluate hand strengths
            p0_strength = self.hand_evaluator.evaluate_hand_strength(
                p0_cards, community_for_eval)[1]
            p1_strength = self.hand_evaluator.evaluate_hand_strength(
                p1_cards, community_for_eval)[1]

            if p0_strength > p1_strength:  # P0 wins
                return final_pot - p0_total_contribution  # P0's net gain
            elif p1_strength > p0_strength:  # P1 wins
                return -p0_total_contribution  # P0's net loss
            else:  # Tie - split pot
                # P0's net from split
                return (final_pot / 2) - p0_total_contribution

    def get_community_cards_count(self, street):
        """Get number of community cards for current street, bounds checking"""
        if street < 0:
            return 0
        elif street >= 3:  # River or beyond
            return 5
        else:
            return [0, 3, 4, 5][street]

    def _action_to_amount(self, action, pot):
        """
        Convert 'bet_small', 'raise_medium', etc. into an absolute chip amount.
        """
        if not (action.startswith('bet_') or action.startswith('raise_')):
            return 0.0
        size = action.split('_', 1)[1]
        return self.BET_MULTIPLIERS.get(size, 0.0) * pot

    def get_min_raise(self, history, accumulated_pot):
        """
        Returns minimum total amount for a valid raise.
        Reverse-engineers from accumulated_pot and history like get_call_amount_from_history.
        """
        if not history:
            return 2  # Big blind minimum

        # Get current call amount using existing method
        call_amount = self.get_call_amount_from_history(
            0, history, accumulated_pot)

        # Reverse-engineer bet amounts by working backwards through history
        bet_amounts = []
        current_pot = accumulated_pot

        for i in range(len(history) - 1, -1, -1):
            action = history[i]
            if action.startswith(('bet_', 'raise_')):
                mult = self._multiplier_for_action(action)
                pot_before = current_pot / mult
                bet_amount = current_pot - pot_before
                # Insert at front to maintain chronological order
                bet_amounts.insert(0, bet_amount)
                current_pot = pot_before

        if not bet_amounts:
            return 2

        # Calculate last full raise amount
        if len(bet_amounts) == 1:
            # Only one bet in history, so the full raise amount is that bet
            last_raise = bet_amounts[0]
        else:
            # Multiple bets/raises, last raise is difference between last two amounts
            last_raise = bet_amounts[-1] - bet_amounts[-2]

        # Minimum raise = call amount + last full raise amount
        return call_amount + last_raise

    def _multiplier_for_action(self, action):
        """Return the 1.33/1.66/2.0 multiplier for bet_/raise_ actions."""
        for size, mult in self.BET_MULTIPLIERS.items():
            if action.endswith(size):
                return 1.0 + mult
        return 1.0

    def get_player_contribution_this_round(self, history, accumulated_pot, current_player):
        """
        Reverse-engineer how much the current player has contributed this betting round.
        Similar to get_call_amount_from_history approach.
        """
        if not history:
            return 0.0

        # Find the last betting action by current_player
        for i in range(len(history) - 1, -1, -1):
            if i % 2 == current_player and history[i].startswith(('bet_', 'raise_')):
                # Found the player's last betting action
                last_action = history[i]

                # Work backwards to find pot before this action
                pot_before_action = accumulated_pot

                # Reverse through all actions after this one
                for j in range(len(history) - 1, i, -1):
                    action = history[j]
                    if action.startswith(('bet_', 'raise_')):
                        mult = self._multiplier_for_action(action)
                        pot_before_action = pot_before_action / mult

                # Player's total contribution is multiplier * pot_before_action
                size = last_action.split('_')[1]
                return self.BET_MULTIPLIERS[size] * pot_before_action

        return 0.0

    def get_call_amount_from_history(self, street, history, accumulated_pot):
        """
        Amount the next player must put in to call.
        history            – sequence of bet_/raise_ on this street
        accumulated_pot    – pot size *after* all history actions
        """
        if not history and street == 0:
            return 1
        # 1) Find the last bet or raise
        last_idx = -1
        for i in range(len(history) - 1, -1, -1):
            if history[i].startswith(('bet_', 'raise_')):
                last_idx = i
                break
        if last_idx < 0:
            return 0.0

        last_act = history[last_idx]
        last_mult = self._multiplier_for_action(last_act)

        # Pot immediately before the last action
        pot_before_last = accumulated_pot / last_mult
        # Incremental chips posted by the last bettor
        last_amount = accumulated_pot - pot_before_last

        # 2) Find the prior bet/raise (if any) to compute its incremental amount
        prev_amount = 0.0
        for j in range(last_idx - 1, -1, -1):
            act = history[j]
            if act.startswith(('bet_', 'raise_')):
                prev_mult = self._multiplier_for_action(act)
                # Pot immediately before that prior action
                pot_before_prev = pot_before_last / prev_mult
                prev_amount = pot_before_last - pot_before_prev
                break

        # 3) Call amount is the difference of those two increments
        call_amt = last_amount - prev_amount
        return max(0.0, round(call_amt, 6))
