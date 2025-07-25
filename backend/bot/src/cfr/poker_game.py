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

    def get_legal_actions(self, street, history, current_pot, current_player):
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

        # Use dedicated preflop logic
        if street == 0:
            return self.get_preflop_legal_actions(street, history, current_pot, current_player)

        # Check betting cap (industry standard: 1 bet + 3 raises = 4 total)
        bet_and_raise_count = sum(1 for action in history
                                  if action.startswith(('bet_', 'raise_')))

        if bet_and_raise_count >= 4:
            return ['fold', 'call']

        # Count raises in current history
        raise_count = sum(
            1 for action in history if action.startswith('raise_'))

        if not history and street == 0:
            actions = ['call', 'fold', 'raise_large']
            return actions

        if street == 0 and len(history) == 1:
            if history[0] == 'call':
                actions = ['check', 'bet_medium', 'bet_large']
            elif history[0] == 'raise_large':  # SB raised, BB can fold/call/reraise
                actions = ['fold', 'call', 'raise_medium', 'raise_large']
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
                committment = self.get_player_contribution_this_round(
                    history, street, current_pot, current_player)
                min_raise = self.get_min_raise(street, history, current_pot)
                for raise_action, raise_amount in potential_raises:
                    # print(
                    #     f"DEBUG: {raise_action}: amount={raise_amount}, commitment={committment}, min_raise={min_raise}, diff={raise_amount-committment}")
                    if raise_amount >= 2 and raise_amount - committment >= min_raise:
                        actions.append(raise_action)
            return actions

        elif last_action == 'call':
            return []  # Round complete

        else:
            return ['fold', 'call']

    def get_preflop_legal_actions(self, street, history, current_pot, current_player):
        """Generate preflop legal actions with BB-based sizing"""

        # Check betting cap first (1 bet + 3 raises = 4 total)
        bet_raise_count = sum(
            1 for action in history if action.startswith(('bet_', 'raise_')))
        if bet_raise_count >= 4:
            return ['fold', 'call']

        # Count raises for raise limit checking
        raise_count = sum(
            1 for action in history if action.startswith('raise_'))

        if not history:  # SB opening action
            actions = ['fold', 'call']
            action_type = self.get_preflop_action_type(history)
            bet_amounts = self.get_preflop_bet_amounts(
                action_type, current_pot)

            # Add all bet sizes (3BB, 5BB, 7BB)
            for size_name in ['small', 'medium', 'large']:
                actions.append(f'bet_{size_name}')
            return actions

        elif len(history) == 1:  # BB's turn after SB action
            if history[0] == 'call':  # SB called
                actions = ['check']
                action_type = self.get_preflop_action_type(history)

                # BB can bet with opening sizes (3BB, 5BB, 7BB)
                for size_name in ['small', 'medium', 'large']:
                    actions.append(f'bet_{size_name}')
                return actions

            elif history[0].startswith('bet_'):  # SB opened
                actions = ['fold', 'call']

                # BB can 3-bet if under raise limit
                if raise_count < 3:
                    action_type = self.get_preflop_action_type(history)

                    if action_type != 'pot_relative':  # Still in BB-multiple phase
                        # 3-bet sizes (6BB, 10BB, 14BB)
                        for size_name in ['small', 'medium', 'large']:
                            actions.append(f'raise_{size_name}')
                    else:  # Switch to pot-relative
                        min_raise = self.get_min_raise(0, history, current_pot)
                        committment = self.get_player_contribution_this_round(
                            history, street, current_pot, current_player)
                        for size_name in ['small', 'medium', 'large']:
                            raise_amount = self.BET_MULTIPLIERS[size_name] * \
                                current_pot
                            if raise_amount >= 2 and raise_amount - committment >= min_raise:
                                actions.append(f'raise_{size_name}')
                return actions

        else:  # Later preflop actions (len(history) >= 2)
            last_action = history[-1]

            if last_action == 'check':
                actions = ['check']
                # After check, can bet with opening sizes
                for size_name in ['small', 'medium', 'large']:
                    actions.append(f'bet_{size_name}')
                return actions

            elif last_action.startswith(('bet_', 'raise_')):
                actions = ['fold', 'call']

                # Can raise if under limits
                if raise_count < 3:
                    action_type = self.get_preflop_action_type(history)

                    if action_type != 'pot_relative':  # Still in BB-multiple phase
                        for size_name in ['small', 'medium', 'large']:
                            actions.append(f'raise_{size_name}')
                    else:  # Switch to pot-relative
                        min_raise = self.get_min_raise(0, history, current_pot)
                        committment = self.get_player_contribution_this_round(
                            history, street, current_pot, current_player)
                        for size_name in ['small', 'medium', 'large']:
                            raise_amount = self.BET_MULTIPLIERS[size_name] * \
                                current_pot
                            if raise_amount >= 2 and raise_amount - committment >= min_raise:
                                actions.append(f'raise_{size_name}')
                return actions

            elif last_action == 'call':
                return []  # Round complete

        return ['fold', 'call']  # Fallback

    def is_round_complete(self, history):
        """Check if betting round is complete - like Leduc round_complete"""
        if not history:
            return False

        if 'fold' in history:
            return True

        # Preflop special case: call-check ends the round
        if len(history) >= 2 and history[-2:] == ['call', 'check']:
            return True

        # Both players checked
        if len(history) >= 2 and history[-2:] == ['check', 'check']:
            return True

        # Any bet/raise followed by call ends the round
        if len(history) >= 2 and history[-1] == 'call':
            prev_action = history[-2]
            if prev_action.startswith(('bet_', 'raise_')):
                return True

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
            history, street, final_pot, 0)
        p1_betting_contribution = self.get_player_contribution_this_round(
            history, street, final_pot, 1)

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

    def get_min_raise(self, street, history, accumulated_pot):
        BASE_MIN_RAISE = 2
        if not history:
            return BASE_MIN_RAISE

        largest_raise = BASE_MIN_RAISE
        last_bet = None
        current_pot = accumulated_pot

        # Work backwards through history to get pot at each action
        for i in range(len(history) - 1, -1, -1):
            action = history[i]
            if action.startswith(("bet_", "raise_")):
                # Calculate pot BEFORE this action
                pot_before_action = current_pot
                for j in range(len(history) - 1, i, -1):
                    if history[j].startswith(('bet_', 'raise_')):
                        mult = self._multiplier_for_action(
                            history[j], street, history[:j+1])
                        pot_before_action = pot_before_action / mult

                # Now calculate the actual bet amount at time of action
                action_size = action.split('_')[1]
                amount = self.BET_MULTIPLIERS[action_size] * pot_before_action

                # Calculate raise size
                if last_bet is None:
                    raise_size = amount
                else:
                    raise_size = amount - last_bet

                if raise_size > largest_raise:
                    largest_raise = raise_size
                last_bet = amount

        return max(BASE_MIN_RAISE, largest_raise)

    def _multiplier_for_action(self, action, street=None, history=None):
        """Return multiplier for reverse-engineering pot evolution with preflop awareness"""

        # Handle preflop BB-based multipliers
        if street == 0 and history is not None:
            action_type = self.get_preflop_action_type(history)

            if action_type != 'pot_relative':  # BB-multiple phase
                # Get the action index to determine which bet this was
                action_index = len(
                    [a for a in history if a.startswith(('bet_', 'raise_'))])

                # Calculate multiplier based on BB amounts
                bet_amounts = self.get_preflop_bet_amounts(
                    action_type, 1.0)  # Use base pot of 1
                size = action.split('_')[1] if '_' in action else 'medium'

                if size in bet_amounts:
                    # For BB-based amounts, multiplier is (pot + bet_amount) / pot
                    return 1.0 + bet_amounts[size]

        # Fall back to pot-relative multipliers (postflop or late preflop)
        for size, bet_mult in self.BET_MULTIPLIERS.items():
            if action.endswith(size):
                return 1.0 + bet_mult

        return 1.0

    def get_player_contribution_this_round(self, history, street, accumulated_pot, current_player):
        """
        Reverse-engineer how much the current player has contributed this betting round.
        Handles preflop blinds and postflop betting actions.
        """

        # Base contribution for preflop blinds
        base_contribution = 0.0
        if street == 0:
            base_contribution = 1.0 if current_player == 0 else 2.0

        if not history:
            return base_contribution

         # Find the last action by current_player (bet, raise, OR call)
        last_player_action = None
        last_player_action_index = -1

        for i in range(len(history) - 1, -1, -1):
            if i % 2 == current_player:
                last_player_action = history[i]
                last_player_action_index = i
                break

        if last_player_action is None:
            return base_contribution

        # Handle betting actions (bet_/raise_)
        if last_player_action.startswith(('bet_', 'raise_')):
            # Work backwards to find pot before this action
            pot_before_action = accumulated_pot

            for j in range(len(history) - 1, last_player_action_index, -1):
                action = history[j]
                if action.startswith(('bet_', 'raise_')):
                    mult = self._multiplier_for_action(
                        action, street, history[:j+1])
                    pot_before_action = pot_before_action / mult

            # Calculate contribution based on street
            if street == 0:  # Preflop
                action_type = self.get_preflop_action_type(
                    history[:last_player_action_index])
                if action_type != 'pot_relative':
                    bet_amounts = self.get_preflop_bet_amounts(
                        action_type, pot_before_action)
                    size = last_player_action.split('_')[1]
                    return bet_amounts[size]

            # Postflop or late preflop (pot-relative)
            size = last_player_action.split('_')[1]
            return self.BET_MULTIPLIERS[size] * pot_before_action

        # Handle call actions
        elif last_player_action == 'call':
            if street == 0:  # Preflop call
                # Find what the player is calling
                last_bet_amount = 0
                for i in range(last_player_action_index - 1, -1, -1):
                    if history[i].startswith(('bet_', 'raise_')):
                        # Reverse-engineer the bet amount
                        pot_at_bet = accumulated_pot
                        for j in range(len(history) - 1, i, -1):
                            if history[j].startswith(('bet_', 'raise_')):
                                mult = self._multiplier_for_action(
                                    history[j], street, history[:j+1])
                                pot_at_bet = pot_at_bet / mult

                        action_type = self.get_preflop_action_type(history[:i])
                        if action_type != 'pot_relative':
                            bet_amounts = self.get_preflop_bet_amounts(
                                action_type, pot_at_bet)
                            size = history[i].split('_')[1]
                            last_bet_amount = bet_amounts[size]
                        else:
                            size = history[i].split('_')[1]
                            last_bet_amount = self.BET_MULTIPLIERS[size] * \
                                pot_at_bet
                        break

                return last_bet_amount  # Player matched the last bet amount
            else:
                # Postflop call - similar logic but simpler
                return self.reverse_engineer_call_amount(history, street, accumulated_pot, last_player_action_index)

        return base_contribution

    def reverse_engineer_call_amount(self, history, street, accumulated_pot, call_index):
        """Helper to reverse-engineer what amount was called"""
        # Find the last bet/raise before the call
        for i in range(call_index - 1, -1, -1):
            if history[i].startswith(('bet_', 'raise_')):
                # Calculate what that bet/raise amount was
                pot_before_bet = accumulated_pot
                for j in range(len(history) - 1, i, -1):
                    if history[j].startswith(('bet_', 'raise_')):
                        mult = self._multiplier_for_action(
                            history[j], street, history[:j+1])
                        pot_before_bet = pot_before_bet / mult

                size = history[i].split('_')[1]
                return self.BET_MULTIPLIERS[size] * pot_before_bet

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
        last_mult = self._multiplier_for_action(
            last_act, street, history[:last_idx+1])

        # Pot immediately before the last action
        pot_before_last = accumulated_pot / last_mult
        # Incremental chips posted by the last bettor
        last_amount = accumulated_pot - pot_before_last

        # 2) Find the prior bet/raise (if any) to compute its incremental amount
        prev_amount = 0.0
        for j in range(last_idx - 1, -1, -1):
            act = history[j]
            if act.startswith(('bet_', 'raise_')):
                prev_mult = self._multiplier_for_action(
                    act, street, history[:j+1])
                # Pot immediately before that prior action
                pot_before_prev = pot_before_last / prev_mult
                prev_amount = pot_before_last - pot_before_prev
                break

        # 3) Call amount is the difference of those two increments
        call_amt = last_amount - prev_amount
        return max(0.0, round(call_amt, 6))

    def get_preflop_action_type(self, history):
        """Determine what type of preflop action this is"""
        if not history:
            return 'open'  # First action
        bet_raise_count = sum(
            1 for action in history if action.startswith(('bet_', 'raise_')))

        if bet_raise_count == 0:
            return 'open'  # SB called, BB can open
        elif bet_raise_count == 1:
            return '3bet'  # First raise/3-bet
        else:
            return 'pot_relative'  # 4-bet and beyond

    def get_preflop_bet_amounts(self, action_type, current_pot):
        """Get bet amounts based on preflop action type"""
        big_blind = 2

        if action_type == 'open':
            return {
                'small': 3 * big_blind,   # 6 chips
                'medium': 5 * big_blind,  # 10 chips
                'large': 7 * big_blind    # 14 chips
            }
        elif action_type == '3bet':
            return {
                'small': 6 * big_blind,   # 12 chips
                'medium': 10 * big_blind,  # 20 chips
                'large': 14 * big_blind   # 28 chips
            }
        else:  # pot_relative
            return {
                'small': 0.33 * current_pot,
                'medium': 0.66 * current_pot,
                'large': 1.0 * current_pot
            }
