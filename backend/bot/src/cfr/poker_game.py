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

    def get_legal_actions(self, street, history, starting_pot, current_player):
        """Generate legal actions using calculated current pot"""

        # Check round completion first
        if self.is_round_complete(history):
            return []

        if 'fold' in history:
            return []

        # Check betting cap (1 bet + 3 raises = 4 total)
        bet_and_raise_count = sum(1 for action in history
                                  if action.startswith(('bet_', 'raise_')))
        if bet_and_raise_count >= 4:
            return ['fold', 'call']

        # Use dedicated preflop logic
        if street == 0:
            return self.get_preflop_legal_actions(street, history, starting_pot, current_player)

        # Postflop logic
        return self.get_postflop_legal_actions(history, starting_pot, current_player)

    def get_preflop_legal_actions(self, street, history, starting_pot, current_player):
        """Preflop actions with pot calculation"""

        # Check betting cap first
        bet_raise_count = sum(
            1 for action in history if action.startswith(('bet_', 'raise_')))
        if bet_raise_count >= 4:
            return ['fold', 'call']

        # Count raises for raise limit checking
        raise_count = sum(
            1 for action in history if action.startswith('raise_'))

        if not history:  # SB opening action
            actions = ['fold', 'call']
            # Add standard opening sizes (3BB, 5BB, 7BB)
            for size_name in ['small', 'medium', 'large']:
                actions.append(f'bet_{size_name}')
            return actions

        elif len(history) == 1:  # BB's turn after SB action
            if history[0] == 'call':  # SB called
                actions = ['check']
                # BB can bet with opening sizes
                for size_name in ['small', 'medium', 'large']:
                    actions.append(f'bet_{size_name}')
                return actions

            elif history[0].startswith('bet_'):  # SB opened
                actions = ['fold', 'call']
                if history[0] == 'bet_small':
                    for size_name in ['small', 'medium', 'large']:
                        actions.append(f'raise_{size_name}')
                if history[0] == 'bet_medium':
                    for size_name in ['medium', 'large']:
                        actions.append(f'raise_{size_name}')
                if history[0] == 'bet_large':
                    actions.append('raise_large')
                return actions

        elif len(history) == 2:
            if history[0] == 'call':
                if history[1].startswith('bet_'):  # SB opened
                    actions = ['fold', 'call']
                    if history[1] == 'bet_small':
                        for size_name in ['small', 'medium', 'large']:
                            actions.append(f'raise_{size_name}')
                    if history[1] == 'bet_medium':
                        for size_name in ['medium', 'large']:
                            actions.append(f'raise_{size_name}')
                    if history[1] == 'bet_large':
                        actions.append('raise_large')
                    return actions

        # Later preflop actions
        last_action = history[-1]

        if last_action == 'check':
            actions = ['check']
            for size_name in ['small', 'medium', 'large']:
                actions.append(f'bet_{size_name}')
            return actions

        elif last_action.startswith(('bet_', 'raise_')):
            actions = ['fold', 'call']
            if raise_count < 3:
                # Calculate if raises are valid
                min_raise = self.get_min_raise(
                    street, history, starting_pot)
                player_contribution = self.get_player_contribution_this_round(
                    history, street, starting_pot, current_player)

                for size_name in ['small', 'medium', 'large']:
                    # Calculate what this raise would be
                    action_type = self.get_preflop_action_type(history)
                    if action_type != 'pot_relative':
                        print(
                            f"DEBUG: history: {history}, starting_pot: {starting_pot}")
                        raise TypeError('Should not be happening')
                    else:
                        current_pot = self.calculate_current_pot(
                            starting_pot, history, street)
                        preflop_multipliers = self.get_preflop_bet_amounts(
                            'pot_relative', current_pot)
                        raise_amount = preflop_multipliers[size_name]

                    if raise_amount >= min_raise and raise_amount > player_contribution:
                        actions.append(f'raise_{size_name}')
            return actions

        elif last_action == 'call':
            return []  # Round complete

        return ['fold', 'call']  # Fallback

    def get_postflop_legal_actions(self, history, starting_pot, current_player):
        """Postflop actions with pot calculation"""

        current_pot = self.calculate_current_pot(
            starting_pot, history, 1)  # postflop street

        if not history:  # First action postflop
            actions = ['check']
            # print(f"DEBUG: starting_pot={starting_pot}, current_pot={current_pot}")
            for size_name, multiplier in self.BET_MULTIPLIERS.items():
                bet_amount = multiplier * current_pot
                # print(f"DEBUG: {size_name}: {multiplier} × {current_pot} = {bet_amount}, >= 2? {bet_amount >= 2}")
                if bet_amount >= 2:
                    actions.append(f'bet_{size_name}')
            # print(f"DEBUG: final actions = {actions}")
            return actions

        # Check for double check
        if len(history) >= 2 and history[-2:] == ['check', 'check']:
            return []  # Round complete

        last_action = history[-1]

        if last_action == 'check':
            actions = ['check']
            for size_name, multiplier in self.BET_MULTIPLIERS.items():
                bet_amount = multiplier * current_pot
                if bet_amount >= 2:
                    actions.append(f'bet_{size_name}')
            return actions

        elif last_action.startswith('bet_'):
            actions = ['fold', 'call']
            # Add raises if under limit
            raise_count = sum(
                1 for action in history if action.startswith('raise_'))
            if raise_count < 3:
                min_raise = self.get_min_raise(1, history, starting_pot)
                player_contribution = self.get_player_contribution_this_round(
                    history, 1, starting_pot, current_player)

                for size_name, multiplier in self.BET_MULTIPLIERS.items():
                    raise_amount = multiplier * current_pot
                    if raise_amount >= min_raise and raise_amount > player_contribution:
                        actions.append(f'raise_{size_name}')
            return actions

        elif last_action.startswith('raise_'):
            actions = ['fold', 'call']
            # Similar raise logic
            raise_count = sum(
                1 for action in history if action.startswith('raise_'))
            if raise_count < 3:
                min_raise = self.get_min_raise(1, history, starting_pot)
                player_contribution = self.get_player_contribution_this_round(
                    history, 1, starting_pot, current_player)

                for size_name, multiplier in self.BET_MULTIPLIERS.items():
                    raise_amount = multiplier * current_pot
                    if raise_amount >= min_raise and raise_amount > player_contribution:
                        actions.append(f'raise_{size_name}')
            return actions

        elif last_action == 'call':
            return []  # Round complete

        return ['fold', 'call']

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

    def is_terminal(self, history, street):
        """Check if game is completely over"""
        if 'fold' in history:
            return True

        # Reached river and betting complete
        if street == 3 and self.is_round_complete(history):
            return True

        return False

    def calculate_current_pot(self, starting_pot, history, street):
        """
        Central function to calculate current pot size from street start and history
        """
        current_pot = starting_pot

        for i, action in enumerate(history):
            if action in ['check', 'fold']:
                continue
            elif action == 'call':
                call_amount = self.get_call_amount_from_history(  # VERIFY
                    street, history[:i], starting_pot)
                current_pot += call_amount
            elif action.startswith('bet_'):
                bet_amount = self.calculate_bet_amount(
                    action, street, starting_pot, history[:i])
                current_pot += bet_amount
            elif action.startswith('raise_'):
                raise_amount = self.calculate_raise_amount(
                    action, street, starting_pot, history[:i], i)
                current_pot += raise_amount

        return current_pot

    def calculate_bet_amount(self, action, street, starting_pot, history_before):
        """Calculate the actual bet amount for bet actions"""
        size = action.split('_')[1]
        current_player = len(history_before) % 2
        if street == 0:  # Preflop
            action_type = self.get_preflop_action_type(history_before)
            bet_amounts = self.get_preflop_bet_amounts(
                action_type, starting_pot)
            target_amount = bet_amounts[size]

            # Subtract current contributio
            current_contribution = self.get_player_contribution_this_round(
                history_before, street, starting_pot, current_player)
            return target_amount - current_contribution
        else:  # Postflop
            pot_before_bet = self.calculate_current_pot(
                starting_pot, history_before, street)
            return self.BET_MULTIPLIERS[size] * pot_before_bet

    def calculate_raise_amount(self, action, street, starting_pot, history_before, action_index):
        """Calculate the additional amount needed for raise actions"""
        size = action.split('_')[1]
        current_player = action_index % 2

        # Calculate target total contribution
        if street == 0:  # Preflop
            action_type = self.get_preflop_action_type(history_before)
            if action_type != 'pot_relative':
                bet_amounts = self.get_preflop_bet_amounts(
                    action_type, starting_pot)
                target_amount = bet_amounts[size]
            else:
                pot_before_raise = self.calculate_current_pot(
                    starting_pot, history_before, street)
                preflop_multipliers = self.get_preflop_bet_amounts(
                    'pot_relative', pot_before_raise)
                target_amount = preflop_multipliers[size]
        else:  # Postflop
            pot_before_raise = self.calculate_current_pot(
                starting_pot, history_before, street)
            target_amount = self.BET_MULTIPLIERS[size] * pot_before_raise

        # Calculate current contribution
        current_contribution = self.get_player_contribution_this_round(
            history_before, street, starting_pot, current_player)

        return target_amount - current_contribution

    def get_utility(self, p0_cards, p1_cards, community_cards, history, street, starting_pot):
        """Calculate utility with pot calculation"""

        # Calculate final pot size
        final_pot = self.calculate_current_pot(
            starting_pot, history, street)

        # Calculate contributions for each player
        p0_contribution = self.get_player_contribution_this_round(
            history, street, starting_pot, 0)
        p1_contribution = self.get_player_contribution_this_round(
            history, street, starting_pot, 1)

        if 'fold' in history:
            # Find who folded
            folder_index = next(i for i, action in enumerate(
                history) if action == 'fold')
            folder_player = folder_index % 2

            if folder_player == 0:  # P0 folded, P1 wins
                return -p0_contribution
            else:  # P1 folded, P0 wins
                return final_pot - p0_contribution

        else:  # Showdown
            community_for_eval = community_cards[:self.get_community_cards_count(
                street)]

            p0_strength = self.hand_evaluator.evaluate_hand_strength(
                p0_cards, community_for_eval)[1]
            p1_strength = self.hand_evaluator.evaluate_hand_strength(
                p1_cards, community_for_eval)[1]

            if p0_strength > p1_strength:  # P0 wins
                return final_pot - p0_contribution
            elif p1_strength > p0_strength:  # P1 wins
                return -p0_contribution
            else:  # Tie
                return (final_pot / 2) - p0_contribution

    def get_community_cards_count(self, street):
        """Get number of community cards for current street, bounds checking"""
        if street < 0:
            return 0
        elif street >= 3:  # River or beyond
            return 5
        else:
            return [0, 3, 4, 5][street]

    def get_min_raise(self, street, history, starting_pot):
        """Calculate minimum raise using forward pot calculation"""

        if not history:
            return 2.0  # Big blind minimum

        # Find the last two bet/raise amounts
        bet_amounts = []

        for i, action in enumerate(history):
            if action.startswith(('bet_', 'raise_')):
                if street == 0:  # Preflop
                    action_type = self.get_preflop_action_type(history[:i])
                    if action_type != 'pot_relative':
                        bet_amounts_dict = self.get_preflop_bet_amounts(
                            action_type, starting_pot)
                        size = action.split('_')[1]
                        bet_amounts.append(bet_amounts_dict[size])
                    else:
                        pot_before = self.calculate_current_pot(
                            starting_pot, history[:i], street)
                        size = action.split('_')[1]
                        preflop_multipliers = self.get_preflop_bet_amounts(
                            'pot_relative', pot_before)
                        bet_amounts.append(preflop_multipliers[size])
                else:  # Postflop
                    pot_before = self.calculate_current_pot(
                        starting_pot, history[:i], street)
                    size = action.split('_')[1]
                    bet_amounts.append(self.BET_MULTIPLIERS[size] * pot_before)

        # Minimum raise is the difference between last two bet amounts
        if len(bet_amounts) >= 2:
            min_raise_increment = bet_amounts[-1] - bet_amounts[-2]
            return bet_amounts[-1] + min_raise_increment
        elif len(bet_amounts) == 1:
            return bet_amounts[-1] + bet_amounts[-1]  # Double the last bet

        return 2.0  # Default minimum

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

    def get_player_contribution_this_round(self, history, street, starting_pot, current_player):
        """Calculate player's contribution this round using forward calculation"""

        # Base contribution for preflop blinds
        if street == 0:
            contribution = 1.0 if current_player == 0 else 2.0
        else:
            contribution = 0.0

        # Build forward through history to find player's last contribution
        for i, action in enumerate(history):
            action_player = i % 2

            if action_player == current_player:
                if action == 'call':
                    call_amount = self.get_call_amount_from_history(
                        street, history[:i], starting_pot)
                    contribution += call_amount

                elif action.startswith(('bet_', 'raise_')):
                    if street == 0:  # Preflop
                        action_type = self.get_preflop_action_type(history[:i])
                        if action_type != 'pot_relative':
                            bet_amounts = self.get_preflop_bet_amounts(
                                action_type, starting_pot)
                            size = action.split('_')[1]
                            # Replace blind contribution
                            contribution = bet_amounts[size]
                        else:
                            pot_before = self.calculate_current_pot(
                                starting_pot, history[:i], street)
                            size = action.split('_')[1]
                            preflop_multipliers = self.get_preflop_bet_amounts(
                                'pot_relative', pot_before)
                            contribution = preflop_multipliers[size]
                    else:  # Postflop
                        pot_before = self.calculate_current_pot(
                            starting_pot, history[:i], street)
                        size = action.split('_')[1]
                        contribution = self.BET_MULTIPLIERS[size] * pot_before

        return contribution

    def get_call_amount_from_history(self, street, history, starting_pot):
        """
        Return the extra chips the next-to-act player must put in to call.
        `history` is the sequence BEFORE the call.
        """
        if not history:
            return 1.0 if street == 0 else 0.0

        current_player = len(history) % 2

        # Find size of last bet/raise
        last_bet_amt = 0
        for i in range(len(history) - 1, -1, -1):
            act = history[i]
            if act.startswith(('bet_', 'raise_')):
                if street == 0:
                    action_type = self.get_preflop_action_type(history[:i])
                    if action_type != 'pot_relative':
                        size = act.split('_')[1]
                        last_bet_amt = self.get_preflop_bet_amounts(
                            action_type, starting_pot)[size]
                    else:
                        pot_before = self.calculate_current_pot(
                            starting_pot, history[:i], street)
                        size = act.split('_')[1]
                        preflop_multipliers = self.get_preflop_bet_amounts(
                            'pot_relative', pot_before)
                        last_bet_amt = preflop_multipliers[size]
                else:
                    pot_before = self.calculate_current_pot(
                        starting_pot, history[:i], street)
                    size = act.split('_')[1]
                    last_bet_amt = self.BET_MULTIPLIERS[size] * pot_before
                break

        # What has this player invested so far this round?
        player_contrib = self.get_player_contribution_this_round(
            history, street, starting_pot, current_player)

        # Extra chips needed to call
        return max(0.0, last_bet_amt - player_contrib)

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
                'small': 0.66 * current_pot,
                'medium': 1.33 * current_pot,
                'large': 2.0 * current_pot
            }
