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
        self.max_raises_per_street = 2  # 1 bet + 2 raises max per street
        self.hand_evaluator = HandEvaluator()  # For hand evaluation
        self.BET_MULTIPLIERS = {'small': 0.33, 'medium': 0.66, 'large': 1.00}
        # Memoization cache for pot/call/contribution calculations.
        # These are pure functions of their inputs, so the cache is valid forever.
        self._calc_cache = {}

    def _acting_player(self, action_index, street):
        """Which player acts at position action_index in this street's history.
        Preflop: SB (0) acts first. Postflop: BB (1) acts first (they are OOP)."""
        offset = 1 if street > 0 else 0
        return (action_index + offset) % 2

    def get_legal_actions(self, street, history, starting_pot, current_player):
        """Generate legal actions. Memoized: same inputs always give the same result."""
        key = ('legal', street, tuple(history), starting_pot, current_player)
        if key in self._calc_cache:
            return self._calc_cache[key]

        if self.is_round_complete(history):
            result = []
        elif 'fold' in history:
            result = []
        else:
            # Check betting cap (1 bet + 2 raises = 3 total)
            bet_and_raise_count = sum(1 for action in history
                                      if action.startswith(('bet_', 'raise_')))
            if bet_and_raise_count >= 3:
                result = ['fold', 'call']
            elif street == 0:
                result = self.get_preflop_legal_actions(street, history, starting_pot, current_player)
            else:
                result = self.get_postflop_legal_actions(street, history, starting_pot, current_player)

        self._calc_cache[key] = result
        return result

    def get_preflop_legal_actions(self, street, history, starting_pot, current_player):
        """Preflop actions with pot calculation"""

        # Check betting cap first
        bet_raise_count = sum(
            1 for action in history if action.startswith(('bet_', 'raise_')))
        if bet_raise_count >= 3:
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
                min_raise = self.get_min_raise(street, history, starting_pot)
                three_bet_amounts = self.get_preflop_bet_amounts('3bet', starting_pot)
                for size_name in ['small', 'medium', 'large']:
                    if three_bet_amounts[size_name] >= min_raise:
                        actions.append(f'raise_{size_name}')
                return actions

        elif len(history) == 2:
            if history[0] == 'call':
                if history[1].startswith('bet_'):  # BB opened after SB limp
                    actions = ['fold', 'call']
                    min_raise = self.get_min_raise(street, history, starting_pot)
                    three_bet_amounts = self.get_preflop_bet_amounts('3bet', starting_pot)
                    for size_name in ['small', 'medium', 'large']:
                        if three_bet_amounts[size_name] >= min_raise:
                            actions.append(f'raise_{size_name}')
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
            if raise_count < 2:
                min_raise = self.get_min_raise(
                    street, history, starting_pot)
                current_pot = self.calculate_current_pot(
                    starting_pot, history, street)
                call_amount = self.get_call_amount_from_history(
                    street, history, starting_pot)
                pot_after_call = current_pot + call_amount
                preflop_multipliers = self.get_preflop_bet_amounts(
                    'pot_relative', pot_after_call)

                for size_name in ['small', 'medium', 'large']:
                    raise_amount = preflop_multipliers[size_name] + call_amount
                    if raise_amount >= min_raise:
                        actions.append(f'raise_{size_name}')
            return actions

        elif last_action == 'call':
            return []  # Round complete

        return ['fold', 'call']  # Fallback

    def get_postflop_legal_actions(self, street, history, starting_pot, current_player):
        """Postflop actions with pot calculation"""

        current_pot = self.calculate_current_pot(starting_pot, history, street)

        if not history:  # First action postflop
            actions = ['check']
            for size_name, multiplier in self.BET_MULTIPLIERS.items():
                if multiplier * current_pot >= 2:
                    actions.append(f'bet_{size_name}')
            return actions

        # Check for double check
        if len(history) >= 2 and history[-2:] == ['check', 'check']:
            return []  # Round complete

        last_action = history[-1]

        if last_action == 'check':
            actions = ['check']
            for size_name, multiplier in self.BET_MULTIPLIERS.items():
                if multiplier * current_pot >= 2:
                    actions.append(f'bet_{size_name}')
            return actions

        elif last_action.startswith('bet_') or last_action.startswith('raise_'):
            actions = ['fold', 'call']
            raise_count = sum(
                1 for action in history if action.startswith('raise_'))
            if raise_count < 2:
                min_raise = self.get_min_raise(street, history, starting_pot)
                call_amount = self.get_call_amount_from_history(
                    street, history, starting_pot)
                pot_after_call = current_pot + call_amount

                for size_name, multiplier in self.BET_MULTIPLIERS.items():
                    raise_amount = multiplier * pot_after_call + call_amount
                    if raise_amount >= min_raise:
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
        Central function to calculate current pot size from street start and history.
        Memoized: same inputs always produce the same result.
        """
        key = ('pot', starting_pot, tuple(history), street)
        cached = self._calc_cache.get(key)
        if cached is not None:
            return cached

        current_pot = starting_pot
        for i, action in enumerate(history):
            if action in ['check', 'fold']:
                continue
            elif action == 'call':
                current_pot += self.get_call_amount_from_history(
                    street, history[:i], starting_pot)
            elif action.startswith('bet_'):
                current_pot += self.calculate_bet_amount(
                    action, street, starting_pot, history[:i])
            elif action.startswith('raise_'):
                current_pot += self.calculate_raise_amount(
                    action, street, starting_pot, history[:i], i)

        self._calc_cache[key] = current_pot
        return current_pot

    def calculate_bet_amount(self, action, street, starting_pot, history_before):
        """Calculate the actual bet amount for bet actions"""
        size = action.split('_')[1]
        current_player = self._acting_player(len(history_before), street)
        if street == 0:  # Preflop
            action_type = self.get_preflop_action_type(history_before)
            bet_amounts = self.get_preflop_bet_amounts(
                action_type, starting_pot)
            target_amount = bet_amounts[size]

            # Subtract current contribution
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
        current_player = self._acting_player(action_index, street)

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
                call_amount = self.get_call_amount_from_history(
                    street, history_before, starting_pot)
                pot_after_call = pot_before_raise + call_amount
                preflop_multipliers = self.get_preflop_bet_amounts(
                    'pot_relative', pot_after_call)
                target_amount = preflop_multipliers[size] + call_amount
        else:  # Postflop
            pot_before_raise = self.calculate_current_pot(
                starting_pot, history_before, street)
            call_amount = self.get_call_amount_from_history(
                street, history_before, starting_pot)
            pot_after_call = pot_before_raise + call_amount
            target_amount = self.BET_MULTIPLIERS[size] * pot_after_call + call_amount

        # Calculate current contribution
        current_contribution = self.get_player_contribution_this_round(
            history_before, street, starting_pot, current_player)

        return target_amount - current_contribution

    def get_utility(self, p0_cards, p1_cards, community_cards, history, street, starting_pot,
                    p0_prev_invested=0.0, p1_prev_invested=0.0):
        """Calculate utility from P0's perspective.

        p0_prev_invested / p1_prev_invested: chips each player put in during
        all streets BEFORE this one.  get_utility adds the current-street
        contribution to get the true total investment.
        """

        final_pot = self.calculate_current_pot(starting_pot, history, street)

        p0_this = self.get_player_contribution_this_round(
            history, street, starting_pot, 0)
        p0_total = p0_prev_invested + p0_this

        if 'fold' in history:
            folder_index = next(i for i, action in enumerate(
                history) if action == 'fold')
            folder_player = self._acting_player(folder_index, street)

            if folder_player == 0:  # P0 folded, P1 wins
                return -p0_total
            else:  # P1 folded, P0 wins
                return final_pot - p0_total

        else:  # Showdown
            community_for_eval = community_cards[:self.get_community_cards_count(
                street)]

            # phevaluator: lower raw score = stronger hand
            p0_raw = self.hand_evaluator.get_raw_hand_value(p0_cards, community_for_eval)
            p1_raw = self.hand_evaluator.get_raw_hand_value(p1_cards, community_for_eval)

            if p0_raw < p1_raw:  # P0 wins
                return final_pot - p0_total
            elif p1_raw < p0_raw:  # P1 wins
                return -p0_total
            else:  # Tie
                return (final_pot / 2) - p0_total

    def get_community_cards_count(self, street):
        """Get number of community cards for current street, bounds checking"""
        if street < 0:
            return 0
        elif street >= 3:  # River or beyond
            return 5
        else:
            return [0, 3, 4, 5][street]

    def get_min_raise(self, street, history, starting_pot):
        """Calculate minimum raise. Memoized: pure function of its inputs."""
        key = ('minraise', street, tuple(history), starting_pot)
        if key in self._calc_cache:
            return self._calc_cache[key]
        result = self._compute_min_raise(street, history, starting_pot)
        self._calc_cache[key] = result
        return result

    def _compute_min_raise(self, street, history, starting_pot):
        """Calculate minimum raise using forward pot calculation"""

        if not history:
            return 2.0  # Big blind minimum

        # Preflop: seed with BB post (2 chips) so the first raise increment is
        # computed as open - BB rather than open + open.
        bet_amounts = [2.0] if street == 0 else []

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
                        call_amount = self.get_call_amount_from_history(
                            street, history[:i], starting_pot)
                        pot_after_call = pot_before + call_amount
                        size = action.split('_')[1]
                        preflop_multipliers = self.get_preflop_bet_amounts(
                            'pot_relative', pot_after_call)
                        bet_amounts.append(preflop_multipliers[size] + call_amount)
                else:  # Postflop
                    pot_before = self.calculate_current_pot(
                        starting_pot, history[:i], street)
                    call_amount = self.get_call_amount_from_history(
                        street, history[:i], starting_pot)
                    pot_after_call = pot_before + call_amount
                    size = action.split('_')[1]
                    bet_amounts.append(self.BET_MULTIPLIERS[size] * pot_after_call + call_amount)

        # Minimum raise is the difference between last two bet amounts
        if len(bet_amounts) >= 2:
            min_raise_increment = bet_amounts[-1] - bet_amounts[-2]
            return bet_amounts[-1] + min_raise_increment
        elif len(bet_amounts) == 1:
            return bet_amounts[-1] + bet_amounts[-1]  # Double the last bet

        return 2.0  # Default minimum

    def get_player_contribution_this_round(self, history, street, starting_pot, current_player):
        """Calculate player's contribution this round using forward calculation. Memoized."""
        key = ('contrib', tuple(history), street, starting_pot, current_player)
        cached = self._calc_cache.get(key)
        if cached is not None:
            return cached

        if street == 0:
            contribution = 1.0 if current_player == 0 else 2.0
        else:
            contribution = 0.0

        for i, action in enumerate(history):
            action_player = self._acting_player(i, street)

            if action_player == current_player:
                if action == 'call':
                    contribution += self.get_call_amount_from_history(
                        street, history[:i], starting_pot)

                elif action.startswith(('bet_', 'raise_')):
                    if street == 0:  # Preflop
                        action_type = self.get_preflop_action_type(history[:i])
                        if action_type != 'pot_relative':
                            bet_amounts = self.get_preflop_bet_amounts(
                                action_type, starting_pot)
                            size = action.split('_')[1]
                            contribution = bet_amounts[size]
                        else:
                            pot_before = self.calculate_current_pot(
                                starting_pot, history[:i], street)
                            call_amount = self.get_call_amount_from_history(
                                street, history[:i], starting_pot)
                            pot_after_call = pot_before + call_amount
                            size = action.split('_')[1]
                            preflop_multipliers = self.get_preflop_bet_amounts(
                                'pot_relative', pot_after_call)
                            contribution = preflop_multipliers[size] + call_amount
                    else:  # Postflop
                        pot_before = self.calculate_current_pot(
                            starting_pot, history[:i], street)
                        call_amount = self.get_call_amount_from_history(
                            street, history[:i], starting_pot)
                        pot_after_call = pot_before + call_amount
                        size = action.split('_')[1]
                        contribution += self.BET_MULTIPLIERS[size] * pot_after_call + call_amount

        self._calc_cache[key] = contribution
        return contribution

    def get_call_amount_from_history(self, street, history, starting_pot):
        """
        Return the extra chips the next-to-act player must put in to call.
        `history` is the sequence BEFORE the call.  Memoized.
        """
        key = ('call', street, tuple(history), starting_pot)
        cached = self._calc_cache.get(key)
        if cached is not None:
            return cached

        if not history:
            result = 1.0 if street == 0 else 0.0
            self._calc_cache[key] = result
            return result

        current_player = self._acting_player(len(history), street)

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
                        raiser_call = self.get_call_amount_from_history(
                            street, history[:i], starting_pot)
                        pot_after_call = pot_before + raiser_call
                        size = act.split('_')[1]
                        preflop_multipliers = self.get_preflop_bet_amounts(
                            'pot_relative', pot_after_call)
                        last_bet_amt = preflop_multipliers[size] + raiser_call
                else:
                    pot_before = self.calculate_current_pot(
                        starting_pot, history[:i], street)
                    raiser_call = self.get_call_amount_from_history(
                        street, history[:i], starting_pot)
                    pot_after_call = pot_before + raiser_call
                    size = act.split('_')[1]
                    last_bet_amt = self.BET_MULTIPLIERS[size] * pot_after_call + raiser_call
                break

        player_contrib = self.get_player_contribution_this_round(
            history, street, starting_pot, current_player)

        result = max(0.0, last_bet_amt - player_contrib)
        self._calc_cache[key] = result
        return result

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
                'small': 9 * big_blind,   # 18 chips (~3× small open)
                'medium': 12 * big_blind,  # 24 chips (~2.4× medium open)
                'large': 16 * big_blind   # 32 chips (large 3-bet / squeeze)
            }
        else:  # pot_relative
            return {
                'small': 0.66 * current_pot,
                'medium': 1.33 * current_pot,
                'large': 2.0 * current_pot
            }
