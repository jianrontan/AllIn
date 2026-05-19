# backend/bot/src/cfr/poker_game.py
from ..abstractions.hand_evaluator import HandEvaluator

STARTING_STACK = 200


class PokerGame:
    """
    Simplified poker game for CFR training - like LeducPoker class
    This is separate from PyPokerEngine gameplay
    """

    def __init__(self):
        self.streets = ['preflop', 'flop', 'turn', 'river']
        self.max_raises_per_street = 2  # max raises per street (1 bet + 2 raises = 3 total)
        self.hand_evaluator = HandEvaluator()
        self.BET_MULTIPLIERS = {'small': 0.33, 'medium': 0.66, 'large': 1.00}
        self._calc_cache = {}

    def _acting_player(self, action_index, street):
        """Preflop: SB (0) acts first. Postflop: BB (1) acts first (they are OOP)."""
        offset = 1 if street > 0 else 0
        return (action_index + offset) % 2

    # ------------------------------------------------------------------
    # Legal action generation
    # ------------------------------------------------------------------

    def get_legal_actions(self, street, history, starting_pot, current_player,
                          p0_stack=None, p1_stack=None, p0_prev=0.0, p1_prev=0.0):
        """Generate legal actions, respecting stack sizes."""
        key = ('legal', street, tuple(history), starting_pot, current_player,
               p0_stack, p1_stack, p0_prev, p1_prev)
        if key in self._calc_cache:
            return self._calc_cache[key]

        if self.is_round_complete(history):
            result = []
        elif 'fold' in history:
            result = []
        else:
            bet_and_raise_count = sum(1 for a in history if a.startswith(('bet_', 'raise_')))
            if 'allin' in history:
                # After allin only fold/call are available
                result = ['fold', 'call']
            elif bet_and_raise_count >= self.max_raises_per_street + 1:
                result = ['fold', 'call']
            elif street == 0:
                result = self.get_preflop_legal_actions(
                    street, history, starting_pot, current_player,
                    p0_stack, p1_stack, p0_prev, p1_prev)
            else:
                result = self.get_postflop_legal_actions(
                    street, history, starting_pot, current_player,
                    p0_stack, p1_stack, p0_prev, p1_prev)

        self._calc_cache[key] = result
        return result

    def _action_cost(self, action, street, history, starting_pot,
                     current_player, p0_prev=0.0, p1_prev=0.0):
        """Chips `current_player` must put in to take `action` given `history`."""
        if action in ('check', 'fold'):
            return 0.0
        if action == 'call':
            return self.get_call_amount_from_history(
                street, history, starting_pot, p0_prev, p1_prev)
        if action == 'allin':
            return self._allin_amount(
                history, street, starting_pot, current_player, p0_prev, p1_prev)
        if action.startswith('bet_'):
            return self.calculate_bet_amount(
                action, street, starting_pot, history, p0_prev, p1_prev)
        if action.startswith('raise_'):
            return self.calculate_raise_amount(
                action, street, starting_pot, history, len(history), p0_prev, p1_prev)
        return 0.0

    def _apply_stack_constraints(self, actions, player_remaining,
                                 street, history, starting_pot,
                                 current_player, p0_prev=0.0, p1_prev=0.0):
        """
        Replace any sized bet/raise the player cannot afford with 'allin'.

        A sized bet/raise whose exact chip cost meets or exceeds the player's
        remaining stack is not a distinct action — it is simply an all-in. Using
        the exact cost (not a pot-multiple heuristic) guarantees no action ever
        commits more chips than the player actually has.
        """
        if player_remaining is None:
            return actions

        needs_allin = False
        filtered = []

        for action in actions:
            if not (action.startswith('bet_') or action.startswith('raise_')):
                filtered.append(action)
                continue

            cost = self._action_cost(action, street, history, starting_pot,
                                     current_player, p0_prev, p1_prev)
            if cost >= player_remaining:
                needs_allin = True
            else:
                filtered.append(action)

        # Short-stack shove safety net: every node that reaches this function
        # permits aggression. If the abstraction produced no sized bet/raise at
        # all (e.g. every sized raise fell below the minimum legal raise), the
        # player can still go all-in — provided the shove is a genuine raise
        # (commits strictly more than a call would).
        if not needs_allin and not any(
                a.startswith(('bet_', 'raise_')) for a in filtered):
            call_amount = self.get_call_amount_from_history(
                street, history, starting_pot, p0_prev, p1_prev)
            allin_amount = self._allin_amount(
                history, street, starting_pot, current_player, p0_prev, p1_prev)
            if allin_amount > call_amount:
                needs_allin = True

        if needs_allin and 'allin' not in filtered:
            # Insert allin before the first bet/raise in the list, or at end
            insert_at = next(
                (i for i, a in enumerate(filtered) if a.startswith(('bet_', 'raise_'))),
                len(filtered)
            )
            filtered.insert(insert_at, 'allin')

        return filtered

    def get_preflop_legal_actions(self, street, history, starting_pot, current_player,
                                  p0_stack=None, p1_stack=None, p0_prev=0.0, p1_prev=0.0):
        """Preflop actions with pot calculation"""

        raise_count = sum(1 for a in history if a.startswith('raise_'))
        player_remaining = (p0_stack if current_player == 0 else p1_stack)

        if not history:
            actions = ['fold', 'call', 'bet_small', 'bet_medium', 'bet_large']
            return self._apply_stack_constraints(
                actions, player_remaining, street, history, starting_pot,
                current_player, p0_prev, p1_prev)

        elif len(history) == 1:
            if history[0] == 'call':
                actions = ['check', 'bet_small', 'bet_medium', 'bet_large']
                return self._apply_stack_constraints(
                actions, player_remaining, street, history, starting_pot,
                current_player, p0_prev, p1_prev)

            elif history[0].startswith('bet_'):
                actions = ['fold', 'call']
                min_raise = self.get_min_raise(street, history, starting_pot, p0_prev, p1_prev)
                three_bet_amounts = self.get_preflop_bet_amounts('3bet', starting_pot)
                for size_name in ['small', 'medium', 'large']:
                    if three_bet_amounts[size_name] >= min_raise:
                        actions.append(f'raise_{size_name}')
                return self._apply_stack_constraints(
                actions, player_remaining, street, history, starting_pot,
                current_player, p0_prev, p1_prev)

        elif len(history) == 2:
            if history[0] == 'call' and history[1].startswith('bet_'):
                actions = ['fold', 'call']
                min_raise = self.get_min_raise(street, history, starting_pot, p0_prev, p1_prev)
                three_bet_amounts = self.get_preflop_bet_amounts('3bet', starting_pot)
                for size_name in ['small', 'medium', 'large']:
                    if three_bet_amounts[size_name] >= min_raise:
                        actions.append(f'raise_{size_name}')
                return self._apply_stack_constraints(
                actions, player_remaining, street, history, starting_pot,
                current_player, p0_prev, p1_prev)

        last_action = history[-1]

        if last_action == 'check':
            actions = ['check', 'bet_small', 'bet_medium', 'bet_large']
            return self._apply_stack_constraints(
                actions, player_remaining, street, history, starting_pot,
                current_player, p0_prev, p1_prev)

        elif last_action.startswith(('bet_', 'raise_')):
            actions = ['fold', 'call']
            if raise_count < self.max_raises_per_street:
                min_raise = self.get_min_raise(street, history, starting_pot, p0_prev, p1_prev)
                pot_now = self.calculate_current_pot(starting_pot, history, street, p0_prev, p1_prev)
                call_amount = self.get_call_amount_from_history(
                    street, history, starting_pot, p0_prev, p1_prev)
                pot_after_call = pot_now + call_amount
                preflop_multipliers = self.get_preflop_bet_amounts('pot_relative', pot_after_call)

                for size_name in ['small', 'medium', 'large']:
                    raise_amount = preflop_multipliers[size_name] + call_amount
                    if raise_amount >= min_raise:
                        actions.append(f'raise_{size_name}')
            return self._apply_stack_constraints(
                actions, player_remaining, street, history, starting_pot,
                current_player, p0_prev, p1_prev)

        elif last_action == 'call':
            return []

        return ['fold', 'call']

    def get_postflop_legal_actions(self, street, history, starting_pot, current_player,
                                   p0_stack=None, p1_stack=None, p0_prev=0.0, p1_prev=0.0):
        """Postflop actions with pot calculation"""

        current_pot = self.calculate_current_pot(starting_pot, history, street, p0_prev, p1_prev)
        player_remaining = (p0_stack if current_player == 0 else p1_stack)

        if not history:
            actions = ['check']
            for size_name, multiplier in self.BET_MULTIPLIERS.items():
                if multiplier * current_pot >= 2:
                    actions.append(f'bet_{size_name}')
            return self._apply_stack_constraints(
                actions, player_remaining, street, history, starting_pot,
                current_player, p0_prev, p1_prev)

        if len(history) >= 2 and history[-2:] == ['check', 'check']:
            return []

        last_action = history[-1]

        if last_action == 'check':
            actions = ['check']
            for size_name, multiplier in self.BET_MULTIPLIERS.items():
                if multiplier * current_pot >= 2:
                    actions.append(f'bet_{size_name}')
            return self._apply_stack_constraints(
                actions, player_remaining, street, history, starting_pot,
                current_player, p0_prev, p1_prev)

        elif last_action.startswith('bet_') or last_action.startswith('raise_'):
            actions = ['fold', 'call']
            raise_count = sum(1 for a in history if a.startswith('raise_'))
            if raise_count < self.max_raises_per_street:
                min_raise = self.get_min_raise(street, history, starting_pot, p0_prev, p1_prev)
                call_amount = self.get_call_amount_from_history(
                    street, history, starting_pot, p0_prev, p1_prev)
                pot_after_call = current_pot + call_amount

                for size_name, multiplier in self.BET_MULTIPLIERS.items():
                    raise_amount = multiplier * pot_after_call + call_amount
                    if raise_amount >= min_raise:
                        actions.append(f'raise_{size_name}')
            return self._apply_stack_constraints(
                actions, player_remaining, street, history, starting_pot,
                current_player, p0_prev, p1_prev)

        elif last_action == 'call':
            return []

        return ['fold', 'call']

    # ------------------------------------------------------------------
    # Terminal / round-complete checks
    # ------------------------------------------------------------------

    def is_round_complete(self, history):
        """Check if betting round is complete."""
        if not history:
            return False
        if 'fold' in history:
            return True
        # Preflop limp: SB calls BB, then BB checks to end preflop
        if len(history) >= 2 and history[-2:] == ['call', 'check']:
            return True
        # Both checked
        if len(history) >= 2 and history[-2:] == ['check', 'check']:
            return True
        # Bet/raise followed by call
        if len(history) >= 2 and history[-1] == 'call':
            prev_action = history[-2]
            if prev_action.startswith(('bet_', 'raise_')):
                return True
        # Allin followed by call or fold
        if len(history) >= 2 and history[-2] == 'allin' and history[-1] in ('call', 'fold'):
            return True
        return False

    def is_terminal(self, history, street):
        """Check if game is completely over."""
        if 'fold' in history:
            return True
        # Allin called = showdown (no more streets needed)
        if len(history) >= 2 and history[-2] == 'allin' and history[-1] == 'call':
            return True
        if street == 3 and self.is_round_complete(history):
            return True
        return False

    # ------------------------------------------------------------------
    # Pot / contribution calculations (now handle 'allin')
    # ------------------------------------------------------------------

    def _allin_amount(self, history_before, street, starting_pot, player, p0_prev, p1_prev):
        """Compute how many chips 'player' puts in when going allin."""
        prev_invested = p0_prev if player == 0 else p1_prev
        contrib_before = self.get_player_contribution_this_round(
            history_before, street, starting_pot, player, p0_prev, p1_prev)
        remaining = STARTING_STACK - prev_invested - contrib_before
        return max(0.0, remaining)

    def calculate_current_pot(self, starting_pot, history, street, p0_prev=0.0, p1_prev=0.0):
        """
        Central function to calculate current pot size from street start and history.
        Memoized: same inputs always produce the same result.
        """
        key = ('pot', starting_pot, tuple(history), street, p0_prev, p1_prev)
        cached = self._calc_cache.get(key)
        if cached is not None:
            return cached

        current_pot = starting_pot
        for i, action in enumerate(history):
            if action in ['check', 'fold']:
                continue
            elif action == 'allin':
                player = self._acting_player(i, street)
                amount = self._allin_amount(history[:i], street, starting_pot, player, p0_prev, p1_prev)
                current_pot += amount
            elif action == 'call':
                current_pot += self.get_call_amount_from_history(
                    street, history[:i], starting_pot, p0_prev, p1_prev)
            elif action.startswith('bet_'):
                current_pot += self.calculate_bet_amount(
                    action, street, starting_pot, history[:i], p0_prev, p1_prev)
            elif action.startswith('raise_'):
                current_pot += self.calculate_raise_amount(
                    action, street, starting_pot, history[:i], i, p0_prev, p1_prev)

        self._calc_cache[key] = current_pot
        return current_pot

    def calculate_bet_amount(self, action, street, starting_pot, history_before,
                             p0_prev=0.0, p1_prev=0.0):
        """Calculate the actual bet amount for bet actions"""
        size = action.split('_')[1]
        current_player = self._acting_player(len(history_before), street)
        if street == 0:
            action_type = self.get_preflop_action_type(history_before)
            bet_amounts = self.get_preflop_bet_amounts(action_type, starting_pot)
            target_amount = bet_amounts[size]
            current_contribution = self.get_player_contribution_this_round(
                history_before, street, starting_pot, current_player, p0_prev, p1_prev)
            return target_amount - current_contribution
        else:
            pot_before_bet = self.calculate_current_pot(
                starting_pot, history_before, street, p0_prev, p1_prev)
            return self.BET_MULTIPLIERS[size] * pot_before_bet

    def calculate_raise_amount(self, action, street, starting_pot, history_before, action_index,
                               p0_prev=0.0, p1_prev=0.0):
        """Calculate the additional amount needed for raise actions"""
        size = action.split('_')[1]
        current_player = self._acting_player(action_index, street)

        if street == 0:
            action_type = self.get_preflop_action_type(history_before)
            if action_type != 'pot_relative':
                bet_amounts = self.get_preflop_bet_amounts(action_type, starting_pot)
                target_amount = bet_amounts[size]
            else:
                pot_before_raise = self.calculate_current_pot(
                    starting_pot, history_before, street, p0_prev, p1_prev)
                call_amount = self.get_call_amount_from_history(
                    street, history_before, starting_pot, p0_prev, p1_prev)
                pot_after_call = pot_before_raise + call_amount
                preflop_multipliers = self.get_preflop_bet_amounts('pot_relative', pot_after_call)
                target_amount = preflop_multipliers[size] + call_amount
        else:
            pot_before_raise = self.calculate_current_pot(
                starting_pot, history_before, street, p0_prev, p1_prev)
            call_amount = self.get_call_amount_from_history(
                street, history_before, starting_pot, p0_prev, p1_prev)
            pot_after_call = pot_before_raise + call_amount
            target_amount = self.BET_MULTIPLIERS[size] * pot_after_call + call_amount

        current_contribution = self.get_player_contribution_this_round(
            history_before, street, starting_pot, current_player, p0_prev, p1_prev)
        return target_amount - current_contribution

    def get_utility(self, p0_cards, p1_cards, community_cards, history, street, starting_pot,
                    p0_prev_invested=0.0, p1_prev_invested=0.0):
        """Calculate utility from P0's perspective."""

        final_pot = self.calculate_current_pot(
            starting_pot, history, street, p0_prev_invested, p1_prev_invested)

        p0_this = self.get_player_contribution_this_round(
            history, street, starting_pot, 0, p0_prev_invested, p1_prev_invested)
        p0_total = p0_prev_invested + p0_this

        if 'fold' in history:
            folder_index = next(i for i, action in enumerate(history)
                                if action == 'fold')
            folder_player = self._acting_player(folder_index, street)

            if folder_player == 0:
                return -p0_total
            else:
                return final_pot - p0_total

        else:  # Showdown (allin+call or river complete)
            # When allin occurred, run out remaining board cards
            if 'allin' in history:
                community_for_eval = community_cards[:5]
            else:
                community_for_eval = community_cards[:self.get_community_cards_count(street)]

            p0_raw = self.hand_evaluator.get_raw_hand_value(p0_cards, community_for_eval)
            p1_raw = self.hand_evaluator.get_raw_hand_value(p1_cards, community_for_eval)

            if p0_raw < p1_raw:
                return final_pot - p0_total
            elif p1_raw < p0_raw:
                return -p0_total
            else:
                return (final_pot / 2) - p0_total

    def get_community_cards_count(self, street):
        if street < 0:
            return 0
        elif street >= 3:
            return 5
        else:
            return [0, 3, 4, 5][street]

    # ------------------------------------------------------------------
    # Contribution / call helpers
    # ------------------------------------------------------------------

    def get_player_contribution_this_round(self, history, street, starting_pot, current_player,
                                           p0_prev=0.0, p1_prev=0.0):
        """Calculate player's contribution this round. Memoized."""
        key = ('contrib', tuple(history), street, starting_pot, current_player, p0_prev, p1_prev)
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
                        street, history[:i], starting_pot, p0_prev, p1_prev)

                elif action == 'allin':
                    amount = self._allin_amount(history[:i], street, starting_pot,
                                                current_player, p0_prev, p1_prev)
                    contribution += amount

                elif action.startswith(('bet_', 'raise_')):
                    if street == 0:
                        action_type = self.get_preflop_action_type(history[:i])
                        if action_type != 'pot_relative':
                            bet_amounts = self.get_preflop_bet_amounts(action_type, starting_pot)
                            size = action.split('_')[1]
                            contribution = bet_amounts[size]
                        else:
                            pot_before = self.calculate_current_pot(
                                starting_pot, history[:i], street, p0_prev, p1_prev)
                            call_amount = self.get_call_amount_from_history(
                                street, history[:i], starting_pot, p0_prev, p1_prev)
                            pot_after_call = pot_before + call_amount
                            size = action.split('_')[1]
                            preflop_multipliers = self.get_preflop_bet_amounts(
                                'pot_relative', pot_after_call)
                            contribution = preflop_multipliers[size] + call_amount
                    else:
                        pot_before = self.calculate_current_pot(
                            starting_pot, history[:i], street, p0_prev, p1_prev)
                        call_amount = self.get_call_amount_from_history(
                            street, history[:i], starting_pot, p0_prev, p1_prev)
                        pot_after_call = pot_before + call_amount
                        size = action.split('_')[1]
                        contribution += self.BET_MULTIPLIERS[size] * pot_after_call + call_amount

        self._calc_cache[key] = contribution
        return contribution

    def get_call_amount_from_history(self, street, history, starting_pot,
                                     p0_prev=0.0, p1_prev=0.0):
        """
        Return the extra chips the next-to-act player must put in to call.
        `history` is the sequence BEFORE the call. Memoized.
        """
        key = ('call', street, tuple(history), starting_pot, p0_prev, p1_prev)
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
            if act == 'allin':
                bet_player = self._acting_player(i, street)
                last_bet_amt = self._allin_amount(
                    history[:i], street, starting_pot, bet_player, p0_prev, p1_prev)
                break
            elif act.startswith(('bet_', 'raise_')):
                if street == 0:
                    action_type = self.get_preflop_action_type(history[:i])
                    if action_type != 'pot_relative':
                        size = act.split('_')[1]
                        last_bet_amt = self.get_preflop_bet_amounts(
                            action_type, starting_pot)[size]
                    else:
                        pot_before = self.calculate_current_pot(
                            starting_pot, history[:i], street, p0_prev, p1_prev)
                        raiser_call = self.get_call_amount_from_history(
                            street, history[:i], starting_pot, p0_prev, p1_prev)
                        pot_after_call = pot_before + raiser_call
                        size = act.split('_')[1]
                        preflop_multipliers = self.get_preflop_bet_amounts(
                            'pot_relative', pot_after_call)
                        last_bet_amt = preflop_multipliers[size] + raiser_call
                else:
                    pot_before = self.calculate_current_pot(
                        starting_pot, history[:i], street, p0_prev, p1_prev)
                    raiser_call = self.get_call_amount_from_history(
                        street, history[:i], starting_pot, p0_prev, p1_prev)
                    pot_after_call = pot_before + raiser_call
                    size = act.split('_')[1]
                    last_bet_amt = self.BET_MULTIPLIERS[size] * pot_after_call + raiser_call
                break

        player_contrib = self.get_player_contribution_this_round(
            history, street, starting_pot, current_player, p0_prev, p1_prev)

        result = max(0.0, last_bet_amt - player_contrib)
        self._calc_cache[key] = result
        return result

    # ------------------------------------------------------------------
    # Min-raise / preflop helpers (unchanged in logic, no allin in these)
    # ------------------------------------------------------------------

    def get_min_raise(self, street, history, starting_pot, p0_prev=0.0, p1_prev=0.0):
        """Calculate minimum raise. Memoized."""
        key = ('minraise', street, tuple(history), starting_pot, p0_prev, p1_prev)
        if key in self._calc_cache:
            return self._calc_cache[key]
        result = self._compute_min_raise(street, history, starting_pot, p0_prev, p1_prev)
        self._calc_cache[key] = result
        return result

    def _compute_min_raise(self, street, history, starting_pot, p0_prev=0.0, p1_prev=0.0):
        if not history:
            return 2.0

        bet_amounts = [2.0] if street == 0 else []

        for i, action in enumerate(history):
            if action.startswith(('bet_', 'raise_')):
                if street == 0:
                    action_type = self.get_preflop_action_type(history[:i])
                    if action_type != 'pot_relative':
                        bet_amounts_dict = self.get_preflop_bet_amounts(action_type, starting_pot)
                        size = action.split('_')[1]
                        bet_amounts.append(bet_amounts_dict[size])
                    else:
                        pot_before = self.calculate_current_pot(
                            starting_pot, history[:i], street, p0_prev, p1_prev)
                        call_amount = self.get_call_amount_from_history(
                            street, history[:i], starting_pot, p0_prev, p1_prev)
                        pot_after_call = pot_before + call_amount
                        size = action.split('_')[1]
                        preflop_multipliers = self.get_preflop_bet_amounts('pot_relative', pot_after_call)
                        bet_amounts.append(preflop_multipliers[size] + call_amount)
                else:
                    pot_before = self.calculate_current_pot(
                        starting_pot, history[:i], street, p0_prev, p1_prev)
                    call_amount = self.get_call_amount_from_history(
                        street, history[:i], starting_pot, p0_prev, p1_prev)
                    pot_after_call = pot_before + call_amount
                    size = action.split('_')[1]
                    bet_amounts.append(self.BET_MULTIPLIERS[size] * pot_after_call + call_amount)

        if len(bet_amounts) >= 2:
            min_raise_increment = bet_amounts[-1] - bet_amounts[-2]
            return bet_amounts[-1] + min_raise_increment
        elif len(bet_amounts) == 1:
            return bet_amounts[-1] + bet_amounts[-1]

        return 2.0

    def get_preflop_action_type(self, history):
        if not history:
            return 'open'
        bet_raise_count = sum(1 for a in history if a.startswith(('bet_', 'raise_')))
        if bet_raise_count == 0:
            return 'open'
        elif bet_raise_count == 1:
            return '3bet'
        else:
            return 'pot_relative'

    def get_preflop_bet_amounts(self, action_type, current_pot):
        big_blind = 2
        if action_type == 'open':
            return {'small': 3 * big_blind, 'medium': 5 * big_blind, 'large': 7 * big_blind}
        elif action_type == '3bet':
            return {'small': 9 * big_blind, 'medium': 12 * big_blind, 'large': 16 * big_blind}
        else:  # pot_relative
            return {
                'small': 0.66 * current_pot,
                'medium': 1.33 * current_pot,
                'large': 2.0 * current_pot,
            }
