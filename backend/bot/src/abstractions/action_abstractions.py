# backend/bot/abstractions/action_abstractions.py
class ActionAbstraction:
    """
    Like ['k', 'b', 'c', 'r', 'f'] but with bet sizing
    """

    def __init__(self):
        self.bet_sizes = {
            'small': 0.33,    # 1/3 pot
            'medium': 0.66,   # 2/3 pot
            'large': 1.0,     # pot bet
        }

    def is_legal_bet_size(self, game_state, multiplier):
        """
        Check if bet size is legal
        min bet = max(big blind, current bet), max = stack
        """
        pot_size = game_state.get('pot_size', 0)
        player_stack = game_state.get('player_stack', 0)
        current_bet = game_state.get('current_bet', 0)
        big_blind = game_state.get('big_blind', 2)

        # Calculate actual bet amount
        if multiplier == 'stack':  # All-in case
            bet_amount = player_stack
        else:
            bet_amount = multiplier * pot_size

        # Minimum bet requirement
        min_bet = max(big_blind, current_bet, 2)

        # Check constraints
        if bet_amount < min_bet:
            return False
        if bet_amount > player_stack:
            return False

        return True

    def abstract_action_history(self, pypoker_actions, game_state, street='preflop'):
        """
        Convert PyPokerEngine action history to simple format
        Leduc history: "kb", "brc", etc.
        Poker: Similar but with bet size categories
        """
        abstracted_history = ""

        for action in pypoker_actions:
            if action['action'] == 'fold':
                abstracted_history += 'f'
            elif action['action'] == 'call':
                abstracted_history += 'c'
            elif action['action'] in ['bet', 'raise']:
                # Abstract bet size to category
                bet_category = self.categorize_bet_size(
                    action, game_state, pypoker_actions, street)
                # 's', 'm', 'l', 'o' for small/med/large/over
                abstracted_history += bet_category[0]
            elif action['action'] == 'check':
                abstracted_history += 'k'

        return abstracted_history

    def categorize_bet_size(self, action, game_state, action_history=None, street='preflop'):
        """
        Determine if bet is small/medium/large with off-tree detection
        Following training structure from poker_game.py
        """
        bet_amount = action.get('amount', 0)
        pot_size = game_state.get('pot_size', 1)
        big_blind = game_state.get('big_blind', 2)

        # Handle all-in case
        player_stack = game_state.get('player_stack', float('inf'))
        if bet_amount >= player_stack * 0.95:  # 95% of stack = all-in
            return 'allin'

        # Calculate pot before this bet for ratio
        pre_bet_pot_size = pot_size - bet_amount
        if pre_bet_pot_size <= 0:
            pre_bet_pot_size = pot_size

        # OFF-TREE DETECTION based on training structure
        if street == 'preflop':
            # Count previous bet/raise actions
            bet_raise_count = 0
            if action_history:
                bet_raise_count = sum(1 for a in action_history
                                      if a.get('action', '').upper() in ['BET', 'RAISE'])

            if bet_raise_count == 0:  # Opening bet
                # Max large open = 14 chips (7BB)
                if bet_amount > 14:
                    return 'large'
            elif bet_raise_count == 1:  # 3-bet
                # Max large 3-bet = 28 chips (14BB)
                if bet_amount > 28:
                    return 'large'
            else:  # 4-bet+
                # Max large = 2.0x pot
                if bet_amount > 2.0 * pre_bet_pot_size:
                    return 'large'
        else:  # Postflop
            # Max large = 1.0x pot
            if bet_amount > 1.0 * pre_bet_pot_size:
                return 'large'

        # Normal categorization
        bet_ratio = bet_amount / pre_bet_pot_size

        if bet_ratio <= 0.49:       # ≤49% pot
            return 'small'
        elif bet_ratio <= 0.7:      # 50-70% pot
            return 'medium'
        else:                       # >70% pot
            return 'large'

    def pypoker_to_cfr_actions(self, pypoker_valid_actions, game_state):
        """
        Convert PyPokerEngine valid_actions to CFR action format
        """
        cfr_actions = []

        for action_info in pypoker_valid_actions:
            action_type = action_info['action']

            if action_type in ['fold', 'call', 'check']:
                cfr_actions.append(action_type)
            elif action_type in ['bet', 'raise']:
                # Determine what sizes are legal for this action type
                for size_name, multiplier in self.bet_sizes.items():
                    if self.is_legal_bet_size(game_state, multiplier):
                        cfr_actions.append(f"{action_type}_{size_name}")

        # Remove duplicates while preserving order
        return list(dict.fromkeys(cfr_actions))

    def cfr_to_pypoker_action(self, cfr_action, valid_actions, round_state, game_state):
        """
        Convert CFR action back to PyPokerEngine format
        Uses valid_actions to get exact amounts
        """
        if cfr_action == 'fold':
            return 'fold', 0

        elif cfr_action == 'check':
            return 'check', 0

        elif cfr_action == 'call':
            # Get call amount directly from valid_actions
            for action in valid_actions:
                if action['action'] == 'call':
                    return 'call', action['amount']
            return 'call', 0

        elif cfr_action.startswith('bet_') or cfr_action.startswith('raise_'):
            action_type = 'raise' if cfr_action.startswith('raise_') else 'bet'
            size_name = cfr_action.split('_')[1]

            # Find the bet/raise action in valid_actions
            for action in valid_actions:
                if action['action'] == action_type:
                    if isinstance(action.get('amount'), dict):
                        min_amt = action['amount']['min']
                        max_amt = action['amount']['max']

                        # Calculate target amount based on game state
                        target_amount = self._calculate_target_amount(
                            size_name, action_type, game_state, round_state
                        )

                        # Clamp to valid range
                        final_amount = max(
                            min_amt, min(target_amount, max_amt))
                        return action_type, int(final_amount)
                    else:
                        # Fixed amount
                        return action_type, action['amount']

            # Fallback: if can't bet/raise, try to call
            for action in valid_actions:
                if action['action'] == 'call':
                    return 'call', action['amount']
            return 'check', 0

        return 'check', 0

    def _calculate_target_amount(self, size_name, action_type, game_state, round_state):
        """
        Calculate target bet/raise amount following training structure
        """
        pot_size = game_state.get('pot_size', 3)
        current_bet = game_state.get('current_bet', 0)
        player_contrib = game_state.get('player_contribution', 0)
        big_blind = game_state.get('big_blind', 2)

        # Determine if we're preflop
        street = round_state.get('street', 'preflop')

        if street == 'preflop':
            # Count bet/raise actions to determine action type
            action_history = round_state.get(
                'action_histories', {}).get('preflop', [])
            bet_raise_count = sum(1 for a in action_history
                                  if a.get('action', '').upper() in ['BET', 'RAISE'])

            if bet_raise_count == 0:  # Open
                sizing = {'small': 6, 'medium': 10, 'large': 14}
                return sizing[size_name]
            elif bet_raise_count == 1:  # 3-bet
                sizing = {'small': 12, 'medium': 20, 'large': 28}
                return sizing[size_name]
            else:  # 4-bet+ (pot relative)
                multipliers = {'small': 0.66, 'medium': 1.33, 'large': 2.0}
                if action_type == 'raise':
                    to_call = current_bet - player_contrib
                    pot_after_call = pot_size + to_call
                    return (multipliers[size_name] * pot_after_call) + to_call
                else:
                    return multipliers[size_name] * pot_size
        else:
            # Postflop: standard pot-relative sizing
            multipliers = {'small': 0.33, 'medium': 0.66, 'large': 1.0}

            if action_type == 'raise':
                to_call = current_bet - player_contrib
                pot_after_call = pot_size + to_call
                return (multipliers[size_name] * pot_after_call) + to_call
            else:
                return multipliers[size_name] * pot_size
