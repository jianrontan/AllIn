class ActionAbstraction:
    """
    Like ['k', 'b', 'c', 'r', 'f'] but with bet sizing
    """

    def __init__(self):
        self.bet_sizes = {
            'tiny': 0.25,     # 1/4 pot
            'small': 0.33,    # 1/3 pot
            'medium': 0.66,   # 2/3 pot
            'large': 1.0,     # pot bet
            'overbet': 1.5,   # 1.5x pot
        }

    def is_legal_bet_size(self, game_state, multiplier):
        """
        Check if bet size is legal
        Based on search results[4]: min bet = max(big blind, current bet), max = stack
        """
        pot_size = game_state.get('pot_size', 0)
        player_stack = game_state.get('player_stack', 0)
        current_bet = game_state.get('current_bet', 0)
        big_blind = game_state.get('big_blind', 1)

        # Calculate actual bet amount
        if multiplier == 'stack':  # All-in case
            bet_amount = player_stack
        else:
            bet_amount = multiplier * pot_size

        # Minimum bet requirement (from search results[4])
        min_bet = max(big_blind, current_bet)

        # Check constraints
        if bet_amount < min_bet:
            return False
        if bet_amount > player_stack:
            return False

        return True

    def abstract_action_history(self, pypoker_actions, game_state):
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
                bet_category = self.categorize_bet_size(action, game_state)
                # 's', 'm', 'l', 'o' for small/med/large/over
                abstracted_history += bet_category[0]
            elif action['action'] == 'check':
                abstracted_history += 'k'

        return abstracted_history

    def categorize_bet_size(self, action, game_state):
        """
        Determine if bet is tiny/small/medium/large/overbet
        """
        bet_amount = action.get('amount', 0)
        pot_size = game_state.get('pot_size', 1)

        pre_bet_pot_size = pot_size - bet_amount

        if pre_bet_pot_size <= 0:
            pre_bet_pot_size = pot_size

        # Handle all-in case
        player_stack = game_state.get('player_stack', float('inf'))
        if bet_amount >= player_stack * 0.95:  # 95% of stack = all-in
            return 'allin'

        bet_ratio = bet_amount / pre_bet_pot_size

        # Categorize based on defined bet sizes
        if bet_ratio <= 0.29:       # ≤ 29% pot
            return 'tiny'
        elif bet_ratio <= 0.49:     # 30-49% pot
            return 'small'
        elif bet_ratio <= 0.7:      # 50-70% pot
            return 'medium'
        elif bet_ratio <= 1.1:      # 71-110% pot
            return 'large'
        else:                       # > 110% pot
            return 'overbet'

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

                # Add all-in option
                if self.is_legal_bet_size(game_state, 'stack'):
                    cfr_actions.append('all_in')

        # Remove duplicates while preserving order
        return list(dict.fromkeys(cfr_actions))

    def cfr_to_pypoker_action(self, cfr_action, game_state):
        """
        Convert CFR action back to PyPokerEngine format
        For use in declare_action() method
        """
        if cfr_action == 'fold':
            return 'fold', 0
        elif cfr_action == 'call':
            required_call = game_state.get('current_bet', 0) - game_state.get('player_contribution', 0)
            player_stack = game_state.get('player_stack', 0)
            call_amount = min(required_call, player_stack)
            return 'call', call_amount
        elif cfr_action == 'check':
            return 'check', 0
        elif cfr_action == 'all_in':
            return 'raise', game_state.get('player_stack', 0)
        elif cfr_action.startswith('bet_'):
            size_name = cfr_action.split('_')[1]
            multiplier = self.bet_sizes[size_name]
            bet_amount = multiplier * game_state.get('pot_size', 0)
            return 'bet', bet_amount
        elif cfr_action.startswith('raise_'):
            size_name = cfr_action.split('_')[1]
            multiplier = self.bet_sizes[size_name]
            bet_amount = multiplier * game_state.get('pot_size', 0)
            return 'raise', bet_amount
        else:
            return 'check', 0
