# backend/bot/src/subgame/off_tree_detector.py
class OffTreeDetector:
    def __init__(self):
        # Blueprint action space
        self.blueprint_actions = {
            'check', 'call', 'fold',
            'bet_small', 'bet_medium', 'bet_large',
            'raise_small', 'raise_medium', 'raise_large'
        }

        # Blueprint bet size ranges (from poker_game.py)
        self.blueprint_bet_sizes = {
            'small': 0.33,   # 33% pot
            'medium': 0.66,  # 66% pot
            'large': 1.0     # 100% pot
        }

        # Define off-tree conditions
        self.off_tree_conditions = {
            'bet_size_tolerance': 0.15,  # 15% tolerance around expected sizes
            'max_raises': 2,
            'unknown_actions': True
        }

    def is_off_tree_situation(self, game_history, pot_sizes_history=None):
        """Detect various off-tree scenarios"""
        off_tree_reasons = []

        # 1. Check for unknown actions
        unknown_actions = self._check_unknown_actions(game_history)
        if unknown_actions:
            off_tree_reasons.extend(unknown_actions)

        # 2. Check betting cap violations (4th raise)
        raise_violations = self._check_raise_limit_violations(game_history)
        if raise_violations:
            off_tree_reasons.extend(raise_violations)

        # 3. Check bet sizing anomalies
        if pot_sizes_history:
            sizing_violations = self._check_bet_sizing_violations(
                game_history, pot_sizes_history)
            if sizing_violations:
                off_tree_reasons.extend(sizing_violations)

        return len(off_tree_reasons) > 0, off_tree_reasons

    def _check_unknown_actions(self, game_history):
        """Check for actions not in blueprint abstraction"""
        unknown_actions = []

        for action in game_history:
            if action not in self.blueprint_actions:
                unknown_actions.append(f"Unknown action: {action}")

        return unknown_actions

    def _check_raise_limit_violations(self, game_history):
        """Check for raises beyond the 3-total cap (1 bet + 2 raises)"""
        violations = []

        # Count raises per street (assuming single street for now)
        bet_raise_count = sum(1 for action in game_history
                              if action.startswith(('bet_', 'raise_')))

        if bet_raise_count > 3:  # 1 bet + 2 raises = 3 max
            violations.append(
                f"Excessive betting: {bet_raise_count} bet/raise actions")

        return violations

    def _check_bet_sizing_violations(self, game_history, bet_amounts_data):
        """Check for bet sizes outside expected ranges using actual amounts"""
        violations = []

        for i, action in enumerate(game_history):
            if action.startswith(('bet_', 'raise_')):
                if i < len(bet_amounts_data):
                    bet_data = bet_amounts_data[i]
                    violation = self._analyze_bet_size_deviation(
                        action,
                        bet_data['pot_size_at_time'],
                        bet_data['amount']
                    )
                    if violation:
                        violations.append(violation)

        return violations

    def _analyze_bet_size_deviation(self, action, pot_size, actual_amount):
        """Analyze if bet size is significantly outside blueprint ranges"""

        # Blueprint expected ratios
        expected_ratios = {
            'small': 0.33,
            'medium': 0.66,
            'large': 1.0
        }

        # Extract size category from action
        size_category = None
        for category in ['small', 'medium', 'large']:
            if category in action:
                size_category = category
                break

        if not size_category:
            return f"Unknown bet size category in action: {action}"

        # Calculate expected amount
        expected_ratio = expected_ratios[size_category]
        expected_amount = expected_ratio * pot_size

        # Calculate deviation percentage
        if expected_amount > 0:
            deviation = abs(actual_amount - expected_amount) / expected_amount

            # Check if deviation exceeds thresholds
            if deviation > 0.25:  # 25% threshold
                return (f"Bet size deviation: {action} expected {expected_amount:.2f}, "
                        f"actual {actual_amount:.2f} ({deviation*100:.1f}% deviation)")

        return None

    def get_actual_bet_amount_from_game_state(self, action, game_state):
        """Extract actual bet amount from game state (to be implemented based on game logic)"""
        # This would integrate with poker_game.py logic
        # For now, return expected amount as placeholder
        pot_size = game_state.get('pot_size', 0)

        if 'small' in action:
            return 0.33 * pot_size
        elif 'medium' in action:
            return 0.66 * pot_size
        elif 'large' in action:
            return 1.0 * pot_size
        else:
            return 0
