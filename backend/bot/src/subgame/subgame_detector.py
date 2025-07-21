# backend/bot/src/subgame/subgame_detector.py
from .confidence_detector import ConfidenceDetector
from .off_tree_detector import OffTreeDetector


class SubgameDetector:
    def __init__(self, blueprint_source, confidence_thresholds=None):
        self.confidence_detector = ConfidenceDetector(
            blueprint_source, confidence_thresholds)
        self.off_tree_detector = OffTreeDetector()
        self.blueprint_source = blueprint_source

        # Additional thresholds for subgame triggering
        self.subgame_thresholds = {
            'high_stakes_pot_ratio': 10,  # Trigger for pots > 10x big blind
            'deep_stack_spr': 15,         # Stack-to-pot ratio > 15
            'complex_betting_threshold': 3  # 3+ bet/raise actions
        }

    def should_trigger_subgame_solving(self, info_set_key, game_history, total_iterations,
                                       game_state=None, pot_sizes_history=None):
        """Comprehensive check for subgame solving triggers"""

        trigger_reasons = []

        # 1. Check visit frequency
        visit_ok, visit_reason = self.confidence_detector.check_visit_frequency_confidence(
            info_set_key, total_iterations)
        if not visit_ok:
            trigger_reasons.append(f"Visit frequency: {visit_reason}")

        # 2. Check strategy uniformity
        strategy_ok, strategy_reason = self.confidence_detector.check_strategy_uniformity_confidence(
            info_set_key)
        if not strategy_ok:
            trigger_reasons.append(f"Strategy uniformity: {strategy_reason}")

        # 3. Check off-tree situation
        is_off_tree, off_tree_reasons = self.off_tree_detector.is_off_tree_situation(
            game_history, pot_sizes_history)
        if is_off_tree:
            trigger_reasons.extend(
                [f"Off-tree: {reason}" for reason in off_tree_reasons])

        # 4. Check high-stakes situations
        if game_state:
            high_stakes_reasons = self._check_high_stakes_situations(
                game_state, game_history)
            if high_stakes_reasons:
                trigger_reasons.extend(high_stakes_reasons)

        # Return trigger decision and reasons
        should_trigger = len(trigger_reasons) > 0
        return should_trigger, trigger_reasons

    def _check_high_stakes_situations(self, game_state, game_history):
        """Check for high-stakes situations requiring precision"""
        reasons = []

        pot_size = game_state.get('pot_size', 0)
        player_stack = game_state.get('player_stack', 0)
        big_blind = game_state.get('big_blind', 2)

        # High pot relative to blinds
        pot_bb_ratio = pot_size / big_blind if big_blind > 0 else 0
        if pot_bb_ratio > self.subgame_thresholds['high_stakes_pot_ratio']:
            reasons.append(f"High stakes pot: {pot_bb_ratio:.1f}x big blind")

        # Deep stack situation
        if pot_size > 0:
            spr = player_stack / pot_size
            if spr > self.subgame_thresholds['deep_stack_spr']:
                reasons.append(f"Deep stack situation: SPR {spr:.1f}")

        # Complex betting pattern
        bet_raise_count = sum(1 for action in game_history
                              if action.startswith(('bet_', 'raise_')))
        if bet_raise_count >= self.subgame_thresholds['complex_betting_threshold']:
            reasons.append(
                f"Complex betting: {bet_raise_count} bet/raise actions")

        return reasons
