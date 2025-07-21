# backend/bot/src/subgame/confidence_detector.py
import numpy as np


class ConfidenceDetector:
    def __init__(self, blueprint_source, confidence_thresholds=None):
        self.blueprint_trainer = blueprint_source
        self.info_sets = blueprint_source.info_sets

        # Default thresholds
        default_thresholds = {
            'min_visit_count': 10,
            'min_visit_frequency': 0.01,  # 1% of total iterations
            'strategy_entropy_threshold': 0.7,  # 70% = very uniform
            'staleness_threshold': 0.2  # 20% of iterations
        }

        # Merge custom thresholds with defaults
        if confidence_thresholds:
            self.thresholds = {**default_thresholds, **confidence_thresholds}
        else:
            self.thresholds = default_thresholds

    def check_visit_frequency_confidence(self, info_set_key, total_iterations):
        """Check if info set has sufficient visit frequency"""
        info_set = self.info_sets.get(info_set_key)

        if not info_set:
            return False, "Unknown information set"

        # Check absolute visit count
        if info_set.visit_count < self.thresholds['min_visit_count']:
            return False, f"Low visit count: {info_set.visit_count}"

        # Check visit frequency relative to training
        visit_frequency = info_set.visit_count / max(1, total_iterations)
        if visit_frequency < self.thresholds['min_visit_frequency']:
            return False, f"Low visit frequency: {visit_frequency:.3f}"

        # Check staleness (not visited recently)
        staleness = total_iterations - info_set.last_visited_iteration
        if staleness > total_iterations * self.thresholds['staleness_threshold']:
            return False, f"Stale information set: {staleness} iterations ago"

        return True, "Sufficient visit frequency"

    def check_strategy_uniformity_confidence(self, info_set_key):
        """Check if strategy is too uniform (indicates low confidence)"""
        info_set = self.info_sets.get(info_set_key)

        if not info_set or not info_set.legal_actions:
            return False, "No strategy data available"

        try:
            # Get average strategy
            avg_strategy = info_set.get_average_strategy(
                info_set.legal_actions)

            # Calculate Shannon entropy
            entropy = -np.sum(avg_strategy * np.log2(avg_strategy + 1e-10))
            max_entropy = np.log2(len(info_set.legal_actions))

            # Normalize to 0-1 scale (0 = decisive, 1 = uniform)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

            # High entropy indicates uniform strategy = low confidence
            if normalized_entropy > self.thresholds['strategy_entropy_threshold']:
                return False, f"Strategy too uniform: {normalized_entropy:.3f}"

            return True, f"Strategy sufficiently decisive: {normalized_entropy:.3f}"

        except Exception as e:
            return False, f"Error calculating strategy uniformity: {e}"
