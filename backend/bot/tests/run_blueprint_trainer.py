# backend/bot/tests/test_confidence_detection.py
from src.cfr.blueprint_trainer import BlueprintTrainer
import sys
import os
import json
from pathlib import Path
import time
import psutil
import traceback
from datetime import datetime

# Add the parent directory to the path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class EnhancedBlueprintTrainer:
    """Enhanced Blueprint Trainer with comprehensive statistics and unified output"""

    def __init__(self, output_dir="analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.start_time = None
        self.end_time = None
        self.process = psutil.Process()

    def train_and_analyze(self, iterations, output_filename=None):
        """
        Complete training pipeline with unified statistics output

        Args:
            iterations: Number of CFR iterations to run
            output_filename: Optional custom filename for output

        Returns:
            dict: Comprehensive training results and statistics
        """
        print("🚀 ENHANCED BLUEPRINT TRAINER - UNIFIED STATISTICS")
        print("=" * 80)

        # Initialize trainer and tracking
        trainer = BlueprintTrainer()
        self.start_time = datetime.now()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        print(f"Starting CFR training for {iterations} iterations...")
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Initial memory usage: {start_memory:.1f} MB")

        # Run training with time tracking
        training_start = time.time()
        expected_value = trainer.train_blueprint(iterations)
        training_end = time.time()

        self.end_time = datetime.now()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # Generate unified statistics
        unified_stats = self._generate_unified_statistics(
            trainer, iterations, expected_value,
            training_end - training_start, start_memory, end_memory
        )

        # Save unified output
        if output_filename is None:
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"blueprint_unified_{iterations}iter_{timestamp}.json"

        output_path = self.output_dir / output_filename
        with open(output_path, 'w') as f:
            json.dump(unified_stats, f, indent=2)

        # Display comprehensive summary
        self._display_training_summary(unified_stats)

        print(f"\n📄 UNIFIED STATISTICS SAVED TO: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

        return unified_stats

    def _generate_unified_statistics(self, trainer, iterations, expected_value,
                                     training_duration, start_memory, end_memory):
        """Generate comprehensive unified statistics"""

        # Base structure with enhanced metadata
        unified_stats = {
            "training_metadata": {
                "iterations": iterations,
                "expected_value": expected_value,
                "training_duration_seconds": training_duration,
                "iterations_per_second": iterations / training_duration if training_duration > 0 else 0,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "memory_usage": {
                    "start_mb": start_memory,
                    "end_mb": end_memory,
                    "delta_mb": end_memory - start_memory
                },
                "total_info_sets": len(trainer.info_sets),
                "export_timestamp": datetime.now().isoformat()
            },
            "visit_statistics": self._calculate_visit_statistics(trainer),
            "strategy_analysis": self._analyze_strategy_quality(trainer),
            "convergence_metrics": self._calculate_convergence_metrics(trainer, expected_value),
            "street_distribution": self._analyze_street_distribution(trainer),
            "hand_bucket_analysis": self._analyze_hand_buckets(trainer),
            "action_pattern_analysis": self._analyze_action_patterns(trainer),
            "normalized_strategies": self._extract_normalized_strategies(trainer)
        }

        return unified_stats

    def _calculate_visit_statistics(self, trainer):
        """Calculate comprehensive visit statistics"""
        visit_counts = []
        total_visits = 0
        visit_frequencies = []

        for info_set_key, info_set in trainer.info_sets.items():
            visit_counts.append(info_set.visit_count)
            total_visits += info_set.visit_count

            # Calculate visit frequency
            if info_set.last_visited_iteration >= 0:
                visit_frequency = info_set.visit_count / \
                    max(1, info_set.last_visited_iteration + 1)
            else:
                visit_frequency = 0
            visit_frequencies.append(visit_frequency)

        if visit_counts:
            visit_counts.sort(reverse=True)
            visit_frequencies.sort(reverse=True)

            # Calculate percentiles
            def percentile(data, p):
                if not data:
                    return 0
                k = (len(data) - 1) * p / 100
                f = int(k)
                c = float(k) - f  # Ensure k is treated as float
                if f == len(data) - 1:
                    return data[f]
                return data[f] * (1 - c) + data[f + 1] * c

            return {
                "total_visits": total_visits,
                "total_info_sets": len(trainer.info_sets),
                "average_visits_per_infoset": total_visits / len(trainer.info_sets),
                "visit_distribution": {
                    "max_visits": max(visit_counts),
                    "min_visits": min(visit_counts),
                    "median_visits": percentile(visit_counts, 50),
                    "p75_visits": percentile(visit_counts, 75),
                    "p90_visits": percentile(visit_counts, 90),
                    "p95_visits": percentile(visit_counts, 95)
                },
                "visit_frequency_distribution": {
                    "max_frequency": max(visit_frequencies),
                    "min_frequency": min(visit_frequencies),
                    "median_frequency": percentile(visit_frequencies, 50),
                    "average_frequency": sum(visit_frequencies) / len(visit_frequencies)
                },
                "zero_visit_count": sum(1 for count in visit_counts if count == 0),
                "low_visit_count": sum(1 for count in visit_counts if 0 < count < 10),
                "high_visit_count": sum(1 for count in visit_counts if count >= 100)
            }

        return {"total_visits": 0, "total_info_sets": 0}

    def _analyze_strategy_quality(self, trainer):
        """Analyze strategy quality with enhanced metrics"""
        total_info_sets = len(trainer.info_sets)
        pure_strategies = 0
        mixed_strategies = 0
        uniform_strategies = 0
        strategy_errors = 0
        entropy_scores = []

        for info_set_key, info_set in trainer.info_sets.items():
            try:
                if not info_set.legal_actions:
                    continue

                avg_strategy = info_set.get_average_strategy(
                    info_set.legal_actions)

                # Verify strategy normalization
                total_prob = sum(avg_strategy)
                if abs(total_prob - 1.0) > 0.001:
                    strategy_errors += 1

                # Calculate entropy for strategy analysis
                import numpy as np
                # Ensure float type
                strategy_array = np.array(avg_strategy, dtype=np.float64)
                # Ensure we have valid probabilities
                strategy_array = np.clip(strategy_array, 1e-10, 1.0)
                entropy = -np.sum(strategy_array * np.log2(strategy_array))
                max_entropy = np.log2(len(avg_strategy))

                if max_entropy > 0:
                    normalized_entropy = float(entropy / max_entropy)
                else:
                    normalized_entropy = 0.0

                entropy_scores.append(normalized_entropy)

                # Classify strategy types
                significant_actions = sum(
                    1 for prob in avg_strategy if prob > 0.05)

                if significant_actions == 1:
                    pure_strategies += 1
                elif normalized_entropy > 0.9:  # Very uniform
                    uniform_strategies += 1
                else:
                    mixed_strategies += 1

            except Exception as e:
                strategy_errors += 1
                print(
                    f"Warning: Error analyzing strategy for {info_set_key}: {e}")

        # Calculate entropy statistics
        if entropy_scores:
            entropy_scores.sort()
            avg_entropy = sum(entropy_scores) / len(entropy_scores)
        else:
            avg_entropy = 0

        return {
            "total_strategies_analyzed": total_info_sets,
            "pure_strategies": {
                "count": pure_strategies,
                "percentage": (pure_strategies / total_info_sets * 100) if total_info_sets > 0 else 0
            },
            "mixed_strategies": {
                "count": mixed_strategies,
                "percentage": (mixed_strategies / total_info_sets * 100) if total_info_sets > 0 else 0
            },
            "uniform_strategies": {
                "count": uniform_strategies,
                "percentage": (uniform_strategies / total_info_sets * 100) if total_info_sets > 0 else 0
            },
            "strategy_errors": strategy_errors,
            "entropy_analysis": {
                "average_entropy": avg_entropy,
                "max_entropy": max(entropy_scores) if entropy_scores else 0,
                "min_entropy": min(entropy_scores) if entropy_scores else 0,
                "entropy_distribution": {
                    "decisive_strategies": sum(1 for e in entropy_scores if e < 0.3),
                    "mixed_strategies": sum(1 for e in entropy_scores if 0.3 <= e < 0.7),
                    "uniform_strategies": sum(1 for e in entropy_scores if e >= 0.7)
                }
            }
        }

    def _calculate_convergence_metrics(self, trainer, expected_value):
        """Calculate convergence-related metrics"""
        total_regret = 0.0
        positive_regrets = 0
        negative_regrets = 0
        max_regret = float('-inf')
        min_regret = float('inf')

        for info_set in trainer.info_sets.values():
            for action, regret in info_set.cumulative_regrets.items():
                regret_value = float(regret)  # Ensure float type
                total_regret += abs(regret_value)
                if regret_value > 0:
                    positive_regrets += 1
                else:
                    negative_regrets += 1
                max_regret = max(max_regret, regret_value)
                min_regret = min(min_regret, regret_value)

        total_regret_entries = sum(len(info_set.cumulative_regrets)
                                   for info_set in trainer.info_sets.values())

        return {
            "expected_value": expected_value,
            "total_cumulative_regret": total_regret,
            "average_regret_magnitude": total_regret / total_regret_entries if total_regret_entries > 0 else 0,
            "regret_distribution": {
                "positive_regrets": positive_regrets,
                "negative_regrets": negative_regrets,
                "max_regret": max_regret if max_regret != float('-inf') else 0.0,
                "min_regret": min_regret if min_regret != float('inf') else 0.0
            },
            "convergence_indicators": {
                "regret_balance": positive_regrets / (positive_regrets + negative_regrets) if (positive_regrets + negative_regrets) > 0 else 0,
                "exploitability_proxy": abs(expected_value)
            }
        }

    def _analyze_street_distribution(self, trainer):
        """Analyze information set distribution by street"""
        street_counts = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}

        for info_set_key in trainer.info_sets.keys():
            if '_flop_' in info_set_key:
                street_counts["flop"] += 1
            elif '_turn_' in info_set_key:
                street_counts["turn"] += 1
            elif '_river_' in info_set_key:
                street_counts["river"] += 1
            else:
                street_counts["preflop"] += 1

        total = sum(street_counts.values())
        return {
            "counts": street_counts,
            "percentages": {street: (count / total * 100) if total > 0 else 0
                            for street, count in street_counts.items()}
        }

    def _analyze_hand_buckets(self, trainer):
        """Analyze hand bucket distribution"""
        hand_buckets = {}

        for info_set_key in trainer.info_sets.keys():
            parts = info_set_key.split('_')
            if parts:
                hand_bucket = parts[0]
                hand_buckets[hand_bucket] = hand_buckets.get(
                    hand_bucket, 0) + 1

        # Sort by frequency
        sorted_buckets = sorted(hand_buckets.items(),
                                key=lambda x: x[1], reverse=True)

        return {
            "distribution": dict(sorted_buckets),
            "most_common": sorted_buckets[:5] if sorted_buckets else [],
            "unique_buckets": len(hand_buckets)
        }

    def _analyze_action_patterns(self, trainer):
        """Analyze action pattern distribution"""
        action_patterns = {}
        all_actions = set()

        for info_set_key, info_set in trainer.info_sets.items():
            # Extract action pattern from key
            parts = info_set_key.split('_')
            if len(parts) > 1:
                action_pattern = parts[-1]
            else:
                action_pattern = "(first_to_act)"

            action_patterns[action_pattern] = action_patterns.get(
                action_pattern, 0) + 1

            # Collect all unique actions
            all_actions.update(info_set.legal_actions)

        # Sort by frequency
        sorted_patterns = sorted(
            action_patterns.items(), key=lambda x: x[1], reverse=True)

        return {
            "pattern_distribution": dict(sorted_patterns),
            "most_common_patterns": sorted_patterns[:10],
            "unique_patterns": len(action_patterns),
            "action_diversity": {
                "unique_actions": sorted(list(all_actions)),
                "total_unique_actions": len(all_actions)
            }
        }

    def _extract_normalized_strategies(self, trainer):
        """Extract normalized strategies with enhanced metadata"""
        normalized_strategies = {}

        for info_set_key, info_set in trainer.info_sets.items():
            try:
                if not info_set.legal_actions:
                    continue

                avg_strategy = info_set.get_average_strategy(
                    info_set.legal_actions)
                normalized_strategy = {action: float(prob) for action, prob in
                                       zip(info_set.legal_actions, avg_strategy)}

                # Enhanced strategy metadata with proper error handling
                if normalized_strategy:  # Check if dictionary is not empty
                    strategy_metadata = {
                        "dominant_action": max(normalized_strategy.items(), key=lambda x: x[1])[0],
                        "max_probability": max(normalized_strategy.values()),
                        "min_probability": min(normalized_strategy.values()),
                        "strategy_sum": sum(normalized_strategy.values())
                    }
                else:
                    # Fallback for empty strategy
                    strategy_metadata = {
                        "dominant_action": "none",
                        "max_probability": 0.0,
                        "min_probability": 0.0,
                        "strategy_sum": 0.0
                    }

                normalized_strategies[info_set_key] = {
                    "legal_actions": info_set.legal_actions,
                    "average_strategy": normalized_strategy,
                    "regrets": {action: float(regret) for action, regret in
                                info_set.cumulative_regrets.items()},
                    "visit_metadata": {
                        "visit_count": info_set.visit_count,
                        "last_visited_iteration": info_set.last_visited_iteration,
                        "visit_frequency": info_set.visit_count / max(1, info_set.last_visited_iteration + 1)
                        if info_set.last_visited_iteration >= 0 else 0
                    },
                    "strategy_metadata": strategy_metadata,
                    "key_parts": info_set_key.split('_')
                }

            except Exception as e:
                print(
                    f"Warning: Error processing strategy for {info_set_key}: {e}")

        return normalized_strategies

    def _display_training_summary(self, unified_stats):
        """Display comprehensive training summary"""
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE TRAINING SUMMARY")
        print("=" * 80)

        # Training metadata
        metadata = unified_stats["training_metadata"]
        print(
            f"Training Duration: {metadata['training_duration_seconds']:.1f} seconds")
        print(
            f"Iterations per Second: {metadata['iterations_per_second']:.2f}")
        print(
            f"Memory Usage: {metadata['memory_usage']['start_mb']:.1f} MB → {metadata['memory_usage']['end_mb']:.1f} MB (Δ{metadata['memory_usage']['delta_mb']:.1f} MB)")
        print(f"Expected Value: {metadata['expected_value']:.6f}")
        print(f"Total Information Sets: {metadata['total_info_sets']}")

        # Visit statistics
        visit_stats = unified_stats["visit_statistics"]
        print(f"\n📊 VISIT STATISTICS:")
        print(f"Total Visits: {visit_stats['total_visits']}")
        print(
            f"Average Visits per Info Set: {visit_stats['average_visits_per_infoset']:.2f}")
        print(f"Visit Distribution: Min={visit_stats['visit_distribution']['min_visits']}, "
              f"Median={visit_stats['visit_distribution']['median_visits']:.1f}, "
              f"Max={visit_stats['visit_distribution']['max_visits']}")

        # Strategy analysis
        strategy_analysis = unified_stats["strategy_analysis"]
        print(f"\n🧠 STRATEGY ANALYSIS:")
        print(
            f"Pure Strategies: {strategy_analysis['pure_strategies']['count']} ({strategy_analysis['pure_strategies']['percentage']:.1f}%)")
        print(
            f"Mixed Strategies: {strategy_analysis['mixed_strategies']['count']} ({strategy_analysis['mixed_strategies']['percentage']:.1f}%)")
        print(
            f"Uniform Strategies: {strategy_analysis['uniform_strategies']['count']} ({strategy_analysis['uniform_strategies']['percentage']:.1f}%)")
        print(
            f"Average Entropy: {strategy_analysis['entropy_analysis']['average_entropy']:.3f}")

        # Street distribution
        street_dist = unified_stats["street_distribution"]
        print(f"\n🃏 STREET DISTRIBUTION:")
        for street, count in street_dist["counts"].items():
            percentage = street_dist["percentages"][street]
            print(f"  {street.capitalize()}: {count} ({percentage:.1f}%)")

        # Hand bucket analysis
        hand_analysis = unified_stats["hand_bucket_analysis"]
        print(f"\n🎯 TOP HAND BUCKETS:")
        for bucket, count in hand_analysis["most_common"][:5]:
            print(f"  {bucket}: {count} info sets")

        # Action patterns
        action_analysis = unified_stats["action_pattern_analysis"]
        print(f"\n🎮 TOP ACTION PATTERNS:")
        for pattern, count in action_analysis["most_common_patterns"][:5]:
            display_pattern = f"'{pattern}'" if pattern else "'(first_to_act)'"
            print(f"  {display_pattern}: {count} info sets")

        print(
            f"\nUnique Actions: {action_analysis['action_diversity']['unique_actions']}")


def run_enhanced_training(iterations=1000):
    """Run enhanced training with specified iterations"""
    try:
        trainer = EnhancedBlueprintTrainer()
        results = trainer.train_and_analyze(iterations)

        print(f"\n🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
        print(
            f"Results saved with {len(results['normalized_strategies'])} strategies")

        return results

    except Exception as e:
        print(f"❌ ENHANCED TRAINING FAILED: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run with different iteration counts as needed
    # run_enhanced_training(100)     # Quick test
    run_enhanced_training(10)  # Standard training
    # run_enhanced_training(3000)  # Extended training
