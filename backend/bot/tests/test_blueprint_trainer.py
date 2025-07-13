from src.cfr.blueprint_trainer import BlueprintTrainer
import sys
import os
import json
from pathlib import Path

# Add the parent directory to the path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def save_normalized_strategies(trainer, filename="normalized_strategies.json"):
    """Save normalized average strategies to JSON file using stored legal_actions"""

    analysis_data = {
        "total_info_sets": len(trainer.info_sets),
        "training_summary": {
            "info_sets_by_street": {"preflop": 0, "flop": 0, "turn": 0, "river": 0},
            "hand_bucket_distribution": {},
            "action_pattern_distribution": {}
        },
        "normalized_strategies": {}
    }

    # Process each information set
    for key, info_set in trainer.info_sets.items():

        # Use stored legal_actions from the info set (much cleaner!)
        if not info_set.legal_actions:
            print(
                f"Warning: Info set '{key}' has no stored legal actions, skipping...")
            continue

        # Get normalized average strategy using stored actions
        avg_strategy = info_set.get_average_strategy(info_set.legal_actions)
        normalized_strategy = {action: float(prob) for action, prob in zip(
            info_set.legal_actions, avg_strategy)}

        # Store the normalized strategy
        analysis_data["normalized_strategies"][key] = {
            # Store what actions this info set represents
            "legal_actions": info_set.legal_actions,
            "regrets": {action: float(regret) for action, regret in info_set.cumulative_regrets.items()},
            "average_strategy": normalized_strategy,
            "key_parts": key.split('_')
        }

        # Update summary statistics
        update_summary_stats(analysis_data, key)

    # Create analysis directory and save
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)

    filepath = analysis_dir / filename
    with open(filepath, 'w') as f:
        json.dump(analysis_data, f, indent=2)

    print(f"\n📊 NORMALIZED STRATEGIES SAVED TO: {filepath}")
    print(f"Total info sets: {len(trainer.info_sets)}")
    print(
        f"Info sets with strategies: {len(analysis_data['normalized_strategies'])}")
    print(f"By street: Preflop={analysis_data['training_summary']['info_sets_by_street']['preflop']}, "
          f"Flop={analysis_data['training_summary']['info_sets_by_street']['flop']}, "
          f"Turn={analysis_data['training_summary']['info_sets_by_street']['turn']}, "
          f"River={analysis_data['training_summary']['info_sets_by_street']['river']}")

    return analysis_data


def update_summary_stats(analysis_data, key):
    """Update summary statistics for the analysis"""

    # Count by street
    if '_flop_' in key:
        analysis_data["training_summary"]["info_sets_by_street"]["flop"] += 1
    elif '_turn_' in key:
        analysis_data["training_summary"]["info_sets_by_street"]["turn"] += 1
    elif '_river_' in key:
        analysis_data["training_summary"]["info_sets_by_street"]["river"] += 1
    else:
        analysis_data["training_summary"]["info_sets_by_street"]["preflop"] += 1

    # Count hand buckets
    parts = key.split('_')
    if parts:
        hand_bucket = parts[0]
        analysis_data["training_summary"]["hand_bucket_distribution"][hand_bucket] = \
            analysis_data["training_summary"]["hand_bucket_distribution"].get(
                hand_bucket, 0) + 1

    # Count action patterns
    if len(parts) > 1:
        action_pattern = parts[-1]
        analysis_data["training_summary"]["action_pattern_distribution"][action_pattern] = \
            analysis_data["training_summary"]["action_pattern_distribution"].get(
                action_pattern, 0) + 1


def test_blueprint_trainer_basic(iterations):
    """Test basic blueprint trainer functionality"""
    print("=" * 80)
    print("TESTING BLUEPRINT TRAINER - BASIC FUNCTIONALITY")
    print("=" * 80)

    trainer = BlueprintTrainer()

    print(f"Starting CFR training for {iterations} iterations...")
    expected_value = trainer.train_blueprint(iterations)

    print(f"\n✅ Training completed successfully!")
    print(f"Expected value: {expected_value:.6f}")
    print(f"Total info sets created: {len(trainer.info_sets)}")

    # Check if info sets have stored legal actions
    info_sets_with_actions = sum(
        1 for info_set in trainer.info_sets.values() if info_set.legal_actions)
    print(f"Info sets with stored legal actions: {info_sets_with_actions}")

    # Save normalized strategies
    analysis_data = save_normalized_strategies(
        trainer, "blueprint_trainer_basic.json")

    # Display sample strategies
    print(f"\n📋 SAMPLE NORMALIZED STRATEGIES:")
    print("-" * 80)

    sample_count = 0
    for key, data in analysis_data["normalized_strategies"].items():
        if sample_count >= 5:  # Show first 5
            break

        print(f"Info Set: '{key}'")
        print(f"  Legal Actions: {data['legal_actions']}")
        strategy = data["average_strategy"]
        regrets = data["regrets"]

        # Show probabilities as percentages
        strategy_percentages = {
            action: f"{prob*100:.1f}%" for action, prob in strategy.items()}
        print(f"  Strategy: {strategy_percentages}")
        print(f"  Regrets:  {regrets}")

        # Verify probabilities sum to 1.0
        total_prob = sum(strategy.values())
        print(f"  Strategy sum: {total_prob:.3f} (should be 1.000)")
        print()

        sample_count += 1

    return trainer, analysis_data


def test_blueprint_trainer_extended(iterations):
    """Test blueprint trainer with more iterations"""
    print("=" * 80)
    print("TESTING BLUEPRINT TRAINER - EXTENDED TRAINING")
    print("=" * 80)

    trainer = BlueprintTrainer()

    print(f"Starting extended CFR training for {iterations} iterations...")
    expected_value = trainer.train_blueprint(iterations)

    print(f"\n✅ Extended training completed!")
    print(f"Expected value: {expected_value:.6f}")
    print(f"Total info sets created: {len(trainer.info_sets)}")

    # Save normalized strategies
    analysis_data = save_normalized_strategies(
        trainer, "blueprint_trainer_extended.json")

    # Analyze strategy quality
    analyze_strategy_quality(analysis_data)

    return trainer, analysis_data


def analyze_strategy_quality(analysis_data):
    """Analyze the quality of learned strategies"""
    print(f"\n🔍 STRATEGY QUALITY ANALYSIS:")
    print("-" * 80)

    total_info_sets = len(analysis_data["normalized_strategies"])
    mixed_strategies = 0
    pure_strategies = 0
    strategy_errors = 0

    for key, data in analysis_data["normalized_strategies"].items():
        strategy = data["average_strategy"]

        # Verify strategy sums to 1.0
        total_prob = sum(strategy.values())
        if abs(total_prob - 1.0) > 0.001:  # Allow small floating point errors
            strategy_errors += 1
            print(
                f"  Warning: Strategy for '{key}' sums to {total_prob:.3f}, not 1.0")

        # Count how many actions have significant probability
        significant_actions = sum(
            1 for prob in strategy.values() if prob > 0.05)  # > 5%

        if significant_actions == 1:
            pure_strategies += 1
        else:
            mixed_strategies += 1

    print(f"Total strategies analyzed: {total_info_sets}")
    print(
        f"Pure strategies (single action): {pure_strategies} ({pure_strategies/total_info_sets*100:.1f}%)")
    print(
        f"Mixed strategies (multiple actions): {mixed_strategies} ({mixed_strategies/total_info_sets*100:.1f}%)")
    print(f"Strategy normalization errors: {strategy_errors}")

    # Show most common action patterns
    action_patterns = analysis_data["training_summary"]["action_pattern_distribution"]
    sorted_patterns = sorted(action_patterns.items(),
                             key=lambda x: x[1], reverse=True)

    print(f"\nMost common action patterns:")
    for pattern, count in sorted_patterns[:10]:
        pattern_display = f"'{pattern}'" if pattern else "'(first to act)'"
        print(f"  {pattern_display}: {count} info sets")

    # Show action diversity
    print(f"\nAction diversity analysis:")
    all_actions = set()
    for data in analysis_data["normalized_strategies"].values():
        all_actions.update(data["legal_actions"])

    print(f"  Unique actions seen: {sorted(all_actions)}")
    print(f"  Total unique actions: {len(all_actions)}")


def test_action_storage():
    """Test that legal actions are being stored correctly"""
    print("=" * 80)
    print("TESTING LEGAL ACTION STORAGE")
    print("=" * 80)

    trainer = BlueprintTrainer()

    # Run minimal training
    trainer.train_blueprint(5)

    print(f"Total info sets: {len(trainer.info_sets)}")

    # Check each info set
    for key, info_set in trainer.info_sets.items():
        print(f"Info set '{key}':")
        print(f"  Stored legal actions: {info_set.legal_actions}")
        print(
            f"  Cumulative regrets: {list(info_set.cumulative_regrets.keys())}")
        print(
            f"  Cumulative strategy: {list(info_set.cumulative_strategy.keys())}")

        # Verify consistency
        if info_set.legal_actions:
            regret_actions = set(info_set.cumulative_regrets.keys())
            strategy_actions = set(info_set.cumulative_strategy.keys())
            stored_actions = set(info_set.legal_actions)

            if regret_actions != stored_actions:
                print(
                    f"  ⚠️ Warning: Regret actions {regret_actions} != stored actions {stored_actions}")
            if strategy_actions != stored_actions:
                print(
                    f"  ⚠️ Warning: Strategy actions {strategy_actions} != stored actions {stored_actions}")
            if regret_actions == strategy_actions == stored_actions:
                print(f"  ✅ All action sets match")
        else:
            print(f"  ❌ No legal actions stored!")

        print()


def run_all_tests():
    """Run all blueprint trainer tests"""
    try:
        print("🚀 STARTING BLUEPRINT TRAINER TESTS")
        print("=" * 80)

        # Test 1: Legal action storage
        # test_action_storage()

        # Test 2: Basic functionality
        # trainer_basic, analysis_basic = test_blueprint_trainer_basic(10)

        # Test 3: Extended training
        trainer_extended, analysis_extended = test_blueprint_trainer_extended(50)

        print("\n🎉 ALL BLUEPRINT TRAINER TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        # print("Check the 'analysis' folder for saved strategy files:")
        # print("  - blueprint_trainer_basic.json (10 iterations)")
        print("  - blueprint_trainer_extended.json (50 iterations)")

    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
