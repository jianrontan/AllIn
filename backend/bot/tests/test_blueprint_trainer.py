# backend/bot/tests/test_blueprint_trainer.py
from src.cfr.blueprint_trainer import BlueprintTrainer
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_blueprint_trainer_basic(iterations):
    print("=" * 80)
    print("TESTING BLUEPRINT TRAINER - BASIC FUNCTIONALITY")
    print("=" * 80)

    trainer = BlueprintTrainer()
    print(f"Starting CFR training for {iterations} iterations...")
    expected_value = trainer.train_blueprint(iterations)

    print(f"\nTraining completed!")
    print(f"Expected value: {expected_value:.6f}")
    print(f"Total info sets: {len(trainer.info_sets)}")

    info_sets_with_actions = sum(
        1 for info_set in trainer.info_sets.values() if info_set.legal_actions)
    print(f"Info sets with stored legal actions: {info_sets_with_actions}")

    print(f"\nSAMPLE STRATEGIES:")
    print("-" * 80)
    for key, info_set in list(trainer.info_sets.items())[:5]:
        if not info_set.legal_actions:
            continue
        avg_strategy = info_set.get_average_strategy(info_set.legal_actions)
        strategy_pct = {a: f"{p*100:.1f}%" for a, p in zip(info_set.legal_actions, avg_strategy)}
        total_prob = sum(avg_strategy)
        print(f"'{key}': {strategy_pct}  (sum={total_prob:.3f})")

    return trainer


def test_blueprint_trainer_extended(iterations):
    print("=" * 80)
    print("TESTING BLUEPRINT TRAINER - EXTENDED TRAINING")
    print("=" * 80)

    trainer = BlueprintTrainer()
    print(f"Starting extended CFR training for {iterations} iterations...")
    expected_value = trainer.train_blueprint(iterations)

    print(f"\nExtended training completed!")
    print(f"Expected value: {expected_value:.6f}")
    print(f"Total info sets: {len(trainer.info_sets)}")

    analyze_strategy_quality(trainer)
    return trainer


def analyze_strategy_quality(trainer):
    print(f"\nSTRATEGY QUALITY ANALYSIS:")
    print("-" * 80)

    total = len(trainer.info_sets)
    pure_strategies = 0
    mixed_strategies = 0
    strategy_errors = 0
    all_actions = set()
    action_patterns = {}

    for key, info_set in trainer.info_sets.items():
        if not info_set.legal_actions:
            continue

        avg_strategy = info_set.get_average_strategy(info_set.legal_actions)

        if abs(sum(avg_strategy) - 1.0) > 0.001:
            strategy_errors += 1

        significant_actions = sum(1 for p in avg_strategy if p > 0.05)
        if significant_actions == 1:
            pure_strategies += 1
        else:
            mixed_strategies += 1

        all_actions.update(info_set.legal_actions)

        parts = key.split('_')
        pattern = parts[-1] if len(parts) > 1 else '(first_to_act)'
        action_patterns[pattern] = action_patterns.get(pattern, 0) + 1

    print(f"Total strategies: {total}")
    print(f"Pure strategies:  {pure_strategies} ({pure_strategies/total*100:.1f}%)")
    print(f"Mixed strategies: {mixed_strategies} ({mixed_strategies/total*100:.1f}%)")
    print(f"Normalization errors: {strategy_errors}")

    sorted_patterns = sorted(action_patterns.items(), key=lambda x: x[1], reverse=True)
    print(f"\nMost common action patterns:")
    for pattern, count in sorted_patterns[:10]:
        print(f"  '{pattern}': {count} info sets")

    print(f"\nUnique actions: {sorted(all_actions)}")


def test_action_storage():
    print("=" * 80)
    print("TESTING LEGAL ACTION STORAGE")
    print("=" * 80)

    trainer = BlueprintTrainer()
    trainer.train_blueprint(5)

    print(f"Total info sets: {len(trainer.info_sets)}")
    for key, info_set in trainer.info_sets.items():
        print(f"Info set '{key}':")
        print(f"  Legal actions:       {info_set.legal_actions}")
        print(f"  Cumulative regrets:  {list(info_set.cumulative_regrets.keys())}")
        print(f"  Cumulative strategy: {list(info_set.cumulative_strategy.keys())}")

        if info_set.legal_actions:
            regret_actions = set(info_set.cumulative_regrets.keys())
            strategy_actions = set(info_set.cumulative_strategy.keys())
            stored_actions = set(info_set.legal_actions)
            if regret_actions == strategy_actions == stored_actions:
                print(f"  All action sets match")
            else:
                if regret_actions != stored_actions:
                    print(f"  Warning: regret actions {regret_actions} != stored {stored_actions}")
                if strategy_actions != stored_actions:
                    print(f"  Warning: strategy actions {strategy_actions} != stored {stored_actions}")
        else:
            print(f"  No legal actions stored!")
        print()


def run_all_tests():
    try:
        print("STARTING BLUEPRINT TRAINER TESTS")
        print("=" * 80)
        test_blueprint_trainer_extended(10)
        print("\nALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
    except Exception as e:
        print(f"TEST FAILED: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
