# backend/bot/tests/test_confidence_detection.py
from src.bot.player import Player
from src.subgame.player_blueprint_adapter import PlayerBlueprintAdapter
from src.subgame.subgame_detector import SubgameDetector
import time


def test_confidence_detection():
    """Test confidence detection integration"""

    # Initialize player (loads blueprint automatically)
    player = Player()

    # Test adapter
    adapter = PlayerBlueprintAdapter(player)
    print(f"Adapter has {len(adapter.info_sets)} info sets")
    print(f"Total training iterations: {adapter.total_training_iterations}")

    # Test subgame detector
    detector = SubgameDetector(adapter)

    # Test scenarios
    test_cases = [
        ("weak_k", ["check"]),
        ("ace_x_weak_made_flop_", []),
        ("unknown_info_set", ["check", "bet_large", "raise_large"])
    ]

    for info_set_key, game_history in test_cases:
        should_trigger, reasons = detector.should_trigger_subgame_solving(
            info_set_key, game_history, adapter.total_training_iterations)

        print(f"\nInfo Set: {info_set_key}")
        print(f"History: {game_history}")
        print(f"Trigger: {'YES' if should_trigger else 'NO'}")
        print(f"Reasons: {reasons}")


def test_real_game_scenarios():
    """Test confidence detection with realistic poker scenarios"""

    player = Player()
    adapter = PlayerBlueprintAdapter(player)
    detector = SubgameDetector(adapter)

    # Real game scenarios from blueprint.json
    real_scenarios = [
        {
            'info_set': 'weak_',  # From blueprint - should have good confidence
            'history': [],
            'expected': 'Should NOT trigger - well trained'
        },
        {
            'info_set': 'ace_x_k',  # From blueprint - low visit count
            'history': ['check'],
            'expected': 'May trigger - low visit count (only 1)'
        },
        {
            'info_set': 'weak_bluff_flop_',  # High visit count scenario
            'history': [],
            'expected': 'Should NOT trigger - well trained'
        }
    ]

    for scenario in real_scenarios:
        should_trigger, reasons = detector.should_trigger_subgame_solving(
            scenario['info_set'], scenario['history'], 1000)

        print(f"\nScenario: {scenario['expected']}")
        print(f"Info Set: {scenario['info_set']}")
        print(f"Result: {'🎯 TRIGGER' if should_trigger else '✅ BLUEPRINT'}")
        print(f"Reasons: {reasons}")


def test_off_tree_scenarios():
    """Test off-tree detection with bet sizing violations"""

    player = Player()
    detector = player.off_tree_detector

    test_cases = [
        {
            'description': '4th raise violation',
            'history': ['bet_small', 'raise_small', 'raise_medium', 'raise_large', 'raise_large'],
            'expected': 'Should trigger - excessive betting'
        },
        {
            'description': 'Unknown action',
            'history': ['check', 'unknown_action'],
            'expected': 'Should trigger - unknown action'
        },
        {
            'description': 'Normal betting',
            'history': ['bet_small', 'call'],
            'expected': 'Should NOT trigger - normal'
        }
    ]

    for case in test_cases:
        is_off_tree, reasons = detector.is_off_tree_situation(case['history'])

        print(f"\nTest: {case['description']}")
        print(f"History: {case['history']}")
        print(f"Result: {'🎯 OFF-TREE' if is_off_tree else '✅ ON-TREE'}")
        print(f"Reasons: {reasons}")
        print(f"Expected: {case['expected']}")


def test_high_stakes_scenarios():
    """Test high-stakes detection"""

    player = Player()
    adapter = PlayerBlueprintAdapter(player)
    detector = SubgameDetector(adapter)

    high_stakes_cases = [
        {
            'description': 'Deep stack situation',
            'game_state': {'pot_size': 10, 'player_stack': 200, 'big_blind': 2},
            'history': ['bet_small'],
            'expected': 'Should trigger - SPR = 20'
        },
        {
            'description': 'High pot situation',
            'game_state': {'pot_size': 50, 'player_stack': 100, 'big_blind': 2},
            'history': ['bet_large'],
            'expected': 'Should trigger - 25x big blind pot'
        },
        {
            'description': 'Normal situation',
            'game_state': {'pot_size': 5, 'player_stack': 50, 'big_blind': 2},
            'history': ['check'],
            'expected': 'Should NOT trigger - normal stakes'
        }
    ]

    for case in high_stakes_cases:
        should_trigger, reasons = detector.should_trigger_subgame_solving(
            'weak_', case['history'], 1000, case['game_state'])

        print(f"\nScenario: {case['description']}")
        print(f"Game State: {case['game_state']}")
        print(f"Result: {'🎯 TRIGGER' if should_trigger else '✅ BLUEPRINT'}")
        print(f"Reasons: {reasons}")
        print(f"Expected: {case['expected']}")


def test_strategy_entropy_cases():
    """Test different entropy scenarios"""

    player = Player()

    # Look at actual entropy values from blueprint
    entropy_cases = [
        'weak_',  # Should have mixed strategy (lower entropy)
        'ace_x_k',  # Check actual entropy from blueprint
        'medium_pair_weak_made_river_ks',  # Check if uniform strategy
    ]

    for info_set_key in entropy_cases:
        if info_set_key in player.info_sets:
            info_set = player.info_sets[info_set_key]
            confidence_info = player.get_confidence_info(info_set_key)

            print(f"\nInfo Set: {info_set_key}")
            print(f"Visit Count: {confidence_info['visit_count']}")
            print(
                f"Strategy Entropy: {confidence_info.get('strategy_entropy', 'N/A'):.3f}")
            print(f"Confidence Level: {confidence_info['confidence']}")


def test_confidence_detection_performance():
    """Measure performance impact of confidence detection"""
    player = Player()

    # Simulate multiple confidence checks
    test_iterations = 1000

    # Test without confidence detection
    start_time = time.time()
    for i in range(test_iterations):
        info_set_key = f"test_key_{i % 10}"
        # Just access info set without confidence check
        pass
    baseline_time = time.time() - start_time

    # Test with confidence detection
    start_time = time.time()
    for i in range(test_iterations):
        info_set_key = f"weak_{i % 10}"
        game_history = ['check'] * (i % 3)
        should_trigger, reasons = player.subgame_detector.should_trigger_subgame_solving(
            info_set_key, game_history, 1000)
    detection_time = time.time() - start_time

    print(f"\nPerformance Analysis:")
    print(f"Baseline time: {baseline_time*1000:.2f}ms")
    print(f"With confidence detection: {detection_time*1000:.2f}ms")
    print(
        f"Overhead per check: {(detection_time - baseline_time)/test_iterations*1000:.3f}ms")

    # Fix: Check for division by zero
    if baseline_time > 0:
        print(f"Overhead factor: {detection_time/baseline_time:.2f}x")
    else:
        print(f"Overhead factor: N/A (baseline too fast to measure)")


def test_pypoker_integration():
    """Test confidence detection with PyPokerEngine format data"""

    # Simulate PyPokerEngine round_state
    mock_round_state = {
        'street': 'flop',
        'pot': {'main': {'amount': 15}},
        'community_card': ['AH', 'KS', '3D'],
        'seats': [
            {'uuid': 'player_1', 'stack': 90, 'state': 'participating'},
            {'uuid': 'player_2', 'stack': 85, 'state': 'participating'}
        ],
        'action_histories': {
            'preflop': [
                {'action': 'SMALLBLIND', 'amount': 1, 'uuid': 'player_1'},
                {'action': 'BIGBLIND', 'amount': 2, 'uuid': 'player_2'},
                {'action': 'CALL', 'amount': 2, 'uuid': 'player_1'}
            ],
            'flop': [
                {'action': 'CHECK', 'amount': 0, 'uuid': 'player_2'},
                # Large bet relative to pot
                {'action': 'BET', 'amount': 10, 'uuid': 'player_1'}
            ]
        }
    }

    player = Player()
    hole_cards = ['AS', 'KD']

    # Test info set creation
    info_set_key = player.game_adapter.create_info_set_key(
        hole_cards, mock_round_state)
    game_history = player._extract_simplified_game_history(mock_round_state)
    game_state = player.extract_game_state(mock_round_state)

    print(f"\nPyPokerEngine Integration Test:")
    print(f"Info Set Key: {info_set_key}")
    print(f"Game History: {game_history}")
    print(f"Game State: {game_state}")

    # Test confidence detection
    if player.subgame_detector:
        should_trigger, reasons = player.subgame_detector.should_trigger_subgame_solving(
            info_set_key, game_history, player.total_training_iterations, game_state)

        print(
            f"Confidence Result: {'🎯 TRIGGER' if should_trigger else '✅ BLUEPRINT'}")
        print(f"Reasons: {reasons}")


def test_threshold_sensitivity():
    """Analyze threshold sensitivity"""

    # Test different threshold values
    threshold_tests = [
        {'min_visit_count': 5, 'strategy_entropy_threshold': 0.5},   # More aggressive
        {'min_visit_count': 10, 'strategy_entropy_threshold': 0.7},  # Current
        # More conservative
        {'min_visit_count': 20, 'strategy_entropy_threshold': 0.9}
    ]

    player = Player()

    for i, thresholds in enumerate(threshold_tests):
        print(f"\nThreshold Set {i+1}: {thresholds}")

        # Create detector with custom thresholds
        adapter = PlayerBlueprintAdapter(player)
        detector = SubgameDetector(adapter, thresholds)

        # Test with known scenarios
        test_keys = ['weak_', 'ace_x_k', 'unknown_info_set']
        trigger_count = 0

        for key in test_keys:
            should_trigger, reasons = detector.should_trigger_subgame_solving(
                key, ['check'], 1000)
            if should_trigger:
                trigger_count += 1

        print(
            f"Trigger rate: {trigger_count}/{len(test_keys)} ({trigger_count/len(test_keys)*100:.1f}%)")


if __name__ == "__main__":
    test_confidence_detection()
    test_real_game_scenarios()
    test_off_tree_scenarios()
    test_high_stakes_scenarios()
    test_strategy_entropy_cases()
    test_confidence_detection_performance()
    test_pypoker_integration()
    test_threshold_sensitivity()
