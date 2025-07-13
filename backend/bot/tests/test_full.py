from src.bot.player import Player
from src.cfr.information_set import InformationSet
from src.cfr.blueprint_trainer import BlueprintTrainer
from src.cfr.poker_game import PokerGame
from src.bot.game_adapter import GameAdapter
from src.abstractions.action_abstractions import ActionAbstraction
from src.abstractions.card_abstractions import CardAbstraction
from src.abstractions.hand_evaluator import HandEvaluator


def test_hand_evaluator():
    """Test 1: Hand Evaluation Foundation"""
    print("=" * 60)
    print("TEST 1: HAND EVALUATOR")
    print("=" * 60)

    evaluator = HandEvaluator()

    # Test different hand types
    test_cases = [
        (['AH', 'KH'], ['QH', 'JH', 'TH'], "Royal Flush"),
        (['AH', 'KS'], ['AC', 'KH', 'QD'], "Two Pair"),
        (['AH', '7S'], ['AC', '7H', '2D'], "Two Pair"),
        (['AH', '7S'], ['KC', 'JH', '2D'], "High Card"),
        (['7H', '7S'], ['7C', 'KH', 'QD'], "Three of a Kind")
    ]

    for hole_cards, community_cards, expected in test_cases:
        hand_type, strength = evaluator.evaluate_hand_strength(
            hole_cards, community_cards)
        draw_potential = evaluator.has_draw_potential(
            hole_cards, community_cards)

        print(f"Hand: {hole_cards} + {community_cards}")
        print(f"  Expected: {expected}")
        print(f"  Actual: {hand_type} (strength: {strength})")
        print(f"  Draw potential: {draw_potential}")
        print()

    print("✅ Hand Evaluator Test Complete\n")


def test_card_abstraction():
    """Test 2: Card Abstraction System"""
    print("=" * 60)
    print("TEST 2: CARD ABSTRACTION")
    print("=" * 60)

    abstraction = CardAbstraction()

    # Test preflop bucketing
    preflop_hands = [
        (['AH', 'AS'], 'premium_pair'),
        (['AH', 'KS'], 'ace_king'),
        (['AH', '7S'], 'ace_x'),
        (['7H', '2S'], 'weak')
    ]

    print("PREFLOP BUCKETING:")
    for hole_cards, expected in preflop_hands:
        bucket = abstraction.preflop_bucket(hole_cards)
        print(f"  {hole_cards} → {bucket} (expected: {expected})")
        assert bucket == expected, f"Expected {expected}, got {bucket}"

    # Test postflop bucketing
    postflop_hands = [
        (['AH', 'KS'], ['AC', 'KH', 'QD'], "strong"),  # Two pair
        (['AH', '7S'], ['AC', '7H', '2D'], "medium"),  # Two pair
        (['AH', '7S'], ['KC', 'JH', '2D'], "bluff"),   # High card
        (['7H', '7S'], ['7C', 'KH', 'QD'], "strong"),  # Three of a kind
    ]

    print("\nPOSTFLOP BUCKETING:")
    for hole_cards, community_cards, expected_category in postflop_hands:
        bucket = abstraction.postflop_bucket(hole_cards, community_cards)
        hand_type, strength = abstraction.hand_evaluator.evaluate_hand_strength(
            hole_cards, community_cards)

        print(f"  {hole_cards} + {community_cards}")
        print(f"    Hand: {hand_type} (strength: {strength})")
        print(f"    Bucket: {bucket}")
        print()

    print("✅ Card Abstraction Test Complete\n")


def test_action_abstraction():
    """Test 3: Action Abstraction System"""
    print("=" * 60)
    print("TEST 3: ACTION ABSTRACTION")
    print("=" * 60)

    abstraction = ActionAbstraction()

    # Test bet size categorization
    test_cases = [
        ({'action': 'bet', 'amount': 25}, {'pot_size': 100,
         'player_stack': 200}, 'small'),  # 25/100 = 0.25
        ({'action': 'bet', 'amount': 66}, {'pot_size': 100,
         'player_stack': 200}, 'medium'),  # 66/100 = 0.66
        ({'action': 'bet', 'amount': 100}, {'pot_size': 100,
         'player_stack': 200}, 'large'),  # 100/100 = 1.0
        ({'action': 'bet', 'amount': 190}, {'pot_size': 200,
         'player_stack': 200}, 'allin'),  # 95% of stack
    ]

    print("BET SIZE CATEGORIZATION:")
    for action, game_state, expected in test_cases:
        category = abstraction.categorize_bet_size(action, game_state)
        ratio = action['amount'] / game_state['pot_size']

        print(
            f"  Bet ${action['amount']} in ${game_state['pot_size']} pot (ratio: {ratio:.2f})")
        print(f"    Category: {category} (expected: {expected})")
        print()

    # Test action history abstraction
    print("ACTION HISTORY ABSTRACTION:")
    action_history = [
        {'action': 'check', 'amount': 0},
        {'action': 'bet', 'amount': 33},
        {'action': 'raise', 'amount': 66}
    ]
    game_state = {'pot_size': 100, 'player_stack': 200}

    abstracted = abstraction.abstract_action_history(
        action_history, game_state)
    print(f"  History: {[a['action'] for a in action_history]}")
    print(f"  Abstracted: '{abstracted}'")
    print(f"  Meaning: k=check, s=small bet, m=medium raise")

    print("\n✅ Action Abstraction Test Complete\n")


def test_game_adapter():
    """Test 4: Game Adapter Integration"""
    print("=" * 60)
    print("TEST 4: GAME ADAPTER INTEGRATION")
    print("=" * 60)

    adapter = GameAdapter()

    # Test round_state to game_state conversion
    print("ROUND_STATE TO GAME_STATE CONVERSION:")
    round_state = {
        'street': 'flop',
        'pot': {'main': {'amount': 150}},
        'seats': [{'stack': 95}, {'stack': 97}],
        'action_histories': {
            'flop': [
                {'action': 'check', 'amount': 0},
                {'action': 'bet', 'amount': 50}
            ]
        }
    }

    game_state = adapter.convert_round_state_to_game_state(round_state)
    print(f"  Round state keys: {list(round_state.keys())}")
    print(f"  Game state: {game_state}")
    print()

    # Test info set key creation
    print("INFO SET KEY CREATION:")
    test_scenarios = [
        # Preflop scenario
        {
            'hole_cards': ['AH', 'KS'],
            'round_state': {
                'street': 'preflop',
                'action_histories': {'preflop': [{'action': 'check', 'amount': 0}]},
                'pot': {'main': {'amount': 3}}
            },
            'description': 'Preflop AK after check'
        },
        # Postflop scenario
        {
            'hole_cards': ['AH', 'KS'],
            'round_state': {
                'street': 'flop',
                'community_card': ['AC', 'KH', '2D'],
                'action_histories': {'flop': [
                    {'action': 'check', 'amount': 0},
                    {'action': 'bet', 'amount': 33}
                ]},
                'pot': {'main': {'amount': 100}}
            },
            'description': 'Flop AK two pair after check-bet'
        }
    ]

    for scenario in test_scenarios:
        info_set_key = adapter.create_info_set_key(
            scenario['hole_cards'], scenario['round_state'])

        print(f"  Scenario: {scenario['description']}")
        print(f"  Hole cards: {scenario['hole_cards']}")
        if 'community_card' in scenario['round_state']:
            print(f"  Community: {scenario['round_state']['community_card']}")
        print(f"  Info set key: '{info_set_key}'")
        print()

    print("✅ Game Adapter Test Complete\n")


def test_poker_game():
    """Test 5: Poker Game Logic"""
    print("=" * 60)
    print("TEST 5: POKER GAME LOGIC")
    print("=" * 60)

    game = PokerGame()

    # Test legal actions
    print("LEGAL ACTIONS:")
    action_scenarios = [
        ([], "First action"),
        (['check'], "After check"),
        (['bet_small'], "After bet"),
        (['bet_small', 'call'], "After bet-call")
    ]

    for history, description in action_scenarios:
        actions = game.get_legal_actions(history)
        print(f"  {description}: {history} → {actions}")

    # Test pot calculations
    print("\nPOT CALCULATIONS:")
    pot_scenarios = [
        (['bet_small'], "Small bet (33% of 3)"),
        (['bet_small', 'call'], "Bet-call"),
        (['bet_small', 'raise_medium'], "Bet-raise"),
    ]

    for history, description in pot_scenarios:
        pot_size = game.calculate_current_pot_size(history)
        print(f"  {description}: {history} → pot = ${pot_size:.2f}")

    # Test stack calculations
    print("\nSTACK CALCULATIONS:")
    for player in [0, 1]:
        for history, description in pot_scenarios:
            stack = game.calculate_player_stack_after_history(player, history)
            print(
                f"  Player {player} after {description}: stack = ${stack:.2f}")

    print("\n✅ Poker Game Test Complete\n")


def save_info_sets_analysis(trainer, filename="info_sets_analysis.json"):
    """Save detailed info sets analysis to file"""
    import json
    from pathlib import Path

    # Create analysis directory
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)

    # Structure the data for analysis
    analysis_data = {
        "total_info_sets": len(trainer.info_sets),
        "info_sets_by_street": {
            "preflop": {},
            "flop": {},
            "turn": {},
            "river": {}
        },
        "hand_bucket_distribution": {},
        "action_pattern_distribution": {},
        "detailed_info_sets": {}
    }

    # Categorize and analyze each info set
    for key, info_set in trainer.info_sets.items():
        # Store detailed info
        analysis_data["detailed_info_sets"][key] = {
            "regrets": dict(info_set.cumulative_regrets),
            "strategies": dict(info_set.cumulative_strategy),
            "key_parts": key.split('_')
        }

        # Categorize by street
        if '_flop_' in key:
            analysis_data["info_sets_by_street"]["flop"][key] = dict(
                info_set.cumulative_regrets)
        elif '_turn_' in key:
            analysis_data["info_sets_by_street"]["turn"][key] = dict(
                info_set.cumulative_regrets)
        elif '_river_' in key:
            analysis_data["info_sets_by_street"]["river"][key] = dict(
                info_set.cumulative_regrets)
        else:
            analysis_data["info_sets_by_street"]["preflop"][key] = dict(
                info_set.cumulative_regrets)

        # Analyze hand buckets
        parts = key.split('_')
        if parts:
            hand_bucket = parts[0]
            analysis_data["hand_bucket_distribution"][hand_bucket] = \
                analysis_data["hand_bucket_distribution"].get(
                    hand_bucket, 0) + 1

        # Analyze action patterns
        if len(parts) > 1:
            action_pattern = parts[-1]
            analysis_data["action_pattern_distribution"][action_pattern] = \
                analysis_data["action_pattern_distribution"].get(
                    action_pattern, 0) + 1

    # Save to file
    filepath = analysis_dir / filename
    with open(filepath, 'w') as f:
        json.dump(analysis_data, f, indent=2)

    print(f"\n📊 INFO SETS ANALYSIS SAVED TO: {filepath}")
    print(f"Total info sets: {len(trainer.info_sets)}")
    print(f"By street: Preflop={len(analysis_data['info_sets_by_street']['preflop'])}, "
          f"Flop={len(analysis_data['info_sets_by_street']['flop'])}, "
          f"Turn={len(analysis_data['info_sets_by_street']['turn'])}, "
          f"River={len(analysis_data['info_sets_by_street']['river'])}")

    return analysis_data


def test_single_iteration_info_sets():
    """Test that single iteration discovers all info sets"""
    print("\n" + "=" * 80)
    print("TESTING SINGLE ITERATION INFO SET DISCOVERY")
    print("=" * 80)

    trainer = BlueprintTrainer()

    # Test with just 1 iteration
    print("Testing with 1 iteration...")
    trainer.train_blueprint(iterations=1)

    iteration_1_count = len(trainer.info_sets)
    print(f"Info sets after 1 iteration: {iteration_1_count}")

    # Save the analysis
    analysis_1 = save_info_sets_analysis(
        trainer, "single_iteration_analysis.json")

    # Test with 2 iterations
    print("\nTesting with 1 additional iteration...")
    trainer.train_blueprint(iterations=1)  # Run 1 more

    iteration_2_count = len(trainer.info_sets)
    print(f"Info sets after 2 iterations: {iteration_2_count}")

    # Check if new info sets were discovered
    if iteration_2_count > iteration_1_count:
        print(f"❌ PROBLEM: New info sets discovered in iteration 2!")
        print(f"   This suggests CFR is not exploring the full tree in iteration 1")
    else:
        print(f"✅ GOOD: No new info sets in iteration 2 (as expected for vanilla CFR)")

    # Test theoretical maximum
    print(f"\n📊 ANALYSIS:")
    print(f"Your system: {iteration_1_count} info sets")
    print(f"Kuhn Poker (reference): 12 info sets")
    print(f"Leduc Hold'em (reference): 288 info sets")
    print(f"Expected for simplified poker: 200-500+ info sets")

    if iteration_1_count < 100:
        print(f"⚠️  WARNING: Your info set count is suspiciously low!")
        print(f"   Possible issues:")
        print(f"   1. CFR not exploring full game tree")
        print(f"   2. Action space too restrictive")
        print(f"   3. Street advancement issues")
        print(f"   4. Early termination bugs")

    return analysis_1


def test_blueprint_trainer():
    """Test 6: Blueprint Training (Enhanced with Storage)"""
    print("=" * 60)
    print("TEST 6: BLUEPRINT TRAINER")
    print("=" * 60)

    trainer = BlueprintTrainer()

    # Test card dealing
    print("CARD DEALING:")
    p0_cards, p1_cards, community_cards = trainer.deal_random_hand()
    print(f"  Player 0: {p0_cards}")
    print(f"  Player 1: {p1_cards}")
    print(f"  Community: {community_cards}")
    print()

    # Test round state creation
    print("ROUND STATE CREATION:")
    history = ['check', 'bet_small', 'call']
    street = 1  # Flop

    round_state = trainer.create_round_state_for_info_set(
        community_cards, history, street)
    print(f"  History: {history}")
    print(f"  Street: {street}")
    print(f"  Round state keys: {list(round_state.keys())}")
    print(f"  Pot size: {round_state['pot']['main']['amount']}")
    print(f"  Action history: {round_state['action_histories']['flop']}")
    print()

    # Test training with analysis
    print("TRAINING WITH INFO SET ANALYSIS:")
    trainer.train_blueprint(iterations=100)

    # Save comprehensive analysis
    analysis_data = save_info_sets_analysis(
        trainer, "full_training_analysis.json")

    # Test single iteration discovery
    test_single_iteration_info_sets()

    print("\n✅ Blueprint Trainer Test Complete\n")


def test_player_integration():
    """Test 7: Player Integration"""
    print("=" * 60)
    print("TEST 7: PLAYER INTEGRATION")
    print("=" * 60)

    # Create player (will try to load strategy)
    player = Player()
    print(f"Player created with {len(player.info_sets)} loaded strategies")

    # Test game state extraction
    print("\nGAME STATE EXTRACTION:")
    mock_round_state = {
        'street': 'flop',
        'pot': {'main': {'amount': 150}},
        'community_card': ['AC', 'KH', '2D'],
        'seats': [
            {'uuid': 'player_1', 'stack': 95},
            {'uuid': 'player_2', 'stack': 97}
        ],
        'action_histories': {
            'preflop': [
                {'action': 'SMALLBLIND', 'amount': 1, 'uuid': 'player_1'},
                {'action': 'BIGBLIND', 'amount': 2, 'uuid': 'player_2'}
            ],
            'flop': [
                {'action': 'CHECK', 'amount': 0, 'uuid': 'player_1'},
                {'action': 'BET', 'amount': 50, 'uuid': 'player_2'}
            ]
        }
    }

    game_state = player.extract_game_state(mock_round_state)
    print(f"  Extracted game state: {game_state}")

    # Test action conversion
    print("\nACTION CONVERSION:")
    mock_valid_actions = [
        {'action': 'fold'},
        {'action': 'call', 'amount': 50},
        {'action': 'raise', 'amount': {'min': 100, 'max': 200}}
    ]

    cfr_actions = player.game_adapter.action_abstractions.pypoker_to_cfr_actions(
        mock_valid_actions, game_state)
    print(
        f"  PyPokerEngine actions: {[a['action'] for a in mock_valid_actions]}")
    print(f"  CFR actions: {cfr_actions}")

    # Test info set key creation
    print("\nINFO SET KEY CREATION:")
    hole_cards = ['AH', 'KS']
    info_set_key = player.game_adapter.create_info_set_key(
        hole_cards, mock_round_state)
    print(f"  Hole cards: {hole_cards}")
    print(f"  Info set key: '{info_set_key}'")

    print("\n✅ Player Integration Test Complete\n")


def run_all_tests():
    """Run all comprehensive tests"""
    print("🚀 STARTING COMPREHENSIVE POKER BOT TESTS")
    print("=" * 80)

    try:
        test_hand_evaluator()
        test_card_abstraction()
        test_action_abstraction()
        test_game_adapter()
        test_poker_game()
        test_blueprint_trainer()
        test_player_integration()

        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Your poker bot system is ready for training and gameplay!")

    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
