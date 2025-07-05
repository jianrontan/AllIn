from src.bot.game_adapter import GameAdapter
from src.abstractions.hand_evaluator import HandEvaluator
from src.cfr.blueprint_trainer import BlueprintTrainer


def test_hand_evaluation():
    """Test hand evaluation works"""
    evaluator = HandEvaluator()
    hand_type, strength = evaluator.evaluate_hand_strength(
        ['AH', 'KH'], ['QH', 'JH', 'TH'])
    print(f"Royal flush: {hand_type}, strength: {strength}")
    assert hand_type == 'straight_flush'


def test_info_set_key_creation():
    """Test info set key creation"""
    adapter = GameAdapter()

    # Test preflop
    round_state = {'street': 'preflop', 'action_histories': {'preflop': []}}
    key = adapter.create_info_set_key(['AH', 'KS'], round_state)
    print(f"Preflop key: {key}")

    # Test postflop
    round_state = {
        'street': 'flop',
        'community_card': ['AS', '7H', '2D'],
        'action_histories': {'flop': []}
    }
    key = adapter.create_info_set_key(['AH', 'KS'], round_state)
    print(f"Postflop key: {key}")


def test_blueprint_trainer():
    """Test trainer can run basic iteration"""
    trainer = BlueprintTrainer()
    trainer.train_blueprint(iterations=5)  # Very small test
    print(f"Created {len(trainer.info_sets)} info sets")


if __name__ == "__main__":
    test_hand_evaluation()
    test_info_set_key_creation()
    test_blueprint_trainer()
