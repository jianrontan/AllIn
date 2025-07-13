import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cfr.blueprint_trainer import BlueprintTrainer

def test_forced_kml_ml_sequences():
    """Force the exact sequences that should create kml and ml patterns"""
    print("=" * 60)
    print("TESTING FORCED KML AND ML SEQUENCES")
    print("=" * 60)
    
    trainer = BlueprintTrainer()
    
    # Test both sequences manually
    test_sequences = [
        (['check', 'bet_medium', 'raise_large'], 'kml'),
        (['bet_medium', 'raise_large'], 'ml'),
    ]
    
    # Use fixed cards to ensure consistency
    p0_cards = ['AH', 'AS']  # Premium hand (should raise)
    p1_cards = ['KH', 'KS']  # Strong hand (should bet/call)
    community_cards = ['AC', 'AD', '2D', '3S', '4C']  # Strong board
    
    for history, expected_pattern in test_sequences:
        print(f"\nTesting sequence: {history}")
        print(f"Expected pattern: '{expected_pattern}'")
        
        # Check if the sequence is even possible
        for i, action in enumerate(history):
            partial_history = history[:i]
            legal_actions = trainer.game.get_legal_actions(partial_history)
            
            print(f"  Step {i+1}: After {partial_history}")
            print(f"    Legal actions: {legal_actions}")
            print(f"    Trying to take: '{action}'")
            print(f"    Action allowed: {action in legal_actions}")
            
            if action not in legal_actions:
                print(f"    ❌ SEQUENCE IMPOSSIBLE: '{action}' not legal!")
                break
        else:
            print(f"  ✅ Sequence is technically possible")
            
            # Test the abstraction
            round_state = trainer.create_round_state_for_info_set(
                community_cards, history, 1)  # Flop
            
            actual_pattern = trainer.game_adapter.extract_betting_history(round_state)
            print(f"  Abstracted to: '{actual_pattern}'")
            print(f"  Matches expected: {actual_pattern == expected_pattern}")
            
            # Now manually run one CFR iteration with this exact sequence
            print(f"  Testing CFR traversal...")
            try:
                utility = trainer.cfr(p0_cards, p1_cards, community_cards, 
                                    history, 1.0, 1.0, 1)
                print(f"  CFR utility: {utility}")
                
                # Check if info sets were created
                for key in trainer.info_sets.keys():
                    if expected_pattern in key:
                        print(f"  ✅ Created info set: '{key}'")
                        
            except Exception as e:
                print(f"  ❌ CFR error: {e}")

def test_why_cfr_avoids_large_raises():
    """Test why CFR learned to avoid large raises after medium bets"""
    print("\n" + "=" * 60)
    print("ANALYZING WHY CFR AVOIDS LARGE RAISES")
    print("=" * 60)
    
    trainer = BlueprintTrainer()
    trainer.train_blueprint(10)  # Quick training
    
    # Look for medium bet scenarios
    medium_bet_info_sets = []
    for key, info_set in trainer.info_sets.items():
        if ('km' in key or '_m' in key.split('_')[-1]) and info_set.legal_actions:
            if 'raise_large' in info_set.legal_actions:
                medium_bet_info_sets.append((key, info_set))
    
    print(f"Found {len(medium_bet_info_sets)} info sets with medium bets and raise_large option:")
    
    for key, info_set in medium_bet_info_sets[:5]:  # Show first 5
        strategy = info_set.get_average_strategy(info_set.legal_actions)
        strategy_dict = dict(zip(info_set.legal_actions, strategy))
        
        print(f"\nInfo set: '{key}'")
        print(f"  raise_large probability: {strategy_dict.get('raise_large', 0):.6f}")
        print(f"  raise_large regret: {info_set.cumulative_regrets.get('raise_large', 0):.3f}")
        
        if strategy_dict.get('raise_large', 0) < 0.001:
            print(f"  ⚠️ CFR learned to almost never raise_large here!")

if __name__ == "__main__":
    test_forced_kml_ml_sequences()
    test_why_cfr_avoids_large_raises()
