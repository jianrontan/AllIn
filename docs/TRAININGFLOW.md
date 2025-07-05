```
🎮 BlueprintTrainer.train_blueprint()
    ↓
📇 Deal Random Cards
BlueprintTrainer.deal_random_hand()
├── Player 0: ['AH', 'KS'] 
├── Player 1: ['QD', 'JC']
└── Community: ['AS', 'KH', 'QC', 'JH', '2D']
    ↓
🧠 CFR Algorithm (The Learning Engine)
BlueprintTrainer.cfr() 
├── "I have AK, what should I do?"
├── Try all possible actions: check, bet_small, bet_large
└── Calculate how much I regret not taking each action
    ↓
🎯 Get Legal Actions
PokerGame.get_legal_actions()
├── Input: history=['check'], street=1 (flop)
└── Output: ['check', 'bet_small', 'bet_medium', 'bet_large']
    ↓
🔑 Create Decision Key  
GameAdapter.create_info_set_key()
├── Input: cards=['AH','KS'], community=['AS','KH','QC']
├── CardAbstraction.get_bucket() → "ace_king_strong_flop"
├── ActionAbstraction.abstract_action_history() → "k" (check)
└── Output: "ace_king_strong_flop_k"
    ↓
🔍 Hand Evaluation
HandEvaluator.evaluate_hand_strength()
├── Input: ['AH','KS'] + ['AS','KH','QC'] 
├── phevaluator.evaluate_cards() → hand strength
└── Output: ("pair", 1) = "I have a pair of Aces"
    ↓
📊 Card Abstraction
CardAbstraction.postflop_bucket()
├── strength=1 (pair) → "weak_made" 
├── But wait, this is top pair!
└── Should probably be "strong"
    ↓
💾 Strategy Storage/Update
InformationSet.get_strategy()
├── Key: "ace_king_strong_flop_k"
├── Actions: ['check', 'bet_small', 'bet_large'] 
├── Current regrets: check=-0.5, bet_small=+0.2, bet_large=+0.8
├── Strategy: check=0%, bet_small=20%, bet_large=80%
└── "I should bet large with top pair most of the time"
    ↓
🔄 Recursive Exploration
BlueprintTrainer.cfr() (recursively)
├── Try: history=['check', 'bet_large'] 
├── Opponent options: ['fold', 'call', 'raise_small']
├── Calculate utilities for each path
└── Update regrets: "betting large was good when opponent folded"
    ↓
🏁 Terminal Evaluation
PokerGame.get_utility()
├── If fold: folder loses → utility = +1 or -1
├── If showdown: HandEvaluator compares hands
└── AK pair vs QJ high card → AK wins → utility = +1
```