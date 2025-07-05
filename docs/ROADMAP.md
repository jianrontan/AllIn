# Poker AI Roadmap

Last updated: 15/6/25

## Phase 1: Training (Blueprint Strategy Creation)

```
📚 **TRAINING PHASE - BLUEPRINT STRATEGY**
    ↓
🏗️ Hand Evaluation Foundation
📁 `src/abstractions/hand_evaluator.py` ✅ **COMPLETED** 
├── evaluate_hand_strength() - Henry R Lee's library integration
├── has_draw_potential() - Flush/straight draw detection
├── get_hand_type_from_value() - Convert evaluator results to readable types
├── get_relative_strength() - Convert to 0-8 strength scale
└── Integration with phevaluator - Tested and working
    ↓
📊 Enhanced Abstractions (Uses Hand Evaluation)
├── **Card Abstraction**
│   📁 `src/abstractions/card_abstraction.py` ✅ **COMPLETED**
│   ├── preflop_bucket() - 8 preflop buckets (premium_pair, ace_king, etc.)
│   ├── postflop_bucket() - Enhanced with hand evaluator
│   │   ├── Uses hand_evaluator.evaluate_hand_strength()
│   │   ├── Returns: "monster"/"strong"/"medium"/"weak_made"/"draw"/"bluff"
│   │   └── Dynamic re-bucketing per street
│   ├── cards_to_string() - PyPokerEngine format handling
│   ├── parse_string_cards() - Handle ['AH', 'KS'] format
│   └── has_pair() - Enhanced with hand evaluator
│
└── **Action Abstraction** 
    📁 `src/abstractions/action_abstraction.py` ✅ **COMPLETED**
    ├── **Bet Sizing Framework**
    │   ├── bet_sizes: small(0.33), medium(0.66), large(1.0), overbet(1.5)
    │   ├── is_legal_bet_size() - Min/max bet validation
    │   └── get_legal_actions() - Context-aware action generation
    ├── **Action History Processing**
    │   ├── abstract_action_history() - Convert complex history to simple patterns
    │   ├── categorize_bet_size() - Classify bets by pot ratio
    │   └── Handle all-in detection (95% of stack threshold)
    ├── **PyPokerEngine Integration**
    │   ├── pypoker_to_cfr_actions() - Convert engine actions to CFR format
    │   ├── cfr_to_pypoker_action() - Convert CFR decisions to engine format
    │   └── Handle edge cases (all-in, insufficient stack)
    └── **Action Types Supported**
        ├── Basic: fold, call, check
        ├── Sized bets: bet_small, bet_medium, bet_large, bet_overbet
        ├── Sized raises: raise_small, raise_medium, raise_large, raise_overbet
        └── Special: all_in
    ↓
🧠 Blueprint CFR Training  
📁 `src/cfr/blueprint_trainer.py` ⚠️ **CREATE NEXT**
├── **Game Simulation Module**
│   ├── Deal random hands (hole cards + community cards)
│   ├── Simulate betting rounds using action abstraction
│   └── Use PyPokerEngine game logic or custom implementation
├── **CFR Algorithm Adaptation** 
│   ├── Port Leduc cfr() method to handle full poker
│   ├── Use enhanced abstractions for info set keys
│   ├── Handle 4 betting rounds (preflop/flop/turn/river)
│   └── Scale reach probabilities across streets
├── **Integration Points**
│   ├── Use CardAbstraction.get_bucket() for info set keys
│   ├── Use ActionAbstraction.get_legal_actions() for available moves
│   ├── Use HandEvaluator for terminal node evaluation
│   └── Use ActionAbstraction.abstract_action_history() for history tracking
└── **Training Management**
    ├── Progress tracking and strategy saving
    ├── Memory management for large strategy sets
    └── Training resumption from checkpoints
    ↓
💾 Blueprint Strategy Storage
📁 `strategies/blueprint_strategy.pkl`
└── Foundational strategy for all game situations
```

## Phase 2: Runtime (Gameplay with Subgame Solving)

```
🎮 **RUNTIME PHASE - GAMEPLAY WITH SUBGAME SOLVING**
    ↓
🎮 PyPokerEngine
📁 `main.py` or `test_integration.py`
├── Game Setup & Rules (setup_config)
├── Card Dealing & Game State Management  
└── Player Method Calls
    ↓
📨 Player.declare_action(valid_actions, hole_card, round_state)
📁 `src/bot/player.py` (current file)
    ↓
🤖 Enhanced Player (Two-Level Decision Making)
📁 `src/bot/player.py` (enhanced)
├── **Level 1: Blueprint Consultation**
│   ├── Load blueprint strategy from `strategies/blueprint_strategy.pkl`
│   ├── Quick lookup for standard situations
│   └── Default strategy for most decisions
│
├── **Level 2: Subgame Solving Decision** 
│   ├── Detect if opponent deviated from expected play
│   ├── Check if computational budget allows subgame solving  
│   └── Choose: Use blueprint OR solve subgame
    ↓
🔄 GameAdapter.create_info_set_key()
📁 `src/bot/game_adapter.py` (current) ✅
├── Convert PyPokerEngine format → CFR format
├── Extract betting patterns 
├── Coordinate abstractions (now uses hand evaluation)
└── Create enhanced info set keys
    ↓
📊 Enhanced Abstractions (Runtime)
📁 `src/abstractions/card_abstraction.py` (enhanced)
├── **Uses Hand Evaluation**: Now calls hand_evaluator functions
├── **Postflop Logic**: "strong"/"medium"/"draw"/"bluff" buckets
├── **No Complex EMD**: Simple equity-based approach per video
└── 📁 `src/abstractions/action_abstraction.py` (current) ✅
    ↓
🔀 **DECISION BRANCH**
    ↓                                    ↓
📘 **PATH A: Blueprint Strategy**    🎯 **PATH B: Subgame Solving**
📁 `src/cfr/information_set.py`      📁 `src/solving/subgame_solver.py` 
(current) ✅                         (needs creation)
├── Load trained strategy            ├── **Safe Subgame Solving**[1]
├── get_average_strategy()           ├── Add adversarial root node[1] 
├── Fast lookup                      ├── Calculate "opt out" values[1]
└── Standard situations              ├── **Depth-Limited Solving**[2]
                                    ├── Look ahead 2-4 streets max
                                    ├── **Finer-Grained Abstraction**
                                    ├── No card abstraction in subgame
                                    ├── Dense action abstraction  
                                    ├── Real-time CFR iterations
                                    └── Returns improved strategy
    ↓                                    ↓
    └─────────🔄 **PATHS CONVERGE** ─────────┘
                        ↓
🎯 Enhanced Action Selection
📁 `src/bot/player.py` (select_action_from_strategy)
├── Probabilistic selection from strategy weights
├── random.choices(actions, weights=strategy)  
├── **Blueprint**: Fast, reliable decisions
├── **Subgame**: Optimized for specific situation
└── Select optimal CFR action format
    ↓
🔄 GameAdapter.cfr_to_pypoker_action()
📁 `src/bot/game_adapter.py` (current) ✅
├── Convert CFR action → PyPokerEngine format
├── Calculate bet amounts (pot multipliers)
├── Handle edge cases (all-in, insufficient stack)
└── Return (action_name, amount)
    ↓
📤 Return to PyPokerEngine
├── Receive (action, amount) tuple
├── Validate action legality
├── Update game state
├── **Track Opponent Deviations** (triggers subgame solving)
├── Progress to next player/street
└── Continue game loop
    ↓
🔄 **Loop Continues** (with Learning)
├── Next player's turn OR
├── Next street (flop/turn/river) OR  
├── Hand completion & results
└── **Opponent Action Analysis** (for future subgame triggers)
```

## Key Files to Create/Enhance
### Immediate Priority (Hand Evaluation)

```
📁 `src/abstractions/hand_evaluator.py` ⚠️ **CREATE FIRST**
└── Foundation for all poker logic

📁 `src/abstractions/card_abstraction.py` ⚠️ **ENHANCE postflop_bucket()**  
└── Update to use hand evaluator
```

### Blueprint Training

```
📁 `src/cfr/blueprint_trainer.py` ⚠️ **CREATE NEXT**
└── Adapt Leduc CFR logic for full poker

📁 `strategies/blueprint_strategy.pkl` 
└── Trained foundational strategy
```

### Advanced (Subgame Solving)

```
📁 `src/solving/subgame_solver.py` 📅 **FUTURE**
└── Real-time strategy improvement

📁 `src/solving/safe_subgame_solver.py` 📅 **ADVANCED**  
└── Implements adversarial root + opt-out values[1]
```

### Implementation Order

Hand Evaluator → Foundation for everything\
Enhanced Card Abstraction → Uses hand evaluator\
Blueprint Trainer → Adapt Leduc CFR\
Load Blueprint in Player → Two-level decision making\
Subgame Solver → Advanced real-time improvement

### Summary Table

| Component | File | Role |
| --------- | ---- | ---- |
| HandEvaluator | src/abstractions/hand_evaluator.py | Evaluates hand strength |
| CardAbstraction | src/abstractions/card_abstraction.py | Buckets hands for abstraction |
| ActionAbstraction | src/abstractions/action_abstraction.py | Buckets actions, history, and legal moves |
| GameAdapter | src/bot/game_adapter.py | Bridges PyPokerEngine/game state to abstractions/infosets |
| InformationSet | src/cfr/information_set.py | Stores regrets/strategies per info set |
| BlueprintTrainer | src/cfr/blueprint_trainer.py | Runs CFR training, orchestrates above components |
| StrategyManager | src/cfr/strategy_manager.py | Saves/loads strategies |
| Player | src/bot/player.py | Main bot, uses all above during gameplay |

#### References
[1: Depth-Limited Solving for Imperfect Information Games](https://dl.acm.org/doi/10.5555/3327757.3327865)\
[2: Safe and Nested Subgame Solving for Imperfect-Information Games](https://proceedings.neurips.cc/paper_files/paper/2017/file/7fe1f8abaad094e0b5cb1b01d712f708-Paper.pdf)