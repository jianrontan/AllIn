# Training Flow

One CFR+ training iteration, end to end. Player 0 = SB/button, player 1 = BB.
Value flows back **from player 0's perspective** throughout. For the bigger
picture see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md); for commands see
[../USER_GUIDE.md](../USER_GUIDE.md).

```
🚀 run_blueprint_trainer.run_training(N)            # tests/run_blueprint_trainer.py
    ↓
🎮 BlueprintTrainer.train_blueprint(N, db=BlueprintDB(...))
    │  alternates updating_player = i % 2 each iteration
    ↓
📇 Deal Random Cards
BlueprintTrainer.deal_random_hand()
├── Player 0: ['HA', 'SK']        # engine SuitRank format
├── Player 1: ['DQ', 'CJ']
└── Community: ['SA', 'HK', 'CQ', 'HJ', 'D2']
    ↓
🧠 CFR Algorithm (external-sampling MCCFR+, DCFR discounting)
BlueprintTrainer.cfr(..., updating_player)
├── updating player: explore EVERY legal action, update regrets
├── opponent:        SAMPLE one action from current strategy
└── returns value from player 0's perspective
    ↓
🎯 Get Legal Actions (stack-aware)
PokerGame.get_legal_actions(street, history, pot, player, stacks…)
├── Input: history=['check'], street=1 (flop)
├── Output: ['check', 'bet_small', 'bet_medium', 'bet_large']
└── Unaffordable sized bets collapse to 'allin'; max 3 aggressions/street
    ↓
🔑 Build Info-Set Key   (via the single source of truth)
keys.make_info_set_key(street, position, preflop_bucket, strength, pattern)
├── CardAbstraction.get_bucket(cards, None)        → preflop bucket  e.g. pf_12
├── CardAbstraction.get_bucket(cards, board[:3])   → strength bucket e.g. 6
├── position = 'ip' (P0) | 'oop' (P1)
├── pattern  = current-street betting only (resets each street)  e.g. "k"
└── Output (postflop): "pf_12_6_ip_flop_k"
    ↓
🔍 Hand / Board Evaluation
HandEvaluator.get_raw_hand_value(hole, board)       # via phevaluator
└── feeds the 8 postflop texture buckets (0 bluff … 7 near-nuts)
    ↓
💾 Strategy (regret matching, CFR+)
InformationSet.get_strategy(legal_actions)          # PURE, no side effects
├── regrets floored at 0; normalized to probabilities
└── e.g. check=0%, bet_small=20%, bet_large=80%
        ↓ (opponent nodes only)
   InformationSet.accumulate_strategy(...)           # builds the average strategy
    ↓
🔄 Recursive Exploration
BlueprintTrainer.cfr(...) on each child
├── updating player: sum regret-weighted child values, update cumulative_regrets
│                    (DCFR discount applied once per info set per iteration)
└── opponent:        recurse only into the sampled action
    ↓
🏁 Terminal Evaluation (player-0 perspective)
PokerGame.get_utility(p0, p1, community, history, street, pot, p0_inv, p1_inv)
├── fold:     folder forfeits; winner takes pot − own contribution
├── showdown: HandEvaluator compares; all-ins run the board out
└── tie:      split the pot
    ↓
💽 Checkpoint
BlueprintDB.save_batch(...) every `checkpoint_every` iterations
└── analysis/blueprint_<timestamp>.db   (resume-able; total_iterations tracked)
```

**Notes**

- Keys are built **only** through `cfr/keys.py` so the trainer and every reader
  (API, evaluator, future subgame solver) stay in sync.
- A key whose legal-action set varies across visits (different stack depths) still
  merges correctly because regrets/strategy are keyed by action name — an action
  only accrues on visits where it was legal. See the M1 limitation in
  [DEVELOPER_GUIDE.md §11](DEVELOPER_GUIDE.md#11-known-limitations).
- After a run, `config.resolve_blueprint_path()` auto-selects the DB with the most
  iterations — no manual promotion step.
