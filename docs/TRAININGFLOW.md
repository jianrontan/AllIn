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
🧠 CFR Algorithm (external-sampling MCCFR+, Linear-CFR-style discounting)
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
├── CardAbstraction.get_bucket(cards, board[:3])   → postflop bucket e.g. 6
│     └── PostflopV2: distribution-aware (potential-aware) bucket —
│         canonicalise (hole,board) → O(log n) lookup in the pre-baked
│         centroid table (flop/turn); river = exact equity → spike → nearest
│         river centroid. Prod v1 = 20 flop / 16 turn / 10 river; v2 = 30/24/10 (dev-served;
│         see the Abstraction note below). Computed once per hand
│         (memoized), lazily on first use of each street.
├── position = 'ip' (P0) | 'oop' (P1)
├── pattern  = current-street betting only (resets each street)  e.g. "k"
└── Output (postflop): "pf_12_6_ip_flop_k"
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
│                    (discount applied once per info set per iteration)
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
└── analysis/blueprints/blueprint_<timestamp>.db   (resume-able; total_iterations tracked)
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
- ⚠️ **`EV(cum)` / `EV(round)` / `EV(sess)` measure the CURRENT regret-matched iterate, which
  CFR does NOT drive to convergence** (it can cycle at a large, seat-imbalanced value forever —
  see BUG-013). They are a *gauge of the evolving strategy*, NOT the served blueprint and NOT a
  strength metric. `EV(cum)` is the lifetime mean (`ev_sum`/`ev_count`, persisted/restored on
  resume); in the **parallel** trainer it reads high and lags (each worker measures the stale
  round-start strategy) so it is labelled `EV(cum,lagged)`, with `EV(round)`/`EV(round,ema)` as
  fresher views. **The dial to watch is `EV(served, avg strategy)`** (printed at every checkpoint,
  `BlueprintTrainer.evaluate_served_ev`) — the self-play value of the *average* strategy you
  actually serve; it settles to a small constant (the button's game-value edge). Even that is only
  a seat-balance/convergence sanity check — **for true strength use the evaluation harness (LBR/BR)**,
  which best-responds; a seat-balanced-but-weak strategy self-plays near the game value too.
- **`shape:` line** (printed at every checkpoint next to `EV(served)`, `src/cfr/strategy_shape.py`) is a
  per-decision **collapse detector** — it reads `OK`, `WARN`, or `COLLAPSE` and reports weak-hand open
  fold%, the open-size concentration, the pf_0-vs-strongest fold gradient, and the BB-vs-5BB fold%. It
  exists because no aggregate metric (EV/LBR/AIVAT) catches a *balanced-but-degenerate* strategy — it
  was added after BUG-014, where a blueprint quietly trained to "open one size with 100% of hands, never
  fold the button" for a week. A healthy run shows weak hands folding a lot with a wide strength gradient;
  `COLLAPSE` means a weak bucket folds <5% while playing one size >75%. Run it on any DB:
  `python scripts/check_strategy_shape.py [--db <path>] [--verbose]` (exit code 2 on COLLAPSE).
- **Selecting the ship snapshot.** A run keeps a snapshot per `TRACK_EVERY`. The scoreboard
  `~/progress.txt` gives **BR** (paired across snapshots — seed 42, same boards — so its
  checkpoint-to-checkpoint deltas are trustworthy) and **LBR** (NOT paired by default: the victim
  *samples* its action and desyncs the deal stream, so LBR swings between checkpoints are noise —
  use `paired=True` to compare two snapshots). Aggregate BR is **reach-weighted toward the
  best-response line**, so it understates RARE-line quality that matters against humans. Two
  read-only probes diagnose a snapshot: `scripts/probe_seat_ev.py` (EV(current) vs EV(served=average),
  split by seat) and `scripts/probe_rare_line_convergence.py` (visit-tail + pairwise strategy-drift
  by position × rarity tier — *is the rare/OOP tail still converging?*). **Don't pick the ship
  snapshot by the BR minimum alone** — BR can bottom while rare lines are still improving; confirm
  with a **human-like match eval** (`scripts/run_maniac_live.py` / `compare_gadget_policies.py --aivat`
  vs maniac/calling-station), which weights lines by human reach.
- Hand evaluation goes through `postflop_features.rank7`, which precomputes card→id ids
  and calls phevaluator's internal evaluator directly (skips per-call string parsing);
  river equity is computed via the vectorized `board_winrates` shared across both
  players on a board. Together these made training ~1.87× faster.
- **Abstraction (v2):** preflop is now **lossless — 169 fine
  buckets** (`NUM_PREFLOP_BUCKETS`), one per canonical hand, collapsed to coarse-10 for postflop
  keys. The postflop bucket count K is **defined by the committed centroids** (`postflop_centroids_*.npz`);
  prod v1 is 20 flop / 16 turn / 10 river, and v2 is **30 / 24 / 10** (dev-served; assets-v2 cutover pending)
  (set by `train_on_cloud.sh:FLOP_BUCKETS/TURN_BUCKETS`). Changing K is an abstraction change → re-fit +
  re-bake + retrain; the stamp guard (`PostflopV2._verify_stamp`) hard-errors on a centroid↔table mismatch.

---

## Parallel & cloud training (large runs)

A full blueprint is a multi-day run, so it's trained **data-parallel** across all cores
(`src/cfr/parallel_trainer.py`) and usually on a cloud box. The per-iteration math above is
unchanged; parallelism only changes *how iterations are batched and merged*:

```
run_blueprint_trainer.py --workers N --merge-every 2000 --menu-mode capped [--gamma 1.0] [--resume <db>]
    ↓  per ROUND (merge_every × N iters):
    ├── master pickles the current blueprint → broadcasts to N workers
    ├── each worker runs `merge_every` external-sampling MCCFR+ iters on the FROZEN round-start strategy
    ├── master `merge_round()` folds the N regret/strategy deltas back in (block Linear-CFR discount,
    │     per-worker CFR+ floor + one master floor) and advances the per-key discount clock once
    └── every `checkpoint_every` it writes the DB + prints EV(served) + the `shape:` collapse line
```

- **Quality vs the single-thread oracle:** parallel is an *approximation* (block discount; workers
  don't see each other's mid-round updates), validated by exploitability — see the `parallel_trainer.py`
  docstring. `merge_every=2000` matches the served blueprint (lower = closer to single-thread, more
  overhead). `EV(cum)` reads high/lagged here (labelled `EV(cum,lagged)`) — watch `EV(served)` + BR/LBR.
- **Resume is exact and discount-safe:** the per-key Linear-CFR clocks live in the DB rows and persist;
  alpha/gamma/menu/training-mode mismatches hard-error on resume. Long runs are trained in chunks that
  each `--resume` the same DB.
- **One-shot cloud driver:** `backend/bot/scripts/train_on_cloud.sh` provisions deps, re-fits + bakes the
  postflop tables, trains in `TRACK_EVERY` chunks with **BR + LBR after each** (→ `~/progress.txt`, the
  convergence scoreboard), and bundles the **5 same-generation serving artifacts** (blueprint renamed to
  `blueprint_final.db` + 2 centroids + 2 baked tables) to `~/result/`. It self-detects root vs a sudo user
  (Hetzner / AWS EC2), forces unbuffered logs, and prints `RSS/sysRAM` per round (needs `psutil`).
- **Operational runbook** (provision a 16 vCPU / 64 GB box → ssh → run → monitor → collect → tear down,
  with the gotchas: `nohup`/`tmux`, the speed-test DB, account limits, cost) is in the private
  `docs/private/EC2_TRAINING_RUNBOOK.md`.
