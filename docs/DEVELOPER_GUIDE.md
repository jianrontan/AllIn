# AllIn — Developer Guide

> **Audience**: Anyone returning to this codebase after time away, or a new contributor.
> This guide explains what every module does and how data flows through the system.
> For day-to-day commands (install, train, run, test) see [`USER_GUIDE.md`](../USER_GUIDE.md);
> for the canonical short reference see [`CLAUDE.md`](../CLAUDE.md); for the running
> bug history see [`backend/bot/docs/BUG_LOG.md`](../backend/bot/docs/BUG_LOG.md).

---

## Table of Contents

1. [What Is AllIn?](#1-what-is-allin)
2. [Repository Structure](#2-repository-structure)
3. [Core Concepts](#3-core-concepts)
4. [Module Reference](#4-module-reference)
5. [Data Flow: Training](#5-data-flow-training)
6. [Data Flow: Inference (Two Paths)](#6-data-flow-inference-two-paths)
7. [The Info Set Key — How It Works](#7-the-info-set-key--how-it-works)
8. [Blueprint Storage — SQLite Schema](#8-blueprint-storage--sqlite-schema)
9. [Evaluation: Measuring Quality](#9-evaluation-measuring-quality)
10. [Testing Strategy](#10-testing-strategy)
11. [Known Limitations](#11-known-limitations)
12. [PlantUML Diagrams](#12-plantuml-diagrams)

---

## 1. What Is AllIn?

AllIn is a heads-up (2-player) No-Limit Texas Hold'em poker AI built around
**Monte Carlo CFR+ (Counterfactual Regret Minimization Plus)**. It has three layers:

| Layer | Stack | Role |
|---|---|---|
| Training | Python (MCCFR+) | Self-plays millions of abstracted hands, accumulates regrets, converges on a near-optimal strategy table |
| Game core / Bot | Pure Python | Drives a live hand and looks up the bot's move in the strategy table |
| UI / API | React + Flask | Interactive strategy explorer and a play-vs-bot frontend |

The strategy training produces is the **blueprint**: a lookup table mapping a
compact description of a situation (the *information set key*) to a probability
distribution over actions (e.g. fold 10% / call 45% / bet 45%). The bot samples
from that distribution to play.

> **Big architectural fact:** the blueprint is stored in **SQLite**
> (`backend/bot/analysis/blueprints/blueprint_<timestamp>.db`), not a JSON file. SQLite gives
> incremental checkpoint/resume during long training runs and lets the API read a
> blueprint (WAL + read-only) while a separate training run is still writing. The
> frontend no longer bundles the blueprint at all — it queries the API.

---

## 2. Repository Structure

```
AllIn/
├── backend/
│   ├── requirements.txt                    # Python deps (incl. hypothesis for tests)
│   ├── bot/
│   │   ├── analysis/                        # Trained blueprints: blueprint_<ts>.db (+ -wal/-shm)
│   │   ├── scripts/
│   │   │   └── compute_preflop_equity.py    # Precompute preflop bucket equities
│   │   ├── docs/
│   │   │   └── BUG_LOG.md                    # Running log of fixed correctness bugs
│   │   ├── src/
│   │   │   ├── config.py                     # resolve_blueprint_path() — picks the active DB
│   │   │   ├── abstractions/
│   │   │   │   ├── card_abstractions.py      # 15 preflop + 12/12/10 postflop (delegates to PostflopV2)
│   │   │   │   ├── postflop_v2.py            # Distribution-aware postflop buckets (table lookup + river runtime)
│   │   │   │   ├── postflop_features.py      # Shared: equity dist, EMD, rank7, board_winrates, centroid_hash
│   │   │   │   ├── canonical.py              # Suit-isomorphism canonicalisation of (hole, board)
│   │   │   │   ├── action_abstractions.py    # Bet sizing + PyPokerEngine⇄CFR conversion
│   │   │   │   └── hand_evaluator.py         # Wraps phevaluator (via postflop_features.rank7)
│   │   │   ├── cfr/
│   │   │   │   ├── blueprint_trainer.py      # MCCFR+ training loop
│   │   │   │   ├── poker_game.py             # Stack-aware abstracted rules engine
│   │   │   │   ├── information_set.py        # Per-situation regret/strategy storage
│   │   │   │   └── keys.py                   # Single source of truth for info-set keys
│   │   │   ├── storage/
│   │   │   │   └── blueprint_db.py           # SQLite wrapper (BlueprintDB)
│   │   │   ├── game/                         # Transport-agnostic live-hand engine (no Flask)
│   │   │   │   ├── game_session.py           # GameSession — drives one hand; maintains opp range tracker
│   │   │   │   ├── range_tracker.py          # RangeTracker — hand-level Bayesian opponent range + confidence
│   │   │   │   ├── bot_strategy.py           # BotStrategy iface + BlueprintStrategy + ConfidenceAwareStrategy
│   │   │   │   ├── session_store.py          # SessionStore iface + InMemorySessionStore
│   │   │   │   └── cards.py                  # Deck + engine⇄display card conversion
│   │   │   ├── bot/
│   │   │   │   ├── player.py                 # PyPokerEngine player (used by test_player)
│   │   │   │   └── game_adapter.py           # PyPokerEngine round_state → info-set key
│   │   │   ├── subgame/                      # Subgame-solving scaffolding (roadmap)
│   │   │   │   ├── confidence_detector.py
│   │   │   │   ├── off_tree_detector.py
│   │   │   │   ├── subgame_detector.py
│   │   │   │   └── player_blueprint_adapter.py
│   │   │   └── evaluation/                   # Measurement harness (scored, not guessed)
│   │   │       ├── best_response.py          # In-abstraction exploitability (BR)
│   │   │       ├── lbr.py                     # Local Best Response — off-tree lower bound (+ BotRange)
│   │   │       ├── match.py                   # Head-to-head match runner
│   │   │       └── aivat.py                   # AIVAT variance-reduced head-to-head estimator
│   │   └── tests/
│   │       ├── run_blueprint_trainer.py      # Training entrypoint (new run / resume)
│   │       ├── run_evaluation.py             # Exploitability CLI
│   │       ├── test_cfr_correctness.py       # CFR correctness + chip-conservation fuzz
│   │       ├── test_poker_game_properties.py # Hypothesis property tests
│   │       ├── test_game_session.py          # Game core
│   │       ├── test_player.py                # Bot vs RandomPlayer (PyPokerEngine)
│   │       └── test_custom_betting.py        # Custom bets + action translation
│   └── api/
│       └── strategy_api.py                   # Flask REST API (start from backend/api/)
└── frontend/
    └── src/
        ├── api.js                            # Single API client (VITE_API_BASE)
        ├── pages/{Home,StrategyLookup,AiGame}.jsx
        └── components/{HandExplorer,KeyExplorer,StrategyResult,PlayingCard}.jsx
```

---

## 3. Core Concepts

### Monte Carlo CFR+ with external sampling

CFR learns a near-optimal strategy by repeatedly self-playing and tracking
*regret* — how much better a player could have done by deviating to a different
action. The **average strategy** over all iterations converges to a Nash
equilibrium. This codebase uses three refinements:

- **CFR+** — cumulative regrets are floored at 0, which speeds convergence.
- **External sampling MCCFR** — each iteration designates one *updating player*.
  At the updating player's nodes we explore *every* legal action; at the
  opponent's nodes we *sample a single* action from the current strategy. This
  turns a full O(|A|^depth) tree walk into roughly O(|A| × depth) per iteration.
- **Discounted CFR+ (Linear-CFR-style, *not* canonical DCFR)** — regrets are
  discounted over time by `((t-1)/t)**alpha`, applied once per info set per
  iteration on first visit, on top of the CFR+ zero-floor. There is no separate
  negative-regret `beta`, and the alpha/gamma clocks advance on each role's own
  visit counts rather than a global `t`. Valid and convergent; just not the
  canonical `t^a/(t^a+1)` DCFR scheme (see `blueprint_trainer.py` for the full note).

**Sign convention:** `cfr()` returns value from **player 0's perspective**
throughout. At a decision node the sign is flipped to compute the acting
player's regret, then flipped back on return. (Getting this convention wrong was
the original correctness bug; it is now consistent — see BUG_LOG.md.)

### Information sets

An *information set* is everything the acting player can observe: their own
(bucketed) hole cards, the (bucketed) board strength, the street, their
position, and the betting actions so far. Many concrete game states collapse to
one info set; CFR stores one strategy entry per info-set key.

### Card abstraction (`abstractions/card_abstractions.py`)

Finer, equity- and texture-driven buckets (the old hand-named buckets are gone):

- **15 preflop buckets** — `pf_0` (weakest) … `pf_14` (strongest), assigned from
  precomputed Monte Carlo equity (`scripts/compute_preflop_equity.py`).
  `pf_14` ≈ TT+.
- **Distribution-aware (potential-aware) postflop buckets** — integers, **12 flop /
  12 turn / 10 river** (`PostflopV2`). Each hand is described by the *distribution* of
  its equity-vs-uniform-range over board runouts (a 30-bin histogram) and clustered by
  Earth Mover's Distance, so hands with equal current equity but different *trajectories*
  (a static made hand vs a polarized draw) get different buckets — which the old
  single-axis 8-bucket board-texture heuristic merged. Pipeline:
  `scripts/compute_postflop_buckets.py` fits centroids → `scripts/bake_postflop_table.py`
  bakes a canonical-situation→bucket table (suit-isomorphism via `canonical.py`,
  centroid-stamped) → `postflop_v2.py` does O(log n) table lookups (flop/turn) and exact
  runtime equity (river). **Replaced the old `BoardTextureEvaluator` (now dead code);
  v1 blueprints are incompatible and must be retrained.**

### Action abstraction (`abstractions/sizing.py` + `action_abstractions.py`)

Sizes live in **`abstractions/sizing.py`** (single source of truth; the engine, the LBR
harness, and the PyPokerEngine path all import it — `tests/test_sizing_consistency.py` guards
against drift). Changing a size is an **abstraction change → retrain required**.

Postflop bet sizes are three fractions of the pot:
`small`=0.33× · `medium`=0.66× · `large`=1.0× (a raise is a fraction of pot-after-call).
Preflop differs:

- **Opens** (first-in raise): BB-anchored — 2 / 2.5 / 3.5 BB → small / medium / large.
- **3-bets AND 4-bets+** (unified): pot-relative, raise-to = `to_call + {0.66, 1.0, 1.5} ×
  pot-after-call`. (Replaces the old absolute 3-bet ladder, which collapsed below the
  min-raise versus a large open — only `large` stayed legal.)
- Off-grid bets (custom human sizes, exploiter bets) map onto this grid via pseudo-harmonic
  action translation (`cfr/translation.py`).

Plus `check`, `call`, `fold`, and `allin`. The engine allows at most **3 sized
aggression actions per street** (1 bet + 2 raises). When a sized bet/raise would
cost a player their whole remaining stack, it collapses to `allin`.

### The blueprint

The output of training is a SQLite DB. Each row is one info set: its cumulative
regrets and cumulative strategy. The *average strategy* (normalized cumulative
strategy) is what inference reads.

---

## 4. Module Reference

### `cfr/keys.py` — info-set key construction *(single source of truth)*

`make_info_set_key(street, position, preflop_bucket, postflop_strength,
bet_pattern)` builds the canonical key, and `action_char(action)` maps an action
to its pattern character. **The trainer and every consumer (the evaluation
harness, a future subgame solver) build keys through this module so they can
never drift.** Change the key format here and everywhere stays in sync.

### `cfr/poker_game.py` — `PokerGame`

A self-contained, stack-aware abstracted rules engine, independent of
PyPokerEngine. Player 0 = SB/button (acts first preflop), player 1 = BB (acts
first postflop). Responsibilities:

- `get_legal_actions(...)` — legal actions for a street/history, **respecting
  remaining stack** (`_apply_stack_constraints` replaces unaffordable sized
  bets with `allin` using exact chip cost).
- `is_terminal` / `is_round_complete` — terminal and round-complete detection,
  including `allin`-then-`call`/`fold`.
- `calculate_current_pot`, `get_player_contribution_this_round`,
  `get_call_amount_from_history`, `_action_cost`, `_allin_amount` — the pot /
  contribution / cost arithmetic. All memoized via `_calc_cache`.
- `get_utility(...)` — chip gain/loss **from P0's perspective** at terminal
  (fold or showdown; all-ins run the board out).

> Several subtle correctness bugs lived in this file's contribution/call math
> (double-counting a re-raise, all-in call costing 0). They are fixed and
> regression-tested; see BUG_LOG.md and `test_poker_game_properties.py`.

### `cfr/information_set.py` — `InformationSet`

Stores `cumulative_regrets` and `cumulative_strategy` (both dicts keyed by
action name) plus discount bookkeeping. Three deliberately separate operations:

- `get_strategy(legal_actions)` — **pure** CFR+ regret matching over *this
  visit's* legal actions. No side effects.
- `accumulate_strategy(legal_actions, strategy)` — adds into the running average.
  Called **only at opponent nodes**, where external sampling supplies the right
  reach weighting (so contributions are added unweighted).
- `get_average_strategy(legal_actions)` — normalizes over `cumulative_strategy`
  for the requested actions. **Reads cumulative_strategy, never a stale stored
  action list** — this was the BUG-001 fix.

Because the dicts are keyed by action, a key whose legal-action set varies
across visits (a postflop key spanning different stack depths) still merges
correctly: an action only accrues regret/strategy on the visits where it was
legal. (See [§11](#11-known-limitations) for the abstraction caveat this implies.)

### `cfr/blueprint_trainer.py` — `BlueprintTrainer`

Orchestrates the training loop. Each iteration deals a random hand and runs
`cfr()` with the updating player alternating (`i % 2`). Builds info-set keys via
`keys.make_info_set_key`. Persists through `BlueprintDB.save_batch` /
`checkpoint_to_db`, and supports `resume_from_db` to continue a run.

### `storage/blueprint_db.py` — `BlueprintDB`

SQLite wrapper. `read_only=True` opens with SQLite `mode=ro` so inference can
read a file a training process holds open. Tables: `info_sets`,
`training_metadata`. `save_batch` checkpoints incrementally;
`load_all_to_memory` rehydrates for resume; `get_average_strategy` /
`get_record` are the inference read path.

### `config.py` — `resolve_blueprint_path()`

Picks the active blueprint automatically: the `analysis/blueprints/blueprint_*.db` with the
highest `total_iterations` (that isn't actively being written), or whatever
`ALLIN_BLUEPRINT_DB` points at. **No manual promotion step.**

### `game/` — transport-agnostic live-hand engine

No Flask imports, so it is reusable if the transport changes (e.g. WebSockets).

- `game_session.py` — `GameSession` drives one full hand through `PokerGame`:
  deals a real deck, applies actions, advances streets, runs showdown. Fully
  JSON-serializable (all state in `self.data`). `advance_bot_turns()` runs the
  bot until it is the human's turn. When given a `strategy_fn`, it also maintains
  a per-hand opponent **range tracker** (observe on human actions, reveal on
  streets) in `self.data['opp_range']`, and exposes the bot's read via
  `public_view().botRead`.
- `range_tracker.py` — `RangeTracker` (Phase 3): a hand-level Bayesian belief over
  the opponent's hole cards (per-hand weights, card removal, Bayesian updates from
  a blueprint opponent-model `strategy_fn`, a confidence score that decays on
  off-model play, and `hero_equity` vs the belief). The input a future river
  subgame solver consumes. (`evaluation/lbr.py:BotRange` is the older sibling;
  not yet merged.)
- `bot_strategy.py` — `BotStrategy` interface + `BlueprintStrategy` (blueprint
  lookup) + `ConfidenceAwareStrategy` (blueprint while the range tracker is
  confident; equity-vs-range fallback when confidence collapses). The interface
  receives full public state — including the bot's `hole_cards` — not just the
  bucketed key, so a subgame-solving strategy is a drop-in replacement.
- `session_store.py` — `SessionStore` interface + `InMemorySessionStore` (a
  Redis/DynamoDB store would drop in for multi-process / AWS).
- `cards.py` — deck plus conversion between **engine format** (`SuitRank`, e.g.
  `HA`) used internally and **display format** (`RankSuit`, e.g. `Ah`) at the
  API/frontend boundary.

### `bot/` — PyPokerEngine path *(test harness only)*

- `player.py` — `Player(BasePokerPlayer)` for PyPokerEngine games; loads the
  blueprint via `resolve_blueprint_path()` and samples from the average strategy.
- `game_adapter.py` — converts PyPokerEngine hole cards + round state into
  info-set keys. **Used by `test_player.py`**, not by the Play-vs-AI product
  path (that uses `GameSession`/`PokerGame` directly).

### `evaluation/best_response.py` — `BestResponseEvaluator`

Computes the blueprint's **exploitability** (how much a perfect counter-strategy
beats it). See [§9](#9-evaluation-measuring-quality).

### `abstractions/hand_evaluator.py` — `HandEvaluator`

Wraps `phevaluator` for O(1) hand strength. Used by the card abstraction and by
showdown utility.

---

## 5. Data Flow: Training

```
run_blueprint_trainer.run_training(N)         # tests/run_blueprint_trainer.py
  BlueprintTrainer.train_blueprint(N, db=...)
    for i in range(N):
      deal_random_hand()                      → p0_cards, p1_cards, community
      cfr(..., updating_player = i % 2)        # P0-perspective value
        PokerGame.get_legal_actions(street, history, pot, player, stacks…)
        keys.make_info_set_key(street, position, preflop_bucket, strength, pattern)
          CardAbstraction.get_bucket(cards, board)   # 15 preflop / 12-12-10 postflop
        InformationSet.get_strategy(legal_actions)   # CFR+ regret matching (pure)
        [updating player] explore all actions → recurse, update regrets (discount, floor 0)
        [opponent]        sample one action  → recurse, accumulate_strategy
    BlueprintDB.save_batch(...) every `checkpoint_every` iterations
```

The active blueprint is then chosen automatically by `resolve_blueprint_path()`.

---

## 6. Data Flow: Inference (Two Paths)

There are **two** inference paths. The product (Play vs AI) does **not** use
PyPokerEngine.

### Path A — Play vs AI (product)

```
Frontend AiGame.jsx
  → Flask /api/game/{new,action,next-hand}
    → GameSession (game/game_session.py) drives the hand through PokerGame
      → BlueprintStrategy.decide(info_set_key, legal_actions, public_state)
        → BlueprintDB.get_average_strategy(key)   # read-only SQLite
      → sample an action, apply, advance, settle
```

### Path B — PyPokerEngine (test harness)

```
PyPokerEngine → Player.declare_action(valid_actions, hole_card, round_state)
  → GameAdapter.create_info_set_key(hole_card, round_state)
  → BlueprintDB.get_average_strategy(key)
  → ActionAbstraction.cfr_to_pypoker_action(...) → (action, amount)
```

### Strategy explorer (read path)

```
Frontend HandExplorer / KeyExplorer
  → Flask /api/strategy or /api/strategy/from-hand
    → BlueprintDB.get_record(key)   # found:false for untrained keys is valid
```

---

## 7. The Info Set Key — How It Works

The key uniquely identifies a situation from the acting player's perspective. It
**includes position** so in-position and out-of-position play are learned
separately. Build keys only via `keys.make_info_set_key`.

### Preflop
```
{preflop_bucket}_{position}_{pattern}

Example: "pf_13_ip_"
  preflop_bucket = pf_13     (strong bucket)
  position       = ip        (button/SB) | oop (BB)
  pattern        = ""        (no actions yet this street)
```

### Postflop
```
{preflop_bucket}_{strength}_{position}_{street}_{pattern}

Example: "pf_9_5_ip_turn_m"
  preflop_bucket = pf_9      (the starting-hand bucket, fixed for the hand)
  strength       = 5         (this street's postflop bucket; 0–11 flop/turn, 0–9 river)
  position       = ip
  street         = turn
  pattern        = m         (opponent bet medium)
```

- `position`: `ip` (button/SB, acts last postflop) or `oop` (BB).
- `pattern`: betting actions **on the current street only** — it **resets each
  street**. Characters: `k`=check, `c`=call, `f`=fold, `s`=small bet/raise,
  `m`=medium, `l`=large, `a`=all-in.

Keeping both the starting-hand bucket and the current strength bucket postflop
bakes a form of range-awareness into the abstraction (a strong board for a
premium starting range plays differently than the same board for a weak range).

---

## 8. Blueprint Storage — SQLite Schema

The blueprint lives in `backend/bot/analysis/blueprints/blueprint_<timestamp>.db` (WAL
mode, so you may also see `-wal` / `-shm` sidecar files during/after a run).

- **`info_sets`** — one row per info-set key, storing the cumulative regrets and
  cumulative strategy (the average strategy is derived by normalizing the latter
  at read time — see `InformationSet.get_average_strategy` and BUG-001).
- **`training_metadata`** — run-level metadata, notably `total_iterations`,
  which `resolve_blueprint_path()` uses to select the active blueprint.

Inference always opens with `read_only=True`; only training opens read/write.
There is no `blueprint.json`/`blueprint.db` to maintain by hand.

---

## 9. Evaluation: Measuring Quality

`evaluation/best_response.py` measures **exploitability** =
BR₀(σ₁) + BR₁(σ₀): seat each side in turn as a "hero" who best-responds while
the other plays the blueprint. A true Nash equilibrium scores 0; a high number
means training hasn't converged or the abstraction is leaky.

It walks the **public** betting tree once per board, carrying a villain-reach
vector over all hands and a per-hero-hand value vector, so one board sample
integrates all hero hands × compatible villain hands at once (low variance,
full-game best response with the hero's exact cards). Card removal is handled at
terminals via O(H) per-card running sums.

Run it before and after a change and watch the number drop:

```bash
cd backend/bot
python tests/run_evaluation.py --samples 1000      # active blueprint
```

Results are in milli-big-blinds per hand (mbb/hand); lower is better.

---

## 10. Testing Strategy

| Test | What it covers |
|---|---|
| `test_cfr_correctness.py` | CFR invariants (regrets ≥ 0, average strategy sums to 1, EV finite/near-zero in self-play) + a hand-rolled random-playout fuzz with chip-conservation and call-cost invariants |
| `test_poker_game_properties.py` | **Hypothesis** property-based tests over a `session_walk` strategy: chip conservation, call/contribution arithmetic, all-in semantics, legal-action shape, street symmetry, terminal/utility bounds, **zero-sum (F2) + terminal pot conservation (F3)**. Shrinks failures to minimal counter-examples |
| `test_game_session.py` | Game core: `GameSession`, `SessionStore`, bot strategy |
| `test_player.py` | PyPokerEngine path: bot vs `RandomPlayer` (asserts chip conservation over a full match) |
| `test_custom_betting.py` | Unrestricted custom bet sizing + pseudo-harmonic action translation |
| `test_storage_and_resolve.py` | `resolve_blueprint_path` selection + `BlueprintDB` checkpoint/resume round-trip |
| `test_action_abstraction_roundtrip.py` | Engine bet sizing ↔ `categorize_bet_size` round-trip (postflop bets, preflop opens) |
| `test_aivat.py` | AIVAT river control variate is zero-mean (unbiasedness) |
| `test_eval_api_smoke.py` | Best-response tree-walk smoke + Flask health endpoint |
| `test_best_response_vectorized.py`, `test_lbr_equity.py`, `test_lbr_range.py`, `test_canonical.py` | Oracle checks (BR showdown vs brute force, LBR equity, canonicaliser) — now collected by pytest |

**Lesson baked into the suite (see BUG_LOG.md):** internally-consistent bugs
(e.g. a call costing 0 chips, which preserves chip conservation on both sides)
slip past aggregate invariants. So each primitive — call, bet, raise, all-in —
has its own *semantic* invariant tied to what the action means in poker, not just
to chip totals. Property-based fuzzing found bugs that two static audits missed.

Run via the built-in console runner (`python tests/<file>.py`) or pytest. The
property tests require `hypothesis` (in `requirements.txt`).

---

## 11. Known Limitations

The "known bugs" that previously filled this section are **fixed** — see
[`backend/bot/docs/BUG_LOG.md`](../backend/bot/docs/BUG_LOG.md) for the full
history (sign convention, multi-visit discount decay, average-strategy readout,
contribution double-count, all-in call cost). What remains are *abstraction
limitations*, not bugs:

### M1 — postflop key omits pot / stack depth (SPR)

The postflop info-set key encodes street, buckets, position, and the
current-street pattern, but **not** the pot size or effective stack depth
(stack-to-pot ratio). Concrete nodes with very different SPRs therefore collapse
to the same key. Consequences:

- CFR converges to the equilibrium of the *abstract* game — a reach-weighted
  blend across the merged states — not the true optimum of any single state.
- Rarely-legal actions get **diluted**: e.g. an `allin` that is correct only in
  short-stack spots accrues strategy on those visits but is averaged against the
  many deep-stack visits where it wasn't legal, so its readout probability
  understates how often you'd actually shove in the short spot.

This is **deferred to the subgame-solving phase**: either add a stack-depth/SPR
bucket to the postflop key, or solve the spot at runtime where the real stacks
are known. The `subgame/` package and the `BotStrategy` interface exist so this
is additive, not a rewrite.

### Inference still depends on PyPokerEngine for the test harness

`test_player.py` runs the bot through a forked PyPokerEngine (heads-up turn-order
fix). The product path (`GameSession`) is already PyPokerEngine-free; eventually
the test harness could drop it too and drive `PokerGame` directly.

---

## 12. PlantUML Diagrams

> ⚠️ **These diagrams predate the SQLite migration, the abstraction overhaul
> (15 preflop / 12-12-10 distribution-aware postflop), the position-aware keys,
> the range tracker, and the `game/` engine.** They still
> convey the broad shape (training vs inference, key assembly) but the class
> names, bucket names, JSON storage, and "bug annotations" are stale. Treat
> §2–§9 above as authoritative and regenerate the `.puml` sources before relying
> on the images. Source files are in [diagrams/](diagrams/).

| Diagram | Source | What it shows |
|---|---|---|
| System Architecture | [diagrams/system_architecture.puml](diagrams/system_architecture.puml) | How Frontend, API, Training, and Bot connect |
| Class Diagram | [diagrams/class_diagram.puml](diagrams/class_diagram.puml) | Classes, fields, methods, relationships |
| Training Sequence | [diagrams/training_sequence.puml](diagrams/training_sequence.puml) | CFR+ iteration step by step |
| Inference Sequence | [diagrams/inference_sequence.puml](diagrams/inference_sequence.puml) | A live decision, end to end |
| Info Set Key | [diagrams/infoset_key.puml](diagrams/infoset_key.puml) | How a key is assembled from cards + history |

---

*Last updated: 2026-05-22 — rewritten for the SQLite blueprint, equity/texture
buckets, position-aware keys, the Flask-free `game/` engine, `cfr/keys.py`, and
the exploitability evaluator.*
