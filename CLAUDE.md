# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AllIn is a heads-up Texas Hold'em poker AI using Monte Carlo CFR+ (Counterfactual Regret Minimization). The trained blueprint strategy lives in a SQLite database under `backend/bot/analysis/blueprints/`, is served by a Flask API, and powers two React frontend features: a strategy explorer and an interactive game against the bot.

### Storage: JSON → SQLite

The blueprint was originally exported as a single `analysis/blueprint.json` file, imported directly into the frontend bundle at build time. That was replaced with SQLite (`analysis/blueprints/blueprint_<timestamp>.db`) because:

- The blueprint grew to ~26k info sets; bundling the JSON bloated the Vite build and shipped the whole strategy to every visitor.
- SQLite supports **incremental checkpointing and resume** during long training runs (`BlueprintDB.save_batch` / `load_all_to_memory`).
- WAL mode + read-only connections let the API and bot **read a blueprint while training is still writing** a separate run.

The frontend no longer imports the blueprint at all — it queries the API, which reads the DB server-side.

## Commands

### Backend API (Flask, port 5000)
```bash
# Must be run from backend/api/ so the sys.path insertion resolves correctly
cd backend/api
python strategy_api.py
```

### Frontend (React/Vite, port 5173)
```bash
cd frontend
npm run dev       # Start dev server
npm run build     # Production build
npm run lint      # ESLint
```

### Training
```bash
# Run from backend/bot/
cd backend/bot

# Quick test
python -c "from tests.run_blueprint_trainer import run_training; run_training(100)"

# New full run — creates a timestamped DB, e.g. analysis/blueprints/blueprint_20260518_160906.db
python -c "from tests.run_blueprint_trainer import run_training; run_training(5000000)"

# Resume an existing run
python -c "from tests.run_blueprint_trainer import run_training; run_training(50000, resume='blueprint_20260518_160906.db')"
```

Training writes a timestamped `analysis/blueprints/blueprint_*.db`. There is **no manual promotion step** — `src/config.py:resolve_blueprint_path()` automatically selects the active blueprint (see Key Constraints).

### Tests
```bash
cd backend/bot
python tests/test_game_session.py          # Game core: GameSession, SessionStore, bot strategy
python tests/test_cfr_correctness.py       # CFR algorithm correctness + chip-conservation fuzz
python tests/test_poker_game_properties.py # Hypothesis property tests for engine invariants
python tests/test_player.py                # CFR_Bot vs RandomPlayer via PyPokerEngine
python tests/test_custom_betting.py        # Unrestricted custom bets + action translation
```

Or run the whole suite under pytest from `backend/bot/`:
```bash
python -m pytest tests/ -q
```

`test_poker_game_properties.py` uses [Hypothesis](https://hypothesis.readthedocs.io/)
(in `requirements.txt`). It can run via the built-in `__main__` console runner
(above) or under pytest: `python -m pytest tests/test_poker_game_properties.py -v`.

### Evaluation (exploitability)
```bash
# Run from backend/bot/. Scores how exploitable a blueprint is (lower = better),
# in milli-big-blinds/hand, via a vectorized best-response walk of the public tree.
python tests/run_evaluation.py                  # active blueprint, 400 board samples
python tests/run_evaluation.py --samples 1000
python tests/run_evaluation.py --db analysis/blueprints/blueprint_20260518_160906.db
```

## Architecture

### Data Flow

```
Training:
BlueprintTrainer.train_blueprint()
  → cfr() [Monte Carlo CFR+ with external sampling]
  → InformationSet (regret/strategy storage)
  → BlueprintDB.save_batch() checkpoints into analysis/blueprints/blueprint_<timestamp>.db

Strategy explorer (read path):
Frontend (HandExplorer / KeyExplorer)
  → Flask API (/api/strategy, /api/strategy/from-hand)
  → BlueprintDB.get_record() looks up the info-set key

Play against the bot:
Frontend (AiGame.jsx)
  → Flask API (/api/game/new | /action | /next-hand)
  → GameSession drives a hand through PokerGame
  → BlueprintStrategy queries the blueprint for the bot's moves
```

### Information Set Keys

The system revolves around a string key that uniquely identifies a poker situation. The format **includes position** (added so in-position and out-of-position play are learned separately):

- **Preflop**: `{bucket}_{position}_{pattern}`
  Example: `pf_13_ip_` (a strong preflop bucket, in position, no actions yet)

- **Postflop**: `{startBucket}_{strength}_{position}_{street}_{pattern}`
  Example: `pf_9_5_ip_turn_m` (preflop bucket pf_9 → strength bucket 5, in position, turn, opponent bet medium)

- `position`: `ip` (button/SB, acts last postflop) or `oop` (BB).
- `pattern`: betting actions **on the current street only** (resets each street). Characters: `k`=check, `c`=call, `f`=fold, `s`=small bet/raise, `m`=medium, `l`=large, `a`=all-in.

### Card Abstractions (`backend/bot/src/abstractions/card_abstractions.py`)

Earlier versions used a handful of named buckets (`premium_pair`, `monster`, etc.). These were replaced with finer, equity- and texture-driven buckets for sharper strategy resolution:

- **15 preflop buckets** — `pf_0` (weakest) … `pf_14` (strongest), assigned from precomputed Monte Carlo equity (`scripts/compute_preflop_equity.py`). `pf_14` is TT+.
- **Distribution-aware (potential-aware) postflop buckets** — **12 flop / 12 turn / 10 river** (`PostflopV2`). Each hand is described by the *distribution* of its equity-vs-uniform-range over board runouts (a histogram), and clustered by Earth Mover's Distance. This separates hands with equal current equity but different *trajectories* (a static made hand vs a polarized draw) — which the old single-axis scheme merged. Pipeline: `scripts/compute_postflop_buckets.py` fits centroids (`analysis/abstractions/postflop_centroids_*.npz`); `scripts/bake_postflop_table.py` bakes a canonical-situation→bucket lookup table (`analysis/abstractions/postflop_table_{flop,turn}.npz`) via suit isomorphism (`src/abstractions/canonical.py`); river is computed at runtime (1-D equity vs a uniform range → spike histogram → nearest river centroid, cached). **This replaced the old 8-bucket `BoardTextureEvaluator` heuristic — blueprints must be (re)trained under it; v1 blueprints are incompatible.**
  - **Committed vs regenerated:** the small **centroids** (`postflop_centroids_*.npz`) are the real inputs and are committed to git. The large **baked tables** (`postflop_table_*.npz` — turn is ~126MB) are **git-ignored** (`.gitignore`) and regenerated from the centroids by running `python scripts/bake_postflop_table.py --street {flop,turn}` from `backend/bot/`. A fresh clone must re-bake before training/inference, or `PostflopV2` falls back to slow per-situation lazy bucketing (it warns once). The **river table is intentionally not baked** (~90M canonical situations → impractical size); river always uses the cached runtime path.
  - **Stale-table guard:** each baked table is stamped with a hash of the centroids it was built from (+ K, bins) via `postflop_features.centroid_hash`; `PostflopV2._verify_stamp` checks it on load — a mismatch (centroids regenerated without re-baking) is a hard error, a legacy stamp-less table warns and proceeds. So you can't silently train/infer on a stale table.

### Action Abstractions (`backend/bot/src/abstractions/action_abstractions.py`)

**Sizes are the single source of truth in `src/abstractions/sizing.py`** — the engine
(`poker_game.py`), the eval harness (`lbr.py`), and the PyPokerEngine path
(`action_abstractions.py`) all import from it (a `tests/test_sizing_consistency.py` guards
against drift). Changing any size is an **abstraction change → retrain required** (existing
blueprints become incompatible).

- **Postflop** bet/raise: `small`=0.33x, `medium`=0.66x, `large`=1.0x pot (a raise is a
  fraction of the pot-after-call) + all-in. Overbets (>pot) are deliberately omitted and left
  to the Phase-4 subgame solver.
- **Preflop open** (first-in raise): BB-anchored ladder — `small`=2BB, `medium`=2.5BB,
  `large`=3.5BB. Small opens are GTO-optimal in heads-up; bigger human opens are handled at
  inference by action translation (`cfr/translation.py`), not by the bot's own ladder.
- **Preflop 3-bet AND 4-bet+** (unified): pot-relative, raise-to = `to_call + {0.66, 1.0, 1.5}
  × pot-after-call`. (The old absolute 3-bet ladder collapsed below the min-raise versus a
  large open; pot-relative scales so all three sizes stay legal at every open size.)
- An unrestricted/off-grid bet (custom human size, or an exploiter) maps onto this grid via
  pseudo-harmonic action translation (Phase 1a).

### CFR Training (`backend/bot/src/cfr/`)

- `blueprint_trainer.py` — `BlueprintTrainer.cfr()` implements Monte Carlo CFR+ with external sampling and Linear-CFR-style regret discounting (`alpha`). Updating player explores all actions; opponent samples one. Checkpoints into a `BlueprintDB`. (The discount is CFR+/Linear-CFR — `((t-1)/t)**alpha` on floored regrets, no negative-regret `beta`, per-role clocks — *not* canonical DCFR; see the trainer docstring.)
- `poker_game.py` — `PokerGame` handles game logic (independent of PyPokerEngine). Player 0 = SB/button, player 1 = BB. Max 3 bet/raise actions per street (1 bet + 2 raises). Handles stack constraints and all-ins.
- `information_set.py` — `InformationSet` stores cumulative regrets and strategy. CFR+ floors regrets at 0.
- `keys.py` — **single source of truth** for info-set key construction (`make_info_set_key`) and the action→pattern-character map (`action_char`). The trainer and every consumer that looks a situation up in the blueprint (the evaluation harness, a future subgame solver) build keys through this module so the two can never drift. Change the key format here and everywhere stays in sync.

### Evaluation (`backend/bot/src/evaluation/`)

Measurement harness for strategy quality — separate from the game/training code.

- `best_response.py` — `BestResponseEvaluator` computes a blueprint's **exploitability** (BR₀(σ₁) + BR₁(σ₀)) in mbb/hand via Monte Carlo board sampling. It walks the *public* betting tree once per board carrying a villain-reach vector over all hands and a per-hero-hand value vector, so one board sample integrates all hero hands × compatible villain hands at once (low variance, full-game best response with exact hero cards). This is the convergence scoreboard: run `tests/run_evaluation.py` before/after a change and watch the number drop. Builds lookup keys via `cfr/keys.py`.

### Storage (`backend/bot/src/storage/`)

- `blueprint_db.py` — `BlueprintDB` wraps the SQLite blueprint. `read_only=True` opens with SQLite's `mode=ro` so inference can read a file a training process holds open. Tables: `info_sets`, `training_metadata`.

### Game Core (`backend/bot/src/game/`)

Transport-agnostic engine for playing against the bot — **no Flask imports**, so it is reusable if the transport later changes (e.g. WebSockets for live online play).

- `game_session.py` — `GameSession` drives one full hand through `PokerGame`: deals a real deck, applies actions, advances streets, runs showdown. Fully JSON-serializable (all state in `self.data`). `advance_bot_turns()` runs the bot until it is the human's turn. Optionally maintains an opponent **range tracker** (below): given a `strategy_fn`, it `observe`s the human's actions and `reveal`s the board, persisting the belief in `self.data['opp_range']`; `bot_public_state()` hands the live tracker, the bot's `hole_cards`, and `to_call` to the strategy, and `public_view().botRead` surfaces the bot's read (confidence + top hands) to the UI.
- `range_tracker.py` — `RangeTracker` (Phase 3): a hand-level Bayesian belief over the **opponent's** hole cards. Per-hand weights over the C(50,2) combos, `reveal()` for card removal, `observe()` for Bayesian updates from a blueprint `strategy_fn` (the assumed opponent model), a **confidence** score in [0,1] that decays on off-model actions (`*= exp(-max(0, (-log p_a) - H))`, surprise vs the model's own entropy — correct for mixed strategies), and `hero_equity()` for the bot's equity vs the belief. JSON `to_dict`/`from_dict`. This is the input a future river subgame solver consumes. (The evaluation harness's `lbr.py:BotRange` is the older sibling of this class; not yet merged — see BUG_LOG cross-cutting note on drift.)
- `bot_strategy.py` — `BotStrategy` interface + `BlueprintStrategy` (blueprint lookup) + `ConfidenceAwareStrategy` (plays the blueprint while the range tracker is confident, falls back to an equity-vs-range decision when confidence collapses; `range_model_fn()` exposes the blueprint as the tracker's opponent model). The interface receives full public state (incl. the bot's `hole_cards`), not just the bucketed key, so a subgame-solving strategy is a drop-in.
- `session_store.py` — `SessionStore` interface + `InMemorySessionStore`. A Redis/DynamoDB-backed store would be a drop-in replacement for multi-process / AWS deployment.
- `cards.py` — deck plus conversion between **engine format** (`SuitRank`, e.g. `HA`) used internally and **display format** (`RankSuit`, e.g. `Ah`) used at the API/frontend boundary.

### Bot / Inference (`backend/bot/src/bot/`)

- `game_adapter.py` — `GameAdapter` converts hole cards + round state into info-set keys.
- `player.py` — `Player(BasePokerPlayer)` for PyPokerEngine games; loads the blueprint via `resolve_blueprint_path()` and samples actions from the stored average strategy.

### Frontend (`frontend/src/`)

- `api.js` — single API client module. Base URL is env-driven (`VITE_API_BASE`).
- `pages/Home.jsx` — landing page.
- `pages/StrategyLookup.jsx` — tab container for two independent tools:
  - `components/HandExplorer.jsx` — enter real cards + a betting line; `/api/strategy/from-hand` returns the key and strategy.
  - `components/KeyExplorer.jsx` — build an info-set key from abstraction dropdowns (or paste one); `/api/strategy` returns the strategy.
  - `components/StrategyResult.jsx` — shared result panel (shared component, but each tool keeps its own state).
- `pages/AiGame.jsx` — interactive heads-up game vs the bot; `components/PlayingCard.jsx` renders cards.

### API Endpoints (`backend/api/strategy_api.py`)

Strategy:
- `GET /api/strategy?key=` — blueprint strategy for an info-set key (`found:false` for untrained keys is a valid answer, not an error).
- `POST /api/strategy/from-hand` — derive the key from real cards + a betting line, then return the strategy.
- `GET /api/abstractions` — bucket/position/street/pattern vocabulary for the Key Explorer dropdowns.

Game:
- `POST /api/game/new` — start a session, deal the first hand.
- `GET /api/game/state?id=` — current redacted state.
- `POST /api/game/action` — apply the human action; the bot then responds.
- `POST /api/game/next-hand` — deal the next hand in a session.

Health: `GET /api/test` — reports status and the active blueprint.

### Environment Variables

- `ALLIN_BLUEPRINT_DB` — explicit path to the blueprint DB (overrides auto-resolution).
- `ALLIN_CORS_ORIGINS` — comma-separated allowed CORS origins (defaults to `localhost:5173`/`5174`).
- `VITE_API_BASE` — frontend API base URL (defaults to `http://localhost:5000`).

## Git

Never add, commit, or push code in this repository, or any commands that is unsafe, read only commands are fine.

## Key Constraints

- The Flask API **must** be started from `backend/api/` — it uses `sys.path.insert(0, backend_dir)` so imports like `from bot.src.bot.game_adapter import GameAdapter` resolve correctly.
- The active blueprint is resolved by `src/config.py:resolve_blueprint_path()`: the `analysis/blueprints/blueprint_*.db` with the highest `total_iterations`, or whatever `ALLIN_BLUEPRINT_DB` points at. There is no `blueprint.json`/`blueprint.db` to maintain by hand.
- Inference always opens the DB with `read_only=True`; only training opens it read/write.
- Each hand of `GameSession` starts both players at `STARTING_STACK` (the blueprint assumes ~200 effective). Cross-hand profit/loss is tracked separately in `human_net`.
- Stakes: `STARTING_STACK = 200` chips with SB=1 / BB=2, i.e. **100 BB effective stacks** — standard heads-up depth. Starting pot is always 3 chips. Pot math is in chips throughout the backend; the frontend displays everything in BB (chips ÷ 2).
- Card formats: engine `SuitRank` (`HA`) internally, display `RankSuit` (`Ah`) at the boundary — convert with `src/game/cards.py`.

## Roadmap Notes

The overall build plan is a phased dependency chain (status as of 2026-05-24). The key
constraint is **Phase 3 must precede Phase 4**: the river solver needs a hand-level range
as input, so the range tracker has to exist before the solver.

- **Phase 0 — Measurement harness** ✅ done. LBR + head-to-head/AIVAT + best-response
  exploitability (`src/evaluation/`); baselined at ~11,256 (BR) / ~3,636 (LBR) mbb/hand.
  Built first so every later change is *scored, not guessed*.
- **Phase 1b — strategy-sum discount (gamma)** ✅ done. Linear-CFR-style avg-strategy
  discount (`gamma=2`) with its own opponent-node clock (`information_set.py`,
  `blueprint_trainer.py`). (Not canonical DCFR — see the trainer docstring.)
- **Phase 1a — Pseudo-harmonic action translation** ✅ done (2026-05-26), shipped together
  with unrestricted custom human bet sizing. Inference-only (no retraining). `cfr/translation.py`
  blends the two bracketing grid sizes (Ganzfried-Sandholm) over a per-node grid; consumed by
  `BlueprintStrategy._state_distribution` (live bot) and mirrored in `lbr.py`'s victim model.
  The human can now bet any legal chip amount via `bet_custom_<total>`/`raise_custom_<total>`
  (engine stores the raise-to total; `{action, amountBb}` API contract; custom-BB box in the UI
  beside the size buttons). Result: LBR **3609 → 1670 mbb/hand** (~54% cut) on the v2 9.15M
  snapshot. Tests: `tests/test_custom_betting.py`.
- **Phase 2 — Potential-aware postflop buckets** 🔄 in progress. This is the distribution-aware
  abstraction work, tracked internally as sub-phases A (cluster centroids) ✅ / B (bake
  canonical→bucket tables + migrate `CardAbstraction` to `PostflopV2`) 🔄 / C (retrain
  blueprint on v2 + re-measure BR/LBR) ⬜. Replaces the old 8-bucket heuristic with 12/12/10.
- **Phase 3 — Hand-level Bayesian range tracker** ⬜. Hand-level (not bucket-level) range,
  hooked into `GameSession` JSON state, with a confidence score that decays on off-tree
  actions. **Prerequisite for Phase 4.** (A bucket-level prototype already exists in
  `evaluation/lbr.py`'s `BotRange`.)
- **Phase 4 — River endgame solver (unsafe, no gadget)** ⬜. The flagship: `RiverSubgameSolver(BotStrategy)`
  builds the small river betting tree, takes both ranges from Phase 3, runs vectorized CFR+,
  reads off the action for the bot's actual hand; falls back to the blueprint on flop/turn.
  Requires injecting the bot's hole cards into `decide()` (`bot_strategy.py:21` does not
  currently receive them) via a per-hand `hand_getter`. v1 is *unsafe* (theoretically
  exploitable via the frozen-range trap); accept that for v1.
- **Phase 5 — Safety + depth** ⬜ (multi-week; only after 0–4 prove out). 5a: reach/gadget
  for the river solver (provably no-more-exploitable than the blueprint). 5b: turn/flop
  depth-limited solving with **blueprint counterfactual values as the leaf value function**
  (needs storing those values — new data — plus the 5a gadget).

Dependency chain: **0 → 1 → 2 → 3 → 4 → 5** (1a and 1b are independent quick wins; 2 is
high-leverage but technically optional before 3; 3 strictly gates 4).

Deferred / low-priority: opponent modeling (only coarse aggregate stats — aggression %,
fold-to-c-bet — off by default; per-bucket modeling is infeasible in one human session due
to data sparsity); strategy purification (cheap inference A/B once the harness exists);
widening the betting abstraction (4th size / 3rd raise) only if LBR reveals missing-line leaks.

Productionization (additive, the interfaces already exist for these):
- Online 1v1 play deployed on AWS — swap `InMemorySessionStore` for a Redis/DynamoDB-backed
  store; the Flask-free game core and `BotStrategy` / `SessionStore` interfaces make this
  additive, not a rewrite.
- Unrestricted human bet sizing — ✅ done (2026-05-26, with Phase 1a). The API accepts
  `{action: 'bet_custom'|'raise_custom', amountBb}`; the engine stores the raise-to chip total
  in `history` as `bet_custom_<total>`/`raise_custom_<total>` (chip-conservation tests intact),
  and off-grid bets are handled by action translation (see Phase 1a above).
