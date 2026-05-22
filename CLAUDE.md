# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AllIn is a heads-up Texas Hold'em poker AI using Monte Carlo CFR+ (Counterfactual Regret Minimization). The trained blueprint strategy lives in a SQLite database under `backend/bot/analysis/`, is served by a Flask API, and powers two React frontend features: a strategy explorer and an interactive game against the bot.

### Storage: JSON → SQLite

The blueprint was originally exported as a single `analysis/blueprint.json` file, imported directly into the frontend bundle at build time. That was replaced with SQLite (`analysis/blueprint_<timestamp>.db`) because:

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

# New full run — creates a timestamped DB, e.g. analysis/blueprint_20260518_160906.db
python -c "from tests.run_blueprint_trainer import run_training; run_training(5000000)"

# Resume an existing run
python -c "from tests.run_blueprint_trainer import run_training; run_training(50000, resume='blueprint_20260518_160906.db')"
```

Training writes a timestamped `analysis/blueprint_*.db`. There is **no manual promotion step** — `src/config.py:resolve_blueprint_path()` automatically selects the active blueprint (see Key Constraints).

### Tests
```bash
cd backend/bot
python tests/test_game_session.py     # Game core: GameSession, SessionStore, bot strategy
python tests/test_cfr_correctness.py  # CFR algorithm correctness checks
python tests/test_player.py           # CFR_Bot vs RandomPlayer via PyPokerEngine
python tests/test_blueprint_trainer.py
```

## Architecture

### Data Flow

```
Training:
BlueprintTrainer.train_blueprint()
  → cfr() [Monte Carlo CFR+ with external sampling]
  → InformationSet (regret/strategy storage)
  → BlueprintDB.save_batch() checkpoints into analysis/blueprint_<timestamp>.db

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
- **8 postflop buckets** — integers `0`–`7` from `BoardTextureEvaluator`, combining phevaluator hand strength with board danger: `0` pure bluff, `1` weak draw, `2` strong draw, `3` combo draw, `4` weakest made, `5` medium made, `6` strong made, `7` near-nuts.

### Action Abstractions (`backend/bot/src/abstractions/action_abstractions.py`)

Bet sizes mapped to: `small`=0.33x pot, `medium`=0.66x pot, `large`=1.0x pot. Preflop sizing differs:
- Opens: 3BB/5BB/7BB → small/medium/large
- 3-bets: 9BB/12BB/16BB → small/medium/large
- 4-bets+: pot-relative (0.66x/1.33x/2.0x)

### CFR Training (`backend/bot/src/cfr/`)

- `blueprint_trainer.py` — `BlueprintTrainer.cfr()` implements Monte Carlo CFR+ with external sampling and DCFR regret discounting (`alpha`). Updating player explores all actions; opponent samples one. Checkpoints into a `BlueprintDB`.
- `poker_game.py` — `PokerGame` handles game logic (independent of PyPokerEngine). Player 0 = SB/button, player 1 = BB. Max 3 bet/raise actions per street (1 bet + 2 raises). Handles stack constraints and all-ins.
- `information_set.py` — `InformationSet` stores cumulative regrets and strategy. CFR+ floors regrets at 0.

### Storage (`backend/bot/src/storage/`)

- `blueprint_db.py` — `BlueprintDB` wraps the SQLite blueprint. `read_only=True` opens with SQLite's `mode=ro` so inference can read a file a training process holds open. Tables: `info_sets`, `training_metadata`.

### Game Core (`backend/bot/src/game/`)

Transport-agnostic engine for playing against the bot — **no Flask imports**, so it is reusable if the transport later changes (e.g. WebSockets for live online play).

- `game_session.py` — `GameSession` drives one full hand through `PokerGame`: deals a real deck, applies actions, advances streets, runs showdown. Fully JSON-serializable (all state in `self.data`). `advance_bot_turns()` runs the bot until it is the human's turn.
- `bot_strategy.py` — `BotStrategy` interface + `BlueprintStrategy` (blueprint lookup). The interface receives full public state, not just the bucketed key, so a future subgame-solving strategy is a drop-in.
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
- The active blueprint is resolved by `src/config.py:resolve_blueprint_path()`: the `analysis/blueprint_*.db` with the highest `total_iterations`, or whatever `ALLIN_BLUEPRINT_DB` points at. There is no `blueprint.json`/`blueprint.db` to maintain by hand.
- Inference always opens the DB with `read_only=True`; only training opens it read/write.
- Each hand of `GameSession` starts both players at `STARTING_STACK` (the blueprint assumes ~200 effective). Cross-hand profit/loss is tracked separately in `human_net`.
- Stakes: `STARTING_STACK = 200` chips with SB=1 / BB=2, i.e. **100 BB effective stacks** — standard heads-up depth. Starting pot is always 3 chips. Pot math is in chips throughout the backend; the frontend displays everything in BB (chips ÷ 2).
- Card formats: engine `SuitRank` (`HA`) internally, display `RankSuit` (`Ah`) at the boundary — convert with `src/game/cards.py`.

## Roadmap Notes

- Future goal: online 1v1 play deployed on AWS, with subgame solving added to the bot. The `SessionStore` / `BotStrategy` interfaces and the Flask-free game core exist so those are additive, not rewrites.
- Also planned (post subgame-solving): unrestricted human bet sizing. The action contract is kept a thin `{action, size}` object so widening it to `{action, amount}` is a localized change in `GameSession` and the API.
