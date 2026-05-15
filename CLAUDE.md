# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AllIn is a heads-up Texas Hold'em poker AI using Monte Carlo CFR+ (Counterfactual Regret Minimization). The trained blueprint strategy is stored in `backend/bot/analysis/blueprint.json` and served via a Flask API and queried directly from the React frontend.

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

# Quick test (100 iterations)
python -c "from tests.run_blueprint_trainer import run_enhanced_training; run_enhanced_training(100)"

# Full production training (~5 hours, 4M iterations)
python tests/run_blueprint_trainer.py

# The active blueprint must be manually copied/symlinked to analysis/blueprint.json
# Timestamped outputs: analysis/blueprint_unified_{N}iter_{timestamp}.json
```

### Tests
```bash
cd backend/bot
python tests/test_player.py           # Run CFR_Bot vs RandomPlayer via PyPokerEngine
python tests/test_blueprint_trainer.py  # Test training pipeline and strategy export
python tests/test_confidence_detection.py  # Run EnhancedBlueprintTrainer analysis
```

## Architecture

### Data Flow

```
Training:
BlueprintTrainer.train_blueprint()
  → cfr() [Monte Carlo CFR+ with external sampling]
  → InformationSet (regret/strategy storage)
  → export to analysis/blueprint_unified_*.json
  → copy to analysis/blueprint.json (active model)

Inference:
Flask API (/api/evaluate-hand)
  → GameAdapter.create_info_set_key()
  → returns infoSetKey to frontend

Frontend (StrategyLookup.jsx)
  → imports blueprint.json directly at build time
  → looks up strategy[infoSetKey].average_strategy
```

### Information Set Keys

The entire system revolves around a string key that uniquely identifies a poker situation:

- **Preflop**: `{card_bucket}_{betting_pattern}`  
  Example: `ace_king_sml` (ace-king hand, small bet then medium bet then large bet)

- **Postflop**: `{starting_hand}_{current_strength}_{street}_{betting_pattern}`  
  Example: `ace_king_strong_flop_km` (ace-king preflop → strong postflop hand, flop, check then medium bet)

Betting pattern characters: `k`=check, `c`=call, `f`=fold, `s`=small bet/raise, `m`=medium, `l`=large

### Card Abstractions (`backend/bot/src/abstractions/card_abstractions.py`)

**8 preflop buckets**: `premium_pair` (AA/KK/QQ), `medium_pair` (JJ-99), `small_pair` (88-22), `ace_king`, `strong_ace`, `ace_x`, `broadway`, `suited_connector`, `weak`

**6 postflop buckets** (based on phevaluator hand strength): `monster` (≥7), `strong` (≥5), `medium` (≥2), `weak_made` (pair), `draw`, `bluff`

### Action Abstractions (`backend/bot/src/abstractions/action_abstractions.py`)

Bet sizes mapped to: `small`=0.33x pot, `medium`=0.66x pot, `large`=1.0x pot

Preflop bet sizing differs from postflop:
- Opens: 3BB/5BB/7BB → small/medium/large
- 3-bets: 6BB/10BB/14BB → small/medium/large  
- 4-bets+: pot-relative (0.66x/1.33x/2.0x)

### CFR Training (`backend/bot/src/cfr/`)

- `blueprint_trainer.py` — `BlueprintTrainer.cfr()` implements Monte Carlo CFR+ with external sampling. Updating player explores all actions; opponent samples a single action.
- `poker_game.py` — `PokerGame` handles game logic for training (independent of PyPokerEngine). Max 4 bet/raise actions per street (1 bet + 3 raises).
- `information_set.py` — `InformationSet` stores cumulative regrets and strategy. CFR+ floors regrets at 0.

### Bot / Inference (`backend/bot/src/bot/`)

- `game_adapter.py` — `GameAdapter` converts hole cards + round state into info set keys.
- `player.py` — `Player(BasePokerPlayer)` loads `analysis/blueprint.json` at startup and uses stored regrets as the average strategy for action selection.

### Frontend Pages (`frontend/src/pages/`)

- `Home.jsx` — landing page
- `StrategyLookup.jsx` — imports `blueprint.json` directly as a static asset; calls Flask API to get the info set key, then looks it up in the bundled JSON
- `AiGame.jsx` — interactive game page

### API Endpoints (`backend/api/strategy_api.py`)

- `POST /api/evaluate-hand` — returns `infoSetKey`, `cardBucket`, `strengthBucket`, `actionPattern`
- `POST /api/get-legal-actions` — returns legal CFR actions for a given game state
- `GET /api/test` — health check

CORS is configured to allow `localhost:5173` and `localhost:5174`.

## Git

Never run any git commands in this repository.

## Key Constraints

- The Flask API **must** be started from `backend/api/` — it uses `sys.path.insert(0, backend_dir)` so imports like `from bot.src.bot.game_adapter import GameAdapter` resolve correctly.
- `analysis/blueprint.json` is the active model file. Timestamped training outputs must be manually promoted to this filename.
- `StrategyLookup.jsx` imports `blueprint.json` via a relative path (`../../../backend/bot/analysis/blueprint.json`) at Vite build time — large blueprint files increase bundle size significantly.
- Starting pot in training is always 3 chips (SB=1 + BB=2). All pot calculations are in chips, but the frontend displays in BB (divides by 2).
