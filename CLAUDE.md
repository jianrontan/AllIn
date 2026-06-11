# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AllIn is a heads-up Texas Hold'em poker AI using Monte Carlo CFR+ (Counterfactual Regret Minimization). The trained blueprint strategy lives in a SQLite database under `backend/bot/analysis/blueprints/`, is served by a Flask API, and powers two React frontend features: a strategy explorer and an interactive game against the bot.

**Production status: live at https://allin.jianrontan.com as of v1.0.0.** Backend runs on AWS Lightsail Containers (Flask + gunicorn), frontend on Cloudflare Pages, state in DynamoDB (sessions, players, leaderboard, hand recaps), auth via Cognito + Google IdP, edge via Cloudflare (DNS, CDN, WAF rate limiting). CI/CD via GitHub Actions (`.github/workflows/backend-deploy.yml` + `frontend-deploy.yml`); the blueprint DB + postflop tables ship as GitHub Release assets (release `assets-v1`).

### Storage: JSON → SQLite

The blueprint was originally exported as a single `analysis/blueprint.json` file, imported directly into the frontend bundle at build time. That was replaced with SQLite (`analysis/blueprints/blueprint_<timestamp>.db`) because:

- The blueprint grew to ~128k info sets (in the served `blueprint_final.db`); bundling the JSON bloated the Vite build and shipped the whole strategy to every visitor.
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

- **Preflop**: `{fineBucket}_{position}_{pattern}` — uses the **fine** 30-bucket id.
  Example: `pf_27_ip_` (a strong preflop bucket — fine buckets run pf_0..pf_29 — in position, no actions yet)

- **Postflop**: `{coarseClass}_{strength}_{position}_{street}_{pattern}` — `startBucket` is the **coarse** 10-class id (`make_info_set_key` collapses the fine bucket; a postflop key never carries a fine id, so `coarseClass` ∈ pf_0..pf_9).
  Example: `pf_9_5_ip_turn_m` (coarse preflop class 9 → strength bucket 5, in position, turn, opponent bet medium)

- `position`: `ip` (button/SB, acts last postflop) or `oop` (BB).
- `pattern`: betting actions **on the current street only** (resets each street). Characters: `k`=check, `c`=call, `f`=fold, `s`=small bet/raise, `m`=medium, `l`=large, `o`=overbet (1.5x pot, postflop only), `2`=overbet2 (2.0x pot, postflop only, capped menu), `x`=xlarge open (5 BB, preflop open only), `a`=all-in. (Single source of truth: `cfr/keys.py:ACTION_CHARS`.)

### Card Abstractions (`backend/bot/src/abstractions/card_abstractions.py`)

Earlier versions used a handful of named buckets (`premium_pair`, `monster`, etc.). These were replaced with finer, equity- and texture-driven buckets for sharper strategy resolution:

- **Decoupled preflop buckets — 30 fine / 10 coarse** (imperfect recall, Libratus/Pluribus structure). Two independent equal-frequency quantilings of the *same* Monte Carlo equity table (`scripts/compute_preflop_equity.py`), both **derived at import** in `card_abstractions.py` from the committed `_PREFLOP_EQUITY` (no bucket literal to maintain):
  - **Fine (30)** — `pf_0` (weakest) … `pf_29` (strongest). Used in **preflop keys only** — sharp preflop play (preflop is just 169 hands, so resolution is cheap). `preflop_bucket()` returns this; `NUM_PREFLOP_BUCKETS = 30`.
  - **Coarse (10)** — class `0` … `9`. The preflop-hand summary carried into **postflop keys** as `startBucket`. `preflop_class()` returns it; `NUM_PREFLOP_COARSE = 10`. The fine→coarse collapse happens **inside `cfr/keys.make_info_set_key` for postflop streets** (via `card_abstractions.FINE_TO_COARSE`), so every caller just passes the fine bucket and a postflop key *cannot* carry a fine id. Because 30 = 3×10 over the same quantiles, the collapse is exact (no fine bucket straddles a coarse boundary; an assertion enforces it).
  - **Why:** `startBucket` multiplies the entire postflop info-set count. Carrying coarse (10) instead of fine (40 in the prior scheme) cuts postflop card-space ~3.7× while letting the flop strength buckets get *finer* (see below). The blueprint stays coarse on the river because the Phase-4 solver refines it there anyway. (Replaced an earlier single-bucket scheme — 15 then 40 — where one bucket did both jobs.)
- **Distribution-aware (potential-aware) postflop buckets** — **20 flop / 16 turn / 10 river** (`PostflopV2`; the per-street K is set by `scripts/compute_postflop_buckets.py --buckets`, baked into the centroids). Each hand is described by the *distribution* of its equity-vs-uniform-range over board runouts (a histogram), and clustered by Earth Mover's Distance. This separates hands with equal current equity but different *trajectories* (a static made hand vs a polarized draw) — which the old single-axis scheme merged. Pipeline: `scripts/compute_postflop_buckets.py` fits centroids (`analysis/abstractions/postflop_centroids_*.npz`); `scripts/bake_postflop_table.py` bakes a canonical-situation→bucket lookup table (`analysis/abstractions/postflop_table_{flop,turn}.npz`) via suit isomorphism (`src/abstractions/canonical.py`); river is computed at runtime (1-D equity vs a uniform range → spike histogram → nearest river centroid). The expensive per-board equity pass (`board_winrates`, ranks all 1081 hands on a board) is memoized on the **canonical (suit-isomorphic) board** in a process-global LRU cache (`postflop_v2._RIVER_BOARD_CACHE`, `OrderedDict`; cap `ALLIN_RIVER_CACHE_BOARDS`, default 100k ≈ 0.26GB/proc) — only 134,459 canonical 5-card boards exist vs ~2.6M concrete (19.3×), so over a long run `board_winrates` runs ~134k times total (≈1.63× faster training; module-global so it survives across parallel worker rounds; a cap below 134,459 degrades gracefully via LRU instead of thrashing). Equity is stored as an exact `uint16` numerator, so the cache is bit-identical to the uncached path (no bucket drift). **This replaced the old 8-bucket `BoardTextureEvaluator` heuristic — blueprints must be (re)trained under it; v1 blueprints are incompatible.**
  - **Committed vs regenerated:** the small **centroids** (`postflop_centroids_*.npz`) are the real inputs and are committed to git. The large **baked tables** (`postflop_table_*.npz` — turn is ~126MB) are **git-ignored** (`.gitignore`) and regenerated from the centroids by running `python scripts/bake_postflop_table.py --street {flop,turn}` from `backend/bot/`. A fresh clone must re-bake before training/inference, or `PostflopV2` falls back to slow per-situation lazy bucketing (it warns once). The **river table is intentionally not baked** (~90M canonical situations → impractical size); river always uses the cached runtime path.
  - **Stale-table guard:** each baked table is stamped with a hash of the centroids it was built from (+ K, bins) via `postflop_features.centroid_hash`; `PostflopV2._verify_stamp` checks it on load — a mismatch (centroids regenerated without re-baking) is a hard error, a legacy stamp-less table warns and proceeds. So you can't silently train/infer on a stale table.

### Action Abstractions (`backend/bot/src/abstractions/action_abstractions.py`)

**Sizes are the single source of truth in `src/abstractions/sizing.py`** — the engine
(`poker_game.py`), the eval harness (`lbr.py`/`match.py`/`cross_match.py`), the PyPokerEngine
path (`action_abstractions.py`), and the river subgame projection (`subgame/`) all import from
it (a `tests/test_sizing_consistency.py` guards against drift). Likewise, every pattern char comes
from `cfr/keys.py:action_char` (which RAISES on an unmapped action rather than defaulting to a
char — a silent default once aliased the real `xlarge` char and corrupted keys). Changing any
size is an **abstraction change → retrain required** (existing blueprints become incompatible).

- **Postflop** bet/raise: `small`=0.33x, `medium`=0.66x, `large`=1.0x, `overbet`=1.5x pot (a
  raise is a fraction of the pot-after-call) + voluntary all-in. The one `overbet` tier (1.5x,
  pattern char `o`) is the only overbet the blueprint trains; larger overbets (2x+) are left to
  the Phase-4 subgame solver's own menu.
- **Preflop open** (first-in raise): BB-anchored ladder — `small`=2BB, `medium`=2.5BB,
  `large`=3.5BB, `xlarge`=5BB. Small opens are GTO-optimal in heads-up; the `xlarge` tier
  (pattern char `x`) is **open-only** — a 4th anchor so big human opens translate against a
  trained bracket rather than clamping. It is NOT offered as a 3-bet/4-bet or postflop size.
- **Preflop 3-bet AND 4-bet+** (unified): pot-relative, raise-to = `to_call + {0.66, 1.0, 1.5}
  × pot-after-call` (3 sizes, no `xlarge`). (The old absolute 3-bet ladder collapsed below the
  min-raise versus a large open; pot-relative scales so all three sizes stay legal at every open
  size.) NOTE: the three node types no longer share one size list — open has 4 (incl. `xlarge`),
  3-bet/4-bet has 3 (`sizing.SIZES`), postflop has 4 (incl. `overbet`) — so iterate the relevant
  dict, not a shared tuple. (Review note 2026-05: the 3-bet/4-bet multipliers are FINE as-is —
  judged as 3-bet-to multiples of the open they are 2.32x / 3.0x / 4.0x, which brackets the GTO
  ~3-4x range well; all sizes stay legal at every open and re-raise depth, incl. a 0.5x 4-bet after
  a 1.5x 3-bet. A possible refinement is a 4th SMALL ~0.5x (≈2.0x open) tier on the 3-bet half only
  to capture the linear-3-bet branch — but justify it with a per-line BR/LBR leak measurement, not
  with node-count symmetry; it's an abstraction change → retrain. Do NOT add a 4th 4-bet size: at
  low SPR 4-bets compress toward one-size-or-jam.)
- An unrestricted/off-grid bet (custom human size, or an exploiter) maps onto this grid via
  pseudo-harmonic action translation (Phase 1a).

### CFR Training (`backend/bot/src/cfr/`)

- `blueprint_trainer.py` — `BlueprintTrainer.cfr()` implements Monte Carlo CFR+ with external sampling and Linear-CFR-style regret discounting (`alpha`). Updating player explores all actions; opponent samples one. Checkpoints into a `BlueprintDB`. (The discount is CFR+/Linear-CFR — `((t-1)/t)**alpha` on floored regrets, no negative-regret `beta`, per-role clocks — *not* canonical DCFR; see the trainer docstring.)
- `poker_game.py` — `PokerGame` handles game logic (independent of PyPokerEngine). Player 0 = SB/button, player 1 = BB. Bet/raise actions per street are capped by `max_raises_per_street` (default 2 → 1 bet + 2 raises = 3 aggressions, the cap the blueprint **trains** under); **LIVE play (`GameSession`) passes `float('inf')` to uncap re-raises** so a human can 5-bet/6-bet+ any amount on any street (training/eval keep the default). Handles stack constraints and all-ins.
- `information_set.py` — `InformationSet` stores cumulative regrets and strategy. CFR+ floors regrets at 0.
- `keys.py` — **single source of truth** for info-set key construction (`make_info_set_key`) and the action→pattern-character map (`action_char`). The trainer and every consumer that looks a situation up in the blueprint (the evaluation harness, a future subgame solver) build keys through this module so the two can never drift. Change the key format here and everywhere stays in sync. It also performs the **fine→coarse preflop collapse for postflop keys** (imperfect recall): callers always pass the fine bucket, and `make_info_set_key` maps it to the coarse class for any postflop street via `card_abstractions.FINE_TO_COARSE`.

### Evaluation (`backend/bot/src/evaluation/`)

Measurement harness for strategy quality — separate from the game/training code.

- `best_response.py` — `BestResponseEvaluator` computes a blueprint's **exploitability** (BR₀(σ₁) + BR₁(σ₀)) in mbb/hand via Monte Carlo board sampling. It walks the *public* betting tree once per board carrying a villain-reach vector over all hands and a per-hero-hand value vector, so one board sample integrates all hero hands × compatible villain hands at once (low variance). The hero best-responds with **exact cards** over the full public tree, but only among the **abstract grid bet sizes** — so the number is exact on cards/tree but restricted on sizing, i.e. a **lower bound** on true exploitability (a size-cheating adversary does better; that's what LBR probes). Good for tracking convergence and comparing blueprints — don't over-claim it as the true game value. This is the convergence scoreboard: run `tests/run_evaluation.py` before/after a change and watch the number drop. Builds lookup keys via `cfr/keys.py`.

### Storage (`backend/bot/src/storage/`)

- `blueprint_db.py` — `BlueprintDB` wraps the SQLite blueprint. `read_only=True` opens with SQLite's `mode=ro` so inference can read a file a training process holds open. Tables: `info_sets`, `training_metadata`.

### Game Core (`backend/bot/src/game/`)

Transport-agnostic engine for playing against the bot — **no Flask imports**, so it is reusable if the transport later changes (e.g. WebSockets for live online play).

- `game_session.py` — `GameSession` drives one full hand through `PokerGame`: deals a real deck, applies actions, advances streets, runs showdown. Fully JSON-serializable (all state in `self.data`). `advance_bot_turns()` runs the bot until it is the human's turn. Optionally maintains an opponent **range tracker** (below): given a `strategy_fn`, it `observe`s the human's actions and `reveal`s the board, persisting the belief in `self.data['opp_range']`; `bot_public_state()` hands the live tracker, the bot's `hole_cards`, and `to_call` to the strategy, and `public_view().botRead` surfaces the bot's read (confidence + top hands) to the UI.
- `range_tracker.py` — `RangeTracker` (Phase 3): a hand-level Bayesian belief over the **opponent's** hole cards. Per-hand weights over the C(50,2) combos, `reveal()` for card removal, `observe()` for Bayesian updates from a blueprint `strategy_fn` (the assumed opponent model), a **confidence** score in [0,1] that decays on off-model actions (`*= exp(-max(0, (-log p_a) - H))`, surprise vs the model's own entropy — correct for mixed strategies), and `hero_equity()` for the bot's equity vs the belief. JSON `to_dict`/`from_dict`. This is the input a future river subgame solver consumes. (The evaluation harness's `lbr.py:BotRange` is the older sibling of this class; not yet merged. The two had drifted on off-grid response — fixed in BUG-008: LBR now mirrors the deployed bot's translation on every street and routes untrained brackets to fold. If you change how the deployed bot responds off-tree, update the LBR victim model in lockstep.)
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
- `POST /api/strategy/river-solve` — ungated river subgame solve for a concrete spot (real cards + history); returns the solved strategy.
- `GET /api/abstractions` — bucket/position/street/pattern vocabulary for the Key Explorer dropdowns.

Game:
- `POST /api/game/new` — start a session, deal the first hand.
- `GET /api/game/state?id=` — current redacted state.
- `POST /api/game/action` — apply the human action.
- `POST /api/game/bot-action` — run the bot's pending turn(s) (split out so the client can reveal the new card first).
- `POST /api/game/next-hand` — deal the next hand in a session.

Leaderboard / accounts:
- `GET /api/stats` — global +EV counter (5s in-process cache; polled by every browser ~60s).
- `GET /api/leaderboard` — ranked board (10s in-process cache).
- `GET /api/me?playerId=` — caller's own curated row (lifetime hands + netBB + bb/100); public-by-UUID by design, returns 0-state for unknown ids.
- `POST /api/player` — set the caller's unique username; rate-limited (10/min/player + 30/min/IP → 429).
- `POST /api/auth/google` — verify a Cognito Google ID token, resolve the canonical account; rate-limited (20/min/IP → 429); generic 401 on bad token (reason logged server-side).

Health: `GET /api/test` (alias `GET /api/healthz`) — returns 200 with `{status:"ok", blueprint, iterations, postflopTables, sessionStore, debugOverlay, riverGadget, purify, commit}` when healthy; **503 with `{status:"degraded", error}`** when the blueprint failed to load at import (a `before_request` guard then 503s every other endpoint while degraded).

### Environment Variables

The full, authoritative list is in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) ("Environment variables"). The essentials:
- `ALLIN_BLUEPRINT_DB` — explicit path to the blueprint DB (overrides auto-resolution).
- `ALLIN_BLUEPRINT_SOURCE` — `local` (default) | `s3`; `ALLIN_BLUEPRINT_S3_URI` paired with the latter.
- `ALLIN_CORS_ORIGINS` — comma-separated allowed CORS origins (defaults to `localhost:5173`/`5174`).
- `ALLIN_SESSION_STORE` / `ALLIN_STORE_BACKEND` — `memory` (default) | `dynamodb` for sessions / leaderboard stores. Entrypoint picks 1 worker if either is memory, 2 if both DynamoDB.
- `ALLIN_DYNAMODB_TABLE` / `ALLIN_PLAYERS_TABLE` / `ALLIN_GLOBAL_TABLE` / `ALLIN_HANDS_TABLE` — table names.
- `ALLIN_SESSION_TTL_SECONDS` (86400) / `ALLIN_HANDS_PER_WINDOW` (500) / `ALLIN_HAND_WINDOW_SECONDS` (3600).
- `ALLIN_COGNITO_REGION` / `ALLIN_COGNITO_USER_POOL_ID` / `ALLIN_COGNITO_APP_CLIENT_ID` — Google-sign-in token validation (unset = `/api/auth/google` 503s, gameplay unaffected).
- `ALLIN_DEBUG_OVERLAY` — `1` exposes the bot-bucket debug overlay. **Code default is `1` (ON for dev / local Docker); MUST set `0` in Lightsail env to hide the live bot's bucket mid-hand.**
- `ALLIN_LOG_LEVEL` (INFO) / `ALLIN_GIT_SHA` (build commit, surfaced in healthz).
- `ALLIN_BLUEPRINT_CACHE_DIR` — where S3 source caches the downloaded blueprint (default OS tempdir; set to a stable mount in containers).
- `ALLIN_RIVER_CACHE_BOARDS` — `PostflopV2` river board LRU cap (default 100k).
- **Gunicorn / entrypoint tuning (Docker only):** `ALLIN_WORKERS`, `ALLIN_THREADS` (4), `ALLIN_TIMEOUT` (120), `ALLIN_GRACEFUL_TIMEOUT` (120), `ALLIN_MAX_REQUESTS` (500), `ALLIN_MAX_REQUESTS_JITTER` (50), `ALLIN_BIND` (`0.0.0.0:5000`).
- `ALLIN_DEBUG` (1) / `ALLIN_DEV_HOST` / `ALLIN_DEV_PORT` — dev server only; irrelevant under gunicorn.
- `VITE_API_BASE` / `VITE_COGNITO_DOMAIN` / `VITE_COGNITO_APP_CLIENT_ID` / `VITE_COGNITO_REDIRECT_URI` — frontend build-time config.

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

The overall build plan is a phased dependency chain (status as of 2026-06-08). The key
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
- **Phase 2 — Potential-aware postflop buckets** ✅ done. The distribution-aware abstraction:
  decoupled 30-fine/10-coarse preflop + **20 flop / 16 turn / 10 river** EMD-clustered postflop
  buckets (`PostflopV2`), baked into the capped run. Replaced the old 8-bucket heuristic.
- **Phase 3 — Hand-level Bayesian range tracker** ✅ done. Hand-level (not bucket-level) range
  (`game/range_tracker.py`), hooked into `GameSession` JSON state, with a confidence score that
  decays on off-tree actions. **Prerequisite for Phase 4.**
- **Phase 4 — Subgame solving** ✅ RIVER shipped / 🧊 turn shelved. The **river** endgame solver
  (`subgame/river_subgame_solver.py`, unsafe v1, no gadget) builds the small river tree, takes both
  ranges from Phase 3, runs vectorized CFR+, reads off the bot's action, and is **served live**
  (EV-gated; falls back to the blueprint pre-river). The bot's hole cards flow through `decide()`'s
  public state. The depth-limited **turn/flop** solver (`subgame/turn_*.py`, `cfv.py`) was built and
  lab-validated (M0–M2, ~98.6% less exploitable in-abstraction) but **SHELVED**: the N0 real-game
  gate failed (lower exploitability did not beat the blueprint head-to-head — a cross-street
  consistency break needing continual re-solving, an architecture rebuild). See
  [docs/DEPTH_LIMITED_SOLVER_PLAN.md](docs/DEPTH_LIMITED_SOLVER_PLAN.md) and
  [docs/NN_LEAF_PLAN.md](docs/NN_LEAF_PLAN.md) (both on hold).
- **Phase 5 — Safety + depth.** 5a ✅ **SHIPPED (2026-06-10)** / 5b ⬜ (multi-week; deferred).
  **5a — safe river re-solving gadget** (`subgame/blueprint_projection.blueprint_cfv` +
  `river_cfr.run_gadget` + `solve_control.solve_river_gadget`): the villain gets a per-hand opt-out
  paying the blueprint's river-entry CFV, so the re-solved bot is provably no-more-exploitable than
  the blueprint. Served via `RiverSubgameSolver(safe_gadget=True, gadget_anchor='auto')`: per spot it
  EXPLOITS the read (unsafe-v1) when a self-check proves it's within the blueprint, else CLAMPS to the
  blueprint-anchored gadget (anchors: `belief`/`blueprint`/`confidence`/`auto`). Validated ≤ blueprint
  on every spot incl. a deliberately-wrong belief (`tests/test_safe_river_gadget.py`); off/A/B compared
  live (`tests/compare_gadget_policies.py`). The blueprint CFV machinery is also the leaf-value piece
  5b needs. See [docs/SAFE_RIVER_SOLVING_PLAN.md](backend/bot/docs/SAFE_RIVER_SOLVING_PLAN.md).
  **5b** — continual-re-solving turn/flop depth-limited solving with **blueprint counterfactual values
  as the leaf value function** — the revival path for the shelved turn solver.

Dependency chain: **0 → 1 → 2 → 3 → 4 → 5** (1a and 1b are independent quick wins; 2 is
high-leverage but technically optional before 3; 3 strictly gates 4).

Strategy purification ✅ **done (2026-06-10)** — `cfr/purification.py` (drop sub-threshold
actions, renormalise; argmax at threshold=1.0). A BR sweep (seed 42, 50 samples) found **1%
optimal** (off 14621 → 1% 14534 mbb; 5% over-purifies, full=argmax catastrophic at 24091).
**Served at `purify_threshold=0.01`** on the blueprint-path play (the opponent model is NOT
purified). Wired into both the live bot and the BR scoreboard (`run_evaluation.py --purify`).
NB this is the blueprint TABLE's exploitability — BR doesn't see the guards/gadget.

Deferred / low-priority: opponent modeling (only coarse aggregate stats — aggression %,
fold-to-c-bet — off by default; per-bucket modeling is infeasible in one human session due
to data sparsity); widening the betting abstraction (4th size / 3rd raise) only if LBR reveals
missing-line leaks.

Productionization (additive, the interfaces already exist for these):
- Online 1v1 play deployed on AWS — swap `InMemorySessionStore` for a Redis/DynamoDB-backed
  store; the Flask-free game core and `BotStrategy` / `SessionStore` interfaces make this
  additive, not a rewrite.
- Unrestricted human bet sizing — ✅ done (2026-05-26, with Phase 1a). The API accepts
  `{action: 'bet_custom'|'raise_custom', amountBb}`; the engine stores the raise-to chip total
  in `history` as `bet_custom_<total>`/`raise_custom_<total>` (chip-conservation tests intact),
  and off-grid bets are handled by action translation (see Phase 1a above).
