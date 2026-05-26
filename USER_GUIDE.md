# AllIn — User Guide

AllIn is a heads-up (1v1) Texas Hold'em poker AI. It trains a *blueprint*
strategy with Monte Carlo CFR+ (Counterfactual Regret Minimization), serves
that strategy through a Flask API, and exposes two things in a React web app:

1. **Play vs the bot** — an interactive heads-up game against the trained AI.
2. **Strategy explorer** — look up what the blueprint does in any situation.

This guide is task-oriented: how to install, train, run, play, and evaluate.
For architecture and internals, see [`CLAUDE.md`](CLAUDE.md).

---

## 1. Prerequisites

- **Python 3.12** (the bot, training, API)
- **Node.js 18+** and npm (the frontend)
- A trained blueprint DB in `backend/bot/analysis/blueprints/` (you train one in step 3,
  or drop in an existing `blueprint_*.db`)

---

## 2. Install

```bash
# Backend (from the repo root)
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

`requirements.txt` includes `hypothesis`, used only by the property tests — it
is not needed at runtime, but installing everything keeps one command simple.

---

## 3. Train a blueprint

Training produces a timestamped SQLite file in `backend/bot/analysis/blueprints/`, e.g.
`blueprint_20260522_160906.db`. There is **no manual "make it active" step** —
the API and bot automatically use the blueprint with the most training
iterations (see "How the active blueprint is chosen" below).

```bash
cd backend/bot

# Quick smoke run (seconds) — verifies the pipeline end to end
python -c "from tests.run_blueprint_trainer import run_training; run_training(100)"

# A real run — this takes a long time; it checkpoints as it goes
python -c "from tests.run_blueprint_trainer import run_training; run_training(5000000)"

# Resume a run that was interrupted (pass the DB filename)
python -c "from tests.run_blueprint_trainer import run_training; run_training(50000, resume='blueprint_20260522_160906.db')"
```

Training checkpoints every 1000 iterations by default, so you can stop
(Ctrl+C) and `resume=` later without losing progress. The number of training
iterations is the main quality knob — more iterations = closer to optimal play
(measure it in step 6).

> **Tip:** you can also just run `python tests/run_blueprint_trainer.py`, which
> kicks off a large default run. Use the `run_training(...)` one-liners when you
> want to control the iteration count.

---

## 4. Run the app

Two processes: the API (port 5000) and the frontend dev server (port 5173).
Use two terminals.

**Terminal 1 — API** (must be started from `backend/api/`):

```bash
cd backend/api
python strategy_api.py
```

Confirm it found a blueprint: open <http://localhost:5000/api/test> — it
reports status and which blueprint DB is active.

**Terminal 2 — frontend:**

```bash
cd frontend
npm run dev
```

Open the URL it prints (default <http://localhost:5173>).

---

## 5. Using the web app

### Play vs the bot
Go to the **Play** page. You're dealt a hand and play heads-up against the AI.

- Stacks are **100 big blinds** each hand (200 chips, SB=1 / BB=2). The UI shows
  everything in **BB** (chips ÷ 2).
- Each hand starts fresh at full stacks; your running profit/loss across hands
  is tracked separately and shown as your net.
- Click an action (check/call/fold or a bet/raise size). The bot responds
  automatically, and the board advances street by street to showdown.
- After a hand ends, deal the next one — the button (who acts first preflop)
  alternates, as in real heads-up.

### Strategy explorer
Go to the **Strategy** page. Two independent tools share it:

- **Hand Explorer** — enter real cards (e.g. `Ah Kd`) and a betting line; it
  shows the resulting info-set key and the blueprint's strategy (action
  probabilities) for that spot.
- **Key Explorer** — build an info-set key from dropdowns (bucket / position /
  street / betting pattern), or paste a key directly, and see the strategy.

If a spot was never reached in training, the explorer reports "not found" —
that is a valid answer, not an error; it just means the blueprint has no data
for that exact abstracted situation.

---

## 6. Evaluate blueprint quality (exploitability)

Exploitability measures how badly a perfect counter-strategy could beat the
blueprint — **lower is better**, reported in milli-big-blinds per hand
(mbb/hand). A true Nash equilibrium would score 0. Run it before and after a
training change to confirm the strategy is actually improving.

```bash
cd backend/bot
python tests/run_evaluation.py                  # active blueprint, 400 board samples
python tests/run_evaluation.py --samples 1000   # more samples = lower variance
python tests/run_evaluation.py --db analysis/blueprints/blueprint_20260522_160906.db
```

---

## 7. Run the tests

```bash
cd backend/bot
python tests/test_game_session.py          # Game core: GameSession, bot strategy
python tests/test_cfr_correctness.py       # CFR correctness + chip-conservation fuzz
python tests/test_poker_game_properties.py # Hypothesis property tests for engine invariants
python tests/test_player.py                # Bot vs RandomPlayer via PyPokerEngine
python tests/test_custom_betting.py        # Unrestricted custom bets + action translation
```

Or run everything under pytest from `backend/bot/`: `python -m pytest tests/ -q`

The property tests can also run under pytest:
`python -m pytest tests/test_poker_game_properties.py -v`.

---

## 8. How the active blueprint is chosen

`src/config.py:resolve_blueprint_path()` picks the `analysis/blueprints/blueprint_*.db`
with the **highest training iterations** (that isn't currently being written
to). So when you finish a longer run, the API/bot switch to it automatically on
restart — no file renaming or promotion step.

To pin a specific blueprint (e.g. to compare two), set an environment variable
before starting the API:

```bash
# macOS/Linux
export ALLIN_BLUEPRINT_DB=/abs/path/to/backend/bot/analysis/blueprints/blueprint_20260522_160906.db
# Windows PowerShell
$env:ALLIN_BLUEPRINT_DB = "C:\Ron\AllIn\backend\bot\analysis\blueprint_20260522_160906.db"
```

---

## 9. Configuration reference

| Variable | Used by | Purpose |
|---|---|---|
| `ALLIN_BLUEPRINT_DB` | API / bot | Pin an explicit blueprint DB path (overrides auto-resolution). |
| `ALLIN_CORS_ORIGINS` | API | Comma-separated allowed CORS origins (default `localhost:5173`/`5174`). |
| `VITE_API_BASE` | frontend | API base URL (default `http://localhost:5000`). |

---

## 10. Troubleshooting

- **API says no blueprint / `/api/test` shows none** — you have no
  `blueprint_*.db` in `backend/bot/analysis/blueprints/`. Train one (step 3) or copy one in.
- **`ModuleNotFoundError` when starting the API** — start it from
  `backend/api/` exactly; it adjusts `sys.path` relative to that directory.
- **Frontend can't reach the API** — confirm the API is on port 5000, or set
  `VITE_API_BASE` to wherever it's running. CORS errors usually mean the
  frontend origin isn't in `ALLIN_CORS_ORIGINS`.
- **`No module named hypothesis`** when running the property tests — `pip install
  -r backend/requirements.txt` (or `pip install hypothesis`).
- **Training seems stuck / want to stop** — Ctrl+C is safe; it checkpoints
  every 1000 iterations. Resume with `resume='<that-db>.db'`.
