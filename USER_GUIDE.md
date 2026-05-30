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
iterations (see [§8](#8-running-a-specific-blueprint) below).

`run_training(...)` is the single entry point for both single-threaded and
parallel runs. Its arguments:

| Argument | Default | Meaning |
|---|---|---|
| `iterations` | — | How many CFR iterations to run this session. The main quality knob — more = closer to optimal (measure with §6). |
| `resume` | `None` | Filename of an existing DB to continue (e.g. `'blueprint_20260522_160906.db'`). `None` starts a fresh timestamped run. |
| `checkpoint_every` | `1000` | Save to the DB every N iterations. Use a larger value (e.g. `50000`) for long runs so DB writes don't dominate. |
| `workers` | `None` | `None`/`1` = single-threaded. `>1` = **parallel** across that many processes (see §3.2). |
| `merge_every` | `2000` | **Parallel only.** Iterations *per worker* between merges (a "round" = `merge_every × workers`). |
| `seed` | `None` | Seed Python's RNG for a reproducible single-threaded run. (Parallel runs are *not* bit-reproducible — see §3.2.) |

### 3.1 Single-threaded (default, reproducible)

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
(Ctrl+C) and `resume=` later without losing progress.

> **Tip:** you can also just run `python tests/run_blueprint_trainer.py`, which
> kicks off a large default run. Use the `run_training(...)` one-liners when you
> want to control the iteration count.

### 3.2 Parallel training (recommended for real runs)

External-sampling MCCFR draws an independent random hand per iteration, so the
work parallelizes across processes. Pass `workers=N` to train across N worker
processes. A parallel run writes a **`blueprint_par_<timestamp>.db`** (the `par`
tag distinguishes it on disk; it still auto-resolves like any other blueprint).

```bash
cd backend/bot

# Parallel run: 30M iterations across 8 workers.
python -c "from tests.run_blueprint_trainer import run_training; run_training(30000000, checkpoint_every=50000, workers=8, merge_every=4000)"

# Resume a parallel run (worker count MAY differ from the original run)
python -c "from tests.run_blueprint_trainer import run_training; run_training(10000000, resume='blueprint_par_20260529_002511.db', workers=8, merge_every=4000)"
```

**Choosing the settings:**

- **`workers`** — set to your **physical** core count (e.g. 8). Hyperthreads
  don't help (the inner loop is pure-Python CPU work). On a laptop expect roughly
  **3–4×** speedup, not N× — thermal throttling and the per-round merge cap it.
  If you run the convergence tracker (§3.3) on the same machine, use
  **`workers=7`** to leave a core for it.
- **`merge_every`** — iterations *per worker* between merges. The round size is
  `merge_every × workers`. **Smaller** = more faithful to single-threaded but more
  per-round overhead; **larger** = higher throughput but a slightly coarser
  approximation. **`4000`** (→ 32k iters/round at 8 workers) is a good default for
  the current abstraction; raise it toward 8000 if you see the per-round overhead
  dominating, lower it if exploitability stalls.
- **`checkpoint_every`** — use `50000` (≈1–2 rounds). The parallel path flushes
  the final partial round automatically, so `total_iterations` is always exact and
  a later resume never replays lost work.

**Important caveats:**

- **Parallel is an *approximation* of single-threaded CFR** ("block Linear-CFR":
  workers accumulate raw regret, the master applies the discount once per merge
  round). It is **validated by exploitability** (§6 / §3.3), **not** by seed
  reproducibility — a `seed=` does *not* make a parallel run bit-reproducible. Judge
  a parallel run by whether its BR/LBR drops, not by comparing it to a single-thread run.
- **Do not mix modes on one DB.** A blueprint trained single-threaded should be
  resumed single-threaded, and a parallel one resumed parallel. The discount
  bookkeeping differs between the two (single-thread counts iterations; parallel
  counts merge rounds), so switching mid-DB makes the discount schedule incoherent.
  Pick one mode per blueprint file.

### 3.3 Track convergence while training

Run the tracker in a **separate terminal** to watch a live training run converge.
At each iteration milestone it takes a consistent snapshot of the live DB (safe
while training is still writing), scores its exploitability (best-response, and
optionally LBR), appends a row to a CSV, and reprints the curve so far.

```bash
cd backend/bot

# Auto-detect the active run; measure every 1M iterations (BR only)
python scripts/track_training.py

# Measure every 2M iterations, and also run the (slower) LBR lower bound
python scripts/track_training.py --every 2000000 --lbr

# Watch a specific DB instead of auto-detect
python scripts/track_training.py --db analysis/blueprints/blueprint_par_20260529_002511.db
```

- It auto-detects the **most-recently-modified** `blueprint_*.db` (your active
  run — works for both `blueprint_*` and `blueprint_par_*`).
- Curve CSV → `analysis/training_curve/training_curve_<run-timestamp>.csv`;
  frozen snapshots → `analysis/blueprints/snapshots/`. Snapshots live in a
  **subfolder**, so they are never mistaken for the active blueprint.
- BR is saved and printed **before** LBR runs, so a slow LBR never costs you the
  BR point. Watch the numbers fall and stop training when they flatten.
- BR is CPU-heavy, so each milestone briefly competes with training for a core.
  Spacing milestones out (`--every 2000000`) and/or using `workers=7` keeps the
  impact small. Run the tracker from the **same code/abstraction** as the training.

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

## 8. Running a specific blueprint

### How the active blueprint is chosen (automatic)

`src/config.py:resolve_blueprint_path()` picks the `analysis/blueprints/blueprint_*.db`
with the **highest training iterations** (that isn't currently being written
to). Both `blueprint_*.db` and `blueprint_par_*.db` (parallel runs) match, and
snapshots under `analysis/blueprints/snapshots/` are excluded. So when you finish
a longer run, the API/bot switch to it automatically on restart — no file
renaming or promotion step.

### See what blueprints you have and their iteration counts

```bash
cd backend/bot
python -c "
import glob, sqlite3
for f in sorted(glob.glob('analysis/blueprints/blueprint_*.db')):
    r = sqlite3.connect(f'file:{f}?mode=ro', uri=True).execute(
        \"SELECT value FROM training_metadata WHERE key='total_iterations'\").fetchone()
    print(f'{int(float(r[0])):>12,}  {f}' if r else f'           ?  {f}')
"
```

The live API also reports the active DB at <http://localhost:5000/api/test>.

### Pin a specific blueprint in the API (override auto-resolution)

`strategy_api.py` has **no command-line flag** for the blueprint. It calls
`resolve_blueprint_path()` once at startup, which checks the
**`ALLIN_BLUEPRINT_DB`** environment variable first — set it to an absolute path
and the API serves exactly that file, skipping the auto-pick. Set it in the
**same shell, before** launching the API:

```bash
# set the var, then start the API in the SAME terminal
export ALLIN_BLUEPRINT_DB=/c/Ron/AllIn/backend/bot/analysis/blueprints/blueprint_par_20260529_002511.db
cd /c/Ron/AllIn/backend/api
python strategy_api.py
```

Or scope it to the single command:

```bash
cd /c/Ron/AllIn/backend/api
ALLIN_BLUEPRINT_DB=/c/Ron/AllIn/backend/bot/analysis/blueprints/blueprint_par_20260529_002511.db python strategy_api.py
```

Three things to know:

- **Read at startup, not per request.** To switch blueprints, change the var and
  **restart** the API.
- **Same shell only.** `export` applies to that terminal; a new terminal won't have
  it. To persist across shells, add the `export` line to `~/.bashrc`.
- **Use an absolute path.** The API runs from `backend/api/`, so a relative
  `analysis/...` path won't resolve. (`/c/...` works; a native `C:\...` path works
  too if quoted.)

**Verify which blueprint is live:** `curl http://localhost:5000/api/test` reports
the `blueprint` filename and `iterations`. The API also prints
`DEBUG: Blueprint: <name> (<n> iterations)` on startup.

**Clear the pin** (back to auto-resolution): `unset ALLIN_BLUEPRINT_DB`, then restart.

> **Why you often need this:** auto-resolution picks the **highest-iteration** DB.
> If an old, abstraction-incompatible blueprint with more iterations is still in
> `analysis/blueprints/`, it will outrank your fresh current-abstraction run —
> pin `ALLIN_BLUEPRINT_DB` to force the right one.

The evaluation harness instead takes the path directly, no env var:

```bash
python tests/run_evaluation.py --db analysis/blueprints/blueprint_par_20260529_002511.db
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
