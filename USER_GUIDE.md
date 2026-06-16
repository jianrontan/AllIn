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

The training script takes plain command-line flags — **prefer these over a
`python -c "..."` one-liner**, which is easy to corrupt when pasted into a shell
(a dropped closing quote leaves you stuck at a `>` prompt). Flag commands have no
quotes or line-continuations, so paste can't break them.

```bash
cd backend/bot

# Quick smoke run (seconds) — verifies the pipeline end to end
python tests/run_blueprint_trainer.py --iterations 100

# A real run — takes a long time; it checkpoints as it goes
python tests/run_blueprint_trainer.py --iterations 5000000

# Resume an interrupted run (just the DB filename, no path)
python tests/run_blueprint_trainer.py --iterations 50000 --resume blueprint_20260522_160906.db
```

`python tests/run_blueprint_trainer.py --help` lists every flag. Ctrl+C is safe
— it checkpoints as it goes, so `--resume` later picks up where it left off.

> **Equivalent `python -c` form** (if you prefer the function API — make sure it
> stays on **one line with the closing `"`**):
> `python -c "from tests.run_blueprint_trainer import run_training; run_training(5000000)"`

### 3.2 Parallel training (recommended for real runs)

External-sampling MCCFR draws an independent random hand per iteration, so the
work parallelizes across processes. Pass `workers=N` to train across N worker
processes. A parallel run writes a **`blueprint_par_<timestamp>.db`** (the `par`
tag distinguishes it on disk; it still auto-resolves like any other blueprint).

```bash
cd backend/bot

# Parallel run: 30M iterations across 8 workers
python tests/run_blueprint_trainer.py --iterations 30000000 --workers 8 --merge-every 4000 --checkpoint-every 50000

# Resume a parallel run (worker count MAY differ from the original run)
python tests/run_blueprint_trainer.py --iterations 10000000 --resume blueprint_par_20260529_002511.db --workers 6 --merge-every 2000

# Capped (Fix-#4) blueprint run — must be trained FROM SCRATCH (no --resume): capped
# DBs trained before the BUG-014 trainer fix are corrupt and cannot be resumed.
python tests/run_blueprint_trainer.py --iterations 30000000 --workers 7 --merge-every 2000 --menu-mode capped --checkpoint-every 70000
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

**Memory (RAM).** Each worker is a separate process holding its own copy of the
blueprint **plus** a river-equity cache that speeds training ~1.6×. The cache is
capped by `ALLIN_RIVER_CACHE_BOARDS` (default 100,000 boards ≈ **0.26 GB per
worker** → ~2.1 GB at 8 workers, ~1.6 GB at 6). Total training RAM also grows with
the blueprint itself as it fills out. If you're tight on RAM (e.g. running a
browser alongside):
- **Use fewer workers** (e.g. `workers=6`) — the biggest saver, since it drops a
  whole blueprint copy *and* a cache per worker, and frees cores for other apps.
- **Lower the cache cap**, e.g. `ALLIN_RIVER_CACHE_BOARDS=60000` (~0.16 GB/worker).
  The cache is **LRU**, so a smaller cap just keeps the hottest boards — it
  degrades gracefully instead of slowing down sharply. Set it before launching
  training: `export ALLIN_RIVER_CACHE_BOARDS=60000`.

**Important caveats:**

- **Parallel is an *approximation* of single-threaded CFR** ("block Linear-CFR":
  workers run canonical CFR+ — flooring regret on every write — and the master applies
  the discount once per merge round). It is **validated by exploitability** (§6 / §3.3), **not** by seed
  reproducibility — a `seed=` does *not* make a parallel run bit-reproducible. Judge
  a parallel run by whether its BR/LBR drops, not by comparing it to a single-thread run.
- **Do not mix modes on one DB.** A blueprint trained single-threaded should be
  resumed single-threaded, and a parallel one resumed parallel. The discount
  bookkeeping differs between the two (single-thread counts iterations; parallel
  counts merge rounds), so switching mid-DB makes the discount schedule incoherent.
  Pick one mode per blueprint file.

### 3.3 Track convergence while training

`scripts/track_training.py` builds a convergence curve (best-response, and
optionally LBR, vs iterations) from a running or finished training run. Each data
point is a **frozen snapshot** of the live DB (taken with SQLite online-backup, so
it's consistent even while training is still writing), so the measurement is valid
whenever it's computed.

Because BR/LBR are **far slower than training** (BR is ~2 min/sample → tens of
minutes to hours per milestone), measuring inline would fall behind and miss
milestones. So the tracker has **three modes** (`--mode`):

| Mode | What it does | Speed |
|---|---|---|
| `snapshot` | At each milestone, freeze the live DB to `snapshots/` and append a row to a manifest. **Never measures.** | seconds/milestone — keeps cadence with training |
| `measure` | Read the manifest and score BR (+ optional LBR) for any snapshot not yet in the curve CSV. | slow — run later / on another core |
| `watch` | Legacy all-in-one: snapshot **and** measure at each milestone (only keeps up for small runs). | as slow as `measure` |

**Recommended workflow — snapshot fast now, measure later** (each in its own terminal):

```bash
cd backend/bot

# 1) SNAPSHOT: freeze the live DB at 500k, then every 2M iterations. Fast; runs
#    alongside training and never blocks on a measurement.
python scripts/track_training.py --mode snapshot --first 500000 --every 2000000

# 2) MEASURE: score the snapshots whenever (concurrently, or after training ends).
#    --follow keeps waiting for new snapshots; drop it to drain the backlog once
#    and exit. 60 BR board samples + the (slower) LBR lower bound.
python scripts/track_training.py --mode measure --samples 60 --lbr --follow
```

Legacy single-terminal form (small runs only):

```bash
# Snapshot AND measure at each milestone, every 1M iterations, BR only
python scripts/track_training.py --mode watch --every 1000000
```

**Flags:** `--first` (first milestone, default = `--every`), `--every` (milestone
spacing), `--samples` (BR board samples), `--lbr` / `--lbr-hands`, `--follow`
(measure mode: keep waiting), `--db` (watch a specific DB instead of auto-detect),
`--stamp` (measure mode: target a specific run's timestamp), `--seed`, `--poll`
(seconds between DB/manifest polls). `--help` lists them all.

**Notes:**

- Both modes **auto-detect** the most-recently-modified `blueprint_*.db` (works for
  `blueprint_*` and `blueprint_par_*`) to derive the run timestamp; `measure` can
  instead target a finished run with `--stamp <YYYYMMDD_HHMMSS>`.
- Outputs are tied to the run timestamp under `analysis/training_curve/`: the
  manifest (`snapshots_<stamp>.csv`, written only by `snapshot`) and the curve
  (`training_curve_<stamp>.csv`, written only by `measure`) — single-writer per
  file, so the two processes never race. Frozen snapshots go to
  `analysis/blueprints/snapshots/` (a **subfolder**, so they're never mistaken for
  the active blueprint).
- **Snapshots land only on checkpoint boundaries** (the DB's iteration counter is
  written at checkpoints), so make `--first`/`--every` reachable given your
  `--checkpoint-every` — e.g. for a 500k first milestone, train with
  `--checkpoint-every 500000` (a divisor of your milestone spacing).
- `measure` writes BR first, then LBR, so a slow/failed LBR never costs you the BR
  point. Both modes are **resumable** — they skip milestones already recorded.
- BR is CPU-heavy; if measuring on the training machine, use `workers=7` for
  training to leave it a core. Run the tracker from the **same code/abstraction**
  as the training.

### 3.4 Regenerate the card abstraction (only when it changes)

The blueprint's buckets come from precomputed files. You only run these when the
**abstraction itself changes** (or on a fresh clone that's missing the baked
tables) — **not** for an ordinary training run. Changing any of them is an
**abstraction change: existing blueprints become incompatible and must be
retrained from scratch** (don't `--resume` an old DB across an abstraction change).

The pipeline has three stages, run from `backend/bot/`:

```bash
cd backend/bot

# 1) Preflop equity table (rarely needed — only to re-roll the Monte Carlo equities).
#    The lossless 169 fine / 10 coarse bucket maps are DERIVED from this table at import,
#    so you normally never touch it. Prints a table to paste into card_abstractions.py.
python scripts/compute_preflop_equity.py

# 2) Fit the postflop cluster centroids (served scheme: 20 flop / 16 turn / 10 river;
#    the in-flight retrain re-fits to 30 / 24 / 10 — set --buckets accordingly).
#    Writes analysis/abstractions/postflop_centroids_<street>.npz (commit these).
python scripts/compute_postflop_buckets.py --street flop  --buckets 20 --situations 3000
python scripts/compute_postflop_buckets.py --street turn  --buckets 16 --situations 3000
python scripts/compute_postflop_buckets.py --street river --buckets 10 --situations 5000

# 3) Bake the canonical-situation → bucket lookup tables from the centroids.
#    Flop + turn only; river is computed at runtime (no table). The turn bake is
#    the long one (~90 min). Tables are git-ignored and stamped with a centroid
#    hash, so a stale table is a hard error — re-bake after any re-fit.
python scripts/bake_postflop_table.py --street flop
python scripts/bake_postflop_table.py --street turn
```

- **Fresh clone:** the centroids are committed but the baked tables are not, so run
  stage 3 once before training/inference. (Without the tables, `PostflopV2` falls
  back to slow per-situation bucketing and warns.)
- **Smoke-test the bake** before the 90-min turn run:
  `python scripts/bake_postflop_table.py --street flop --limit-boards 20` (runs
  without saving — confirms the centroids load and the pipeline is healthy).
- After regenerating, you **must** train a fresh blueprint (no resume) and
  re-measure (§3.3 / §6) — old blueprints key on the old buckets.

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
- Click an action (check/call/fold or a bet/raise size), or type a **custom
  amount** in the bet box to bet/raise any legal size. Re-raising is **uncapped**
  in live play: you can 5-bet/6-bet+ (any amount, any street) until someone is
  all-in — the bot answers an all-in with an exact equity-vs-range decision, and a
  non-jam deep raise with a conservative (call/fold) fallback until the deep-raise
  solver lands. (Training stays capped at 3 aggressions/street; only the live game
  is uncapped.) The bot responds automatically and the board advances to showdown.
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

## 6. Evaluate blueprint quality

There are five evaluation tools, each answering a different question. Most run from
`backend/bot/` and take `--db <path>` (default: the active blueprint); the
blueprint-vs-blueprint ones take `--db-a`/`--db-b`. `--help` lists every flag.
**BR and LBR are the GTO scoreboards; the match/maniac tools are quick
strength/sanity checks, not equilibrium measures.**

### 6.1 Best-response exploitability (`run_evaluation.py`)

The convergence scoreboard. Measures how badly a perfect counter-strategy could
beat the blueprint **within the betting abstraction** — **lower is better**, in
milli-big-blinds per hand (mbb/hand); a Nash equilibrium scores 0. It's exact on
cards but restricted to the grid bet sizes, so it's a **lower bound** on true
exploitability. Run it before/after a change to confirm the strategy improved.

```bash
cd backend/bot
python tests/run_evaluation.py                  # active blueprint, 400 board samples
python tests/run_evaluation.py --samples 1000   # more samples = lower variance
python tests/run_evaluation.py --db analysis/blueprints/blueprint_20260522_160906.db
```

> **It's slow** — BR is ~2 min per board sample, so 400 samples is hours. For a
> long run, prefer the snapshot/measure tracker (§3.3) over a one-off invocation.

### 6.2 LBR — local best response (`run_lbr.py`)

The **off-tree** complement to BR: LBR is allowed to make *any* bet size (not just
the grid), so it catches exploitability that BR misses — how much an opponent who
"size-cheats" off the abstraction can win. Also mbb/hand, lower is better.

```bash
python tests/run_lbr.py --hands 3000
python tests/run_lbr.py --hands 5000 --db analysis/blueprints/blueprint_par_20260529_002511.db
```

### 6.3 Head-to-head match + AIVAT (`run_match.py`)

Plays **two blueprints** against each other (seats swapped for fairness) and
reports A's win rate both **raw and AIVAT-corrected** (variance-reduced — always
on, no flag). With no `--db-a`/`--db-b` it's the active blueprint vs itself (expect
~0, a sanity check). Use it to compare two runs of the **same** abstraction; for
different abstractions use §6.4 instead (it loads each side under its own snapshot).

```bash
python tests/run_match.py --hands 20000 \
    --db-a analysis/blueprints/blueprint_A.db \
    --db-b analysis/blueprints/blueprint_B.db
python tests/run_match.py --hands 10000        # self-play (active vs itself) sanity
```

### 6.4 Cross-abstraction blueprint comparison (`run_cross_match.py`)

The faithful way to ask "is run B actually better than run A?" when they were
trained under **different abstractions** — it loads each side under **its own**
abstraction snapshot, so neither side mis-keys.

```bash
python tests/run_cross_match.py --hands 40000
```

> **Heads-up:** a cross-match between different abstractions measures real playing
> strength, but the weaker side can lose to a **coverage hole** rather than to worse
> strategy (see BUG-007 in the bug log). Read it alongside BR/LBR, not alone.

### 6.5 Quick sanity vs an all-in maniac (`run_maniac.py`)

The fastest "is something obviously broken?" check: the blueprint vs a naive
maniac that **never folds** and spams aggression (`--profile jam` = always all-in,
`medium` = always bet/raise medium, `mixed` = 50/50; `all` runs every profile). A
correct blueprint must *profit* hugely off a maniac — if it doesn't, there's an
exploitable hole that BR/LBR can miss (this is the class of leak behind BUG-007).
Reports the blueprint's win rate in mbb/hand (raw, high-variance — use many hands).

```bash
python tests/run_maniac.py --profile jam --hands 40000
python tests/run_maniac.py --hands 40000   # --profile all (default): every profile
```

> **Phase-4 (river solver) has its own scoreboards**, run from `scripts/` (slow,
> offline): `run_solver_lbr.py` (LBR exploitability of the solver vs the blueprint
> on the same deals — the go/no-go win) and `measure_river_exploitability.py`
> (blueprint river-exploitability baseline). See each script's `--help`.

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
| `ALLIN_RIVER_CACHE_BOARDS` | training / bot | Max boards in the per-process river-equity cache (default `100000` ≈ 0.26 GB/process; LRU). Lower it to save RAM (see §3.2). |
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
