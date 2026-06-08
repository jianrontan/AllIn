# AllIn: Game-Theory-Optimal Heads-Up Poker AI

AllIn is a **production-grade artificial intelligence** for heads-up No-Limit
Texas Hold'em, built on **Monte Carlo CFR+** (Counterfactual Regret Minimization)
— the same family of self-play, regret-minimization algorithms behind
championship-level poker bots. It approximates **game-theory-optimal (GTO)**
strategy through millions of iterations of self-play, serves that strategy
through a **Flask** API, and exposes it in an interactive **React** platform.

---

## 🎯 AI & Machine Learning Overview

### 🧠 The Intelligence Engine
- **Monte Carlo CFR+ with external sampling**: each iteration samples chance and
  opponent actions, walking one trajectory through the game tree instead of the
  full exponential tree — making millions of training iterations tractable.
- **Discounted CFR+ (Linear-CFR-style)**: time-discounted updates — regret discount
  **α = 1.5**, strategy-sum discount **γ = 2.0** — for faster, more stable convergence
  toward a Nash equilibrium. (CFR+ with a `((t-1)/t)^α` discount on floored regrets and
  a separate `t^γ`-weighted average strategy — not the canonical DCFR α/β/γ scheme.)
- **Self-play reinforcement learning**: no human data and no hand-crafted
  heuristics — the strategy emerges purely from **regret minimization**.
- **Multi-layer abstraction**: a hierarchical state representation built from
  **decoupled 30-fine / 10-coarse equity-based preflop buckets + distribution-aware
  (potential-aware) postflop buckets (20 flop / 16 turn / 10 river)** clustered by
  Earth Mover's Distance over equity distributions.

### 📊 Trained Blueprint (active model)
```
Served blueprint (capped run, 25M-iteration snapshot — see Deployment):
├── Algorithm:          Monte Carlo CFR+ with external sampling + Linear-CFR-style discount
├── Training iterations: 25,550,000 (the least-exploitable snapshot; pinned via ALLIN_BLUEPRINT_DB)
├── Info sets:          128,177 (trained situations stored)
├── Game:               Heads-up NLHE, 100 BB effective stacks (SB 1 / BB 2)
└── Storage:            SQLite (incremental checkpoint + resume)
```

### 🔬 Algorithmic Architecture
```
Training Pipeline:
Random self-play deal → Monte Carlo CFR+ traversal → regret/strategy update →
SQLite checkpoint → automatic active-blueprint selection → API inference
```

**Core technologies (actually used):**
- **NumPy** for vectorized numerical computing (regret matching, the
  exploitability evaluator)
- **phevaluator** — high-performance C hand-strength library
- **SQLite (WAL mode)** for incremental, resumable strategy storage
- **Flask** REST API · **React + Vite** frontend
- **Hypothesis** property-based testing for engine correctness

---

## 🚀 Why CFR+? (Algorithmic Highlights)

- **External sampling** turns a full game-tree traversal into a single sampled
  path per iteration — the key to scaling to millions of iterations.
- **CFR+ regret flooring** (clamping cumulative regrets at 0) accelerates
  convergence over vanilla CFR.
- **Position-aware information sets** learn in-position and out-of-position play
  separately.
- **Stack-aware game engine** models real chip costs, all-ins, and side-stack
  constraints — not a toy abstraction.
- **Exploitability evaluator** measures how far the blueprint is from
  unexploitable (best-response, in milli-big-blinds/hand) so convergence is
  *measured*, not assumed.

---

## 🛠 Technical Stack

### 🐍 AI / ML Backend
- **Python 3.12** — core development language
- **NumPy** — vectorized regret matching and best-response evaluation
- **phevaluator** — O(1) hand evaluation via precomputed tables
- **SQLite** — blueprint persistence with checkpoint/resume + read-while-writing

### 🧮 Algorithms
- **Monte Carlo CFR+** with **external sampling** and **Linear-CFR-style** discounting
- **Nash-equilibrium approximation** through iterative self-play
- **Feature engineering**: equity-based card bucketing, action abstraction, and
  position-aware information-set keys

### 🌐 Full-Stack Integration
- **Flask API** — strategy lookup + live game endpoints
- **React + Vite frontend** — strategy explorer and play-vs-bot table
- **PyPokerEngine** — used in the test harness for bot-vs-bot simulation
- **phevaluator** — fast showdown evaluation

---

## 🎯 Key Features

### 🤖 Strategy Engine
- **Fast inference**: direct blueprint lookup from SQLite, no per-decision search.
- **Distribution-aware abstractions**: 30-fine/10-coarse decoupled preflop + 20/16/10 potential-aware postflop buckets (EMD-clustered equity distributions).
- **Mixed-strategy output**: probability distributions over fold / call / bet /
  raise / all-in, sampled at play time.
- **Honest "unknown" handling**: situations never reached in training report
  `found: false` rather than guessing.

### 📊 Interactive Platform
- **Strategy Explorer** — look up the blueprint's play for any spot:
  - *Hand Explorer*: enter real cards + a betting line, see the resulting
    info-set key and strategy.
  - *Key Explorer*: build an info-set key from abstraction dropdowns (or paste
    one) and see the strategy.
- **Play vs the Bot** — an interactive heads-up table against the trained AI,
  100 BB deep, with full action and pot tracking.

### 🔬 Quality & Correctness
- **Exploitability scoring** via a vectorized best-response walk of the public
  game tree (`tests/run_evaluation.py`).
- **Property-based testing** (Hypothesis) over the engine's semantic invariants —
  chip conservation, call/all-in arithmetic, legal-action shape — backed by a
  documented [bug log](backend/bot/docs/BUG_LOG.md).

---

## 🛠 Getting Started

### Prerequisites
- **Python 3.12**
- **Node.js 18+** (frontend)
- **Git**

### 1. Clone
```bash
git clone https://github.com/jianrontan/AllIn.git
cd AllIn
```

### 2. Backend + API
```bash
# Install Python dependencies
cd backend
pip install -r requirements.txt

# Start the inference API (must run from backend/api/)
cd api
python strategy_api.py        # http://localhost:5000
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev                   # http://localhost:5173
```

### 🎓 Train your own blueprint
```bash
cd backend/bot

# Quick smoke run (seconds)
python -c "from tests.run_blueprint_trainer import run_training; run_training(100)"

# A real run — checkpoints as it goes; resume any time with resume='<db>.db'
python -c "from tests.run_blueprint_trainer import run_training; run_training(5000000)"
```
Training writes a timestamped `backend/bot/analysis/blueprints/blueprint_*.db`. The API and
bot automatically use the blueprint with the most iterations — **no manual
promotion step**.

### 📊 Using the platform
1. Open the frontend at `http://localhost:5173`.
2. **Strategy Explorer**: enter a hand + betting line (or build an info-set key)
   and get the GTO strategy with probabilities.
3. **Play vs the Bot**: play heads-up against the AI and watch how it responds.

### 📈 Measure blueprint quality
```bash
cd backend/bot
python tests/run_evaluation.py --samples 1000   # exploitability in mbb/hand (lower = better)
```

---

## 🗺 Roadmap

- ✅ **Blueprint training** — Monte Carlo CFR+ with SQLite checkpoint/resume
- ✅ **Serving + Play-vs-bot** — Flask API + React platform
- ✅ **Exploitability evaluation** — best-response convergence scoreboard
- ✅ **Hand-level Bayesian range tracker** — opponent-range belief with confidence
- ✅ **River subgame solving** — real-time re-solving of the river with full
  pot/stack information and the live range (the shippable real-time-solving feature)
- 🧊 **Turn/flop depth-limited solving** — built and validated in the lab, but
  **shelved**: it lowered exploitability yet did not beat the blueprint in real
  games (a cross-street consistency problem needing continual re-solving). See ROADMAP.
- 📅 **Online 1v1 play on AWS** — DynamoDB session store, Cloudflare Pages frontend,
  +EV leaderboard (unrestricted human bet sizing already shipped)

See [docs/ROADMAP.md](docs/ROADMAP.md) for detail, and
[docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for the architecture.

---

## 📚 Documentation

| Doc | Purpose |
|---|---|
| [USER_GUIDE.md](USER_GUIDE.md) | Install, train, run, play, evaluate |
| [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) | Architecture and module reference |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Phase status and what's next |
| [docs/TRAININGFLOW.md](docs/TRAININGFLOW.md) | One CFR+ iteration, end to end |
| [CLAUDE.md](CLAUDE.md) | Canonical short reference for contributors |
| [backend/bot/docs/BUG_LOG.md](backend/bot/docs/BUG_LOG.md) | Correctness bug history |
