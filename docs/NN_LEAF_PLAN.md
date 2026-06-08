# NN Leaf Plan — fast turn + flop solving via a learned leaf

A neural net that **predicts the leaf-value matrix**, replacing the slow per-solve
matrix build (~20–54 s) with an instant forward pass — so the turn (and then flop)
solver runs live in ~1–2 s, ships as a tiny model (server now, browser later), and
unlocks the **flop** (where a bake is impractical). Successor to the M4 bake path.

Status: ⛔ **ON HOLD — N0 GATE FAILED (2026-06-07).** The whole ladder above Rung 1 is
NOT being built. Keep this plan as the resume path *if* the turn solver is later made to
beat the blueprint in real games (a research track, not a serving build).

**N0 RESULT (the gate that stops everything):** head-to-head vs the blueprint, 250 hands,
`scripts/measure_turn_match.py` → river-stack **+1801 ± 1363**; turn-stack **−611 ± 1169**
at honest fidelity (n=64 / 500 iters), and **−260** at n=20 — **two independent negative
runs, ~1.3σ BELOW the river-only stack.** No positive signal; if anything it hurts. Why,
despite M2's −98.6% exploitability: lower exploitability ≠ higher EV vs a *non-adaptive*
opponent (the blueprint), and the **EV-gate self-grades** with the same coarse leaf →
passes losing deviations. → **STOP. Ship Rung 1 (25M blueprint + river solver). Do NOT
build the bake or the NN.** The de-risk worked: a ~14-min check, not a months-long NN
project, caught this. Re-run N0 only after a turn-solver *design* change (sharper leaf /
non-self-grading EV gate).

---

## 1. The key design insight (why this is cheap, not DeepStack-expensive)

DeepStack's CFV net is **range-conditional**: opponent range is a network *input*, so
each board needs millions of range samples (~10M total) with *equilibrium* CFV targets →
$1000s / GPU-months. **We do NOT need that.**

**Our leaf is range-INDEPENDENT.** The M0 design (see DEPTH_LIMITED_SOLVER_PLAN.md) is a
**matrix** `M0[hero_bucket, villain_bucket]` that a live solve dots with the current
reach. So:

- **One matrix covers all ranges** for a (board, SPR) → no range sampling.
- The net is a plain **regression**: `(board features, SPR) → M0 (64×64)`. No range input,
  no continual-re-solving — a **drop-in replacement for `turn_leaf_matrix_both`**.
- **Targets are our M2-validated rollout matrices** (not equilibrium CFVs), so the bar is
  "reproduce the leaf we already proved works," not "be superhuman."

→ Think of it as **"compress + generalize the bake into a small model,"** not "train
DeepStack." Data-gen ≈ the bake's compute (or less, from a board *sample*), training is
minutes, the artifact is ~MBs.

## 2. Technical design

- **Net I/O:** input = a **suit-isomorphic board encoding** + **SPR** (scalar; continuous,
  so no SPR bins needed); output = `M0` as **quantile-indexed** 64×64. Quantile indexing
  (equal-freq strength bins) gives a board-INDEPENDENT meaning to cell `[i,j]` = "avg
  payoff of hero's i-th strength quantile vs villain's j-th" — a consistent regression
  target across boards.
- **Live solve path (unchanged except the leaf):** compute the partition live
  (`turn_strength`, ~0.1–2 s) → `tb_idx`; net forward pass → `M0_hat` (instant); enforce
  `M1 = -M0_hat^T` (exact zero-sum); then the existing `TurnCFR` dots them with live reach.
  So the net replaces ONLY the expensive matrix build; everything else is the validated
  M0–M2 machinery.
- **Output structure / loss:** Huber loss on matrix cells; optionally exploit structure
  (monotonicity in strength, the `-M0^T` symmetry) to shrink the net / improve accuracy.
- **Accuracy metric:** rel-RMSE of `M0_hat` vs held-out true `M0` (the M0 coarseness
  metric), AND — the real test — the solver's out-of-bucket exploitability with the net
  leaf vs with the true matrix.

## 3. Milestones (gated)

- **N0 — Validate the turn solver's VALUE (prerequisite; cheap; no NN).** Confirm the
  n=64 turn solve beats the blueprint enough to justify any of this — via the M2 gate
  (already strong, −98.6%) + a slow real-game check (n=64 B2/LBR, background). **GATE: if
  it doesn't beat the blueprint, STOP — neither NN nor bake is worth it.**
- **N1 — Turn data-gen pipeline.** Script: sample `(board, SPR)`, compute `M0` via
  `turn_leaf_matrix_both`, save `(board_encoding, SPR, M0)`; checkpointable. Suit-iso board
  featurization. Measure real per-matrix time → pick sample size. *(~hours–2 days laptop
  compute, $0.)*
- **N2 — Turn net + training.** Regression net `(board, SPR) → M0`; train (minutes–hours);
  validate on held-out boards. **GATE: held-out `M0_hat` rel-RMSE within the M0 band
  (~≤15%); accuracy worst on paired boards is the watch-item.**
- **N3 — Integrate net leaf.** Swap `leaf_matrix_fn` for the net forward pass in
  `TurnSubgameSolver`; live turn solve → ~1–2 s. **GATE: re-run the out-of-bucket
  exploitability gate with the NET leaf — must still pass (net approximation doesn't break
  the solve).**
- **N4 — Real-game gate + latency (the payoff).** Because solves are now FAST, a full LBR
  gate is finally tractable. **HARD GATE (D8): LBR strictly drops on-vs-off** at servable
  latency; strategy-shape probe clean.
- **N5 — Flop net (cascade).** Reuse N1–N4 one street up: the flop leaf = turn+river
  continuation, its data generated using the **turn net as the leaf** (cascade). Flop
  data-gen → flop net → integrate → LBR gate. *(More compute than turn, still ~$0 laptop.)*
- **N6 — Deploy.** Ship the net(s) (~MBs) + solver. **No 5 GB bake** → cheap server box;
  browser/WASM stays open (small model is shippable, unlike the bake).

## 4. Cost / time (corrected 2026-06-07 — NOT DeepStack-scale)

| | compute | calendar (with Claude Code) | $ |
|---|---|---|---|
| Data-gen (turn) | ~hours–2 days laptop, checkpointable/overnightable | — | $0 |
| Net training | minutes–hours | — | $0 (laptop/Colab) |
| Turn net (N1–N4) | as above | **~1.5–2 weeks** (mostly coding + the validate/iterate loop; machine idle most of it) | ~$0 |
| Flop net (N5) | more (cascade) | **+~1–2 weeks** | ~$0 |

NOT 1–3 months / not $100s+ — that was the wrong (DeepStack range-conditional, superhuman,
10M-example) anchor. The "~weeks" is **development time**, not a continuous training grind
(contrast blueprint training, which DID run continuously for days): data-gen is a finite,
chunkable batch job; the net trains in minutes.

## 5. Risks (honest)

1. **Generalization to unseen boards** — will `M0_hat` be good on boards not in the sample?
   Unknown until N2. Mitigation: add boards (more free laptop hours); worst case degrades to
   "store the bake" (still works, just bigger/server-only).
2. **The "good solver" bar** — a faithful DeepStack reimpl *lost* to Slumbot; CFV solvers
   are finicky. Our bar is lower (beat OUR blueprint, not Slumbot), and we use *rollout*
   targets (lower-risk than equilibrium CFVs), but N4 (LBR) is where it's truly proven.
3. **Flop cascade complexity** — N5 is genuinely more code + a deeper leaf object.
4. **Featurization** — encoding boards suit-isomorphically so the net generalizes (not
   wasting capacity on equivalent boards) needs care.
5. **Opportunity cost** — biggest single feature on top of a working river bot; gated on N0.

## 6. Relationship to the bake (and deployment)

The **NN supersedes the bake** as the chosen path (the user wants flop, which the bake
can't practically do). The bake remains the **fallback** if the net fails to generalize
(N2). Deployment: a ~MB model on a cheap server (no 5 GB SSD bake needed) — and, unlike the
bake, **browser/WASM-shippable later** (the only path to "serve many for ~$0"). See
`deployment-plan` / `poker-bot-deployment-feasibility` memories, DEPLOYMENT.md.

## 6b. De-risking (bake-first; the NN is the *easy* kind of NN)

The single biggest de-risk: **the bake and the NN are the same project up to the last
step.** Both need the identical CFR data-gen (compute `M0` matrices); the bake STORES them,
the NN LEARNS them. So:

1. **N0** validate value.
2. **Data-gen** the `M0` matrices (pure CFR/Python — the comfortable part).
3. **Bake the turn** (store) → a **working, shippable turn solver, zero NN risk** (server-
   side). This is a safe milestone AND a **correctness oracle** (compare any NN output to it).
4. **Train the turn NN** on the *same* data (optional upgrade): smaller, faster, flop- and
   browser-capable. **Accuracy is checkable BEFORE going live** (held-out `M0̂` vs the bake).
   If it's good → swap it in; if not → keep the bake. **You are never stuck with nothing.**
5. **Flop** then *needs* a net (can't bake), but its data is generated with the turn
   bake/net as the leaf — by which point the NN pipeline is proven on the turn.

Why the NN risk is smaller than it sounds:
- It's **supervised regression** (fit examples, minimize error) — the most beginner-
  friendly, well-trodden ML task. NOT reinforcement learning / self-play / the finicky
  DeepStack range-conditional + continual-re-solving machinery (that's the version that
  *lost to Slumbot*; we're not building it).
- **CFR stays the core.** The NN is one small bolt-on (a value lookup); all the actual
  game-solving is the CFR you're comfortable with.
- **Claude Code writes the PyTorch**; you direct + review + understand it (it's explainable).
- The bake fallback means a failed NN costs *time*, never the feature.

## 6d. ROOT CAUSE of the N0 failure (why no backend can fix it)

Not a broken layer — the blueprint, range tracker, and river solver are each fine *alone*.
The failure is a **cross-layer consistency break that only appears when the turn solver is
inserted.** Every solve assumes *"the bot plays the blueprint on every OTHER street."* The
turn solver violates that in two ways:

1. **It lies to the river solver about the bot's own range.** The river solver computes the
   bot's hand range from "the bot played the blueprint up to here." When the turn solver
   *deviates* from the blueprint on the turn, the bot reaches the river with a *different*
   range — but the river solver still assumes the blueprint range → it now mis-balances
   value/bluffs and plays *worse*. (This is why adding the turn solver makes the whole
   stack lose vs the river-only stack: it degrades the previously-great river solver.)
2. **The turn plans for a river that won't happen.** The turn leaf = "value if the river is
   played by the *blueprint*," but the bot actually plays the *river solver*. So turn bets
   are optimized for a continuation the bot won't use.

Why the **river** solver works but the **turn** doesn't: the river is the *last* street, so
"blueprint up to here" is TRUE and there's no downstream to corrupt — it's self-consistent.
The turn is *non-terminal* → both assumptions break. This is precisely the problem
**continual re-solving** (DeepStack/Libratus) exists to solve; our one-shot,
blueprint-anchored solves are only valid on the final street.

**FIX = make the turn and river solvers consistent** (share the real range + real
continuation values across streets = continual re-solving). That's an architecture rebuild,
NOT a bug fix and NOT a backend (bake/NN) choice — the bake and NN reproduce the same
inconsistent strategy, so neither helps.

## 6c. Fallback ladder (every rung is a complete, shippable product)

The de-risk in one picture: a ladder where you can **stop at any rung** and have a working
bot, and each failure **falls back to the rung below**. You never end up with nothing.

| Rung | What ships | Reached if… | Fallback if the next step fails |
|---|---|---|---|
| **0. Blueprint + river solver** (Slumbot-class) | ✅ **works TODAY** | always | — (this is the floor) |
| **1. + turn BAKE** (server-side turn solving) | N0 passes + data-gen + store | low-risk (deterministic CFR storage) | drop to Rung 0 |
| **2. + turn NN** (faster, ~MB, browser-able) | NN reproduces the bake on held-out boards | if NN inaccurate → **keep the bake (Rung 1)** | stay at Rung 1 |
| **3. + flop NN** (flop solving) | flop net passes its LBR gate | if flop net fails → **turn+river only** | stay at Rung 2 |

Failure-mode mapping:
- **NN "not feasible" (won't generalize)** → fall back to the **bake (Rung 1)**; turn solver
  still ships server-side. Flop is then blocked (can't bake flop) → accept turn+river, or
  invest more data/features in the net.
- **NN "slow"** → not a real risk: a forward pass is ~ms; the slow part was the *build*,
  which both bake and NN eliminate. The live solve's other costs (partition ~1–2 s + CFR
  ~1–2 s) are leaf-backend-independent and acceptable.
- **N0 fails (turn solve doesn't beat blueprint)** → **STOP**; ship Rung 0 (river bot +
  25M blueprint) — already a complete product. No turn/flop work wasted beyond N0 (cheap).

Effort to climb each rung: Rung 1 ~1 wk + data-gen; Rung 2 +~1 wk; Rung 3 +~1–2 wks. Stop
whenever the payoff/effort stops being worth it — the product is whole at every rung.

## 7. Hard rule

**N0 first.** Do not build the data-gen/net until the turn solver's value is confirmed —
it's cheap to check and it gates the entire project.
