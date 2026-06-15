# Depth-Limited Turn Subgame Solver — Implementation Plan

Status: 🧊 **SHELVED (2026-06-07) — validated in the lab, no real-game value.** M0–M2 are
DONE and stand as a validated lab result (turn CFR+ with a reach-conditioned bucketed leaf cut
in-abstraction exploitability ~98.6–98.9%), but the **M4 N0 real-game gate FAILED**: head-to-head
vs the blueprint the turn solver scored **−611 mbb** while the river-only stack won **+1801 mbb**
— lower exploitability did NOT translate to higher EV vs a non-adaptive blueprint. Root cause is a
cross-street consistency break (see M4 below + [NN_LEAF_PLAN.md](NN_LEAF_PLAN.md) §6d); the real
fix is continual re-solving, an architecture rebuild, **not** a bake or NN leaf. Do not resume
serving work until a turn-leaf / EV-gate redesign re-passes N0. Scope was **turn first** (flop
deferred). See [ROADMAP.md](ROADMAP.md) Phase 4.

> The body below is the original implementation plan, kept for the revival path. Read it as the
> *design that was built and validated through M2*, not as active/next work.

> **Two reviews (codebase-feasibility + method-soundness) reshaped this plan.** The biggest
> change: the leaf value must be **reach-conditioned** (a hero×villain value object dotted with
> the *live* solver reach), NOT a per-bucket scalar — a scalar CFV is computed under the
> *blueprint's* range and is systematically wrong once the solve shifts the range (exploitable).
> Also added: a **measure-first milestone** before any build, a strengthened M0, an out-of-leaf
> adversary for the M2/exploitability gates, and a corrected codebase-reuse map. See §11.

---

## 1. Goal
Generalize the live **river** subgame solver one street up to the **turn**: solve from the turn
at runtime with the real pot/stacks/ranges, including the bot's own overbets/5-bets, instead of
reading the coarse blueprint. The river solver works only because the river ends in a showdown
(exact terminals). The turn has a street ahead, so the missing piece is a **leaf value function**
at the depth limit (turn betting closed → river to come).

## 2. Why a leaf value function (= depth-limited solving)
Solve the current street to a **depth limit** and substitute a **value function** there. That
value is the blueprint's **counterfactual value (CFV)** for continuing, made robust (multi-valued,
§6) so the solver can't over-exploit a single assumed continuation. CFVs derive from the stored
*average* strategy → recomputable offline → **no retrain**.

## 3. Locked v1 design decisions
| # | Decision | Rationale |
|---|---|---|
| D1 | **Turn only** for v1; flop later (measure-first) | Better range info + one street of leaf-summary. |
| D2 | **Reach-conditioned leaf** (per hero-bucket × villain-bucket continuation value), evaluated against the **live solver reach** at the leaf. Bucketed (per blueprint river-bucket). | A per-bucket SCALAR CFV is only correct if the solver's villain range == the blueprint's, which it never is (review P0). The leaf must be a value *object* dotted with live reach. Escalate to per-hand only if M0(c) shows bucket coarseness dominates. |
| D3 | Multi-valued leaf: opponent picks worst of a **K-set**. K=1 only to bootstrap the pipeline; the **evaluator uses an out-of-leaf BR from M2 onward** (see D9). | K-set and range-mismatch are the same problem from two sides; K>1 restores soundness against a shifting/deviating opponent. |
| D4 | **Solver menu = the river solver's menu** (incl. all-in; a turn jam is near-terminal) | Matches the established path + `blueprint_projection` EV-gate mapping. |
| D5 | EV-gate, SPR/cost-gate, confidence-gate **as on the river** (SPR re-calibrated for turn tree sizes) | Reuse proven gates. |
| D6 | Build on the **latest capped snapshot**; **re-extract leaf values on the final converged blueprint** at ship | Leaf values are blueprint-specific; extractor *correctness* is blueprint-agnostic. |
| D7 | Thread `db_menu_mode` = **capped** everywhere | Mismatch = BUG-011 class. |
| D8 | **Serve v1 live only if M4 LBR strictly drops** vs blueprint — NOT on the confidence gate alone | Confidence defends vs *off-model* play; it does NOT defend vs an in-model opponent exploiting the unsafe (small-K) solve. If LBR doesn't drop, don't serve until the M6 gadget. |
| D9 | All exploitability/eval gates use an **out-of-leaf best response** (the BR may continue richer than the leaf grants the bot) | Else the gate self-grades an exploitable strategy as good (the over-bluff trap: good in-model, loses out-of-model). |

## 4. Components (corrected NEW vs EXTEND; file targets)
| Component | New/Extend | File · notes (review-corrected) |
|---|---|---|
| **A. Offline CFV/leaf extractor** | **NEW (heavy)** | `scripts/bake_cfv.py` + `src/subgame/cfv.py`. Per (turn/river-entry key) a **hero-bucket × villain-bucket continuation-value object**, expectation over river runouts under blueprint play. **No turn→river runout traversal exists in the codebase** (the trainer deals one fixed board/iter; `best_response.py` fixes a 5-card board). This is a from-scratch runout-averaging value propagator over the river subtree per turn context — **substantially heavier than baking the postflop tables** (which is bucketing-only, no tree walk). Threads the capped menu (D7). |
| **B. Leaf value lookup** | NEW | `src/subgame/cfv.py` — runtime: dot the per-hand continuation object with the **live solver reach**; opponent takes worst of K. |
| **C. Turn tree** | **NEW module derived from `river_tree.py`** | `src/subgame/turn_tree.py`. Needs a **third terminal kind = LEAF** (distinct from fold/showdown), a 4-card board, the equal-stack invariant (holds at turn — §9), own aggression cap. Most of `_build`/`_terminal`/`_add_aggressive` is touched → derive, don't "extend". |
| **D. Depth-limited CFR+** | EXTEND `river_cfr.py` | Leaf branch in **BOTH `_terminal` AND `_terminal_one`** (the latter drives `exploitability`/`_br_value`) + a 4-card-board basis (**`H=1128`, not 1081**; `compatible_mass`/`_COMPATIBLE=990` are 5-card-only — need a 4-card variant). "Reuse `exploitability` unchanged" is **wrong** without the `_terminal_one` leaf branch. |
| **E. TurnSubgameSolver** | EXTEND `river_subgame_solver.py` | `_solver_inputs`/`decide` are **river-gated** (`street=='river'`) — add a turn branch. Note the existing **turn all-in guard already fires on the turn** and runs *before* the solve — design the overlap (guard handles near-terminal jams; solver handles deep non-jam betting). |
| **F. Turn-entry ranges** | EXTEND `range_tracker`/`game_session` | Only `river_entry_*` snapshots exist today; add `turn_entry_*` (two lines at `street==2`) + `_turn_path_specs` + JSON round-trip. Genuinely easy. |
| **G. Turn EV-gate baseline** | EXTEND `blueprint_projection.py` | **Parameterize the hardcoded river `3`** (`make_info_set_key(3,…)`, `ba['strg'][3]`, `ba['groups'][3]`) to street 2. |
| **H. Turn-exploitability harness** | NEW | `scripts/measure_turn_exploitability.py` — **must use the D9 out-of-leaf BR**, else meaningless. |
| **I. LBR victim update** | EXTEND `evaluation/lbr.py` | BUG-008 lockstep: the LBR victim must mirror the deployed turn-solver behavior or M4's LBR number is invalid. |

**Unchanged:** blueprint, trainer, the live river solver, the engine. Additive behind `BotStrategy`.

## 5. Milestones (the spine)
- **M-pre — MEASURE FIRST (before any build; cheap, doable now).** (1) Add a turn-entry tracker
  snapshot and compare **belief sharpness/confidence at turn entry vs river entry** — if turn-entry
  beliefs are routinely low-confidence, the confidence-gate defers most spots and the feature's
  value collapses; know this first. (2) Quantify the **addressable band**: the `_facing_allin_guard`
  already covers near-terminal turn jams, and the SPR gate skips high-SPR — so the full solver's
  marginal value is confined to *non-all-in, deep-but-gateable-SPR turn betting*. Measure how often
  that band actually occurs. **GATE: if the band is thin or beliefs are weak, prefer the cheaper
  alternative (one-ply "bet-or-not + size" using a static CFV leaf + live range for the immediate
  action only, no full subtree CFR) and stop here.**
- **M0 — Leaf extractor + REAL validation (the keystone gate). ✅ DONE (2026-06-05) — GO.**
  Built A (`src/subgame/cfv.py`: exact per-hand rollout leaf + reach-conditioned bucket matrix +
  strength partition + card-removal-aware reconstruction). Validated with THREE checks, not one:
  (a) exact leaf zero-sum residual 2e-16 (pipeline correct); (b) error under a **deliberately
  extreme full→broadway villain shift** vs a true rollout under that shift; (c) per-hand-rollout
  vs bucketed-leaf coarseness. **Results across 4 turn boards** (`scripts/validate_cfv*.py`):
  reach-conditioning is **essential** — frozen per-bucket scalar **430–800%** vs reach-conditioned
  matrix **13–16%** under the shift (~30–50×, confirms D2). The blueprint's ~11–16 turn buckets are
  too coarse (14–33%); **leaf resolution is a free lever** → a **~128-bin strength-partitioned,
  card-removal-aware** matrix leaf gives **~9–15% coarseness**. **Review follow-up
  (`scripts/measure_leaf_accuracy.py`, 6 textures × 3 SPRs × 3 shift shapes incl.
  draw-heavy/polarized ORTHOGONAL to the strength partition, FrozenBlueprint snapshot): worst
  under-shift STABLE rel-RMSE = 7.8%; the orthogonal draw-heavy shift (6.0%) is NOT worse than the
  rank-correlated broadway shift (4.7%)** → the leaf is robust beyond the easy case (the earlier
  "13–16%" was narrow-range denominator inflation; the stable metric — error / no-shift RMS —
  normalizes that out). Caution: per-pair POT-normalized error hits ~20–28% on stress textures
  (flush/paired at high SPR, largely SPR-gate-skipped) — carry into M2. v1 leaf = that finer matrix;
  **2-D feature (strength×potential)** is the escalation if M2 flags the leaf. Cost ≈ 2× a river
  solve to build once/solve, then a dot per iter. **GATED on (b)/(c) — passed.**
- **M1 — Turn tree (C). ✅ DONE (2026-06-05).** `src/subgame/turn_tree.py` (`TurnTree`/`TurnNode`/
  `build_turn_tree`/`is_leaf`) mirrors `river_tree.py`'s betting scaffold + chip accounting exactly,
  importing its sized-edge label format + menu/caps (no drift), with ONE difference: a non-fold
  close (check-check / call) is a **depth-limit LEAF** (river to come), not a showdown, carrying
  `(final_pot, leaf_stacks)` for the M0 leaf value fn. An all-in-and-called turn line closes with
  `leaf_stacks==(0,0)` → the leaf fn yields pure equity-to-river → **no special case**. The tree is
  board-agnostic (the 4-card board enters at the M2 CFR basis, not the betting tree). Equal-stack
  invariant enforced AND shown to propagate to equal leaf stacks (the inner-river precondition).
  Tests `tests/test_turn_tree.py` (15, all green; reuse the river chip-conservation/aggression/
  dedup/min-raise/all-in suite + leaf-classification + all-in-called-zero-behind + leaf
  chip-conservation incl. behind + positional alternation + pinned node/leaf counts). River suite
  still green (import-only reuse).
- **M2 — Depth-limited CFR+, K=1 (D + B). ✅ DONE (2026-06-06) — GATE PASSED.**
  - **Stage 1 — leaf plumbing ✅.** `showdown_kernel.build_turn_board_arrays` (4-card `H=1128`
    basis, no showdown ranks — landmine #1); `cfv.leaf_value_vec` (vectorized card-removal-aware
    reach-conditioned leaf, the solve-time form of `bucketed_measure_leaf_cr`);
    `cfv.turn_leaf_matrix_both` (M0 via one `_eval`/bucket; **M1 := -M0^T ENFORCED** — the
    independently-averaged M1 differs by the hero-collapse normalization and silently breaks the
    leaf's zero-sum property → turns the subgame non-zero-sum). Validated
    (`scripts/validate_turn_leaf_vec.py`): M0==per-seat 0.0, M1==-M0^T 0.0, vectorized==dict 4.6e-16.
    NOTE: offline measurement vs a LIVE-training DB must read through a frozen snapshot
    (`FrozenBlueprint`) — two builds seconds apart otherwise read different blueprint values.
  - **Stage 2 — solver ✅.** `subgame/turn_cfr.py` `TurnCFR(RiverCFR)` overrides ONLY the two
    terminal evaluators (leaf → M0/M1 dotted with live reach; fold → 4-card pot transfer); all CFR+
    mechanics inherited (sync-safe). Validated (`scripts/validate_turn_cfr.py`): leaf zero-sum
    exact, root E0+E1 == 0 (2e-16), depth-limited exploitability monotonically decreasing
    (2512→100 mbb/hand at 400 iters — a MECHANICS check, floor not established). Direct engine
    tests `tests/test_turn_cfr.py` (5, synthetic leaf): runs on the 4-card basis w/o showdown
    fields, leaf-cache built once per distinct leaf, exact root zero-sum, converges.
    NOTE: `TurnCFR.exploitability` is overridden purely to DOCUMENT that it measures the
    DEPTH-LIMITED gap (BR pinned to the blueprint river INSIDE the leaf), not true subgame
    exploitability — don't read it as a quality number.
  - **Stage 3 — out-of-BUCKET (in-model-river) exploitability GATE 🔄 CODE-COMPLETE (2026-06-06),
    real run PENDING.** Pieces built + light-validated: (a) turn blueprint projection
    (`build_turn_board_arrays(board4, cards)` now carries pf/strg2/groups2;
    `blueprint_projection.blueprint_turn_strategy_on_tree` at street 2 — landmine #5; validated
    100% turn-key hit on a sample board); (b) `turn_cfr.ExactLeafTurnCFR` — overrides `_terminal_one`
    so the BR walk values leaves by the EXACT `turn_leaf_value_exact` rollout (NOT the bucketed
    matrix the solver used) → the solver can't game its own bucketing; (c)
    `scripts/measure_turn_solve_gate.py` — per turn spot, exact-leaf exploitability of SOLVED vs
    BLUEPRINT. Tiny smoke (1 board, 3 rivers, n=16, 60 iters): blueprint 8232 → solved 2319 mbb
    (−72%), GATE PASS — confirms plumbing + direction, NOT the magnitude. **MEDIUM-fidelity gate
    (2026-06-06, 3 boards, 8 rivers, n=64, 500 iters, 1-core-pinned): 3/3 PASS -- blueprint mean
    7904 -> solved mean 109 mbb (-98.6%), solved ~0 (near-unexploitable out-of-bucket); the smoke's
    -72% was just an under-converged solve.** Delta is reliable (same exact-leaf grader both sides);
    only the absolute scale is approximate at 8 rivers. **FULL run (4-6 boards, 16 rivers, n=128, 800
    iters; ~1-3h heavy) deferred until the BR sweep frees the box -- confirms absolute numbers.**
    **FULL-fidelity gate ✅ (2026-06-06, all 6 textures, 16 rivers, n=128, 800 iters, parallel
    1-core-pinned): 6/6 PASS — blueprint mean 7244 -> solved mean 77 mbb (-98.9%), solved ~0 on
    every texture.** M2 GATE PASSED. (Still out-of-bucket / in-model-river — true-game robustness is
    M3 K-set + M4 LBR.) Still out-of-bucket but IN-MODEL-RIVER (rollout assumes blueprint river); a river-
    deviating villain is only caught by M4 LBR → NECESSARY not SUFFICIENT, does not license serving.
    Budget-granularity fix still TODO (matters for M4 live, not the offline gate).
    **GATE: solved < blueprint exploitability vs the exact-leaf adversary.**
  - **Measurement hygiene (review): all M0/M2 offline scripts read through
    `storage.blueprint_db.FrozenBlueprint`** (a memoizing snapshot) so multi-pass measurement vs a
    LIVE-training DB is internally consistent. M0 accuracy numbers should be (re)taken on a static
    snapshot; the structural-shift sweep is `scripts/measure_leaf_accuracy.py`.
- **M3 — Multi-valued leaf, K=2–4 (B).** K-set = {blueprint, blueprint-BR, per-hand-rollout
  extreme} as a starting default. Re-measure (more robust; exploitability vs out-of-leaf BR drops).
- **M4 — `TurnSubgameSolver` live (E+F+G+I). ⏸️ PAUSED (2026-06-07) — cost/benefit not justified.**
  - **Stage 1 ✅ — the class.** `subgame/turn_subgame_solver.py` `TurnSubgameSolver(RiverSubgameSolver)`
    adds ONLY the turn path; inherits the all-in guard, EV gate, action-mapping, river solve +
    blueprint fallback (factored `_run_guard`/`_gate_and_pick` in the parent — river tests 11/11).
    decide() routes turn→turn-solve, else→parent; turn SPR gate; off-grid turn sizes snap. Tests
    `tests/test_turn_subgame_solver.py` (5). Leaf models blueprint-river (conservative); real river
    robustness is the inherited river SOLVER at serve time (why M3/K-set skipped).
  - **Stage 2 ✅ — wired turn-entry fields** (`turnEntryPot/turnEntryStacks/turnPath` + turn-entry
    range snapshots) into `game_session.bot_public_state`, mirroring the river-entry fields
    (`_street_path_specs` generalizes `_river_path_specs`). Game-session tests 11/11.
  - **LATENCY WALL:** a turn solve is **~20-54s** (n=64/8r=54s, n=48/6r=26s, n=32/6r=19s; measured,
    1-core; leaf-build-bound — ~12 leaves x n x rivers river-evals). >> the river solver's ~8s. NOT
    servable as-is; needs **baked leaf matrices** (deploy = small box → parallel-build hurts
    concurrency; baking at n=32 ~1GB is the deployment-friendly path).
  - **B2 head-to-head (Stage-3 stand-in, 250 hands ea., SERVABLE fidelity n=20):** river-stack
    **+1801 ± 1363** vs blueprint; turn-stack **−260 ± 1469** vs blueprint. Inconclusive on variance
    (~1σ) BUT **no positive turn edge over the deployed river stack.** `scripts/measure_turn_match.py`.
  - **THE INFLECTION:** value is proven only at HIGH fidelity (M2, n=128, ~50s/unservable); at the
    SERVABLE fidelity (low n) a coarse leaf can make the turn solve no-better-than-blueprint and the
    EV-gate (judging with that same coarse leaf) waves bad deviations through; AND M3 showed the
    marginal value over the river solver is small (river solver already gives river robustness). So:
    **high cost (baking) for uncertain marginal value.** A full LBR gate (D8) is infeasible at high
    fidelity (days) and tests the no-edge version at low fidelity.
  - **DECISION (2026-06-07): PAUSE live serving.** Keep the validated M0-M2 work + the class. Pivot
    to higher-value/lower-cost: ship the 25M blueprint, deploy the (already-valuable) river solver.
    Revisit turn/flop solving only if baking infra exists or the blueprint plateau becomes the
    binding constraint. **Stage 3 (LBR gate) NOT run** — gated behind a decision to resume.
- **M5 — Flop (later).** Reuse M1–M4 one street up, only if a measurement shows it beats the
  blueprint enough.
- **M6 — Safety gadget (Phase 5a).** Reach/opt-out gadget → provably no-more-exploitable.

## 6. Key concepts
- **CFV:** value of a hand if the game reaches this spot, under blueprint play from here, weighted
  by **opponent+chance** reach (not own). **It is defined relative to a specific opponent reach** →
  the leaf must be re-weighted by the *live* reach (D2), not frozen.
- **Multi-valued leaf / K-set:** opponent picks the worst-for-hero of K continuations; restores
  soundness against a range that the solve has shifted. K=1 = the exploitable single continuation.
- **Safe vs unsafe:** unsafe trusts the frozen root range (a thinking opponent adapts → over-bluff
  trap); safe gives the opponent a blueprint-value opt-out (gadget) → provably no-more-exploitable.

## 7. Cost & gating
Turn trees are bigger than river; reuse the SPR gate (**re-calibrate** the node-count→budget
threshold for turn) + a node/time cap, and **fix the per-block budget check** (currently only after
a `check_every` block → can overrun). The offline extractor is a heavy per-blueprint batch pass
(re-run on blueprint change, D6).

## 8. Risks & limitations (re-prioritized)
- **P0 — leaf range-dependence.** A scalar bucket CFV breaks correctness under the solve's range
  shift; the leaf must be reach-conditioned (D2). *Most fundamental — more than coarseness.*
- **P0 — self-grading gates.** M0(a)/a frozen-leaf M2 look good but prove little; use out-of-leaf
  adversaries (D9, M0 b/c). NOTE (review): even the M2 Stage-3 gate is only out-of-BUCKET, still
  in-model-river — the genuinely out-of-model check is M4 LBR (villain deviates downstream); Stage-3
  is necessary not sufficient and does NOT license serving.
- **P0 — measurement vs a live-training DB.** Multi-pass offline measurement read different
  blueprint values mid-run (a real artifact that produced a spurious mismatch in review). FIXED:
  all M0/M2 scripts read via `FrozenBlueprint`; re-take M0 numbers on a static snapshot.
- **P1 — leaf coarseness** bounds turn sharpness at the blueprint's river-bucket resolution
  (per-hand rollout = v2 escalation).
- **P1 — weak turn-entry belief** + thin addressable band → the feature may not beat the blueprint
  (M-pre gates this).
- **P1 — unsafe v1** exploitable by an in-model opponent → serve only on LBR drop (D8), gadget at M6.
- **P2 — cost/latency/extraction effort** under-estimated; M0/M1/M2 are multi-week, not warm-ups.

## 9. Resolved / open
- **Equal-stack invariant: RESOLVED — holds at turn entry** (stacks reset per hand + symmetric
  prior action), so no lift needed; confirm as an M1 acceptance check.
- Open: the K continuation set (M3); per-hand rollout leaves (v2 if M0 b/c demand); flop (M5).

## 10. References
Brown & Sandholm, *Depth-Limited Solving for Imperfect-Information Games* (Modicum); *Safe and
Nested Subgame Solving*. Code: `river_subgame_solver.py`, `river_tree.py`, `river_cfr.py`,
`solve_control.py`, `blueprint_projection.py`, `range_inputs.py`, `src/evaluation/showdown_kernel.py`,
`range_tracker.py`.

## 11. Codebase landmines (from the feasibility review — address in the noted milestone)
1. **`build_board_arrays` assumes a 5-card board** (`H=1081`, `_COMPATIBLE=990`); turn = 4-card →
   `H=1128`, and `board[:5]` silently returns 4 cards. Affects the whole turn-CFR basis. (M1/M2)
2. **Three terminal evaluators**, not one: `_terminal`, `_terminal_one`, and the `exploitability`
   path — the leaf must live in all that matter, or the M2 gate measures a showdown leaf. (M2)
3. **EV-gate / `hand_action_evs` normalize by 5-card `compatible_mass`** — leaf values must be in
   the same units or gate margins are off-scale. (M2/M4)
4. **No turn-entry snapshot exists**; `_solver_inputs`/`decide` are river-gated; the turn all-in
   guard already fires on the turn — design the overlap. (E/F)
5. **`blueprint_projection.py` hardcodes river `3`** in 3 spots. (G)
6. **SPR gate is river-calibrated** (`SOLVER_MAX_SPR=6.0` from river node counts) — recalibrate;
   the budget-granularity bug bites harder. (M2)
7. **LBR victim lockstep** (BUG-008) — update `lbr.py` for the turn solver or M4's LBR is invalid.
8. `showdown_kernel.py` is in `src/evaluation/`, not `src/subgame/` (doc fix).

## 12. REVIVAL — Stage 1: the consistency rebuild (continual re-solving). Started 2026-06-15.

The N0 failure (NN_LEAF_PLAN §6d) is NOT a broken layer — it is three CROSS-layer
inconsistencies that only appear once the turn solver sits above the river solver. Each is
fixed at an identified site. The 5a gadget machinery (`blueprint_cfv`, `river_cfr.run_gadget`)
is the foundation. Heavy validation (N0′) is deferred to post-retrain (free CPU + the right
blueprint; leaf values re-extract on the final blueprint per D6). Order: 1a → 1b → 1c → Stage 2.

**STATUS (2026-06-15): all of Stage 1 (1a + 1b + 1c) is IMPLEMENTED and smoke-validated; turn
5/5 + river 14/14 tests green. The only thing left for the revival decision is the heavy Stage-2
N0′ gate (`ExactLeafTurnCFR(adversary='river_br')` exploitability + a power-budgeted AIVAT match),
which is deliberately deferred until the 100M retrain finishes and the CPU frees up.**

**1a — Own-range chaining ✅ IMPLEMENTED (2026-06-15, validation deferred to post-retrain).**
5-part chain: `RangeTracker.observe_solved(prob_by_hand)` (per-hand multiply, no confidence
update) + `_pick_engine_action` records `chosenTreeAction` + `TurnSubgameSolver._attach_hero_range_update`
(builds `last_debug['heroRangeUpdate']` = solved per-hand P(played action) ONLY when the EV gate
DEVIATED) + `advance_bot_turns` threads it + `apply_action(solved_hero_probs=)` routes the bot-range
update through `observe_solved`. Turn-only (the river is last-street, no downstream). Unit-tested
(test_range_tracker `test_observe_solved_per_hand_update`); range-tracker 23/23, game-session 15/15,
no regression. HEAVY validation (turn deviation → river-entry hero range differs from blueprint →
N0′) deferred to free CPU + the retrained blueprint.
- Break: `game_session._advance_street` snapshots `river_entry_bot = d['bot_range']`, the hero
  tracker updated by observing the bot's actions under the BLUEPRINT model. After a turn-solver
  deviation the bot reaches the river with a DIFFERENT range, but the river solver still assumes
  the blueprint range → it mis-balances value/bluffs and plays worse (this is what degrades the
  previously-great river solver, per §6d thread 1).
- Fix: when the bot's turn action came from the turn SOLVE, update `bot_range` with the solver's
  per-hand strategy at the turn node (`solver.average_strategy(node)` -> [H,A]) instead of the
  blueprint model. `solve_turn_for_action` already holds the solved CFR in `info['cfr']`; expose
  the [H,A] solved strategy (+ the node) so `game_session.apply_action`'s bot path observes the
  bot's action under THAT strategy for the hero range. (Villain/opp_range update is unchanged —
  it already observes the bot's realized action.)

**1b — CFV chaining ✅ IMPLEMENTED (2026-06-15, validation deferred).** As predicted it mostly
fell out of 1a + the 5a gadget: `game_session` sets `turn_deviated` when the turn solve deviates
(set in apply_action, exposed as `turnDeviated` in bot_public_state's river block, reset per hand);
the river `_solver_inputs` threads it; `solve_for_action(turn_deviated=)` forces the gadget to the
**'belief' anchor** (clamp the villain to its blueprint-river-CFV opt-out at the turn-solved hero
reach [1a] + tracked villain = the turn's promise) instead of auto-exploiting; `_safe_solve` gained
an `anchor=` override. So after a turn deviation the river honors the turn's promise -> the turn plan
is a valid lower bound (Burch/Brown nested-safe argument). No regression (river 14/14, game 15/15).
NB: the standalone-safety vs consistency tension (belief-clamp isn't wrong-belief-robust) is a
Stage-4 (turn-root gadget) concern, not 1b.
- Break: the inherited river path solves the gadget anchored to `blueprint_cfv` at the TRACKER
  reaches; but the turn solve promised the villain specific CFVs (§6d thread 2: "the turn plans
  for a river that won't happen"). The river must honor the turn's promise.
- Fix: this LARGELY FALLS OUT of 1a. The turn leaf is, by construction, blueprint river-play
  under the turn-SOLVED entry ranges -> the per-realized-river opt-out IS
  `blueprint_cfv(river_tree, ba, raw, turn_solved_reach0, turn_solved_reach1, villain_seat)` =
  the EXACT 5a function called with the turn-solved river-entry reaches (which 1a now produces)
  instead of the tracker reaches. So: ensure `_safe_solve`'s gadget anchor uses the turn-solved
  river-entry reach when the river follows a turn deviation. New code is small once 1a lands.

**1c — Exact-leaf EV gate ✅ IMPLEMENTED (2026-06-15, validation deferred to post-retrain).**
- Break: `_gate_and_pick` -> `hand_action_evs` reads the solver's BUCKETED leaf (TurnCFR._terminal
  matrix) -> the gate SELF-GRADES (a deviation that looks good under the coarse leaf passes even if
  it loses under the true continuation; §6d / N0 "EV-gate self-grades").
- Fix (landed): `_gate_and_pick` now calls `self._hand_action_evs(info, node, row)` (was the bare
  `hand_action_evs`). The base `RiverSubgameSolver._hand_action_evs` keeps the bucketed behaviour
  (river leaves are exact already); `TurnSubgameSolver._hand_action_evs` OVERRIDES it to build an
  `ExactLeafTurnCFR`, copy the solved `strat_sum`, and read the bot row out of
  `node_action_values` valued by the EXACT blueprint river rollout. `ExactLeafTurnCFR._terminal`
  was added (two-sided exact leaf via `_terminal_one` per seat) so `node_action_values` grades on
  the true continuation, not the bucket matrix. `solve_turn_for_action` stashes
  `board4`/`rivers`/`ba_cache` in `info` for the exact path.
- Validated (smoke, 2026-06-15): on a turn spot the exact EVs differ from the bucketed EVs
  (gate is genuinely active, not a no-op), e.g. `allin` exact −32.85 vs bucketed −36.29; cost
  ~17s/solve at solve fidelity (offline gate, not live). `test_turn_subgame_solver.py` 5/5 and
  `test_river_subgame_solver.py` 14/14 green (no regression on either path under the new EV call).

**Stage 2 — the redesigned N0′ gate (what decides if this ships). INSTRUMENTS CODED 2026-06-15;
the heavy RUN is deferred to post-retrain.**
- (i) Exploitability gate — `ExactLeafTurnCFR(adversary='river_br')` is the out-of-MODEL-river
  adversary (BR also deviates on the river) -> exposes the frozen-range trap the fixes target.
  ALREADY CODED: `scripts/measure_turn_solve_gate.py --adversary river_br` (blueprint-turn vs
  solved-turn exploitability under the river-BR leaf). Run on free cores post-retrain.
- (ii) Power-budgeted real-game match — `scripts/measure_turn_match.py` (AIVAT, ~5-10k hands; the
  250-hand ±1363 mbb run was hopeless): the consistency-fixed turn stack must beat the RIVER-ONLY
  stack (+1801 mbb baseline), not just the blueprint. **REWRITTEN 2026-06-15** so it is a REAL N0′
  instrument: it now drives every hand through `advance_bot_turns` (so the 1a/1b/1c continual-
  re-solving chaining is LIVE, exactly like serving) and adds `--aivat` (c1+c2+c3 live-AIVAT).
  The OLD version drove a bare `apply_action(a)` loop that bypassed `advance_bot_turns` ->
  it measured the turn solver WITHOUT the consistency fixes (a silent false signal); fixed.
  - **AIVAT c2 (river-runout CV) added 2026-06-15** for more power per hand (the 250-hand run was
    hopeless on variance). c2 needs B's turn-board range; the match feeds it the bot's OWN on-model
    `river_entry_opp` RangeTracker snapshot (duck-types as LBR's BotRange via `.hands`/`.w`) --
    more accurate than replaying LBR's BotRange (BUG-008 drift) and valid because the opponent IS
    the blueprint that belief models. `aivat._hand_variates` prefers `rec['river_range']` over the
    events replay; unbiased for any fixed range (conditional mean 0). Verified: c2 fires on every
    river-reaching hand, beta(c2) non-zero, ~43% var-reduction on an 80-hand probe.
  - **Latent 1a breakage fixed:** the shared AIVAT record wrapper `compare_gadget_policies.
    _play_and_record.wrapped(action)` (reused by `test_aivat_live`) didn't accept the
    `solved_hero_probs=` kwarg 1a's `advance_bot_turns` now always passes -> crash mid-hand,
    uncaught because the test wasn't re-run after 1a. Wrapper now accepts+forwards it.
  - **BUG found+fixed by the rewrite's in-match smoke (2026-06-15):** `_gate_and_pick` is SHARED
    between the turn and the inherited river path, so a `TurnSubgameSolver` handling a RIVER
    decision dispatched into the 1c turn override of `_hand_action_evs` with river `info` (a base
    `RiverCFR`, no `tb_idx`) -> `AttributeError`. Fix: the turn override defers to `super()` unless
    `info` carries `'board4'` (only turn info stashes it). Neither suite caught it (river tests use
    the base solver; turn tests only make a turn decision) -> regression test
    `test_turn_solver_handles_river_decision` added (turn 6/6). In-match smoke now clean:
    `turn_solver` + `river_solver` both fire, 1a chains in-match (heroRangeUpdate observed).
- If it fails WITH consistency fixed, the M-pre thin-band explanation wins -> stop for good.

**Test strategy (CPU-light now; heavy N0′ post-retrain):** per-fix unit smokes on a single turn
spot -- 1a: river-entry hero range differs from blueprint after a forced turn deviation; 1c:
exact-leaf EV differs from the bucketed EV on a spot. NOT the heavy match (that fights training).
