# Turn-Solver Latency Plan — v3

Status: DRAFT v3 (2026-06-20), after 4-agent review. Scope: the **turn** only (the river clears 12 s and
is left alone). Related: [NN_LEAF_PLAN.md](NN_LEAF_PLAN.md), [DEPTH_LIMITED_SOLVER_PLAN.md](DEPTH_LIMITED_SOLVER_PLAN.md).

## 0. Budget & levers (v3 — the grade is now first-class)

- **12 s is the IDLE target** (one solve, full CPU). Under contention the active (within-cap) solves may
  **LOAD LONGER** -- a generous wall cap kept < the client/LB timeout -- so they COMPLETE rather than
  degrade to blueprint. (Viable ONLY once the latency work makes the leaf fast; on the un-optimized Micro,
  2-way contention is too slow → blueprint.) Beyond the cap, overflow gets **instant blueprint**
  (non-blocking acquire), NOT a wait. Reconcile today's 10/16/12/30 into: idle target 12, a generous solve
  wall cap, non-blocking permit acquire.
- **TWO things must fit that 12 s, budgeted SEPARATELY: the SOLVE and the GRADE.** (The exact gate grade
  is ~as expensive as the solve, is uncacheable across BR calls, and the NN does NOT speed it — on a cold
  board the GRADE is the deeper bottleneck. This is the review's central finding.)
- **"Preserve buckets" applies to the SOLVE only, and is CONDITIONAL on Phase 0.** The GRADE deliberately
  runs at *lower river fidelity* than the solve (with a variance-aware margin, §2). If Phase 0 shows the
  cold solve+grade can't fit 12 s with the solve's buckets preserved, we reduce — **grade rivers first,
  solve buckets last.**
- **Lever priority:**
  - SOLVE: cache + parallel (no quality loss) → NN leaf → bounded menu → lower solve `n_buckets`/`leaf_rivers` (last).
  - GRADE: cache + parallel → **sample rivers** (deliberately coarser; variance-margined) — the grade is
    a conservative gate, not a fidelity-critical strategy.

### TWO different NNs — do not conflate
- **Leaf NN (Phase 2, WE DO THIS):** small (~MB) net predicting the leaf *value* (board → matrix). The bot
  **still solves in real time** — the NN just makes the leaf instant. Cheap (~weeks; a regression to the
  rollout matrix already computed). "Help the solve."
- **DeepStack policy/CFV net (Phase 6 alternative, REJECTED):** large net predicting the *solved strategy*
  directly → **no real-time solving**. $1000s / GPU-months / research-grade. "Replace the solve." NOT built.
- "Scale = cheap solves" means the **leaf-NN-accelerated real-time solve**, not a baked/learned policy.

## 1. Problem (corrected by review)

- Cold leaf build dominates; the heavy primitive is **`build_board_arrays`** (Python loop over ~1081
  hands × 4 streets, per river), **cached only per-solve** → every solve is cold w.r.t. the leaf.
- **TWO bottlenecks:** the SOLVE leaf AND the **GRADE** (`turn_leaf_value_exact` — a full per-river
  blueprint rollout, **reach-dependent so uncacheable across BR calls**, "offline-only" per the code).
  The NN speeds the solve; **nothing cheap speeds the grade except sampling fewer rivers.**
- Deep-SPR timeout = **leaf-COUNT** explosion (more bet sizes → more distinct `(final_pot, leaf_stacks)`
  leaves), not just CFR-tree.

## 2. The GRADE as a first-class target (v3 headline)

- The grade must fit 12 s **separately** from the solve. Speed it with: (a) the same `build_board_arrays`
  cache + parallel, and (b) **sampling fewer rivers** than the solve.
- **Sampling is safe ONLY with a variance-aware gate:**
  - Margin ≥ **k·SE of the sampled grade** (NOT the fixed 4× non-converged margin — that's for CFR
    convergence, a different error source; using it for sampling variance is unjustified).
  - Grade rivers **disjoint from the solve's sampled rivers** (so the grade independently checks the
    solve's river-selection bias — same rivers would inherit the solve's low-river bias).
  - **Never grade with the NN** (correlated errors = self-grading = the N0 root cause). The grade stays
    on the real rollout, just coarser + margined.
- **Trade-off (DESIGN APPROVAL #1):** a coarser, variance-margined grade is **more conservative** → it
  passes **fewer** turn deviations → **more decisions fall back to blueprint**. We accept a slightly
  less-aggressive turn solver to keep it safe + live.

## 3. Phases

### Phase 0 — Profile BOTH the solve AND the grade (cold + warm) — ~1-2 days

**FIRST RESULTS (2026-06-20, `scripts/profile_turn_solve.py`, 1 solve @ SPR 2.1, n_buckets=24/leaf_rivers=4):**
cold 18.5 s / warm 14.7 s. Breakdown: **SOLVE leaf build (`turn_leaf_matrix_both`) = 12.5 s** (of which
`_eval`, the per-bucket river-tree eval, = 9.9 s over **34,574 calls**); CFR ≈ 3.5 s; **GRADE
(`turn_leaf_value_exact`) = only 2.4 s.**
- **The GRADE is NOT the bottleneck** — refutes the review's "grade is the deeper, uncacheable problem."
  At 2.4 s it fits; grade-sampling + variance-margin is **NOT load-bearing** (kept available, not required).
- **The SOLVE leaf dominates** (12.5 s), driven by `_eval` = `leaf_rivers × n_buckets × tree`. Cache +
  parallel + NN all target exactly this.
- **Warm barely helps** (per-solve `ba_cache` discarded) → every solve is cold → a cross-solve canonical
  cache is the first lever. The leaf build is **eager/pre-CFR**, so the 12 s budget can't cap it.
- **DRILL-DOWN RESULT:** the hot primitive is **`blueprint_projection._project` = 10.8 s** of the 12.5 s
  leaf (projecting the blueprint onto each river tree, per river/leaf). **`build_board_arrays` does NOT
  rank** — the plan/review named the wrong primitive. `_project` is **reach-INDEPENDENT** (board +
  pot/stacks + blueprint only) → **canonically cacheable across solves**, and is currently discarded
  every solve (per-solve `ba_cache`). So the DOMINANT cost is recoverable by the Phase-1 no-quality-loss
  **cross-solve canonical cache** (`scripts/bench_turn_solve.py`). (Cache key = canonical board + pot/
  stacks + blueprint VERSION.)

**FRESH-BOARD FLOOR (2026-06-20, `bench_turn_solve.py --fresh 10`, wall-clock, no cProfile):** 10 DISTINCT
boards @ SPR ~2 → **mean 14.0 s / p50 14.0 / p90 15.8** (low SPR = best case; deep turns worse). KEY
IMPLICATION that REFRAMES the cache:
- The `_project` cache keys on the canonical board → it only HITS on a **repeat board**. A single player
  rarely repeats a turn board → the cache's value is **cross-player popular-board warmup at SCALE**, LOW
  at current low traffic. **So the cache does NOT speed up the current fresh-board experience (~14 s).**
- **Parallel is useless on the 0.25-vCPU Micro** (sub-one-core, nothing to parallelize across).
- ⇒ **The cold-fresh case (the actual current experience) needs the NN leaf** (eliminates the 10.8 s
  `_project` on EVERY board) OR a bounded menu (fewer leaves) OR fewer buckets. The cache is still worth
  building (cheap, no quality loss, pays at scale) but is NOT the "fast now" lever — **the NN is.**
- **BUILD-ORDER FLIP:** for fast-on-the-current-box, NN/bounded-menu > cache. Cache = a scale optimization.

Per-stage timing: `build_board_arrays` (count × ms), the per-bucket `_eval` loop, CFR, **and the exact
grade**, cold vs warm, plus **per-solve peak RSS**. **Output gates the "preserve buckets" directive** — if
the cold grade alone can't fit its share of 12 s, "preserve buckets" is revisited here, not asserted.

### Phase 1 — Cache + parallelize (no quality loss) — ~1 week
- **NEW canonical cross-solve LRU on `build_board_arrays`** (board-only, suit-isomorphic; **NOT** a reuse
  of `_RIVER_BOARD_CACHE` — different payload). **Thread-safe** (locked, like the existing singleton-BOT
  caches) — an unlocked `OrderedDict` LRU corrupts under `--threads`. Applies to solve AND grade.
- **Parallelize the leaf build** (rivers × buckets, embarrassingly parallel). NB on a fractional-vCPU box
  this competes with concurrency (Phase 5): one fast parallel solve vs N slow ones — favor the former.
- **Post-recycle warm-up** (workers recycle ~daily → cold-cache thundering herd) or accept the first
  post-recycle minute degrades to blueprint.

### Phase 2 — NN leaf (SOLVE) + fast independent GRADE — ~weeks
- **NN leaf** regresses the range-independent `M0` (board → `[B,B]`; verified range-independent). Speeds
  the **solve** only. **VERSION-STAMPED to the blueprint** (`centroid_hash`-style) + **stale-NN fallback
  to the exact build** (the 30/24 v2 re-fit is done + dev-served → a blueprint swap invalidates `M0` targets).
  Healthz exposes NN version + blueprint stamp; an `ALLIN_NN_LEAF` flag + kill-switch.
- **Fast grade** = cache + parallel + sampled rivers (§2), never the NN. N0′ must validate **with the NN
  leaf backend served**, not the exact leaf.

### Phase 3 — Bounded action menu (only if 12 s still missed)
Coarser *solve* menu. **Re-opens N0′** (changes served play). Verify the gate's blueprint-baseline
projection still represents the blueprint on the coarse menu.

### Phase 4 — Lower SOLVE `n_buckets`/`leaf_rivers` (last resort, conditional on Phase 0)

### Phase 5 — Concurrency / serving on the cheap box (REWRITTEN to reality)
The box is **Lightsail Micro: ~0.25 vCPU, ≤1 GB, 2 gunicorn workers**. The solve semaphore is
**per-worker** (`threading.BoundedSemaphore`), so the env count is **× workers box-wide**.
- **Box-wide cap = 2 simultaneous solves** (directive). The semaphore is PER-WORKER, so with 2 gunicorn
  workers that's **`ALLIN_SOLVE_PERMITS=1` env-PINNED** (1 × 2 workers = 2 box-wide; setting env=2 → 4
  box-wide, which 0.25 vCPU can't do). Pin it -- the auto-default reads the *host* core count. Account for
  the SEPARATE explorer-river pool too. Within the 2: each solve COMPLETES, loading longer under contention
  (§0); beyond the 2 → instant blueprint. cap>2 needs a bigger box (2 solves already split 0.25 vCPU).
- **NO pot-prioritized admission in v1** (DESIGN APPROVAL #3): a semaphore exposes only a count, and the
  2 workers can't see each other's in-flight solves — "top-k active pots" needs cross-process state. The
  EV gain of "biggest pots first" is second-order vs the cap. v1 = cap + graceful fallback (FCFS). Revisit
  pot-priority only with workers=1 + a lock-guarded heap, or on a bigger box.
- **BUILD the non-blocking blueprint fallback** (it's the OPPOSITE of today): currently no-permit does
  `acquire(timeout=30)` → wait 30 s → 503. Change to `acquire(blocking=False)` → **play blueprint inline,
  no solve** (needs a "blueprint-only this turn" flag through `advance_bot_turns`/`decide`). This is the
  single most concrete actionable item and it is **unbuilt**.
- **Memory budget per permit** (not just CPU): each solve holds leaf matrices + `ba` arrays on top of
  2 workers × 127 MB tables + 52 MB river LRU on ≤1 GB — measure peak RSS (Phase 0) before raising the cap.

### Phase 6 — Scale-out = cheap solves, NOT a baked policy (RE-SCOPED)
- **Re-scoped (DESIGN APPROVAL #4):** baking the *solved policy* is **intractable** (the policy is
  range-conditional — CFR × the live tracker range — not a board function). Only the range-*independent*
  leaf is bakeable, which is **just the Phase-2 NN's target**. And "a net that predicts the solved policy
  directly" is the **DeepStack range-conditional CFV net that NN_LEAF_PLAN explicitly rejected**
  ($1000s / GPU-months / lost-to-Slumbot).
- **So scale = the Phase-2 leaf NN makes each real-time solve cheap → near-arbitrary concurrency follows
  from cheap solves + bigger boxes, not from eliminating real-time solving.** There is no separate "bake
  the policy" phase. (If we ever want true no-real-time serving, it's the rejected DeepStack project — a
  deliberate, separate, expensive decision, not this plan.)

## 4. Correctness invariants (v3)
- **Grade-exact + variance-aware margin**, independent of the solve's leaf; never the NN (§2).
- **"No worse than blueprint" = GTO/safe-gadget arm ONLY**, and **in-model-river** (model-relative, not
  absolute → N0′/LBR is the real oracle). The **exploit arm** (belief-anchored, no floor) is **carved
  out** — validated by live A/B, not the gate. The `multivalued_leaf` branch is out of scope (keep the
  NN on the exact-graded served path).
- **NN lifecycle:** blueprint-version stamp + stale fallback + healthz + kill-switch.
- **Cross-street `1a`/`1b`** read the SOLVED strategy / propagate `turn_deviated` → downstream of the leaf
  backend → survive both phases (but only *validated* by N0′; never proven to fix N0 yet).
- **Fallback:** any error/timeout → blueprint. Non-converged solve → 4× margin (for convergence, NOT a
  substitute for the grade's variance margin).

## 5. Validation — N0′ is a BLOCKER for served default-on
- vs the **blueprint opponent that failed N0** (reproduce the negative, show ≥0) **plus** an adaptive
  opponent — a maniac-only gate is vacuous.
- **Thousands of hands + a power calc** (N0's 250 hands were noise: −260 ± ~1169), paired-AIVAT.
- **Run with the served backend** (NN leaf + sampled grade), not the exact leaf.
- **Golden-spot regression:** gated served action within ε of the exact-leaf solver on a fixed battery.
- Any Phase-3 menu change re-opens N0′. Phase 6 gated on N0′ **and** a frozen solve config.

## 6. Success criteria
- Bot turn decision (solve + grade) **≤ 12 s at any SPR**, with the SOLVE's `n_buckets`/`leaf_rivers`
  preserved **IF Phase 0 allows** (else reduce grade rivers first, solve buckets last).
- **Cap=2 box-wide** (env=1/worker × 2 workers) sustainable post-latency-work; peak RSS within ≤1 GB;
  within-cap solves load longer under contention, beyond-cap → instant blueprint. cap>2 = bigger box.
- **N0′ decisively ≥ blueprint** (with the NN backend served) + golden-spot passes.

## 7. Design decisions (RESOLVED 2026-06-20)
1. **GRADE fidelity: TEST FIRST** (Phase 0) — measure the cold-grade cost, then decide full vs sampled
   rivers (variance-margined sampled gate only if the full grade can't fit). Not pre-decided.
2. **Concurrency: box-wide cap = 2** (`ALLIN_SOLVE_PERMITS=1` per-worker × 2 workers). Within-cap solves
   LOAD LONGER under contention and complete (not blueprint); beyond-cap → instant blueprint. cap>2 = bigger box.
3. **Pot-priority: DROPPED for v1** (FCFS + blueprint fallback).
4. **Scale: cheap solves + bigger box** (the Phase-2 leaf NN). NOT the DeepStack range-conditional policy net.
5. **"Preserve buckets": conditional on Phase 0** — decided by measurement; reduce grade rivers before
   solve buckets if forced.

## 8. Open questions
- Grade: how many rivers, and the `k` in the variance margin (calibrate in Phase 0).
- Is Phase 1 (cache+parallel, buckets preserved) alone enough at 12 s, deferring the NN?
- NN data-gen cost vs the *current* deeper live tree (`LIVE_TURN_MAX_AGGRESSIONS=5`) + 30/24 re-fit.
- Cross-worker concurrency: stay per-worker (cap×workers) or move to a shared primitive?
