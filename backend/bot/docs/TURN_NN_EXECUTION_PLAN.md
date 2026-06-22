# Turn NN-Leaf — Execution Plan (river-fidelity decision + cloud run)

> ❌ **CLOSED — DEAD END (2026-06-22). DO NOT EXECUTE.** Real-time turn solving lost −66 mbb/hand H2H; the
> NN here would learn the *blueprint* leaf, which is the broken target. Turn solver UNWIRED from serving.
> The only viable revival is a *range-conditional CFV net* (DeepStack-scale $$$), not this plan. Kept for
> the record. Verdict: `docs/private/ROADMAP.md` "Dead Ends"; analysis below + `TURN_BAKE_VS_NN_SPEC.md`.

Status: DRAFT (2026-06-21), for review. Decides the OPEN question — **how many rivers** to train the NN
target on — by MEASURING the EV cost of lower fidelity, then runs the cloud data-gen + train + validate.
Builds on: TURN_BAKE_VS_NN_SPEC.md, TURN_LATENCY_PHASE0_FINDINGS.md, the verified generator
(`src/subgame/turn_leaf_gen.py`), and the measured noise floor (12 riv → 0.20, 32 riv → ~0.12, 48 → 0).

## ⚠️ 2-AGENT REVIEW (2026-06-21) — THREE blockers SIT ABOVE the river question
The river-fidelity decision is **downstream of bigger, cheaper-to-answer blockers**. Resolve these FIRST:
1. **DOES THE TURN SOLVE EVEN WIN? (N0′ / edge).** It FAILED head-to-head once (N0); the 1a/1b cross-street
   fixes are UNVALIDATED. **If the turn solve's edge over the blueprint ≈ 0, the entire NN/bake/river
   effort is moot.** Phase 1 must measure the EDGE first and stop here if it's ~0.
2. **THE GATE LATENCY (real, but OPTIMIZABLE — corrected 2026-06-21).** The NN makes the SOLVE leaf
   instant but the EV gate is a separate exact rollout the NN CANNOT speed (grading with the NN = N0
   self-grading). Naive scale of Phase-0's 2.4s@4-riv = ~19s@32 — over budget. BUT the gate's DOMINANT
   cost is `blueprint_strategy_on_tree` (the SAME reach-INDEPENDENT projection that dominates the solve);
   only the final `RiverCFR._eval` (cheap) uses the reach (cfv.py:101-104). So the gate IS optimizable:
   (a) the **canonical-board projection cache** — the no-quality-loss Phase-1 lever — speeds BOTH solve
   AND gate (the "gate is uncacheable" claim was wrong: only the cheap final value is reach-dependent);
   (b) **cheaper-river gate** (fewer rivers + variance margin, NN-independent → no self-grading); (c) **skip
   the gate on "keep"** (only grade actual deviations — most spots keep the blueprint). The ~19s figure
   assumes NO caching; with the projection cache the gate is a fraction of that. (Cache is SHARED by the
   bake — a shared cost/win, not a bake-vs-NN differentiator.)
3. **30/24 BLUEPRINT CUTOVER.** The NN/bake target + partition + gate all key on the SERVED blueprint
   version. The NN trains on 52.5M/30-24, but prod serves an older 20/16-era snapshot. **Until prod serves
   the 30/24 (assets-v2 cutover — currently "later"), the stamp mismatches → self-disables → the run is
   wasted.** Hard-gate the cloud gen on "30/24 is FINAL and SERVED."

**Net: the river-fidelity work assumes a turn solve that (a) wins and (b) can serve in budget — neither is
established. Measure the EDGE + the GATE cost (both cheap) BEFORE the 8h cloud gen.**

## NO BOX MIGRATION (the NN path's whole point — confirmed)
Every constraint we found stays on the CURRENT box (dev `snap_52500000`; prod = the existing Lightsail
Container) — NO disk-backed migration (that's the BAKE's cost, which the NN exists to avoid):
- **NN leaf:** ~MB net, numpy-only inference, ships IN the existing Docker image. No storage migration.
- **Data-gen/train:** a TEMPORARY cloud compute box (or the dev PC) for the one-time run → produces the
  ~MB net. Not a serving box. No migration.
- **Partition (32-river):** compute-only (the cheap rank pass), no storage.
- **Gate:** kept cheap by the **cheaper-river gate + skip-on-keep + a runtime projection LRU** — all
  COMPUTE, no new storage. (A *baked* projection table WOULD need a disk-backed box → we do NOT bake it;
  the runtime LRU + cheaper-river gate keep the gate on the current box.)
- The only non-migration dependency is the obvious one: prod must eventually serve the 30/24 blueprint the
  NN trained on (the assets-v2 ship — NOT a box change, just shipping the blueprint you already have).
CAVEAT: "no migration" is the NN path's PROPERTY; the NN path's RISK is whether it works (edge + floor +
featurization). The trade stays: BAKE = certain-but-migration; NN = no-migration-but-unproven.

## The open decision: river fidelity (N)
`N` = rivers used for the NN's M0 target AND (by the consistency gate) the served solver's partition + EV
grade. Higher N is a 3-way trade:
- **Floor ↓** (more learnable): 12→0.20, 24→~0.14, 32→~0.12, 48→0 (deterministic, no sampling).
- **Leaf quality ↑** (closer to the true 48-river value): single-sample error vs truth ≈ floor/√2
  (32 riv ≈ 8.5%; the CURRENT live solve at 4 riv ≈ 25% — so any N≥24 is a big upgrade on the status quo).
- **Cost ↑** (linear in N): gen time AND the serving EV-gate latency both scale ~linearly with N
  (32 riv gate ≈ 5 s; 48 riv ≈ 10 s, tight in the 12 s budget).

**We do NOT pick N by intuition — Phase 1 measures the EV cost of each N and we pick the knee.**

## Phase 1 — EDGE + EV-GAP + GATE-COST (one harness; redesigned per review) — ~1-2 h
**1a. THE EDGE (gate everything on this) — measured at HIGH fidelity, NOT 4 rivers.** The CURRENT solve
uses 4 rivers (~25% leaf error) → of course it's weak; measuring ITS edge tests the thing we're fixing.
So measure the edge using the **48-river (best-possible) leaf**: does a WELL-RESOLVED turn solve beat the
blueprint? (gold-48-leaf solve EV − blueprint EV, graded on the exact rollout, on DEVIATING spots).
**If even the 48-river solve's edge ≈ 0 → STOP, the turn-solve concept is dead.** If 48-river HAS edge but
4-river doesn't → that PROVES the fidelity/NN work is the point. (This is also the cleanest single experiment.)
**1b. THE EV-GAP (choose N), redesigned:** gold = **48-leaf with the SERVED partition + gate held at N**
(vary ONLY the leaf the NN controls — not partition, not gate). Metric = EV of the **GATED SERVED ACTION**
under N-leaf vs 48-leaf, **graded on a common exact-48 rollout**, computed **only on spots where the gold
DEVIATES** (averaging over gate-KEEP spots dilutes to a meaningless ~0). Report per-SPR, oversampling
high-SPR (where leaf error is worst). Threshold is **RELATIVE**: N-leaf EV-gap ≤ ~10-20% of the edge (1a),
not an absolute 2-3 mbb. Measurement is DETERMINISTIC (fixed crc32 sample, deterministic CFR) → ~40 spots
resolve it exactly; the risk is COVERAGE not variance, so stratify across the served SPR range.
**1c. THE GATE COST (the binding constraint).** For each N∈{16,24,32}, capture the **wall-clock of the
exact grade** directly (the harness already grades each spot — free). This replaces the plan's ungrounded
~5s/~10s table with measured numbers. If the EV-sufficient N's gate misses the 12s budget → invoke the
cheaper-river gate (default) and re-measure; if still impossible → the turn solve can't serve live (BAKE
doesn't help — shared gate). Old Phase-1 text below superseded by 1a-1c.

## (superseded) original EV-gap text
The metric that matters is NOT rel-RMSE on M0 (Agent A axis 1) — it's the bot's ROOT EV. For ~40 turn
spots (varied board × SPR, low+mid SPR weighted by real play), solve the turn subgame TWICE per spot:
- once with the **N-river leaf** (the candidate), once with the **48-river "true" leaf** (the gold).
- Measure: **root-EV gap (mbb/hand)** and root strategy L1, per N.
Sweep **N ∈ {16, 24, 32, 40}**. Output: an EV-gap-vs-N curve.
- **Decision rule:** pick the smallest N whose EV-gap is **negligible** (target < ~2-3 mbb root EV — i.e.
  far below the turn solve's own edge over the blueprint) AND whose gate latency fits the budget.
- This converts "is 0.12 bad?" into a number. Expected: the gap is tiny by N≈24-32 (gate-protected +
  CFR-averaged), so full 48 is likely unnecessary — but we let the data decide, not the hunch.

## Phase 2 — FIDELITY DECISION
Fill the table from Phase 1 + known costs, then pick N:
| N | floor | leaf err vs truth | gen (cloud ~8h@32) | gate latency | EV-gap (Phase 1) |
|---|-------|-------------------|--------------------|--------------|------------------|
| 24 | ~0.14 | ~10% | ~6 h | ~4 s | (measure) |
| 32 | ~0.12 | ~8.5% | ~8 h | ~5 s | (measure) |
| 40 | ~0.10 | ~7% | ~10 h | ~7 s | (measure) |
| 48 | 0 | 0 (true) | ~12 h | ~10 s (tight) | 0 by defn |
- **Gate-cost escape hatch (option):** decouple the gate from N — grade on a CHEAPER river sample (fewer
  than N) with a variance-aware margin (Agent A). Lets a high-N NN keep a cheap gate, at the cost of a
  small gate/solve inconsistency. Consider only if a high N is needed but its gate blows the budget.

## Phase 3 — CLOUD DATA-GEN at chosen N — ~8-12 h, one cloud night
- Box: the blueprint cloud box (~2.5× the dev PC; 16 vCPU). All 16,432 canonical boards × SPR grid × N
  rivers via `scripts/nn_leaf_datagen.py` (`NN_LEAF_RIVERS=N`, parallel, resumable, mmap-shared tables).
- ~$3-5, frees the dev PC, full fidelity (9 SPR, all boards — no trimming needed on cloud).
- Output `m0_data.npz` + a STAMP (blueprint version, N, n_buckets=24, SPR grid, feature-version).

## Phase 4 — TRAIN (with Agent-A methodology fixes) — ~30 min
- **Train on ALL boards** (we serve all 16k → train error is the deployment metric). Hold out a small set
  ONLY as a learnability/compressibility DIAGNOSTIC (held-out ≫ train ⇒ not compressible ⇒ BAKE).
- **Per-SPR Y normalization** (one scalar per SPR slice — preserves each matrix's zero-sum shape, fixes
  the low-SPR under-fit the global scale caused).
- Report: **train rel-RMSE (deployment)** + per-SPR breakdown + held-out (diagnostic) + the **downstream
  root-EV gap** on held-out spots (Phase-1 machinery reused — the real go/no-go).
- Net ~MB; export numpy weights + stamp for serving (no torch in the image).

## Phase 5 — SERVE + N0′ (blocking) 
- Wire the NN leaf into `turn_subgame_solver` (`leaf_fn` → net; returns `(M0, -M0ᵀ)`); served solver at
  `leaf_rivers=N` (partition + gate consistency — the release gate). Machine-checkable stamp enforced.
- **N0′ head-to-head** (thousands of hands, paired-AIVAT, vs the blueprint opponent that failed N0) —
  BLOCKS serving default-on. Golden-spot regression (gated served action within ε of the exact-leaf solve).

## Success / go-no-go
- Phase 1: EV-gap at the chosen N < ~2-3 mbb → fidelity sufficient.
- Phase 4: train rel-RMSE comfortably < the floor-implied ceiling AND downstream EV-gap small → net works.
  (If train rel-RMSE can't beat ~0.15 even at a low-floor N → featurization/capacity problem → BAKE.)
- Phase 5: N0′ ≥ blueprint → ship.
- **Fallback at any failed gate:** BAKE (cheap VPS Hetzner/Oracle), which is floor-immune + N0′-free.

## Open questions (for review)
1. Is the Phase-1 EV-gap protocol sound — does root-EV gap vs the 48-river gold correctly capture the
   live cost of N-river leaf, given the served gate also uses N (not 48)? Should the gold be "48-river
   everything" or "48-river leaf + N-river gate"?
2. Is ~40 spots enough for a stable EV-gap, or does the high variance of turn EV need more / AIVAT?
3. The gate-cost escape hatch (cheaper gate than N): does the variance margin reliably preserve "no worse
   than blueprint", or does the solve/gate river mismatch reintroduce the N0 self-grading risk?
4. Per-SPR normalization: does normalizing each SPR slice separately distort cross-SPR comparability the
   net needs (the net takes SPR as an input feature)?
