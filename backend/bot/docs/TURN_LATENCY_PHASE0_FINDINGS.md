# Turn-Solver Latency — Phase 0 Findings

Date: 2026-06-20. Scripts: `scripts/profile_turn_solve.py` (cProfile attribution),
`scripts/bench_turn_solve.py` (wall-clock latency, `--fresh`/`--repeat`).
Config: served snapshot `snap_52500000.db`, `n_buckets=24`, `leaf_rivers=4`, SPR ~2 (low end of the gate).

## Measurements

### Where the time goes (cProfile, 1 solve)
| Stage | Cold | Note |
|-------|------|------|
| Total | 18.5 s | (cProfile inflates ~25%; real wall ~14 s — see below) |
| **SOLVE leaf build** (`turn_leaf_matrix_both`) | **12.5 s** | the dominant cost |
| └ **`blueprint_projection._project`** | **10.8 s** | THE hot primitive (project blueprint onto each river tree) |
| CFR iterations | ~3.5 s | |
| **GRADE** (`turn_leaf_value_exact`) | **2.4 s** | the exact-rollout gate |
| `build_board_arrays` | — | did NOT rank — the plan/review named the wrong primitive |

### Fresh-board floor (wall-clock, 10 DISTINCT boards)
mean **14.0 s** / p50 14.0 / p90 15.8 / min 13.2 / max 15.8. Consistent → a hard ~14 s cold floor at low
SPR (deep turns are worse: bigger tree → more leaves → more `_project`).

## Conclusions

1. **The GRADE is NOT the bottleneck** (2.4 s). Refutes the review's "grade is the deeper, uncacheable
   problem." Grade-sampling + variance-margin is NOT load-bearing (kept available, not required).
2. **The bottleneck is `_project`** (10.8 s of 12.5 s) — projecting the blueprint onto each river tree,
   per (leaf × river). It is **reach-independent** (board + pot/stacks + blueprint) → cacheable, and is
   currently discarded every solve (per-solve `ba_cache`).
3. **The cross-solve canonical cache only helps REPEAT boards.** A single player rarely repeats a turn
   board → the cache's value is cross-player popular-board warmup **at scale**, LOW at current traffic.
   So the cache does **not** speed up the current fresh-board experience (~14 s).
4. **Parallel is useless on the 0.25-vCPU Micro** (sub-one-core).
5. ⇒ **The cold-fresh case — the actual current experience — needs the NN leaf** (removes `_project` on
   every board) OR a bounded menu (fewer leaves) OR fewer buckets (last resort). The cache is a SCALE
   optimization, not the "fast now" lever.

## Build-order implication (vs the v3 plan)
v3 assumed the cache was the highest-leverage first build. Phase 0 shows that's true **at scale** but NOT
for the current box/traffic. For "fast on the current Micro": **NN leaf (or bounded menu) first**; cache
is a cheap, no-quality-loss scale add that can land anytime (version-stamped to the blueprint).

## BAKE vs NN vs in-memory cache (clarified 2026-06-20)
The leaf matrix `M0` is range-independent → it can be **baked OFFLINE**, not just cached in memory.
Baking is strictly better than the in-memory cache: the cache only helps REPEAT boards (scale), a baked
table removes `_project` on **every** board incl. cold/fresh — which is the ~14 s problem.
- **Bake the table:** ~`[24×24]` floats (~2.3 KB) × ~20k canonical turn boards × ~75 leaf pot/stack
  configs ≈ **~3–5 GB**. No training; just run the existing leaf build offline. **Server-scale — does NOT
  fit the 0.25 vCPU / 1 GB Micro** (needs a bigger box / mmap).
- **NN leaf = the baked table COMPRESSED to ~MB** (a smooth `board → M0` net) → fits the Micro. Same idea,
  small size; costs data-gen + training.
- **Decision pivot:** bigger box → **bake** (least engineering, no training). Cheap Micro → **NN** (the
  compression). In-memory cache → demoted to a minor scale add.
- **NN effort is mostly COMPUTE, not labor:** data-gen ~hours-to-a-day (parallelizable cloud box),
  training ~hours, then the N0′ match (slow, but any turn change needs it) + wiring (~days). Not "weeks"
  of work — calendar weeks dominated by validation/iteration.

NOTE: `docs/NN_LEAF_PLAN.md` is referenced in CLAUDE.md but does NOT exist in the repo — earlier size/
time figures were from memory, not a committed doc. Sizes above are first-principles estimates to verify.

## 2-AGENT VERIFICATION (2026-06-20)
**Bake viability — CONFIRMED with corrections:**
- M0 range-independent → bakeable. The "CFR re-reads M0 every iteration" fear is **REFUTED**: `TurnCFR.
  _leaf_cache` + eager pre-build → M0 read **once per distinct leaf per solve** (~26-100), not ×iters.
  Sparse ~mid-KB reads ≈ single-digit ms vs 14 s → SSD/mmap premise sound.
- CORRECTION: M0 is **float64 → ~6-10 GB** (cast to float32 at bake → ~3-5 GB, free).
- BLOCKER to handle: the existing mmap path extracts `.npz`→`.npy` AT RUNTIME, which fails on prod's
  **read-only** container dir → falls back to a **full in-RAM load** → a multi-GB table OOMs the 1 GB box.
  Fix: ship **pre-extracted `.npy`** (no startup write) or a writable volume.
- RISK: live leaf `(pot, stacks)` is **continuous**; a baked table is a discrete grid → off-grid =
  miss = slow `_project` rebuild. The "~75 configs/board" is UNVERIFIED → the enumeration must measure it.

**NN cost — my 0.9 s/sample was WRONG; corrected:**
- Per-sample = one `turn_leaf_matrix_both`, cost set by river fidelity: **~0.17 s at 4 rivers** (reproduce
  CURRENT live behavior) / **~4-8 s at 44 rivers** (higher fidelity). NOT a full 14 s solve.
- ⇒ ratio to a blueprint run: **~1/12-1/20 at 4-river** (cheaper than I said) / **~1/3 at 44-river**.
  To make the *current* solve fast you reproduce 4-river → cheap. Upgrading fidelity costs live too
  (separate decision). Sample count ~200k is reasonable (codebase trains on 30k-250k elsewhere).
- BAKE NUANCE: the bake reproduces the deterministic 4-river M0 **exactly** (bit-identical to live); the
  NN must learn a less-smooth board→M0 (the per-board river sample jumps) → a point FOR bake over NN.

**Both bake AND NN hinge on ONE unmeasured number: the live leaf `(pot, stacks)` config distribution.**
The enumeration (next) measures it → real table size + grid-miss answer + confirms/refutes ~75/board.

## ENUMERATION RESULTS (2026-06-20, `scripts/enumerate_turn_leaves.py`)
- **The turn betting tree is BOARD-INDEPENDENT** — leaf configs depend only on `(pot_entry, stacks, menu)`.
- **Pot-normalization COLLAPSES the continuum to a 1-D SPR axis:** at SPR=3, pots {40,100,173,250} give
  IDENTICAL normalized leaf sets. ⇒ bake key = `(canonical board, SPR-bucket)` — **clean, no grid-miss**
  (Agent 1's top risk eliminated). A continuous live pot maps to its SPR; M0 scales with pot.
- Leaf count per tree grows with SPR: **3 @ SPR 0.5 → 66 @ SPR 8 (the gate)**; **467 distinct normalized
  configs** over an SPR 0.5–8 grid (step 0.25).
- **Canonical turn boards = 16,432.** ⇒ bake size (467 configs × `[B×B]` float32):
  **B=24 (served) → 17.7 GB; B=16 → 7.9 GB; B=48 → 70.7 GB.**
- **VERDICT: "3-5 GB on SSD" was ~4× LOW — it's ~17.7 GB at served fidelity.** Access pattern is fine,
  but the 18 GB *artifact* is too big to ship/store on the cheap Lightsail Container. ⇒ **bake needs a
  bigger box (≥~20 GB SSD); the cheap box needs the NN** (compress 17.7 GB → ~MB). Decision is box-size-
  driven. Exhaustive bake data-gen ≈ 7.7M M0 × 0.17 s / 16 cores ≈ ~23 h; NN trains on a subset (~hours).

## Open / next
- Measure the fresh-board floor at HIGH SPR (near the gate of 8) — quantify the worst case.
- Decide the cold-case lever: NN leaf (no fidelity loss, ~weeks) vs bounded menu (sizing fidelity, ~days).
- After any lever lands: re-run `bench_turn_solve.py --fresh` and compare to the 14 s floor.
