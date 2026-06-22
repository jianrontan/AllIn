# Turn Leaf: BAKE vs NN — Pipeline Specs

> ❌ **CLOSED — DEAD END (2026-06-22). DO NOT BUILD.** Both pipelines accelerate the *blueprint*-leaf turn
> solve, which lost −66 mbb/hand H2H (optimistic, self-inconsistent leaf). Accelerating a losing strategy
> doesn't help. Turn solver UNWIRED from serving. Only viable revival = range-conditional CFV net
> (DeepStack-scale). Kept for the record. Verdict: `docs/private/ROADMAP.md` "Dead Ends".

Both make the turn solve fast by replacing the real-time `_project`/`turn_leaf_matrix_both` (10.8 s of
12.5 s, Phase 0) with a precomputed leaf value `M0`. M0 is range-independent and keyed on
`(canonical turn board, SPR-bucket)` (pot-normalized → no grid-miss; enumeration: 16,432 boards × 467
configs). The two differ in HOW M0 is stored/served.

## NOISE-FLOOR RESULT (2026-06-21, `scripts/measure_m0_floor.py`, 90 boards) — DECISIVE
The 12-river M0 target's rel-RMSE between two different river draws (= the floor a smooth net can't beat):
**OVERALL floor 0.202** (per SPR: 0.184 / 0.199 / 0.209 / 0.217), and **bias(12 vs 44-river gold) 0.158**.
- **⇒ a small NN CANNOT reach the <=0.15 target at 12 rivers — the TARGET is too noisy** (not the net/
  features). An 8h run would bottom out ~0.20. (This validated Agent A's axis-5 warning for ~minutes of
  compute; the overnight NN run was NOT launched.)
- To make the NN target learnable it must be DETERMINISTIC/smooth = **ALL 44 rivers** (floor→~0) → ~4×
  data-gen (multi-day cloud) + a ~10s 44-river serving gate (tight) + still needs N0′ + better features.
- **The BAKE is unaffected by the floor** (it stores the EXACT M0, no smoothing) → 3rd strike for the NN
  (probe 0.486 → floor 0.20). With cheap storage found (Hetzner CX22 $4.59/mo / Oracle Free $0, both hold
  the table + 8× compute), the BAKE is the clean path. ALSO: bias 0.158 means the live 4-river solve's
  leaf values are ~20%+ off truth → the served solve should use MORE rivers regardless (quality + N0).

## BUILD STATUS (2026-06-20)
- **M0 generator built + VERIFIED** (`src/subgame/turn_leaf_gen.py`): reproduces the live solve's leaf
  **bit-identically** (maxdiff 0.0) and M0 is **exactly scale-linear** (`M0(k·pot)=k·M0`, maxdiff 0.0) →
  the `(board, leaf-SPR) → M0/pot` key is exact. Per-sample cost: ~1.0 s cold (recomputes board setup) /
  **~0.17 s board-amortized** (rivers+partition computed once per board, reused across SPR samples).
- **52.5M (`snap_52500000`, 30/24) is the FINAL blueprint** (user) → the NN trains on the final, NO
  re-train needed; ships with the assets-v2 (30/24) serving cutover (the NN's blueprint must = served).
- torch 2.6 + sklearn available LOCALLY (not in requirements) → serving uses numpy-only inference
  (exported weights), no torch in the image.
- **FEASIBILITY PROBE (2026-06-20, `scripts/nn_leaf_feasibility.py`, 600 boards × 8 SPR): DISCOURAGING.**
  Held-out-by-board rel-RMSE **0.486** vs mean-baseline **0.529** — the NN barely beat a constant (+0.043).
  Threshold was <0.15 promising / >0.5 problem → this is the bad zone. CAVEATS (under-powered, not a clean
  refutation): only 600/16,432 boards, crude 28-dim hand features, an off-the-shelf MLP that didn't
  converge. Also data-gen was **3.1 s/board** (2× the 0.17 s estimate) → a real-scale dataset needs the
  parallel cloud box, not local.
- **DECISION: lean PIVOT TO BAKE.** The NN showed weak signal + needs real ML iteration (features/arch/
  data) with uncertain payoff + still must clear N0′ — three risks. The BAKE is GUARANTEED (bit-identical,
  the generator is verified) and costs only the box migration (EC2+EBS gp3 ~$2.40/mo volume + a small
  instance). One improved NN attempt (richer features + real net, cloud) is defensible ONLY if avoiding
  the box migration outweighs the ML uncertainty — cap at one shot. (User's call pending.)

## ⚠️ SERVING-CONFIG RELEASE GATE (verified 2026-06-20)
The NN is trained on **12-river** M0 (`NN_LEAF_RIVERS=12`). The served `TurnSubgameSolver` MUST then run
**`leaf_rivers=12`** — because the strength PARTITION (`tb_idx`) and the exact EV GATE
(`turn_leaf_value_exact`) are both computed over the river sample, so a 12-river-trained NN + a 4-river-
served solver = a SILENT bucket-definition mismatch on every turn leaf (value-corrupting, not a crash).
The NN replaces only the bucketed M0 (`leaf_fn`); partition + gate stay on the solver's 12-river sample.
The stamp MUST record `leaf_rivers=12` and be enforced against the served config. (Today the served/
datagen solver is at leaf_rivers=4 — must bump to 12 before the NN goes live. Harmless for datagen, which
generates M0 via the standalone 12-river generator.)

## Shared upstream (both need this)
- **M0 generator:** a function `(canonical_board, leaf_config) → [B×B] float32`, = one
  `turn_leaf_matrix_both` at the deterministic 4-river sample. ~0.17 s/matrix (Phase 0). Embarrassingly
  parallel (independent per board/config). Run on the existing 16-vCPU EC2 training box.
- **Stamp:** blueprint version + `n_buckets` + `leaf_rivers` + menu + SPR-grid → invalidate on any change
  (mirror `centroid_hash` for the postflop tables). A stale stamp → fall back to the live `_project`.

---

## PIPELINE A — BAKE (exhaustive table, mmap from SSD)

**Idea:** precompute ALL `16,432 boards × 467 configs = 7.7M` matrices into one `.npy`, mmap it, look up.

1. **Generate** (offline, EC2 16-vCPU): 7.7M × 0.17 s / 16 ≈ **~23 h**, one-time. Output a structured
   `.npy`: shape `[16432, 467, B, B]` float32, indexed by `(board_id, config_id)`. **17.7 GB @ B=24.**
   Plus an index: `canonical_board → board_id`, `(SPR-bucket, leaf-shape) → config_id`.
2. **Stamp + ship:** GitHub Release asset (like the blueprint), 17.7 GB. Or bake into an EBS snapshot.
3. **Serve** (disk-backed instance — see infra): ship PRE-EXTRACTED `.npy` (NOT `.npz` — the runtime
   extract step fails on a read-only dir and would full-load → OOM). mmap it. At solve time:
   canonicalize the board → `board_id` + suit-perm; per leaf, map `(final_pot, stacks) → SPR-bucket →
   config_id`; read `M0 = table[board_id, config_id]`; apply the suit-perm to relabel back to concrete.
4. **Fallback:** board/config miss or stamp mismatch → the live `_project` build (the existing path).
5. **Validate:** **golden-spot = bit-identical** to the current solve (same M0, same everything) → the
   bot's play is UNCHANGED → **pure latency optimization, NO N0′ head-to-head needed.** (Huge: the bake
   can't regress EV because it doesn't change strategy.) Just a correctness check (baked M0 == live M0)
   + a serving-latency smoke.

**Pros:** no ML, no training; **bit-identical → no N0′ gate** (the thing that killed the project before);
conceptually simple.
**Cons:** **17.7 GB artifact** (slow ship/store); **requires migrating serving off Lightsail Containers**
to a disk-backed instance; ~23 h one-time gen; re-bake (+23 h, +18 GB) on every blueprint/abstraction
change (incl. the in-flight 30/24 retrain).

---

## PIPELINE B — NN LEAF (compress the table to a net)

**Idea:** train a small net `(board features, SPR) → [B×B]` ≈ the same M0; ~MB; fits the current box.

1. **Generate training data** (offline, EC2): SAMPLE ~100k–500k `(board, SPR)` pairs (the net
   generalizes → a subset, not all 7.7M) → `(features, M0)`. ~100k × 0.17 s / 16 ≈ **~18 min**
   (500k ≈ ~1.5 h).
2. **Featurize + train:** suit-iso canonical board features + SPR scalar → small MLP/CNN regressing the
   576-vector (`B=24`). ~**1 h** (CPU/Colab). Output: a ~MB net + the stamp.
3. **Serve:** ship the ~MB net IN the existing Docker image (no infra change). Replace
   `turn_leaf_matrix_both` with net inference (~ms). `ALLIN_NN_LEAF` flag + healthz version + kill-switch.
4. **Fallback:** stamp mismatch / error → live `_project`.
5. **Validate:** (a) held-out **rel-RMSE** of `M0_hat` vs exact M0; (b) **N0′ head-to-head** (thousands
   of hands, paired-AIVAT, vs the blueprint opponent) — REQUIRED, because the net APPROXIMATES M0 → it
   DOES change strategy slightly → must prove ≥ blueprint. This is the slow, blocking gate.

**Pros:** ~MB → **fits the current cheap Lightsail Container, no infra migration**; cheap gen (~hours);
cheap to re-train on a blueprint change.
**Cons:** approximation error → **N0′ validation REQUIRED** (the blocking gate); the per-board river
sample makes `board→M0` less smooth → may need more samples / careful featurization; ML iteration risk.

---

## Decision axes

| | BAKE (A) | NN (B) |
|---|---|---|
| Artifact | 17.7 GB | ~MB |
| Serving infra | **migrate** → disk-backed Lightsail Instance / EC2+EBS gp3 | **current** Lightsail Container |
| RAM needed | ~none extra (mmap, ~230 KB/solve paged) | net in RAM (~MB) |
| SSD needed | ~20 GB | none |
| One-time compute | ~23 h (exhaustive) | ~hours (subset + train) |
| Behavior change | **none (bit-identical)** | slight (approximation) |
| **N0′ gate** | **NOT needed** | **REQUIRED (blocking)** |
| Re-do on blueprint swap | re-bake ~23 h + 18 GB | re-train ~hours |
| Main cost | infra migration + big artifact | ML work + N0′ validation |

## AWS for the bake's SSD
- Serving: **Lightsail Instance** (~$20/mo = 4 GB / 80 GB SSD, or ~$40 = 8 GB / 160 GB) OR **EC2 t3.medium
  + 30 GB gp3 EBS**. NOT Lightsail Containers (no controllable SSD). NOT S3 (not mmap-able) / EFS (NFS
  latency hurts random reads).
- Data-gen (both pipelines): the existing 16-vCPU EC2 training box — no new resource.

## Open
- Exhaustive bake (7.7M) vs the NN's sampled subset — the bake's 23 h gen is the same M0 generator, just
  run over the full grid. Could also bake a SMALLER table (coarser SPR grid / fewer buckets) to shrink
  below ~8 GB if we want bake-on-a-smaller-disk.
- B (buckets) is the B² lever on bake size (24→17.7 GB, 16→7.9 GB) — but lowering it is the "last resort"
  quality cut.
