# Phase 6 — abstraction upgrade (v3 blueprint) — PLAN, awaiting sign-off

Branch `rust-core`. Status 2026-11-09: **signed off (Arm A, served-code
changes behind `key_format`, Arm B deferred); 6.1 measured — see "6.1
results"; bucket count (4 → 5) awaiting confirmation.**
Parent plan: [RUST_CORE_PLAN.md](RUST_CORE_PLAN.md) (Phases 0–5 done).

## Why now

- **Retraining the current abstraction cannot beat v2.** The first 100M Rust run
  reproduced v2 (matched its preflop at ~55M) and paired LBR found no snapshot
  better than v2 (55M: −131 ± 135 mbb, z ≈ −1; later snapshots drift slightly
  worse). v2 is converged *for its buckets*; the remaining error is the
  abstraction, not iterations.
- **The cost objection is gone.** SPR buckets were rejected in 2026-06 because a
  2.5× bigger tree meant 2.5× of a ~2-day cloud run. The Rust trainer does
  100M iterations in ~46 min on the M6 (~37k it/s sustained), so even a 3–6×
  tree is an evening, not a week.
- **There is a known, concrete leak the current key cannot fix:** BUG-027 — in a
  4-bet pot the bot called a 75 BB flop shove with second pair 98% of the time,
  because the postflop key resets the pattern each street and cannot tell a
  4-bet pot from a limped pot.

## What changes

### Arm A — SPR buckets on flop and turn (main candidate)

Add the **stack-to-pot ratio at the start of the street** to every flop and turn
key: SPR = effective stack ÷ pot when the street begins.

- **Buckets (default, from ROADMAP item 8):** `<2 / 2–4 / 4–8 / >8` (K = 4).
  Rough map at 100 BB: limped and single-raised pots → `>8`; 3-bet pots → `4–8`
  or `2–4`; 4-bet pots → `<2`. Step 6.1 checks the real distribution before
  the boundaries are frozen — in particular whether `>8` should split into
  `8–16 / >16` to separate limped from raised pots.
- **Streets:** flop + turn only. Preflop SPR is already implicit in the preflop
  pattern. The river is excluded for now: the served river solver uses the
  real pot and stacks. (Caveat: the blueprint river still anchors the safe
  gadget and is the fallback; revisit if 6.4 shows river leaks.)
- **The SPR is a pure function of the betting history** (fixed 200-chip stacks,
  fixed blinds), so every caller that has the history can compute it — no new
  state to thread through `GameSession`.
- **Key format:** one new token before the pattern, e.g.
  `pf_6_25_ip_flop_r3_a` (`r0..r3`). The Rust packed key has 3 spare bits
  (61 of 64 used), enough for up to 7 buckets.
- **Size:** flop + turn are 148k of v2's 195k keys. Worst case ×4 on that
  slice → ~640k keys (3.3×). Real count is lower (not every bucket is
  reachable with every hand class), measured in 6.1.

### Arm B — finer flop/turn card buckets (second, only after A is scored)

Raise the per-street strength K (today flop 30 / turn 24 / river 10), e.g. to
50 / 40 / 10. This is a re-fit of the centroids + re-bake of the tables
(`compute_postflop_buckets.py`, `bake_postflop_table.py`). It sharpens hand
resolution but does **not** address BUG-027 or the equity-vs-uniform blind
spot (BUG-023), so it is the lower-priority lever. Combined with A: ~5–7×
v2's keys.

### Explicitly out of scope

- Range-aware (non-uniform) equity buckets — the deeper fix for BUG-023/027's
  second half; a research project.
- New bet sizes (a 4th 3-bet size, etc.) — only with a per-line leak measured.
- Real-time turn solving — closed (Dead Ends).

## Budget (M6, measured throughput)

| Config | Info sets (est.) | Iterations for v2-level per-key convergence | Wall time at ~37k it/s |
|---|---|---|---|
| v2 (current) | 195k | 52.5M | ~25 min |
| A: SPR K=4 flop+turn | ≤ 640k | ≤ 175M | ≤ 80 min |
| A + B | ~1–1.3M | ~350M | ~2.5–3 h |

Run each arm to ~2× its estimate with 5M (or 10M) snapshots, and pick the ship
snapshot by evaluation, not by the latest iteration. Throughput may drop as the
table outgrows the CPU cache — the first 10 minutes of the run will say.
RAM: ~300 B/info set in Rust → ~400 MB at 1.3M; fine. Serving reads the DB per
key through SQLite (no full load), so a ~250–500 MB DB costs disk and image
size, not RAM — check the Lightsail image budget before a cutover.

## Steps

**6.1 Measure first (no training, ~minutes).**
- Distribution of street-start SPR on flop and turn in v2 self-play, weighted
  by **chips at stake**, not just decision count. (The 2026-06 Measurement A
  counted decisions — 9% low-SPR — which under-weights big pots, exactly where
  BUG-027 loses money.)
- Exact reachable key count for Arm A (enumerate SPR buckets over v2's real
  flop/turn keys from self-play, instead of the ×K upper bound).
- A **4-bet-pot probe**: how often v2 stacks off with medium-strength flop
  hands facing a shove in 3-bet/4-bet pots. This is the BUG-027 fingerprint
  and becomes a per-checkpoint probe like the strategy-shape check.
- Output: frozen SPR boundaries + go/no-go for Arm A.

**6.1 results (2026-11-09)** — `scripts/measure_street_spr.py`, 400k v2
self-play hands (17 s on 10 workers):

- **The money is at low SPR, even though most decisions are not.** By decision
  count only 14% of flop decisions start at SPR < 4, but weighted by pot size
  it is 50%; on the turn, 47% of pot-weighted decisions start at SPR < 2. The
  2026-06 Measurement A ("9% low-SPR → not worth it") counted decisions and
  missed this.
- **Preflop pot type maps cleanly onto flop SPR bands:** limped 100% > 48;
  single-raised 8–32 (93% in 12–32); 3-bet 2–16 (mostly 4–12); 4-bet+ < 8
  (85% < 4). ROADMAP's `<2 / 2–4 / 4–8 / >8` would put limped, single-raised
  and a third of 3-bet pots in one bucket (~90% of flop decisions) — it wastes
  the split.
- **Reachable-key multiplier** (distinct keys seen with vs without the bucket):

  | Bucketing | Flop | Turn |
  |---|---|---|
  | K4 `<2/2–4/4–8/>8` (ROADMAP) | 1.62× | 1.87× |
  | K5 `<2/2–4/4–8/8–16/>16` | 1.92× | 2.23× |
  | **P5 `<2/2–5/5–12/12–40/>40`** (pot-type aligned) | **1.88×** | **2.16×** |
  | P4 `<4/4–12/12–40/>40` | 1.75× | 1.94× |

  P5 on v2's full key set: flop 80.6k → ~151k, turn 67.6k → ~146k, total
  ~344k info sets (**1.77× v2**, well under the 3.3× worst case) → ~95M
  iterations for v2-level per-key convergence, ~45 min.
- **Facing an all-in** (realized value of calling, exact showdown): no large
  leak *in self-play* — the losing cells (medium/weak hands calling in
  3-bet/4-bet pots, mostly on the turn) sum to about −25 mbb/hand. Caveat:
  self-play understates BUG-027, because v2's own shoving range is the same
  blended range; a human who shoves only the nuts in a 4-bet pot (the real
  BUG-027 hand) punishes the blended call far harder.
- **Recommendation:** go, with **P5** (5 buckets, one per pot type:
  `>40` limped, `12–40` single-raised, `5–12` 3-bet, `2–5` 4-bet,
  `<2` jam zone) on flop and turn — costs about the same as K5 and separates
  the pot types K4 merges. *This changes K from the signed-off 4 to 5 —
  needs confirmation.*

**6.3/6.4 results (2026-11-09, overnight)** — 200M spr5 Rust run (1h00m,
404k info sets, snapshots every 10M; scripts in the session scratchpad
`overnight/`). Candidates vs served v2, with a same-abstraction control
(Rust v2-format 55M) to show what plain retraining noise looks like:

| vs v2 | SPR 50M | SPR 100M | SPR 150M | SPR 200M | control (v2-format 55M) |
|---|---|---|---|---|---|
| Head-to-head, AIVAT, 240k hands (mbb/hand, >0 = candidate wins) | −59 ± 34 | −31 ± 33 | −49 ± 33 | +47 ± 33 | +55 ± 34 |
| Paired LBR, 10k hands (difference, lower = better) | −49 ± 141 | +185 ± 169 | +227 ± 179 | +221 ± 190 | +114 ± 101 |
| Best response, 24 boards (total; v2 = 14,028) | — | 13,347 | — | 13,651 | — |

- **Best response, 96 paired boards (4 seed blocks):** SPR 100M is **5.6% less
  exploitable than v2** (−784 ± 72 mbb/hand; as SB −1.8%, **as BB −8.0%**),
  lower in all 4 blocks. The control is **+1.8%** (slightly worse than v2), so SPR
  100M beats plain retraining by ~1,035 mbb/hand. The gain is concentrated on
  the BB/OOP side — the side whose BR had plateaued on the v2 abstraction.
  Caveat: BR is exact on cards but restricted to the grid sizes, and measures
  the blueprint table (not the river solver or guards).
- **Head-to-head and LBR: no detectable difference** — every candidate,
  including the control, is within the ±35 (H2H) / ±150 (LBR) noise band.
- **BUG-027 is NOT fixed, and SPR buckets cannot fix it.** Under the capped menu
  an all-in only arises from a stack clamp, so "facing a flop lead shove" only
  occurs at low SPR: v2's key `pf_6_25_ip_flop_a` was already effectively a
  4-bet-pot key. The spr5 blueprint reaches only `..._flop_r0_a` there and
  learns the same 98–99% call. The real causes are the other two in BUG-027:
  equity-vs-uniform strength buckets and humans shoving far narrower than the
  equilibrium range — candidates for range-aware buckets or the exploitation
  layer, not SPR. (Elsewhere the spr5 blueprint does call shoves by pot odds:
  medium flop hands call 84% at SPR < 2 vs 36% at SPR > 40, against v2's
  blended 62%.)
- **Preflop drift:** weak-hand SB fold falls 74% → 40% over the run (v2 68%);
  BR prefers 100M over 200M, so the ship candidate is **SPR 100M**.
- **Gate (6.4):** BR improved ✔, LBR/H2H neutral ✔, BUG-027 fingerprint gone ✘ —
  but that criterion rested on a wrong premise (above). Decision for the user.

**6.2 Implement the key change, backward compatible.**
- `make_info_set_key(..., spr_bucket=None)`; `None` keeps today's format, so
  v2 keeps working unchanged. The blueprint DB is stamped with a
  `key_format` metadata value (`v2` | `spr4`), and every reader builds keys in
  the format of the blueprint it opened.
- Same change in Rust `keys.rs` + key packing, one new parity fixture.
- Python callers (14 sites): trainer, `game_adapter`, `game_session`,
  `range_tracker`, `tilt`, `replay`, `blueprint_projection`, and the eval
  harness (`lbr`, `lbr_solver`, `best_response`, `match`, `cross_match`), plus
  `fit_opponent_models` / `measure_*` scripts. **These include served code**
  — the sign-off below covers touching them on this branch behind the
  `key_format` switch.
- `cross_match` must build each side's keys in that side's format, so a v3
  blueprint can play v2 head-to-head.
- **Key parsers by token position** misread the 7-token spr5 keys and must be
  updated with the opponent-model re-fit: `exploitation/opponent_model.py`
  (`_levels`: `pat = p[5]` would read `r3`), `scripts/build_opponent_model.py`,
  `scripts/analyze_key_coverage.py`, `scripts/compare_jam_fingerprint.py`.
- **Done (2026-11-09):** every blueprint reader now fails closed on a key format
  it can't build (`keys.require_key_format`: API, `BlueprintStrategy` and so the
  river solver, LBR, BR, match, cross_match), and auto-resolution
  (`config.resolve_blueprint_path`) only picks v2 blueprints. Each reader is
  switched to `('v2', 'spr5')` as it learns to build spr5 keys.
- **River solver consumers.** The river solver itself is bucket-free (exact
  hands, real pot and stacks), but it reads the blueprint in three places, and
  all three must use the new keys:
  1. its **input ranges** come from the range tracker, which updates on every
     flop/turn action through blueprint keys → those lookups need the SPR
     bucket, or the ranges are built from the wrong (or missing) strategies;
  2. the **safe gadget's opt-out values** (`blueprint_projection`, river-entry
     CFV) look up river keys — unchanged under Arm A (river excluded), but they
     change under Arm B (new river K / tables) and the turn-projection path
     uses turn keys;
  3. the **guards and fallback** in `river_subgame_solver` (untrained-key
     checks, blueprint fallback) look keys up directly.
  Arm B's new tables reach the solver's bucket groups automatically through
  `CardAbstraction`, so the tables and the blueprint must always ship together
  (the existing centroid-hash stamp enforces this). Test: a river solve on a
  v3 blueprint gives the same ranges as building them by hand from v3 keys.
- Tests: key round-trip for both formats; SPR computed from history equals the
  engine's pot/stack at street start (property test); parity green.

**6.3 Train Arm A** (~1.5–3 h incl. 2× margin), snapshots every 10M, with the
strategy-shape and 4-bet-pot probes at each checkpoint.

**6.4 Evaluate Arm A vs v2** — all paired, same deals:
- LBR (both blueprints; comparable because the action grid is unchanged).
- BR on a small sample count — **>10 min per sample; wall time stated before
  running.**
- Head-to-head v3 vs v2 with AIVAT via `cross_match` — the most sensitive test.
- The 4-bet-pot probe: BUG-027 must be fixed.
- Gate: no BR/LBR regression **and** a positive or neutral head-to-head **and**
  the BUG-027 fingerprint gone. Otherwise stop and report.

**6.5 Arm B (optional):** only if A passes or clearly stalls on card
resolution. Re-fit centroids, re-bake tables, train A+B, evaluate the same way
against the best A snapshot.

**Then Phase 7 (separate sign-off):** v3 bundle — re-fit opponent models on the
new keys (they self-disable on an abstraction mismatch until then), update
`/api/abstractions` + KeyExplorer (frontend — explicit go-ahead needed),
publish `assets-v3`, Docker/CI, live A/B, cutover.

## Risks

- **Opponent-model sparsity.** SPR in the key splits the human data ~K ways.
  The model's empirical-Bayes backoff (blueprint → strength×pos×street → full
  key) absorbs this, but the personal layer will read thinner; check in Phase 7.
- **Eval comparability.** BR/LBR stay comparable only while the action grid is
  identical — another reason not to change sizes in the same release.
- **Value may be small on average.** BUG-027 lives in rare big pots; if 6.1
  shows little money there, Arm A's gain may not show in LBR at all — the
  4-bet-pot probe and head-to-head carry the decision.

## Decisions needed (sign-off)

1. Arm A as specified (K = 4, flop + turn, boundaries frozen after 6.1)?
2. Touch served Python code on this branch behind the `key_format` switch (no
   behaviour change for v2 blueprints)?
3. Arm B deferred until Arm A is scored?
