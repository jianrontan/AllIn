# Phase 6 — abstraction upgrade (v3 blueprint) — PLAN, awaiting sign-off

Branch `rust-core`. Status 2026-11-09: **draft for review — nothing implemented.**
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
