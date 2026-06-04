# Poker AI Roadmap

Last updated: 2026-06-03

Status legend: ✅ done · 🚧 in progress · 📅 planned

This roadmap tracks the arc from a static blueprint to online, subgame-solving
play. For how the current system works see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md);
for commands see [../USER_GUIDE.md](../USER_GUIDE.md). Speculative, not-yet-committed
features (WASM client-offload, bot personality layer) live in [IDEAS.md](IDEAS.md); the
AWS go-live plan, +EV leaderboard, and cost estimates live in [DEPLOYMENT.md](DEPLOYMENT.md).

> **The target architecture (2026-05-28), in one paragraph.** This is the
> Libratus/Pluribus shape, and we are most of the way to it. A coarse **blueprint**
> plays the common cases. **Real-time subgame solving**, *anchored to the blueprint*
> (warm-started from it, ranges modelled on it, EV-gated against it), refines the
> spot whenever play goes off-abstraction or into a high-leverage endgame. Off-tree
> opponent bets are handled by **solving a new subgame** (Libratus's rule: re-solve
> on every off-tree bet — it beats action translation), with **pseudo-harmonic
> translation as the cheap fallback** on streets we choose not to solve. The bot is
> **not capped at pot-sized bets**: it should overbet and 5-bet+ where correct, which
> the solver supplies. Two things gate the finish line: (1) an **all-in anchor in the
> betting abstraction** (the cheap, retrain-only fix for the BUG-007 jam-response
> hole), and (2) **depth-limited solving with blueprint leaf values** — the single
> capability that unlocks the bot's own overbets/5-bets on the flop and turn. The
> river solver already exists and already overbets; everything else hangs off (2).

---

## ✅ RESOLVED (2026-06-03) — "bot plays poorly vs humans": diagnosis + redesign plan

> **RESOLUTION SUMMARY (2026-06-03).** The investigation below is closed. Outcome:
>
> - **Fix #1 (all-in over-call guard)** ✅ shipped (inference-only, all streets).
> - **Fix #2 (parallel merge jam-bias / BUG-009)** ⛔ **REVERTED (2026-06-04, BUG-014).** The raw-regret
>   worker store broke CFR+ re-activation → blueprints collapsed to "open xlarge with 100% of hands, never
>   fold the button." Reverted to per-worker CFR+ flooring (A/B-confirmed: pf_0 fold 74% vs 1%). The
>   +0.0014 jam bias it chased is covered structurally by Fix #4. **All capped DBs trained after it are
>   corrupt → retrain from scratch.** See BUG-014.
> - **Fix #3 (SPR-aware sizing / SPR buckets)** ❌ **decided AGAINST.** Measurements A+B killed it:
>   only ~9% of decisions are low-SPR (A), and SPR buckets cost 1.7–2.5× training for the same
>   per-key convergence (B). The over-jamming was the proposal *menu*, not SPR-blind keys.
> - **Fix #4 (bet-menu cap + drop voluntary all-in)** ✅ shipped + **LBR-VALIDATED at matched iters
>   (2026-06-03, 3000 hands, seed 42, lower = better):**
>
>   | blueprint | abstraction / menu | iters | LBR (mbb/hand) |
>   |---|---|---|---|
>   | pre-redesign (old 15-bucket) | OLD | 9M | **3,114** (old-harness anchor, BR 13,936) |
>   | control-4M | NEW / control | 4M | **3,337** |
>   | **capped-5M** | NEW / capped | 5M | **2,204** |
>   | control-33M (incumbent) | NEW / control | 33M | **1,917** |
>
>   → capped is **34% less exploitable than control at matched iters** (the 4M gate's −35% reproduced).
>   **Ship gate:** the capped 28M retrain (`blueprint_par_capped_20260603_013018.db`, in progress) must
>   come in **below the incumbent's 1,917** — on track (2,204 at only 5M, with a 34% structural edge).
>   BR not run (≈9 min/sample → impractical; LBR is the right probe for a sizing change).
>
> - **EV-gauge scare (2026-06-02/03) — RESOLVED, was never a bug.** The trainer's `EV(cum)`/`EV(round)`/
>   `EV(sess)` are computed on the **current regret-matched iterate**, which CFR does *not* drive to
>   convergence (capped's cycles, seat-imbalanced P0+46/P1−34, gauge ≈ +6 forever). The **served**
>   strategy is the *average* iterate, which is fine (~0 self-play value, same as control). Proven by
>   `scripts/probe_seat_ev.py`, and the gauge is **anti-correlated** with LBR (capped's scary +4.68
>   gauge is *less* exploitable than control's nice +0.37). **Fix:** `BlueprintTrainer.evaluate_served_ev`
>   now prints `EV(served, avg strategy)` at every checkpoint — the dial to watch (a seat-balance /
>   convergence check, NOT a strength metric → use LBR for strength). See the `ev-cum-investigation` memo.
> - **Also shipped this arc:** BUG-011 (capped served through a control engine → stray jam), BUG-012
>   (parallel Ctrl+C hang), `menu_mode` threaded through every train/eval/serve consumer via
>   `db_menu_mode`, parallel BR (`--workers`), and the `capped_no2` ablation arm (wired, not yet trained).
>
> The original diagnosis is retained below as the project narrative.

---

The 30.5M new-abstraction blueprint (`blueprint_par_20260529_233056.db`, served via
`ALLIN_BLUEPRINT_DB`) **overjams** (preflop + low-SPR flop) and **overcalls all-in
jams**, and more iterations don't fix it (convergence slowing). Two A/B tests
isolated the causes. **Three distinct root causes, only one is a retrain:**

| Symptom | Root cause (evidenced) | Fix |
|---|---|---|
| Over-**jams**, worst at marginal / low-SPR spots | **(2) parallel-trainer floor bias** — TEST 1: merge_every 4000 vs 250 jams 1.22× in aggregate, **up to 3.5–5.5× at marginal keys** (e.g. `pf_28_oop_cmm` 0.563 vs 0.160). It's a BIAS not a convergence gap → more iters don't help. | **Fix #2** corrected merge |
| Over-**jams** specifically as SPR drops | **(3) bet-sizing menu collapse** — TEST 2: P(allin) ~doubles as the sized-bet menu thins (0.155 at 4 sized opts → 0.234 at 3 → 0.30 at 0). Pot-fraction sizes ≥ stack collapse into all-in, so GTO mass in the gap rounds to a jam. Structural, not training. | **Fix #3** SPR-aware sizing (± SPR buckets) |
| Over-**calls** human jams | **(1) un-adapted GTO + SPR-blind key (M1)** — TEST 1: call-off freq IDENTICAL across merge arms (1.02×) → **NOT the trainer**. The blueprint plays a fixed GTO call freq that ignores the range tracker's read of *this* opponent. | **Fix #1** extend the facing-jam equity guard to all streets |

> **Tests of record:** TEST 1 (parallel bias) = merge_every 4000 vs 250 A/B at 500k
> iters, jam 1.22× / call 1.02×. TEST 2 (SPR collapse) = P(allin) vs menu-richness on
> the served DB. Both confirm *mechanism + direction*, not the exact magnitude in the
> 30.5M DB. Scripts: `scripts/_tmp_spr_fingerprint.py` (TEST 2).

**Key insight (2026-05-31, user observation) — separate the PROPOSAL menu from the RESPONSE path.**
Before voluntary all-in was added at every node, the bot handled human aggression *surprisingly well*
on blueprint + pseudo-harmonic translation alone: a human jam was off-tree, translation mapped it to
the nearest well-trained big-bet response, and the bot never generated its *own* stray jams (jam
wasn't in its proposal menu). Adding voluntary all-in everywhere handed the floor-bias (#2) and the
menu-collapse (#3) a degenerate action to over-weight, and diluted per-action data — in exchange for
a benefit translation already delivered. Translation is "unsound" only versus a *size-cheating*
adversary (the LBR leak); a human who just shoves is playing the exact line translation handles most
robustly. **Principle: keep a tight, well-trained proposal menu + a robust response path (translation
+ Fix #1) for off-tree opponent bets.** This motivates Fix #4 below.

**The four fixes (ranked; do all):**
1. ✅ **DONE (2026-05-31) — Over-calling guard extended to ALL streets incl. preflop.** `RiverSubgameSolver._facing_allin_guard` now fires on preflop/flop/turn (river still routes through the full solver). Inference-only, NO retrain, fixes the in-game over-calling immediately, and finally makes the range tracker pay off (call iff `hero_equity` vs the tracked opponent range beats pot odds; confidence-gated → defers to blueprint when the read is untrusted). Cheapest, highest felt-impact. A called preflop/flop/turn jam is near-terminal (board runs out) → pure equity, no solve. Empty-board preflop is accepted (`board is None` reject, not falsy-empty); `n_runouts` = 600 preflop / 200 flop. `bot_public_state()` now carries `seat`. Tests: `tests/test_allin_guard.py` (15/15, incl. 3 preflop cases).
2. ⛔ **REVERTED 2026-06-04 (BUG-014) — see the RESOLVED block at the top of this file. The change
   below was a NET LOSS: storing raw worker regret broke CFR+ re-activation and collapsed blueprints to
   "open xlarge with 100% of hands, never fold." Per-worker CFR+ flooring is restored. The text below is
   kept as the (wrong) historical diagnosis.** ~~✅ DONE (2026-06-01) — Bias-corrected parallel merge.~~ *The bug is WORKER-local flooring: each worker's `cfr()` floored its own cumulative on write (`blueprint_trainer.py:251`), clamping a losing action's negative regret to 0 before the master saw it → no cross-worker cancellation = upward jam bias. Fix: at line 251 apply the CFR+ write-floor ONLY in single-thread (`discount_enabled=True`); in worker mode store the raw signed sum. The master keeps its per-round `max(0, decay*base+increment)` floor (single-thread floors on every write = canonical CFR+, so that floor was correct). Shrinks the floor-granularity approximation, doesn't remove it. Validated: `tests/test_parallel_trainer.py` 29/29 incl. `test_worker_mode_stores_raw_regret` + `test_merge_cross_worker_cancellation`; harness `scripts/validate_parallel_merge.py` — **run at scale 2026-06-01** (200k iters/arm, `merge_every=2000`, 16 workers): jam bias (parallel − single) = **+0.0014** (single 0.2941 vs parallel 0.2955 over 10,804 jam-action keys) → **bias confirmed gone** (was ~1.22× aggregate pre-fix); TV-distance mean 0.224 (not bit-identical — different RNG streams + partly converged, the bias-relevant number is the aggregate jam delta); 3.76× speedup. The harness gates on cumulative-strategy mass, not `visit_count` (which counts iterations single-thread but merge-rounds in parallel).* Workers report **raw unfloored regret increments**; master applies the CFR+ floor ONCE at merge → restores cross-worker cancellation at full `merge_every=4000` throughput (no speed penalty). MUST be validated: the A/B harness shows arm-A ≈ arm-B AND both match a single-thread reference (closes the validation gap the roadmap flagged and we skipped). Parallel is NOT scrapped — it stays an approximation pushed below noise; fallback if validation fails is smaller `merge_every` (slower but correct). Subtlety: a worker still needs floored regrets for its own within-chunk regret-matching, so "report raw deltas" changes within-chunk behavior → exactly why validation gates it.
3. **SPR-aware bet sizing (the redesign — UNCONFIRMED, measure first).** Replace fixed pot-fractions at low SPR with sizing that doesn't collapse to {small, jam} (geometric / fraction-of-remaining-stack, as solvers do); possibly add SPR buckets on **flop+turn only** (NOT preflop — preflop SPR is already implicit in the betting pattern; M1/SPR-loss is a postflop-pattern-reset problem). SPR buckets (learn different strategies) and SPR-aware sizing (fix the action menu) are DIFFERENT and both may be needed.
4. **Bet-menu cap + drop the voluntary all-in node (abstraction change → retrain; batch with #3).** Implements the proposal/response split above. **✅ VALIDATED — LBR gate passed (2026-06-02): at equal 4M budget (same seed), capped is ~35% LESS exploitable than control (LBR +2177.3 vs +3336.8 mbb/hand, Δ −1159.5). Earns the full 33M retrain.** *(Engine + trainer wiring done + reviewed: two-path parity 0 mismatches, control byte-identical, no lost jam. Fingerprint confirmed the cap structurally removes the high-SPR voluntary jam 100%→0%; jam-frequency itself is convergence-limited, so LBR — the off-grid/size-cheating probe — was the decisive metric. BR skipped for cost. Both arms 4M-under-converged so the absolute number is large; the 33M capped retrain is the shippable bot.)* `sizing.POSTFLOP_BET_MULT_CAPPED` (5-size incl. `overbet2`=2.0×, char `'2'`); `PokerGame(postflop_menu=, voluntary_allin=)` (default = control, byte-identical); `BlueprintTrainer(menu_mode='control'|'capped')` stamped to DB + `run_blueprint_trainer --menu-mode` + `blueprint_capped_*` tag + parallel-worker payload + resume menu-guard. Launch: `python tests/run_blueprint_trainer.py --iterations N --workers W --menu-mode capped`. **Preflop decision (resolved): the `voluntary_allin` flag is GLOBAL — it also drops the deep-stack voluntary preflop open-jam/limp-jam.** Kept global on purpose: a 100BB preflop open-jam is never GTO (push/fold is <~20BB), and genuine short-SPR 5-bet jams still emerge via the pot-relative re-raise clamp, so it costs ~nothing. **Serving/eval consumers threaded ✅ (2026-06-02):** `db_menu_mode(db)` auto-derives the arm at every boundary — `GameSession`+API (BUG-011 fix: a capped blueprint served through a control engine stray-jammed in live play), `lbr.py`, `best_response.py`, `match.py`+`run_match`, `action_abstractions`/`game_adapter`/`player` (PyPokerEngine), `translation.postflop_grid_for`, `strategy_api._PATTERN_CHARS`. Shared `sizing.is_capped_mode()` predicate so `capped`+`capped_no2` both drop the voluntary all-in. BR CLI gained `--workers` (parallel, bit-identical to serial). **A third arm `capped_no2`** (capped menu minus the 2.0× tier, still no voluntary all-in) is wired as the clean one-variable test of the 2.0× tier's value — train it to matching iters on CONVERGED arms and LBR-compare vs `capped`.
   - **Postflop:** `POSTFLOP_BET_MULT = {small 0.33, medium 0.66, large 1.0, overbet 1.5, overbet2 2.0}` and **remove the separate voluntary all-in action**. All-in *emerges* when the top tier clamps to the stack — so it still appears at low SPR, but stops competing for regret mass at high SPR. EV loss from the 2x cap is tiny (>2x-pot flop/turn bets are ~never GTO; polarized-river jams are the solver's job) — confirm with BR/LBR. **Off-tree handoff when FACING a bet >2x:** river → subgame solver; flop/turn → translation (no turn/flop solver yet); an actual all-in on any street → Fix #1. **Impl subtlety:** a tier that clamps to all-in should still map to pattern char `a` (not the tier char), so two physically identical all-ins share one key.
   - **Preflop re-raise:** maybe add `2.0` on the **3-bet branch only** — for 4-bets `2.0x` ≈ a jam at 100BB and CLAUDE.md already rejects a 4th 4-bet size (low-SPR 4-bets compress to one-size-or-jam). Measure before committing.
   - **Preflop open:** leave the 4-tier BB-anchored ladder (2 / 2.5 / 3.5 / 5BB) **unchanged**. Opens are BB-anchored, not pot-anchored, so the 2.0x-overbet idea doesn't map; a 10BB open is never +EV as a *proposal*, and `xlarge=5BB` + translation already anchor big human opens as a *response*.

**Measurement plan to design the #3 redesign with numbers (do all, cheap — no BR/cloud/full-retrain):**
- **Measurement A — SPR spread per key.** ✅ **DONE (2026-06-01)** — `scripts/measure_spr_spread.py`, 40k self-play hands on the served `blueprint_par_20260529_233056.db` (82,897 postflop decisions, 441 keys ≥50 visits). **Verdict: SPR card-buckets are NOT high-leverage → favor configs #1/#2 (menu-cap / SPR-aware sizing, no buckets).** Postflop SPR-regime split `[jam<1 / low1-3 / mid3-6 / deep>6] = [3.5% / 5.6% / 8.3% / 82.6%]` → only **9.1%** of decisions are low-SPR, **82.6% genuinely deep**. "Drowned-low" (low/jam decision mass trapped in a ≥60%-deep-dominated key — the over-jam-leak fingerprint of a shove averaged into a pot-bet key) = just **7.7%**. Straddle 39.3% / weighted-crossover 16.1% over-count benign mostly-deep-with-a-tail keys; crossover rises by street (flop 11.6% → turn 19.0% → river 23.8%, so turn>flop if buckets ever added). **Implication:** the over-jamming isn't mainly SPR-blind keys averaging regimes (too few low-SPR decisions) — it's the proposal *menu* offering a degenerate all-in at deep/mid SPR that bias #2 over-weighted → **Fix #4 (menu cap) is the targeted fix, not SPR buckets.** Caveat: 9.1% slightly under-counts true low-SPR frequency (early jams never reach a low-SPR postflop node), but deep-dominance is too strong to flip the call.
- **Measurement B — info-set budget per candidate scheme.** ✅ **DONE (2026-06-01)** — `scripts/measure_infoset_budget.py`, projecting each config's structural multiplier onto the served blueprint's REAL reachable-key counts (4,200 preflop + 70,493 postflop = 74,693; flop 29,063 / turn 25,387 / river 16,043). Result: **#0/#1/#2 = 1.00× (74,693)** — menu-cap and SPR-aware sizing change chip amounts / which actions exist, NOT the key *structure* (menu-cap even shrinks the pattern set by dropping the all-in action), so they cost **zero** extra budget. **#3 (K=2 buckets) = 1.73× (129,143); #4 (K=3) = 2.46× (183,593)** — SPR buckets multiply the flop+turn slice. **A + B verdict: SPR-bucket configs #3/#4 are OUT** — A showed low leverage (<10% low-SPR decisions), B shows high cost (1.7–2.5× training for the same per-key convergence). The favored fix (#4 menu-cap, the redesign A pointed to) is also the *free* one (1.00×). So C only needs to compare #0 (control, bias-free) vs the menu-cap/SPR-aware arm — no SPR-bucket retrain.
- **Measurement C — does SPR-aware sizing ALONE fix it?** Re-fit just the sizing (geometric low-SPR), short train, re-run TEST 2's fingerprint: did P(allin) stop spiking as the menu thins? If yes, #3 needs no SPR card-buckets (cheapest outcome).
  - **Arm C2 (Fix #4) — bet-menu cap + dropped all-in node.** Retrain a short run with the 2.0x cap and no voluntary all-in action; re-run TEST 2's fingerprint + BR/LBR. Does P(allin) drop without exploitability rising? This is the empirical test of whether the old proposal/response split genuinely played better.

**Candidate abstraction configs being measured (the grid A/B/C run over).** The **card** abstraction (30/10 preflop, 20/16/10 postflop) is held FIXED as the control — it's not the suspect; the **bet-sizing/SPR** axis is. Two knobs move:
- **Knob 1 — SPR on the key (Fix #3 bucket half / the M1 fix):** `none` (current SPR-blind key) | `K SPR buckets on FLOP+TURN only` (NOT preflop — SPR implicit in the betting pattern; NOT river — solver's job), with K ∈ {2, 3} candidates.
- **Knob 2 — bet-sizing menu:** `current` ({0.33,0.66,1.0,1.5} **+ a separate voluntary all-in node**) | `menu-cap` (Fix #4: {0.33,0.66,1.0,1.5,2.0}, **drop the all-in node** — all-in emerges only when the top tier clamps to stack) | `SPR-aware sizing` (Fix #3: low-SPR sizes that don't collapse to {small, jam} — geometric / fraction-of-remaining-stack).

| # | SPR buckets | Sizing menu | What it isolates |
|---|---|---|---|
| 0 | none | current + all-in node | **control** (= served blueprint) |
| 1 | none | menu-cap (Fix #4) | does dropping the all-in node ALONE cut P(jam)? |
| 2 | none | SPR-aware (Fix #3 sizing) | does sizing ALONE fix it, no card-buckets? (cheapest win) |
| 3 | K=2 flop+turn | current + all-in node | does SPR *bucketing* alone help? |
| 4 | K=3 flop+turn | menu-cap | the "full" candidate |

How the three measurements use the grid: **A trains NOTHING** (pure diagnostic on the *existing* blueprint's self-play) — its SPR-spread output CHOOSES K and which streets, and can kill configs 3–4 before any training (if SPRs barely spread, `none` wins). **B is a COUNTING pass only** (reuse the decouple info-set counter): a postflop key is `coarse(10) × strength × position(2) × pattern`, so K SPR buckets multiply the flop+turn slice ~K× — B answers "does config 4 fit a trainable budget or does K=3 blow it up 3×?". **C trains the 1–2 survivors** of A+B for a short run + re-runs the TEST-2 fingerprint: if config 2 (SPR-aware sizing, no buckets) flattens the jam spike → ship it, skip card-buckets (cheapest); else fall to config 4. **Open params** A must set first: K (2 vs 3) and the SPR-aware sizing formula (geometric vs fraction-of-remaining-stack). **Do NOT pre-commit configs 3/4 before A says the SPR spread is real.**

**Architecture decisions locked this session:**
- **"Solve from the turn" is OFF the table for now.** A solve-to-showdown turn solver is ~46× a river solve → minutes, infeasible (river already ~10s); AND the solver depends on the blueprint for ranges/warm-start/EV-baseline, so the blueprint can't drop turn/river. The solver stays **river-only** until depth-limited solving exists.
- **North star (a) (user, 2026-05-31): a turn AND flop depth-limited subgame solver running in <10s like the river does now.** Feasible ONLY via depth-limited solving with a **leaf value function** (blueprint counterfactual values, recomputable offline) — NOT solve-to-showdown. This is the Phase-4 "depth-limited turn/flop (leaf values)" hard lift; it's the eventual target, not a near-term fix.

**Sequencing:** ship #1 (immediate, no compute) → build+validate #2 → run Measurements A/B/C → decide #3 from numbers → (later) build the depth-limited turn/flop solver toward north-star (a). Don't commit the #3 abstraction redesign before the measurements.

---

## Phase 1 — Blueprint training ✅ COMPLETE

A heads-up blueprint is trained with Monte Carlo CFR+ and stored in SQLite.

| Component | File | Status |
|---|---|---|
| Hand evaluation | `src/abstractions/hand_evaluator.py` (phevaluator) | ✅ |
| Card abstraction | `src/abstractions/card_abstractions.py` — **decoupled imperfect-recall preflop: 30 fine buckets (`pf_0..pf_29`, preflop keys) / 10 coarse classes (postflop `startBucket`)** + **distribution-aware (potential-aware) postflop buckets: 20 flop / 16 turn / 10 river** (`PostflopV2`, EMD-clustered equity distributions) | ✅ |
| Preflop equity precompute | `scripts/compute_preflop_equity.py` | ✅ |
| Postflop bucket pipeline | `scripts/compute_postflop_buckets.py` (fit centroids) → `scripts/bake_postflop_table.py` (bake canonical→bucket tables, centroid-stamped) → `src/abstractions/{postflop_v2,postflop_features,canonical}.py` | ✅ |
| Action abstraction | `src/abstractions/action_abstractions.py` — small/medium/large + preflop ladders + all-in. **`menu_mode` toggle (`control` \| `capped` \| `capped_no2`, `src/abstractions/sizing.py`): `capped` adds the 2.0× `overbet2` tier and DROPS the voluntary all-in node (Fix #4); auto-derived at every boundary via `db_menu_mode`.** | ✅ |
| Abstracted rules engine | `src/cfr/poker_game.py` — stack-aware, all-ins, 3 aggressions/street | ✅ |
| Info-set keys | `src/cfr/keys.py` — single source of truth, position-aware | ✅ |
| CFR+ trainer | `src/cfr/blueprint_trainer.py` — external-sampling MCCFR+, Linear-CFR-style discounting (regret + strategy-sum; not canonical DCFR) | ✅ |
| Regret/strategy storage | `src/cfr/information_set.py` | ✅ |
| Persistence | `src/storage/blueprint_db.py` — SQLite, WAL, checkpoint/resume | ✅ |
| Active-blueprint resolution | `src/config.py:resolve_blueprint_path()` | ✅ |

**Outcome:** training writes `analysis/blueprints/blueprint_<timestamp>.db`; the API/bot
auto-select the DB with the most iterations. Correctness has been hardened
through a documented bug hunt ([../backend/bot/docs/BUG_LOG.md](../backend/bot/docs/BUG_LOG.md))
plus Hypothesis property tests.

> **Known limitation carried forward (M1):** the postflop key omits pot/stack
> depth (SPR), so different stack depths collapse to one key. This is the main
> motivation for Phase 3 (subgame solving). See
> [DEVELOPER_GUIDE.md §11](DEVELOPER_GUIDE.md#11-known-limitations).

---

## Phase 2 — Serving + Play vs the bot ✅ COMPLETE

A Flask API serves the blueprint and a React frontend plays against it.

| Component | File | Status |
|---|---|---|
| Flask-free live-hand engine | `src/game/game_session.py` (`GameSession`) | ✅ |
| Bot strategy interface | `src/game/bot_strategy.py` — `BotStrategy` + `BlueprintStrategy` | ✅ |
| Session store interface | `src/game/session_store.py` — `InMemorySessionStore` (Redis/DynamoDB drop-in) | ✅ |
| Card format conversion | `src/game/cards.py` — engine `SuitRank` ⇄ display `RankSuit` | ✅ |
| REST API | `backend/api/strategy_api.py` — strategy lookup + game endpoints | ✅ |
| Strategy explorer UI | `frontend/.../HandExplorer.jsx`, `KeyExplorer.jsx` | ✅ |
| Play-vs-bot UI | `frontend/.../AiGame.jsx` | ✅ |
| Measurement harness | `src/evaluation/` — best-response exploitability (`best_response.py`), LBR off-tree lower bound (`lbr.py`), head-to-head + AIVAT variance reduction (`match.py`, `aivat.py`); CLIs `tests/run_evaluation.py` / `run_lbr.py` / `run_match.py` | ✅ |

**Design intent:** the `game/` engine has **no Flask imports** and the
`BotStrategy` / `SessionStore` interfaces are deliberately thin so the later
phases (subgame solving, online play, AWS) are additive, not rewrites. The
`BotStrategy` interface already receives full public state, not just the bucketed
key, so a subgame solver is a drop-in replacement.

---

## Phase 3 — Hand-level range tracking ✅ COMPLETE

The prerequisite for subgame solving: a hand-level Bayesian belief over the
opponent's hole cards, which the river solver consumes as its input range.

| Component | File | Status |
|---|---|---|
| Hand-level Bayesian range tracker | `src/game/range_tracker.py` (`RangeTracker`) — per-hand weights, card removal, blueprint-model Bayesian updates, confidence score, equity-vs-range | ✅ |
| GameSession integration | `game_session.py` — per-hand tracker, `observe` on human actions, `reveal` on streets, persisted in session JSON; river-entry snapshots fed to the solver (`bot_public_state`) | ✅ |
| Confidence-aware consumer | `bot_strategy.py` (`ConfidenceAwareStrategy`) — blueprint while confident, equity-vs-range fallback when confidence collapses | ✅ |
| "Bot's read" UI | `public_view().botRead` + `AiGame.jsx` panel (confidence + top hands) | ✅ |

> The earlier `src/subgame/{off_tree_detector,subgame_detector,confidence_detector,player_blueprint_adapter}.py`
> prototypes were **deleted (2026-05-28)** — they were a closed cluster referenced
> by nothing. The live off-tree trigger is `RiverSubgameSolver._solver_inputs` + the
> EV gate, not those detectors.

---

## Next blueprint redesign — card + betting abstraction 🔄 IN PROGRESS (one fresh retrain)

> **SUPERSEDED IN PART (2026-05-29) — see `docs/ABSTRACTION_REDESIGN_HANDOFF.md`.**
> The card abstraction below ("preflop 15→40, postflop 12/12/10 unchanged") was
> revised into a **decoupled imperfect-recall** scheme: **30 fine** preflop buckets
> (preflop keys only) + **10 coarse** classes (postflop `startBucket`), and postflop
> strength **20 / 16 / 10** (finer flop). The decouple cuts postflop card-space ~3.7×
> vs carrying the full preflop bucket, which pays for the finer flop. The **betting**
> changes in this section (xlarge open, overbet tier, voluntary all-in) are UNCHANGED
> and still ship in the same retrain. The notes below are kept for the betting bundle
> + rationale; for the card-abstraction numbers and the fine/coarse key contract, the
> handoff doc is the source of truth.

> **Status (2026-05-29).** Betting bundle fully implemented + bug-swept. Card
> abstraction now on the decoupled 30-fine/10-coarse + 20·16·10 scheme (code done;
> postflop centroids pending re-fit/re-bake — the paused step).
> Training path: preflop fine/coarse decoupled buckets (`card_abstractions.py`, maps
> derived from `compute_preflop_equity.py`'s equity table), 4th open `xlarge`/`x` + postflop overbet `overbet`/`o`
> + voluntary all-in everywhere (`sizing.py`, `keys.py`, `poker_game.py` BOTH the
> history and threaded `state_*` paths). Measurement/inference mirrors all updated:
> `best_response.py` (engine-derived, auto-adapts), `lbr.py` + `match.py` +
> `cross_match.py` victim models (now context-aware: opens are `bet_*` incl. xlarge,
> 3-bet/4-bet `raise_*`×3, postflop incl. overbet, voluntary all-in — this also fixed
> a *pre-existing* bug where opens were modelled as `raise_*` and looked up uniformly),
> `action_abstractions.py` (PyPokerEngine path), `translation.POSTFLOP_GRID` (added the
> `('o',1.5)` overbet bracket), `blueprint_projection.tree_action_char` (overbet char),
> API `/abstractions` `_SIZE_CHAR`/`_PATTERN_CHARS` (+`x`/`o`; fixed a preflop-open
> 500), and the explorer UI vocab. `keys.action_char` now RAISES on an unmapped action
> (removed the silent `'x'` default that aliased `bet_xlarge`). A 3-agent review found
> the engine core clean (oracle 262k checks / 0 fail; cap + chip-conservation intact).
> Gate green across 100+ tests + training/LBR/match/PyPokerEngine smokes. The parallel
> trainer needs **no** change (merge keys on info-set-key + action-name). All existing
> `blueprint_*.db` are abstraction-incompatible → fresh run (don't resume).

The next blueprint bundles **all** the abstraction changes below into one retrain
(they all invalidate existing blueprints, so they ship together and are measured as
a bundle). Headline: sharper preflop cards (15→~40 buckets), a voluntary all-in
anchor everywhere, a 4th preflop open, and **one postflop overbet tier** — modelled
on Libratus's blueprint (potential-aware cards + a few pot-fraction/multiple bet
sizes incl. overbets). The richer-but-deeper menu is **made affordable by
parallelising training** (see "Training cost + parallelism" at the end).

### Card abstraction

| Axis | Now | New | Why |
|---|---|---|---|
| Preflop buckets | 15 (`pf_0..pf_14`) | **~40** | 15 ≈ 11 hands/bucket (merges AKs/AQs/AJs); ~40 ≈ 4/bucket — much sharper conditioning. The preflop bucket is the `startBucket` prefix on **every** postflop key, so it also sharpens postflop. |
| Postflop strength | 12 / 12 / 10 | unchanged (candidate to bump) | Keep for this retrain to isolate the preflop+betting deltas; bump later if LBR says postflop is the residual leak. |

> **Lossless 169-hand preflop (Libratus-style) considered and deferred.** It is the
> zero-preflop-error ideal, but 169/15 ≈ **11.3×** the whole tree (the bucket prefixes
> every key) → ~11× the iterations. ~40 captures most of the resolution at ~2.7× cost.
> Revisit 169 only if preflop card error is shown to be the binding leak.

> **No SPR/pot dimension in the blueprint — by design, Libratus-consistent.** Stacks
> reset to a fixed depth each hand (ours and Libratus's both), so pot + remaining stack
> at any node are a deterministic function of the betting history — SPR needs no separate
> axis. Libratus assumes perfect recall of actions *in actual play* and uses **no card
> abstraction + real pot/stacks during real-time solving**; SPR is recovered at solve
> time, not baked into the blueprint. Our key is lossier here (the **M1 limitation**: the
> per-street pattern resets and uses bucketed sizes, so different-pot paths to a street
> can collapse to one key) — but we recover SPR the same way: the subgame solver already
> takes the real `riverEntryPot`/`riverEntryStacks`. So we do **not** add an SPR bucket.

### Betting abstraction

**Why.** A faithful head-to-head (`src/evaluation/cross_match.py`) showed the
current small-open blueprint losing ~200 mbb/hand to the older big-open blueprint.
Root cause (verified): the bot has **zero trained response to off-grid big opens
or any preflop all-in** — **0 `pf_*_*_a` keys** exist. The engine only offers
`allin` preflop when a sized raise is unaffordable, and at 100 BB the 3-raise cap
means sized raises never exhaust the stack, so a preflop jam is **never reachable**
in training. This is an action-grid hole, not a postflop or training defect (the
new-vs-new mirror is ~break-even OOP). It is **not** a reason to revert to big
opens — small opens are GTO-correct in HU and the redesign fixed a real 3-bet-collapse
bug; the fix is to make big sizes/jams *representable* (Libratus's dense early-street
abstraction + translation — not preflop solving).

**The change** (abstraction change → fresh retrain required; cannot resume an
existing run):

| Node | Sizes |
|---|---|
| Open (first-in) | `2 / 2.5 / 3.5 / 5` BB (a 4th, larger open — new size char **`x`**, so the open alphabet is `s/m/l/x`) **+ allin** |
| 3-bet | `0.66 / 1.0 / 1.5 × pot` **+ allin** |
| 4-bet | `0.66 / 1.0 / 1.5 × pot` **+ allin** |

- Keep the small GTO opens (2/2.5/3.5) untouched and **add** a 5 BB anchor. The
  open grid does double duty — the bot's *own* opens **and** the buckets it reads
  *opponent* opens through — so *adding* a size serves both, where *re-spacing*
  3 slots forces a trade-off between them.
- Make `allin` an always-available preflop aggressive action at every node
  (open / 3-bet / 4-bet), not just when forced. Fills the jam-response hole and
  lets the bot 3-/4-bet-jam itself.
- 3-bet/4-bet multipliers **unchanged** (facing-large-3-bets is already well
  trained: ~375 keys / 5.6M visits; the pot-relative ladder scales correctly).

**Postflop — add ONE overbet tier + voluntary all-in.**

| Node | Now | New |
|---|---|---|
| Postflop bet/raise | `0.33 / 0.66 / 1.0 × pot` (+ stack-forced all-in only) | `0.33 / 0.66 / 1.0 / **1.5** × pot` **+ voluntary all-in** |

- **1.5× pot only — not 1.5× *and* 2.0×.** Each extra postflop size compounds across
  flop/turn/river patterns (≈2–2.5× the postflop pattern space per added size). One
  overbet (1.5×) captures the bulk of overbet value (polarized turn/river barrels);
  2.0× is rarer and ~1.5× extra cost for diminishing return.
- **2.0×+ lives in the SOLVER menu, not the blueprint.** This is the Libratus split:
  the blueprint carries *one* overbet so the bot is never helpless on flop/turn; the
  river solver (already `1.5×`; widen to `2.0×` cheaply since it only solves the spot,
  not the whole tree) and the future turn/flop solver carry the richer overbet menu
  where it actually matters. Fine where it counts, coarse globally.
- Voluntary all-in applies postflop too (same `'a'` char; same engine change as the
  preflop anchor). **Not redundant with 1.5×** — they cover different SPR regimes. A
  1.5× bet auto-converts to all-in (`_apply_stack_constraints`) only when `1.5×pot ≥
  stack`, i.e. **SPR ≤ 1.5**; above that a jam is strictly larger and strategically
  distinct (jam = deny implied odds/end decisions; 1.5× = keep fold equity + money
  behind). The useful band is **SPR ~1.5–3** (4-bet pots, later streets of 3-bet pots);
  at high SPR all-in just trains to ~0 freq, and the engine dedups the low-SPR overlap,
  so the marginal cost is small. The blueprint also needs shove mass as the **river
  solver's warm-start prior** (`river_cfr.warm_start`) and for turn/flop + solver-fallback
  play where it's the only decision-maker.
- New postflop size char needed (e.g. **`o`** for the 1.5× overbet) in `cfr/keys.py`
  + `sizing.py` (`POSTFLOP_BET_MULT`), mirrored in `poker_game.py` / `action_abstractions.py`
  / the translation grid.

**Covers / doesn't.** Covers opens, 3-bets, 4-bets, and all **jams**. It does
**not** cover **non-jam 5-bets+** — those are beyond the 3-raise cap and not
all-in (a small-sizing line can 5-bet to ~25 BB with ~75 BB behind). Those rare
deep reraises are handled by **subgame solving** (Phase 4), not by abstracting
them (rare lines train thinly → poor strategy).

**Decided against:** a real-time per-opponent *self-improver* (Libratus-style
overnight hole-patching) — infeasible here, since one session per human gives
per-opponent sizing data too sparse to estimate.

**Exactly what the all-in anchor touches (confirmed 2026-05-28).** `allin` already
maps to char `'a'` in `cfr/keys.py`, so the **key format does not change**. The work
is in three places:
1. **Engine — `cfr/poker_game.py` (the real change).** Today `get_preflop_legal_actions`
   / `get_postflop_legal_actions` emit only `check/fold/call/bet_*/raise_*`; `allin`
   is injected *only* by `_apply_stack_constraints` when a sized bet exceeds the stack.
   At 100 BB that almost never fires → ~0 `_a` keys (the BUG-007 hole). Fix: append
   `allin` as a **voluntary** aggressive action at every betting node (deduped when a
   sized bet already equals the stack). **This grows the action set → abstraction
   change → fresh retrain.**
2. **Translation grid — `game_session.py` (and `evaluation/lbr.py`'s mirror) where the
   per-node grid for `translate_bet` is built.** Append an `('a', allin_eff_frac)`
   bracket so an opponent bet above the top sized bet **interpolates toward all-in**
   instead of clamping to the top grid char. The static `translation.POSTFLOP_GRID`
   reference stays as-is (all-in's fraction is stack-dependent, computed per node).
   *This only becomes meaningful once (1) is retrained* — until the blueprint has
   trained `_a` responses, `blend()` routes the all-in bracket's weight to the fold
   fallback (`_blend_lookup` returns `{}` for an untrained key).
3. **PyPokerEngine path — `abstractions/action_abstractions.py`** mirrors the grid;
   `tests/test_sizing_consistency.py` guards drift.

### Training cost + parallelism

Info-set count (≈ iterations to reach equal convergence, since MCCFR must visit each
set enough times) scales ~linearly with the abstraction size. Rough multipliers vs the
**current** 15-bucket / 3-size blueprint:

| Change | ≈ iteration multiplier |
|---|---|
| Preflop 15 → ~40 buckets (prefixes every key) | ~2.7× |
| Preflop: voluntary all-in + 4th open (`x`) | ~1.5× (preflop keys only) |
| Postflop: +1.5× overbet + voluntary all-in | ~2.5× (postflop pattern space) |
| **Bundle (the proposed redesign)** | **≈ 6–7× more iterations** |
| If 2.0× overbet were also added | ≈ 9–10× (why it's deferred to the solver) |

These are order-of-magnitude (the rare-line tail dominates *full* convergence; common
spots converge sooner). So if the current blueprint trains to convergence in **N**
iterations, budget **~6–7N** for the redesign.

**Parallelism makes the bundle ~wall-clock-neutral.** External-sampling MCCFR is
independent sampled iterations → embarrassingly parallel. A realistic **~5×** on an
8-physical-core laptop (thermal throttling caps it below 8×) brings 6–7× iterations
back to **~1.3× today's wall-clock** for a far richer blueprint. **IMPLEMENTED** in
`cfr/parallel_trainer.py` (see measured facts + recommended settings below); the design
notes that follow document how it works:
- `multiprocessing` (processes, not threads — the inner loop is pure Python, the GIL
  would serialise threads). W ≈ physical cores. Single-process path stays intact.
- Each worker runs a chunk into its own in-memory `InformationSet` dict; **periodic
  merge** sums `cumulative_regrets`, `cumulative_strategy`, **and the discount clocks
  `visit_count` / `strategy_visit_count`** (the clocks are the easy-to-miss part — the
  Linear-CFR `((t-1)/t)**alpha` discount is keyed on them; omit them and the schedule
  breaks). Re-broadcast the merged dict, resume.
- **Correctness caveat (it's an *approximation* of single-threaded Linear-CFR+):**
  regrets are floored at write (`blueprint_trainer.py:233`), so summing two floored
  series ≠ a single floored series; and each worker discounts on its own clock, so the
  merged weighting is slightly off. This is exactly the drift the data-parallel-CFR
  literature tolerates — bound it by merging at the existing checkpoint cadence. All
  workers must share identical `alpha`/`gamma` (the `blueprint_trainer.py:375-384`
  guard already refuses a mid-blueprint schedule change).
- **Validate** against a small single-thread run (same exploitability) before trusting
  it for the big retrain.

**Files:** `sizing.py`, `cfr/poker_game.py` (voluntary `allin` + 4th open `x`),
`cfr/keys.py` (`x` only — `a` already exists), `abstractions/action_abstractions.py`,
`game_session.py` + `evaluation/{match,lbr}.py` (all-in translation bracket + grid
mirror), frontend explorer vocab.
**Verify post-retrain:** open-jam trains to ~0 frequency; dedup `allin` when a sized
raise already equals stack; re-baseline preflop; and read the BR/LBR delta as a
*bundle* (preflop + postflop changed together) — keep the preflop change minimal so
it can't masquerade as a postflop regression.

### Measured training-perf facts (2026-05-29) + recommended run settings

The parallel trainer is now IMPLEMENTED (`cfr/parallel_trainer.py`, `run_training(..., workers=)`),
not just designed. Measured on the new decoupled abstraction (30-fine/10-coarse + 20·16·10),
baked tables present, single-thread:

| Quantity | Value |
|---|---|
| Per-iteration compute (baked tables, single-thread) | **~12 ms/iter (~84 it/s)** |
| Baseline broadcast size | **156 bytes/info-set** (≈8.9 MB at 57k sets) |
| Broadcast cost grows as | **blueprint size × workers** — `pool.map` re-pickles the baseline once per worker each round |
| New-scheme info-set count | **57k at just 20k iters, still climbing** (richer betting tree → likely converges well past the old ~41k) |

The tension: each merge round costs ≈ `(N_infosets × 156 B) × workers` of pickling + IPC, and
you want the round's *compute* to dwarf it. At 12 ms/iter that's easy to arrange by sizing the
round large enough.

**Recommended command (8-core laptop):**
```
python -c "from tests.run_blueprint_trainer import run_training; \
run_training(20000000, checkpoint_every=50000, workers=8, merge_every=4000)"
```
- **`merge_every=4000`** → 32,000 iters/round (4000 × 8 workers). Round compute ≈ 4000 × 12 ms
  ≈ 48 s/worker vs a broadcast of only ~3–6 s even at ~150k info-sets → **<15% overhead**, while
  32k/round is a negligible slice of a 20M run so the discount-timing + write-floor bias stay
  small. The old default `merge_every=2000` (16k/round) would push overhead to ~25–35% as the
  blueprint fills — wasteful now that iters are 12 ms, not microseconds.
- **`workers=8`** (or **7** if running `track_training.py` concurrently, so BR/LBR doesn't steal a
  training core). Don't exceed physical cores — hyperthreads don't help pure-Python CFR and only
  multiply the per-round pickle cost.
- **`checkpoint_every=50000`** — ~1–2 rounds; DB writes are trivial next to 48 s rounds. (If using
  the snapshot tracker at a 2M cadence, keep `checkpoint_every` a divisor of the snapshot interval
  — snapshots can only land on checkpoint boundaries.)
- **20M is a planning figure, not a target.** Stop when BR/LBR flattens (`track_training.py
  --every 2000000`). The new scheme has *more* info-sets (betting expansion) but a ~3× *smaller*
  postflop card-space, so per-info-set convergence is faster — expect the knee in the low tens of
  millions. At ~5× parallel speedup, 20M ≈ **~13–14 h**.

### Where training time actually goes (2026-05-30 profiling) + the river-board cache

After the redesign retrain, a `cProfile` pass (`scripts/profile_evaluator.py`) over real training
turned up a surprising distribution that re-ordered the remaining speed levers:

| Finding | Number |
|---|---|
| Hand evaluator (`rank7`/phevaluator `_evaluate_cards`/`hash_quinary`) share of training | **only ~17%** |
| Share of all evals coming from **river bucketing** (`board_winrates`, 1081 hands/board) vs showdown | **~99% vs ~1%** (~930 evals/iter, ~2 of them showdown) |
| Pure-Python CFR loop (`cfr` walk + `get_strategy` + threaded state engine) | **the dominant cost** |
| Parallel speedup on the 8-core laptop | **~3.7×** (312 vs 84 it/s) — capped by thermal throttle, the serial master merge, and hyperthreads not helping pure-Python |

**Consequence for "Lever B" (swap pure-Python phevaluator for a compiled evaluator like eval7).**
Because the evaluator is only ~17% of training, a compiled evaluator buys **only ~1.14× on
training** — it was *re-scoped from a training lever to an inference/solver lever* (the river solver
and range tracker are eval-dominated, so eval7 helps live latency / the solver's time budget there).
The genuine 5–10× training lever is **compiling the CFR inner loop itself** (Cython / Rust `nogil`),
not the evaluator — that's the P3 follow-up. (Lever B, if ever done, still needs an
ordinal-equivalence proof so it does *not* force a re-bake/retrain.)

**The win actually taken: a canonical river-board equity cache.** Since ~99% of training evals are
`board_winrates` ranking every hand on a river board, and equity is suit-invariant, we cache that
pass on the **canonical (suit-isomorphic) board** instead of the concrete one:

| | |
|---|---|
| Concrete 5-card boards | 2,598,960 |
| **Canonical** 5-card boards (the cache key space) | **134,459** → **19.3× fewer** |
| `board_winrates` calls over a 30M-iter run, canonical key | ~134k **total** → **99.7% hit rate** after a ~1.5% warmup |
| **Measured training speedup** (warm vs no-cross-iter-reuse, single-thread) | **1.63×** (`scripts/measure_river_speedup.py`) |

Implementation notes (`abstractions/postflop_v2.py`, `abstractions/canonical.py:canonical_board_perm`):
- **Resume-safe, NOT an abstraction change.** Equity is stored as a `uint16` numerator
  (`2·990·equity` is an exact integer in `[0,1980]`; `s/1980.0` reconstructs the *bit-identical*
  float64 the pre-cache path produced), so every river **bucket is unchanged** — `keys.py`,
  centroids, and `sizing.py` are untouched. An in-progress blueprint can just **resume** under the
  new code; no retrain. (An earlier `float32` store flipped ~1/5000 bin-edge buckets — caught by the
  3-agent audit and replaced with the exact `uint16`.)
- **Module-global, on purpose.** The parallel trainer builds a fresh `BlueprintTrainer`/`PostflopV2`
  every merge round, so an *instance* cache would go cold every ~4000 iters (concrete≈canonical in a
  short window → ~1.0×). A module global lives for the worker process's lifetime, so each persistent
  worker warms it across rounds. **LRU eviction** (`OrderedDict`, default cap
  `ALLIN_RIVER_CACHE_BOARDS=100_000` ≈ 0.26 GB/process → ~2.1 GB at 8 workers, ~1.6 GB at 6) — a cap
  below the 134,459 canonical boards degrades gracefully (keeps the hottest boards) instead of
  thrashing; raise toward 134,459 on a roomy box to cache them all. Tests:
  `tests/test_river_board_cache.py`.
- **Open follow-up:** the inference API serves one shared `PostflopV2`/cache across Flask threads
  (`threaded=True`); the cache is a pure memo so it can never return a *wrong* value, but the
  check/clear/insert is non-atomic — add a lock or a per-session instance before go-live.

---

## Phase 4 — Subgame solving 🚧 IN PROGRESS

Improve on the blueprint at runtime by re-solving the current spot with full
information (real pot, real stacks, the Phase-3 range), fixing the M1 abstraction
loss. **Long-term goal: handle any number of reraises and any bet size — incl. the
bot's own overbets and 5-bets+ — on any street.** Blueprint for the common cases,
on-demand solving wherever play goes off-abstraction or into a high-leverage endgame
(bounded in practice by stack depth: at 100 BB you get ~4–5 raise levels before
someone is all-in).

**Solving cost scales with the number of streets remaining, not SPR.** A **river**
solve is cheap *regardless of SPR* because it terminates at the showdown kernel —
exact terminal values, no continuation needed. A **turn/flop/off-tree-preflop** solve
has streets ahead, so it cannot solve to the end cheaply and needs a **leaf value
function** at the depth limit. That single capability — not SPR — is the dividing line
between what we have and what we don't.

| Component | File | Status |
|---|---|---|
| River endgame solver (v1, *unsafe*) | `src/subgame/river_subgame_solver.py` — small river tree, vectorized CFR+, ranges from `RangeTracker`, blueprint warm-start (`river_cfr.warm_start` + `blueprint_projection.py`), EV-gated; **served live** (`bot_public_state` feeds it river-entry pot/stacks/ranges/path) | ✅ built (~24× less river-exploitable than the blueprint — see caveat below); **already overbets** (menu includes 1.5× pot + all-in) |
| River nested off-grid sizing | inject the opponent's *exact* bet size as a real tree edge instead of snapping it to the nearest menu size (Libratus nested solving) | ✅ (2026-05-30) `RiverTree.inject_realized_edge` + `_navigate` tries injection before snapping; the bot's decision node now faces the TRUE off-grid pot. Falls back to snapping when the size is really an all-in / below min-raise / already on-grid. Tests in `test_river_tree.py` + end-to-end. |
| Low-SPR deep-raise / 5-bet+ solving (any street) + **all-in-terminal guard** | reuse the river machinery for near-terminal deep nodes, incl. **preflop non-jam 5-bets+** and any beyond-cap reraise — the bot can re-raise off-abstraction here. **Highest-value slice (file 2026-05-30): guard every all-in.** A flop/turn all-in that gets called runs the board out with no further decisions → it's *near-terminal*, so it solves with the cheap river machinery (equity-vs-range, **no leaf-value function**). | 🔄 **FACING-a-jam DONE (2026-05-30)**; proposing-a-jam + non-jam deep-raise still 📅. `RiverSubgameSolver._facing_allin_guard`: on flop/turn, when the call commits the whole stack (`to_call ≥ bot_stack` — near-terminal regardless of pre-jam SPR), override the blueprint with the pot-odds-correct call/fold from `RangeTracker.hero_equity` vs the live belief. Confidence-gated (`guard_confidence`, defers to blueprint on an off-model/untrusted belief — the high-SPR-overbet safety) + chip-margin gated. Runs BEFORE the river path in `decide()`. Tests: `test_allin_guard.py` (12, incl. high-SPR overbet acts-when-trusted / defers-when-untrusted). Directly kills the stray ~1% blueprint jams that dump a stack. (River all-ins already covered: solver runs on every river decision.) PROPOSING a jam needs the opponent's calling model + a value for the non-jam alternative → deferred. |
| **Depth-limited turn/flop solving (leaf values)** | blueprint counterfactual values as **multi-valued** leaf states; **this is what lets the bot overbet/5-bet on the flop and turn** | 📅 (the hard lift, gates the rest) |
| Safe / nested subgame solving | adversarial root + opt-out values (gadget) — provably no-more-exploitable than the blueprint, all streets | 📅 |

> **Measurement caveat on the "~24×" figure (2026-05-30).** That number came from
> `scripts/measure_river_exploitability.py` when the measurement tree used the *solver's*
> menu (`{0.5,0.75,1.0,1.5}`), which SNAPPED the blueprint's `{0.33,0.66,1.0,1.5}` bets up
> to the nearest solver size — measuring a distorted blueprint playing sizes it never chose,
> which **overstated the leak removed (flattered the solver)**. FIXED: the harness now builds
> the measurement tree on the **blueprint's own size grid** so the blueprint projects with
> zero snapping and both strategies are scored on the same grid (verified: each blueprint bet
> maps to a distinct exact edge). The ~24× is therefore an **upper bound pending a re-run** on
> the unbiased harness; the relative result (solver ≪ blueprint river-exploitability) holds,
> the exact multiple will move. Re-measure on a current-abstraction blueprint after the retrain.

Approach: when the spot is **structurally off-abstraction** (raise count beyond the
cap, or an off-grid size outside translation's bracket) or is a high-leverage endgame,
and the compute budget allows, solve a **depth-limited** subgame with a **finer action
abstraction (incl. overbets) and real stacks** instead of reading the blended
blueprint. The bot's hole cards already flow through `public_state`, so the solver
drops in via the existing `BotStrategy` interface. Solve quality is bounded by the
input ranges (Phase 3); the safe/gadget layer makes a solve provably
no-more-exploitable than the blueprint. Background: [1], [2].

**Design notes (carry into the build):**

- *The river solver already solves per-decision (no cache).* `RiverSubgameSolver.decide`
  calls `solve_for_action` on **every** river decision — it does not solve once at
  river-entry and replay. So "nested sizing" is **not** about adding re-solving (that
  already happens); it's about *how the opponent's off-grid bet enters the tree*.
  **DONE (2026-05-30):** `_navigate` now tries `RiverTree.inject_realized_edge` (splice the
  exact realized size as a real tree edge, subtree built with the standard menu) before
  falling back to `_match_edge` snapping; so the bot's decision node faces the TRUE pot, not
  a snapped one. Snapping remains the fallback only when the exact size can't be a distinct
  legal edge (really an all-in, below min-raise, or already on-grid). This is a separate
  layer from blueprint pseudo-harmonic translation (`cfr/translation.py`), which is the
  blueprint's off-grid handling — the inject/snap is the *solver's* path reconstruction.
- *Leaf values use multi-valued states, and are NOT gated on training.* A single
  blueprint value at the depth limit is **unsafe**: if the bot deviates inside the
  subgame, the opponent adapts beyond the leaf and the fixed value never sees it. The fix
  (Brown–Sandholm depth-limited solving, used by Modicum/Pluribus) is to let the opponent
  pick among **several blueprint continuation strategies** at the leaf (multi-valued
  states) — Modicum used ~4 on the flop, ~10 preflop, and solved from the flop in ~700
  core-hours. These continuation values are the blueprint's **counterfactual values
  (CFVs)**, a function of the stored average strategy → **recomputable offline from a
  finished blueprint** by a tree traversal. So the next (bucket) retrain does **not** need
  CFV instrumentation; checkpointing CFVs during training is an optional speed
  optimization to add when this item is built, not a now-or-never decision.
- *Lifting the live aggression cap — HUMAN side ✅ DONE (2026-06-03); bot's own deep raises
  still pending the solver.* "Train shallow (capped), play uncapped, solve the deep tail":
  `max_raises_per_street` is now a `PokerGame` constructor param (default **2** = the trained
  cap, kept everywhere in training/eval) and `GameSession` builds its LIVE engine with
  `float('inf')`, so a **human can 5-bet/6-bet+ any amount on any street** (bounded by stack;
  combines with the Phase-1a custom-bet sizing). The blueprint stays capped. The bot's
  responses route as designed: a faced **all-in** 5-bet → the near-terminal equity guard;
  a faced **non-jam** deep raise → a **passive stopgap** (`bot_strategy._distribution` now
  falls back to check/call/fold on an untrained key, never uniform-over-legal — so the uncap
  can't make the bot stray-raise/jam from an untrained node). The bot does **not** yet
  *propose* its own 5-bets+ (it has ~0 average-strategy mass on beyond-cap raises) — that, and
  a principled non-jam-deep-raise response, are the remaining solver work below.
- *All-in-terminal guard (filed 2026-05-30, user idea) — the cheapest, highest-value entry
  point to this row.* **Why it's cheap:** an all-in that gets called has no further decisions
  — the board just runs out — so its value is pure **equity-vs-range to the river**, the same
  near-terminal case the river solver already handles. No betting subtree, **no leaf-value
  function** (that's what makes it independent of the hard depth-limited item below it).
  **Trigger:** in `bot_strategy`/`game_session`, before the bot commits a blueprint-proposed
  `allin` on the **flop or turn** (river already routes through the solver), or when it faces
  one, run a one-shot equity solve and **EV-gate** the override (reuse the existing gate).
  **Why it matters:** it directly removes the stray ~1% undertrained-bucket jams that can dump
  a whole stack — a catastrophic-loss guard worth having before go-live. **Build notes:** (1)
  the flop/turn case needs runout enumeration/sampling to the river (the river solver assumes a
  complete 5-card board), so reuse `showdown_kernel`/`board_winrates` over the remaining
  runouts, weighting by the `RangeTracker` belief; (2) facing-an-all-in is fold-or-call, a
  single equity compare; proposing-an-all-in compares jam-EV vs the blueprint's best non-jam
  action; (3) it does NOT need the aggression-cap lift or CFVs, so it can ship before the rest
  of this row.

---

## Phase 5 — Online 1v1 play on AWS 📅 PLANNED

Deploy for real-time online heads-up play.

- Swap `InMemorySessionStore` for a Redis/DynamoDB-backed store (multi-process).
- Consider a WebSocket transport for live play (the `game/` engine is already
  transport-agnostic).
- ~~**Unrestricted human bet sizing**~~ ✅ done (2026-05-26). The human can bet any
  legal chip amount: `{action:'bet_custom'|'raise_custom', amountBb}`; the engine
  stores the raise-to total in `history`, and off-grid bets are mapped onto the
  trained grid by pseudo-harmonic action translation (`cfr/translation.py`).

---

## Filed for later — variable / arbitrary stack depth (real H2H solving) 📦 DEFERRED

Everything above assumes **fixed 100 BB reset each hand** (Libratus/DeepStack were both
evaluated this way). Supporting non-reset matches (grind 100→0), deep, shallow, or any
combination — so people can solve *real* H2H games — is filed as a post-Step-5 expansion.
**Not** being built now; current focus stays fixed-100 BB. When revisited, it splits into:

- **Engine work (finite, no new theory):** stop resetting stacks each hand; add
  all-in-for-less + uncalled-bet return (HU has no side pots); **lift the equal-stack
  invariant in `river_tree.py`** (it currently raises on unequal river-entry stacks).
- **Depth generalization (the hard part) = make Step-5's leaf-value function depth-aware.**
  Three tiers: (1) real-stack solving + fixed-100 BB blueprint as prior (cheap; great near
  100 BB, degrades far off it); (2) a grid of blueprints by depth (separate DBs, nearest
  selected, solver interpolates); (3) a **DeepStack-style learned value network over
  (ranges, pot, stacks)** — no blueprint, re-solve every decision, any depth natively (the
  gold standard, ≈ building DeepStack). Shallow is easy (small near-terminal trees); deep
  is hard (more streets, the aggression cap + coarse size grid bite, needs a deeper menu).

## References

[1] [Depth-Limited Solving for Imperfect-Information Games](https://dl.acm.org/doi/10.5555/3327757.3327865)
[2] [Safe and Nested Subgame Solving for Imperfect-Information Games](https://proceedings.neurips.cc/paper_files/paper/2017/file/7fe1f8abaad094e0b5cb1b01d712f708-Paper.pdf)
