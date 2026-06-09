# Bug Log

A running record of notable bugs found in the AllIn poker-AI codebase, their
root causes, fixes, and the lessons learned. Kept for future debugging
reference and as a project narrative.

**Entry format** — each bug gets: ID, date, area, severity, status, a one-line
summary, the symptom, root cause, a concrete walkthrough, the fix, why it
wasn't caught earlier, retrain impact, and lessons. Append new bugs at the top.

---

## BUG-023 — Bot over-jams a counterfeited / range-dominated made hand (the proposing-side leak)

| | |
|---|---|
| **Date** | 2026-06-09 |
| **Area** | Card abstraction (postflop strength buckets) + blueprint strategy quality |
| **Severity** | Medium (bot stacks off light on the BETTING side) · **Status** | Open — abstraction/training limitation; **no cheap inference fix** (retrain / Phase-4 proposal-side solver) |

**Summary.** The bot jammed 88 BB on the turn with `2d3h` on `2c 6s Qc Qs` — i.e. a **counterfeited bottom two pair** (`QQ22`: the QQ is on the board, so the bot's real edge is just a pair of 2s) — facing a 3-bet preflop + flop bet + turn bet, and got stacked by `AA`. The blueprint's trained strategy at the key `pf_0_9_ip_turn_m` is **`allin 48%`** — the single most likely action.

**Root cause.** The postflop strength buckets cluster hands by their equity *distribution* vs a **uniform** range. Verified directly:
- `2d3h` on `2c6sQcQs` → turn strength bucket **9**, equity vs a UNIFORM range = **0.54** ("looks like a decent two pair").
- equity vs the villain's ACTUAL range (3-bet + double-barrel = strong/made) = **0.028** (~3%, drawing dead).

The bucket can't see *range-relative* strength, and on a **paired board** a junk pair gets a "two pair" label and a middling bucket, lumped with genuinely strong made hands whose bucket-average strategy (jam) is correct for *them* and badly wrong for this weak member. So the blueprint over-jams it.

**Why the inference guards don't catch it.**
- It's a **trained** node (the key has a real strategy) and the bot is **proposing** a raise — the deep-raise/all-in guards fire only when *facing* a bet at an *untrained* node, and by design **never propose aggression** (the BUG-011 stray-raise rule).
- The range tracker was at **2% confidence** (the human's line was off the blueprint model → confidence collapsed → belief stayed ~uniform), and vs uniform the hand looks fine (0.54), so an equity check wouldn't veto it either. Knowing the hand is dead requires modeling that the *betting* range is strong — i.e. an opponent calling/range model.

**Path (not a cheap heuristic).**
- Retrain with finer / paired-board-aware postflop buckets that separate counterfeited weak two-pair from genuine value hands (so the weak tier's strategy isn't "jam").
- OR the **Phase-4 proposal-side subgame solver**, which computes the bot's jam EV vs the actual range.
- A range-blind veto on the bot's own jams is exactly the BUG-011 stray-aggression risk *and* would nerf legitimate value jams — not safe.

**Lesson.** Equity-vs-uniform buckets measure "how good is this hand vs a *random* hand," not "vs the range that actually arrived." On paired/dynamic boards that gap is enormous (0.54 → 0.03 here) and surfaces as **over-jamming**. The guards added in BUG-021/022 fix the bot *folding* premiums and *calling* off light (the **response** side); the **betting** side (proposing light) is a separate, abstraction-rooted class — the over-jam half of the known "bot plays poorly: overjams + overcalls" issue.

---

## BUG-022 — Bot calls off 100 BB with T8o vs a jam: off-menu all-in keeps confidence at 100% on a uniform belief

| | |
|---|---|
| **Date** | 2026-06-09 |
| **Area** | Live serving — `RangeTracker.observe` + the faced-all-in equity guards (`src/game/range_tracker.py`, `src/subgame/river_subgame_solver.py`) |
| **Severity** | High (bot stacks off with trash) · **Status** | Fixed (uncommitted) |

**Summary.** A user saw the bot **call a 99 BB all-in with T8o** (over its own 1.5 BB open). The
debug showed "Read confidence: **100%**" on a junk **uniform** range. Chain:
1. A 100 BB jam over a min-open is an **off-menu** action — at that depth `allin` isn't a menu
   3-bet, so the custom raise-to-stack normalizes to `allin`, which isn't in the abstract legal set.
2. `RangeTracker.observe` saw an action **not in `legal`** and **returned early** — so it never
   updated the range *and never decayed confidence*. The bot held a **uniform belief at 100%
   confidence** (zero read, maximal trust).
3. `_facing_allin_guard` fired (confidence 100% ≥ 0.2) and computed T8o equity **vs uniform ≈ 0.53**
   > pot odds ≈ 0.49 → **call**. (No blueprint to fall back on either: the `capped` blueprint
   (`voluntary_allin=False`) never trained *any* facing-a-jam node — 0 keys contain an all-in.)

The math was right given the belief, but the belief ("opponent jams any-two") is absurd: nobody
jams 100 BB over a min-open that wide. **Equity-vs-uniform massively over-estimates because real
all-in ranges are selected/strong.**

### Fix (inference-only, no retrain), two parts
- **Part 1 — `RangeTracker.observe`: an off-menu action now COLLAPSES confidence** (`*= 0.1`, below
  the guards' 0.2 trust threshold). An action the model can't represent is maximally off-model;
  leaving confidence untouched was the root inconsistency. Keeps the prior range (can't reweight on
  an unrepresentable action), but stops *trusting* it → the all-in guard defers.
- **Part 2 — `_jam_range_equity`: a faced all-in with an uninformed (collapsed) read is judged vs a
  top-20% JAM range, not uniform** (`river_subgame_solver._facing_deep_raise_guard`, preflop only —
  the preflop bucket is equity-ordered; postflop EMD strength buckets aren't, left on the uniform
  floor as a separate fix). Chosen over a flat equity cushion after measuring that vs-uniform and
  vs-a-strong-range *rank hands differently*: the cushion would still stack off dominated hands
  (KQo/A5s/55) that look fine vs random; the top-X% range folds them. Premiums always still call.

### Measured (tests/run_maniac_live.py `--style shove`, off-menu jams over opens)
1500 hands: the deep guard handled **1407** faced all-ins via the jam-range floor; **loose call-offs
(stack committed with a sub-top-40% hand) = 0** (the T8o class); `allin_guard` fired 0× (confidence
correctly collapsed); no crashes. Tradeoff: folds 77 to a 100 BB shove (conservative side of X=20%;
tunable).

**Why not caught.** The maniac harness's `jam` style shoves only via the *in-menu* `allin` (when
already near-committed), which `observe` *does* process → confidence decays → guard defers. A real
human jams *off-menu* over a small open, the path the harness never exercised. And the harness only
measured bad *folds*, never bad *calls*. Added: a `shove` style (off-menu jams) and a loose-call-off
metric.

**Lesson.** A uniform belief means *no information* — it must never be held at high confidence. An
off-model action the abstraction can't even represent is the strongest possible "I don't understand
this opponent" signal; treat it as such (collapse confidence), and when committing the whole stack
with no read, assume a *realistic* (selected) opponent range, not a uniform one.

### Hardening + gap-closure (2026-06-09)
Three follow-ups completed the fix and closed the postflop gap (all in `river_subgame_solver.py`):
- **Postflop turn gap.** A faced all-in on the TURN with an uninformed read now uses the same
  top-20% jam range, ranked by each EMD strength bucket's **centroid-MEAN equity** (`_postflop_means`
  / `_combo_strength`). (Earlier I wrongly thought the EMD buckets were unusable for ranking — they
  are: the centroid means are equity-ordered *and* draw-aware, since each centroid is an equity-vs-
  runout histogram. The bucket *index* isn't guaranteed ordered after a re-bake, so ranking is by the
  derived mean.) River stays solver-owned; flop rarely all-in.
- **B1.** An uninformed *money-behind* preflop 5-bet+ (not just an all-in) is also beyond-cap and faces
  a selected range, so it uses the jam range too.
- **Informativeness gate.** `_trust_read` = confident AND the belief actually concentrated off uniform
  (inverse-Simpson `eff_n/n_live < INFORMATIVE_RATIO`). Belt-and-suspenders for a uniform-belief-at-
  high-confidence edge. NOTE the threshold lesson: it was first set at 0.95, which wrongly discarded a
  genuine *mild* read (a 1.5x tilt is ratio ~0.97) and reverted to un-adapted blueprint play at trained
  all-in nodes — a regression a review agent caught. Raised to **0.99** (reject only ~uniform). The
  gate is largely redundant with the off-menu confidence collapse, so it rarely fires.

**A/B validation (tests/run_maniac_live.py, served bot through uncapped GameSession, seed 42, current
vs `--cripple2` = the pre-fix faced-all-in behavior):**

| Opponent | Current (all fixes) | `cripple2` (old) |
|---|---|---|
| `shove` (any-two jam, 10k) | +1.56 BB/h · 0 loose call-offs · 71 prem-folds | +0.31 · **2913** · 326 |
| `widejam` (top-50% jam, 10k) | **+1.05** BB/h · 0 · 34 | **−3.80** · 1333 · 146 |
| `maxbet` (max non-all-in presser, 6k) | +4.32 BB/h · 0 · 0 | (headline) |

The fix turns a **−3.80 loss into a +1.05 win** vs a realistic jammer and zeroes stack-off-light; it
even beats the old behavior vs an any-two shover (not-folding-premiums dwarfs the marginal over-fold).

**MEASUREMENT NOTE (important).** LBR / best-response (`tests/run_evaluation.py`) do NOT measure any of
these guards: they evaluate the *blueprint* on the *capped* (3-aggression) public tree, while the guards
live only in the served `RiverSubgameSolver`/`GameSession` path and fire on uncapped/off-tree/uninformed
nodes the eval tree can't reach. The blueprint DB is unchanged, so a before/after BR run returns an
identical number (hours wasted). The live `run_maniac_live.py` harness (`--style shove|jam|maxbet|
widejam`, `--cripple2`) is the correct tool for inference-guard changes.

---

## BUG-021 — Bot folds premiums (incl. AA) at beyond-cap live nodes: the passive coin-flip fallback

| | |
|---|---|
| **Date** | 2026-06-09 |
| **Area** | Live serving — `RiverSubgameSolver`/`BlueprintStrategy` fallback (`src/subgame/river_subgame_solver.py`, `src/game/bot_strategy.py`) |
| **Severity** | High (bot folds the nuts in live play) · **Status** | Fixed (uncommitted) |

**Summary.** A user saw the bot fold AA to a 5-bet. Root cause: training caps aggression at 3/street
(`max_raises_per_street=2`), but LIVE `GameSession` uncaps re-raises, so a human 5-bet/6-bet reaches
info-set keys (e.g. `pf_29_ip_slll`) that are **never in the blueprint**. `BlueprintStrategy._distribution`
then returns its PASSIVE fallback — uniform over `{check,call,fold}` — which facing a bet is 50/50
call/fold, hand-strength-blind. So the bot folds the strongest possible hand half the time.

### Walkthrough
Verified against `blueprint_final.db`: `pf_29_ip_sl` exists (the 4-bet node, 1825 visits) but
`pf_29_ip_sll` and `pf_29_ip_slll` are **MISSING** — one aggression past the 3-cap. The near-terminal
all-in guard (`_facing_allin_guard`) didn't save it: it only fires when `to_call >= bot_stack`, and a
5-bet-to-90 leaves ~8 BB behind, so it deferred → blueprint fallback → 50/50 → fold.

### Fix (inference-only, no retrain)
`_facing_deep_raise_guard` in `river_subgame_solver.py`, run in `decide()` after the all-in guard: at an
UNTRAINED key facing any bet/raise/**jam**, decide call/fold by equity vs the opponent range — tracked
range when `confidence>=guard_confidence`, else a UNIFORM-range floor (premiums dominate every range, so
they never fold). Never raises (no stray untrained aggression). Plus: a cached uniform floor (latency),
a safe-call-on-exception path, and an AA/KK-never-fold-preflop floor (`_premium_no_fold`) for the residual
CFR-noise fold at trained nodes.

### Measured (tests/run_maniac_live.py — drives the REAL served bot through live uncapped GameSession)
| Opponent (8–20k hands, seed 42) | Bot BB/hand | premium folds | beyond-cap folds |
|---|---|---|---|
| maxbet presser, pre-fix | +1.70 | 375 (1.88%) | **300** |
| maxbet presser, post-fix | **+4.11** | 76 (0.38%) | **0** |
| jam presser, #2 (untrained-jam) OFF | +0.55 | 56 | **27** (folds AKs/AJs to 5-bet shoves) |
| jam presser, #2 ON | +1.17 | 30 | **0** (guard fired 76× on jams) |

The remaining post-fix premium folds are all at *trained* keys and mostly legitimate (JJ/AQ/77 fold to
deep aggression). `#4` translation-overflow-fold and `#5` first-to-act-value were investigated with an
off-grid `overbet` maniac and found already-closed (0 leak folds in 1500 hands). `C3` (skip the river
EV-gate at untrained keys, where the baseline is this same garbage) shipped alongside.

**Why not caught.** The "uncapped live re-raises" feature (a human can 5-bet+) and the passive fallback
were each correct in isolation; their interaction (beyond-cap keys → fallback → fold the nuts) only shows
up in live play past the trained tree, which no test or eval exercised — the capped match/LBR engines
can't even reach the node. The new `run_maniac_live.py` harness (uncapped, drives the served bot) is the
gap-filler.

**Lesson.** A "passive, never-propose-aggression" fallback is safe for *bets* but catastrophic for *folds*
(folding the nuts). When you cap something in training but uncap it in serving, the out-of-distribution
serving nodes need an explicit policy (here: equity vs range), not a uniform default. And measure the
SERVED bot through the SERVING path — the eval harnesses ran the capped tree and were blind to this.

---

## BUG-020 — Explorer endpoints 500'd on a non-list `actions`/`history` payload instead of 400

| | |
|---|---|
| **Date** | 2026-06-09 |
| **Area** | Strategy explorer API (`/api/strategy/from-hand`, `/api/strategy/river-solve`) |
| **Severity** | Low · **Status** | Fixed |

A client sending `actions` (or a `history` street) as a non-list (e.g. a string) made the
pattern builders iterate it and call `.get(...)` on a character, raising and surfacing as a 500.
Fix: `isinstance(..., list)` guards return a clean 400. Lesson: validate request-field *types*,
not just presence.

---

## BUG-019 — Hand Explorer off-grid postflop translation ignores the served menu mode → shows a different line than the live bot

| | |
|---|---|
| **Date** | 2026-06-08 |
| **Area** | Strategy explorer API (`backend/api/strategy_api.py:196-197`) |
| **Severity** | High (silent — explorer reports a *wrong* strategy vs what the bot actually plays, for the capped blueprint that is being served) |
| **Status** | Open — fix pending |

**Summary.** `_postflop_pattern` translates an off-grid human bet against the bare
`translation.POSTFLOP_GRID` (tops out at `'o'` = 1.5× pot), hardcoded — it ignores `BLUEPRINT_MENU_MODE`.
The module already exposes `translation.postflop_grid_for(menu_mode)` (→ `POSTFLOP_GRID_CAPPED` with the
`'2'` = 2.0× tier), and the *game* endpoints use the menu-aware path (`menu_mode=BLUEPRINT_MENU_MODE` at
lines 423/451) — but the explorer's from-hand path does not.

### Walkthrough
`BLUEPRINT_MENU_MODE = db_menu_mode(BLUEPRINT_DB)` resolves to `'capped'` for the served 25M snapshot
(memory: capped-blueprint-ship-25m). A Hand-Explorer query with a postflop bet between 1.5× and 2.0× pot
is mapped to `'o'` only — it never blends toward / queries the trained `'2'` keys. So the explorer reports
a different blend than the live bot (whose translation *does* use the `'2'` tier). The explorer's entire
purpose is "show what the bot does," so a silent divergence here defeats the feature.

### Proposed fix
`grid = translation.postflop_grid_for(BLUEPRINT_MENU_MODE)` once, pass it to both `translate_bet` and
`nearest_char`. Inference/UI only — no retrain.

**Why not caught.** The capped menu and `postflop_grid_for` helper were added for *serving*; the explorer's
own pattern builder predates the menu split and was never re-pointed at the menu-aware grid. No test
asserts explorer-translation == live-bot-translation.

**Lesson.** When a "single source of truth" helper is introduced (`postflop_grid_for`), grep for every
hardcoded use of the thing it replaced (`POSTFLOP_GRID`) — one stale call site reintroduces the drift the
helper was meant to kill. Same failure class as the key-format drift fixed in commit 9a85056.

---

## BUG-018 — Key Explorer offered street-illegal pattern chars; the paste path bypassed the per-street guards

| | |
|---|---|
| **Date** | 2026-06-08 |
| **Area** | Key Explorer (`frontend/src/components/KeyExplorer.jsx`) |
| **Severity** | Low · **Status** | Fixed |

The pattern-char buttons offered every char regardless of street (`o`/`2` are postflop-only,
`x` is a preflop open), so a user could build a structurally impossible key shown as a
misleading `found:false`; a pasted/typed key bypassed the per-street bucket clamp + char
legality entirely. Fix: per-street `PREFLOP_CHARS`/`POSTFLOP_CHARS` filtering + a
`canonicalize()` step on the paste/lookup path. Lesson: an editable mirror of a guarded
builder must re-apply the same guards.

---

## BUG-017 — Key Explorer leaks a fine preflop bucket into postflop keys on street switch

| | |
|---|---|
| **Date** | 2026-06-08 |
| **Area** | Strategy explorer frontend (`frontend/src/components/KeyExplorer.jsx:70-83`) |
| **Severity** | Med (silent — builds an invalid postflop key → always `found:false`, plus a desynced dropdown) |
| **Status** | Open — fix pending |

**Summary.** `sync()` re-clamps the postflop `strength` bucket on a street change but **not** `bucket`
(the start/preflop bucket). The bucket dropdown offers the **fine** list (pf_0..pf_29) preflop and the
**coarse** list (pf_0..pf_9) postflop, but `bucket` state is shared across streets. So: pick `pf_27`
preflop → switch Street to flop → `composeKey` emits `pf_27_<strength>_ip_flop_` — a postflop key carrying
a **fine** id the coarse-keyed blueprint never wrote. The `<select value="pf_27">` also renders blank
(value outside the coarse `<option>` list), desyncing the visible dropdown from the composed key.

### Walkthrough
Postflop keys must carry the coarse class (pf_0..pf_9) — the fine→coarse collapse happens server-side in
`keys.py:make_info_set_key`, but the Key Explorer builds the key string client-side, so it must enforce the
same legality. It clamps `strength` (line 77) but forgot `bucket`.

### Proposed fix
In `sync`, on a preflop↔postflop transition clamp/reset `bucket` into the target street's list (reset to a
valid coarse default, or map fine→coarse), mirroring the existing `strength` clamp. Frontend only — no
retrain.

**Why not caught.** The `strength` clamp made it *look* handled; the bucket list silently swaps fine↔coarse
between streets and nobody re-validated the carried value. No test composes a key across a street switch.

**Lesson.** When two dropdowns swap their option sets based on a third control (street), every dependent
value must be re-validated on that control's change — clamping one and not its sibling is the classic
half-fix. Flagged independently by all 3 review agents (backend, frontend, and the fix-audit pass).

---

## BUG-016 — Sub-stack custom bet mis-snapped to all-in (translation), corrupting the next key

| | |
|---|---|
| **Date** | 2026-06-04 |
| **Area** | Serving / action translation (`game/game_session.py`) |
| **Severity** | Med (silent — no crash; mis-records the pattern char → wrong next-decision key + range update) |
| **Status** | Fixed (inference-only, no retrain) |

**Summary.** `_node_grid` put an `'a'` (all-in) edge into the nearest-char snapping grid whenever `allin`
was legal. A **sub-stack** custom bet (money still behind) whose pot-fraction landed closest to that all-in
edge got `char='a'` / `snapped_action='allin'` — so `bet_pattern` recorded a shove that didn't happen and
the range tracker observed `'allin'`, feeding a wrong info-set key into the bot's next decision. Worst near
the top of the menu (when stack ≈ 2.0× pot, the `'2'` and `'a'` grid edges nearly coincide); broad in the
control arm (`allin` always legal), narrow in capped (`allin` legal only after a clamp). Found by the
BUG-015 follow-up audit.

### Fix
`_node_grid(legal, include_allin=False)` for the sub-stack snap path (an at/above-stack custom is already
normalized to `'allin'` by `_validate_custom`), so a sub-stack near-shove snaps to the largest **sized**
char, not `'a'`. Falls back to the full grid only if no sized edge exists. Tests: custom-betting +
game-session + range-tracker suites (34 pass).

### Residual (deferred, low-consequence)
A custom/sized bet leaving only a tiny **stub** behind (e.g. 2.0× when stack ≈ 2.0× pot → ~3 BB behind)
still records as the sized tier rather than all-in. The principled fix is a single "near-all-in" stub
threshold used in BOTH the engine clamp (`_apply_stack_constraints`, currently exact `cost >= stack`) and
this translation snap — an abstraction change to bundle with the next retrain. Magnitude is small (rare
SPR≈2 spot, ~3 BB stub), so it's left for now.

---

## BUG-015 — Range tracker 500'd on an emergent/custom all-in not in the abstract legal menu

| | |
|---|---|
| **Date** | 2026-06-04 |
| **Area** | Serving / range tracker (`game/range_tracker.py`, `evaluation/lbr.py`) |
| **Severity** | High (500s a live hand mid-pot) |
| **Status** | Fixed |

**Summary.** `RangeTracker.observe` did `ai = legal.index(action)`. Playing the capped bot, the river
solver jammed via a **custom raise-to-stack**, which `GameSession.apply_action` normalizes to `'allin'`
(its char is `'a'`). But at a deep-stack river node under the capped menu (`voluntary_allin=False`),
`'allin'` enters the engine's `legal` list **only via stack-clamp** — which didn't fire (stacks deep,
pot small) — so `legal` held only sized raises. `observe` then did `legal.index('allin')` →
`ValueError: 'allin' is not in list` → `POST /api/game/bot-action` 500.

### Root cause
The action vocabulary and the abstract `legal` list are inconsistent for a *voluntary* all-in in capped
mode: a player (bot solver or human) can make an all-in via a custom raise-to-stack even when the node's
abstract menu has no `'allin'` edge. `observe` assumed the action was always a member of `legal`. A
capped-era latent bug (the capped menu makes `'allin'` absent from `legal` far more often), surfaced by
playing the fresh capped blueprint.

### Fix
`observe` now no-ops when `action not in legal` (keep the prior range + confidence rather than crash) —
an off-menu action has no opponent-model column to condition on. Applied to **both** `RangeTracker.observe`
and `lbr.py:BotRange.observe` to preserve the BUG-008 live↔victim lockstep. Tests:
`test_range_tracker.py:test_offmenu_action_does_not_crash` (+ lbr-range/game-session suites, 30 pass).
Note: the bot's action wasn't committed (observe raised before `history.append`), so a crashed session is
recoverable — restart the API and continue/deal next.

### Lessons
Any `legal.index(action)` on a served path must tolerate an action outside the abstract menu — custom and
emergent all-ins routinely fall outside it under the capped abstraction.

---

## BUG-014 — Fix #2 (parallel raw-regret merge) broke CFR+ re-activation → blueprints collapse to "open xlarge with every hand"

| | |
|---|---|
| **Date** | 2026-06-04 (introduced 2026-05-31 as "Fix #2", `a55d78c`) |
| **Area** | Parallel training (`cfr/blueprint_trainer.py` worker regret store, `cfr/parallel_trainer.py`) |
| **Severity** | High (every blueprint trained after the change is strategically degenerate) |
| **Status** | Fixed (Fix #2 reverted to per-worker CFR+ flooring) |

**Summary.** "Fix #2" had parallel workers store **raw, unfloored** cumulative regret (master floors once
per merge round) to remove a small upward jam bias from per-worker flooring. It backfired: raw storage
**breaks CFR+ re-activation** — an action driven negative can't pop back the instant it earns one positive
regret, so within a chunk it stays suppressed and the strategy **collapses onto whatever action dominated
early**. Every blueprint trained after `a55d78c` opens `bet_xlarge` (5 BB) with ~80-99% of *every* preflop
bucket and **never folds the button** (pf_0 trash fold ~0-1%), getting worse with iterations.

### Symptom
User played the capped retrain and saw the bot open 5 BB with literally every hand. The strategy explorer
confirmed `pf_4_ip_` → 97% `bet_xlarge`, and the pattern held across all 30 buckets and all snapshots
(5M/14M/21M), worsening over time — i.e. not undertraining.

### Root cause
CFR+ floors cumulative regret at 0 on every write, so a dominated action re-enters the strategy the instant
it earns one positive regret (fast re-activation — the reason CFR+ mixes). Fix #2 removed the **per-worker**
write-floor; the master only floors the **summed** per-round delta. Once a dominated action's merged regret
hits 0 (fold, small opens), workers keep reporting net-negative chunk deltas, and `max(0, decay·0 +
negative) = 0` pins it there permanently. Early in training, before the BB learns to defend, opening the
biggest size is locally best — so the strategy locks onto xlarge and fold/small-open can never climb back.

### Fix
Revert to **per-worker CFR+ flooring**: `cfr()` floors `max(0.0, prior+regret)` on every write in worker
mode too (removed the `discount_enabled` gate). `merge_round` is unchanged (sums signed increments + master
floor) — it now operates on floored cumulatives, the canonical data-parallel CFR+ path that trained the sane
`blueprint_par_20260529_233056`. Single-thread is byte-identical (it always floored). The +0.0014 jam bias
Fix #2 chased is negligible and covered structurally by Fix #4 (capped menu drops the voluntary all-in node).

### Proof (decisive A/B — `scripts/ab_fix2_revert.py`, 500k iters, seed 1, single variable)
`RAW (Fix #2)`: pf_0 xlarge 32% / large 64% / **fold 1%** (collapsed).
`FLOORED (revert)`: pf_0 xlarge 5% / **fold 74%** (sane, matches `233056`'s 79% at 504k).

### Why it took a week to catch
Every aggregate metric is blind to a *balanced-but-degenerate* strategy: EV(cum/round) measures the current
iterate; EV(served) checks only seat balance (a collapsed strategy is balanced → ~0); LBR at 5M looked
*good* (+2204) because it probes postflop off-tree sizing and is blind to a balanced wide preflop open,
while the Fix-#4 postflop win dominated the scalar; the **BR pilot actually flagged it** (8-sample, ~3×
normal) but was dismissed as noise and the full BR skipped for cost; AIVAT is head-to-head variance
reduction, not a pathology detector. Only a human inspecting the actual opens caught it.

### Retrain impact
**All blueprints trained after `a55d78c` are corrupt and cannot be resumed** — the collapsed regrets are
baked in. Retrain from scratch on the reverted trainer. Serve `233056` (pre-fix) meanwhile.

### Lessons
(1) A "fix" validated on one narrow metric (jam-action keys) can introduce a worse regression elsewhere
(opens) — validate the *strategy shape*, not just the target symptom. (2) No scalar aggregate (EV, LBR,
AIVAT) catches strategy collapse; add a cheap per-info-set sanity probe (fold%, open-size histogram) to
convergence tracking. (3) Take BR seriously even when expensive — its pilot was right.

### Process fix shipped (2026-06-04)
The lesson-(2) sanity probe was built: **`src/cfr/strategy_shape.py`** prints a `shape: OK|WARN|COLLAPSE`
line at **every training checkpoint** (parallel + single-thread, next to `EV(served)`) and is also a
standalone CLI — `python scripts/check_strategy_shape.py [--db <path>] [--verbose]` (exit 2 on COLLAPSE).
It detects the collapse fingerprint on BOTH preflop nodes (open `pf_N_ip_` and BB-vs-5BB `pf_N_oop_x`):
weak hands fold <5% while one size >75%, or a dead pf_0-vs-strongest fold gradient. Validated: COLLAPSE on
the archived broken DB, OK on the fresh fixed run. The forensic audit of the old DB also surfaced a SECOND
collapse node (BB-vs-5BB-open → raise_large ~99%, masked by the open-collapse); both are cleared in the
new run (`blueprint_par_capped_20260604_114512`: pf_0 open fold 78% / BB-vs-5BB fold 82% by ~1.2M).

### Tests
`test_worker_mode_stores_raw_regret` → rewritten as `test_worker_mode_floors_regret` (asserts workers
floor). Removed `test_floor_bias_direction_old_inflates_high_variance` + `_merge_old_floored` (existed only
to justify Fix #2; now backwards). Added `tests/test_strategy_shape.py` (3 — flags collapse, passes
healthy, no false-alarm on empty). 15/15 `test_parallel_trainer`, 3/3 `test_strategy_shape` pass.

---

## BUG-013 — EV gauge measured the non-converging CURRENT iterate, not the served strategy (a day-long false alarm)

| | |
|---|---|
| **Date** | 2026-06-02 / 03 |
| **Area** | Observability / training telemetry (`cfr/blueprint_trainer.py`, `cfr/parallel_trainer.py`) |
| **Severity** | Medium (no wrong *output* — a misleading *metric* that cost ~a day chasing a non-bug) |
| **Status** | Fixed (added `EV(served)` gauge) |

**Summary.** The trainer's `EV(cum)` / `EV(round)` / `EV(sess)` print the sampled root value of the
**current regret-matched strategy** (`get_strategy`, blueprint_trainer.py:263). The capped blueprint's
current iterate sits at a large, seat-imbalanced value (P0 +46 / P1 −34 → gauge ≈ +6) and **does not
converge** — by CFR theory only the *average* iterate (the served blueprint, `get_average_strategy`) is
guaranteed to. So the gauge read ~+7 "forever" and *looked* like a broken/diverging blueprint, when the
**served** strategy was actually fine (~0 self-play value, same as control).

### Symptom
User watched `EV(round)` climb to +6/+8 and stay there across millions of iters ("why the fuck is the EV
so high compared to last time?"). Past (control) runs had settled near +1, so capped looked broken.

### Root cause
Two compounding facts: (1) the gauge measures the current iterate, which CFR doesn't converge — it can
cycle indefinitely; (2) the capped abstraction's current iterate is *more* lopsided than control's (its
two traverser-halves don't cancel: control +47.2/−46.4 → ~0; capped +40.1/−30.7 → +4.7). At matched iters
control's gauge reads +0.37 while capped's reads +4.68 — **13× apart**, purely from the abstraction, not
iteration count (the first wrong hypothesis offered was "you're comparing 5M to 33M").

### Fix
`BlueprintTrainer.evaluate_served_ev(n, seed)` + `_rollout_avg` — self-play sampling the **average**
strategy (RNG-isolated, deterministically seeded for a paired trend). Printed as `EV(served, avg
strategy)` at every checkpoint in both the single-thread and parallel loops. Relabeled `EV(cum)` →
`EV(cum,lagged)` and added `EV(round)`/`EV(round,ema)` earlier in the arc. `scripts/probe_seat_ev.py`
contrasts current vs served per snapshot (shares `evaluate_served_ev`, no drift).

### Proof it was a false alarm
`probe_seat_ev.py`: served EV ≈ 0 across capped 4M/7M/9.76M (−2.6/+0.8/−0.6) vs control's +0.37 — same
ballpark. And LBR is **anti-correlated** with the gauge: capped-5M (scary +4.68 gauge) is 1,100 mbb/hand
*less* exploitable than control-4M (nice +0.37 gauge). Choosing a bot by the gauge picks the worse one.

### Lessons
A convergence metric must measure the iterate you actually *serve* (the average), not the current one.
The old commit message that claimed to "fix the EV gauge" had only *relabeled* it — a relabel is not a
fix; this entry is the real fix. For strength always use LBR/BR (best-responds), never any EV self-play
gauge (seat-balanced-but-bad strategies self-play near the game value too).

---

## BUG-012 — Parallel trainer Ctrl+C hung forever and orphaned workers (Windows)

| | |
|---|---|
| **Date** | 2026-06-02 |
| **Area** | Training orchestration (`cfr/parallel_trainer.py`) |
| **Severity** | Medium (no data corruption, but Ctrl+C did nothing for minutes and left worker processes hogging cores, slowing every subsequent diagnostic) |
| **Status** | Fixed (3-part) |

**Summary.** On Windows, Ctrl+C on a parallel training run never returned control and left
`SpawnPoolWorker` processes alive. Three causes, all needed fixing: (1) Ctrl+C is delivered to the whole
process group → every worker raised `KeyboardInterrupt` mid-`cfr()` **and the Pool's maintenance thread
respawned them** (an endless loop); (2) `pool.map()` blocks uninterruptibly on Windows, so the master
never surfaced the interrupt; (3) the master needed to `terminate()` (not `close()`) the pool.

### Fix
(a) `_worker_init()` sets `signal.SIGINT = SIG_IGN` so **workers ignore Ctrl+C** (only the master gets it;
no respawn loop); pool created with `initializer=_worker_init`, master SIGINT restored right after. (b)
`pool.map` → `map_async(...).get(timeout=1.0)` polling loop (swallow `TimeoutError`) so the main thread
stays interruptible. (c) `except KeyboardInterrupt → pool.terminate(); join()`. Verified: ~0.2s interrupt,
zero orphans; `test_parallel_trainer` green. Work since the last checkpoint is discarded (correct
interrupt semantics — resume picks up from the checkpointed iteration).

### Lessons
Pool worker signal handling + a non-blocking `get(timeout=)` are both mandatory for an interruptible
Windows `multiprocessing.Pool`; either alone leaves the hang.

---

## BUG-011 — Capped blueprint served through a control-menu engine → uniform-random all-in

| | |
|---|---|
| **Date** | 2026-06-02 |
| **Area** | Serving / inference (`game/game_session.py`, `api/strategy_api.py`) |
| **Severity** | High (visible in live play — the bot made an indefensible move) |
| **Status** | Fixed |

**Summary.** A `capped`-menu blueprint (Fix #4: voluntary all-in dropped, 2.0× tier added)
was served through a `GameSession` that hard-coded `PokerGame()` — the **control** engine
with `voluntary_allin=True`. So the live engine offered the bot an `allin` action at a
high-SPR node that the capped blueprint **never trained**; the blueprint lookup fell back to
**uniform over legal actions**, `allin` drew a random slice, and the bot shoved ~95 BB into a
checked-down turn. An **inference** bug, not a training bug — same blueprint file, no retrain
needed to fix.

### Symptom
User playing the served bot: "how the fuck did the bot just all-in me on the turn??" A
deep-stacked, high-SPR turn jam with no strategic basis. The bot-debug overlay showed the
blueprint strategy at that key as a flat-ish spread including `allin` — the fingerprint of a
uniform fallback, not a trained decision.

### Root cause
The `menu_mode` toggle (control vs capped) was threaded into the **training** path and the
**eval** path (BR/LBR) but the **serving** path was deferred — and `GameSession` was the
serving consumer that actually mattered. With `PokerGame()` defaulting to control:
1. The engine's legal-action set at a high-SPR node included the voluntary `allin`.
2. The capped blueprint has **zero** mass there (it was trained with that action removed —
   verified earlier: 0% of high-SPR capped keys contain `allin`).
3. `BlueprintStrategy._distribution` returns **uniform over legal actions** for an action the
   stored strategy doesn't cover → `allin` got ~1/n probability → sampled → stray jam.

### The fix
- `GameSession.__init__` / `.new` / `.from_dict` gained `menu_mode`; it builds the capped
  `PokerGame` (`postflop_menu=POSTFLOP_BET_MULT_CAPPED, voluntary_allin=False`) when serving a
  capped (or `capped_no2`) blueprint, via the shared `sizing.is_capped_mode` predicate.
- `strategy_api` derives `BLUEPRINT_MENU_MODE = db_menu_mode(BLUEPRINT_DB)` once at startup and
  passes it to both `GameSession` call sites — so the served engine can never disagree with the
  served blueprint (auto-derived from the DB stamp, not a hand-set flag).
- Verified: capped snapshot → `GameSession.game.voluntary_allin=False`, the high-SPR turn menu
  has no `allin`; control default byte-identical; `test_game_session` 8/8; capped `from_dict`
  round-trip intact. (The server must be restarted to load it.)

### Why it wasn't caught earlier
The serving consumers were explicitly deferred ("not needed for the trainer/BR/LBR gate"), and
the menu mismatch only manifests when a *capped* blueprint is *served* — which first happened
when the user pinned a capped snapshot via `ALLIN_BLUEPRINT_DB` to play it. No test served a
capped blueprint through `GameSession`. (Same class as BUG-008: a consumer's action model
drifting from the deployed/served abstraction.)

### Retrain impact
None — inference-only. The capped blueprint was correct on disk; only the serving engine was
misconfigured.

### Lessons
A per-artifact abstraction flag (`menu_mode`) must be **auto-derived from the artifact** at
*every* boundary that consumes it — training, eval, AND serving — not hand-set. The moment one
consumer defaults the flag instead of reading it, you get a silent menu mismatch that surfaces
as "the model did something insane." The `db_menu_mode(db)` helper is the anti-drift mechanism;
the bug was a consumer that hadn't been wired to it yet. When you defer threading a flag through
"non-critical" consumers, the serving path is not non-critical — it's the one the user sees.

---

## BUG-010 — Validation harness gated on `visit_count`, comparing zero keys (a near-miss false-pass)

| | |
|---|---|
| **Date** | 2026-06-01 |
| **Area** | Tooling (`scripts/validate_parallel_merge.py`) |
| **Severity** | Medium (measurement tool; nearly produced a false "all clear") |
| **Status** | Fixed |

**Summary.** The harness written to *validate* BUG-009's fix filtered the info-sets it compares
on `visit_count >= min_visits`. But `visit_count` counts **iterations** in single-thread and
**merge-rounds** in parallel — and a 200k-iter parallel run has only ~7 rounds — so every
parallel key failed the threshold and the harness compared **0 keys**, printing "No shared
high-traffic info-sets" after a 44-minute run.

### Symptom
A full-length run that produced a real, correct speedup number (3.93×) but **no** jam-bias
comparison — the one number the run existed to produce. Crucially, it failed *loud* ("No shared
info-sets") rather than silently emitting a spurious 0.0 bias; a slightly different gate could
have reported a misleading "looks fine."

### Root cause
`visit_count` is an overloaded clock (documented in `blueprint_trainer.resume_from_db`'s cross-mode
guard, but the implication for *cross-mode comparison* was missed). Single-thread bumps it once per
iteration; the parallel master bumps it once per merge round (`merge_round`). They are not
comparable magnitudes, so any fixed visit threshold that's meaningful for one mode excludes the
other entirely.

### The fix
Gate on **cumulative-strategy mass** (`sum(info.cumulative_strategy.values()) >= --min-mass`),
which accumulates comparably in both modes — the same noise filter `scripts/test_merge_bias.py`
already used. Re-run compared 22,092 shared keys and produced the bias number (+0.0014).

### Why it wasn't caught earlier
The harness was smoke-tested only at tiny N where, by luck, single-thread `visit_count` was also
small — the gate happened to pass a few keys, hiding the cross-mode incompatibility that only
appears once `merge_every*workers` makes rounds ≪ iterations.

### Retrain impact
None (tooling only).

### Lessons
A validation harness needs its own sanity check: "did it actually compare a non-zero, plausible
number of items?" An overloaded field (`visit_count` = iterations *or* rounds) is a cross-mode
trap — when comparing two modes, filter on a quantity defined identically in both.

---

## BUG-009 — Parallel CFR+ worker-local flooring biased the blueprint toward high-variance actions

| | |
|---|---|
| **Date** | 2026-06-01 |
| **Area** | Training (`cfr/parallel_trainer.py`, `cfr/blueprint_trainer.py`) |
| **Severity** | High (corrupted the served blueprint — the bot over-jammed) |
| **Status** | Fixed (validated at scale) |

**Summary.** In data-parallel MCCFR+, each worker ran the full single-thread `cfr()`, which floors
cumulative regret at 0 on **every write** (canonical CFR+). Inside a worker that floor clamped a
losing action's negative regret to 0 *before the master ever saw it*, so a negative regret from one
worker could never **cancel** a positive regret from another worker within the same merge round.
The result was a bounded **upward bias on high-variance actions** (all-in / jam) that grew with
`workers × merge_every` — a *bias*, not a convergence gap, so more iterations never fixed it.

### Symptom
The served (parallel-trained) bot **over-jammed** — preflop and low-SPR flop — and the behavior
didn't improve with +10M more iterations. An A/B over `merge_every` (the diagnosis's TEST 1)
measured the large-round arm jamming ~1.22× aggregate, up to 5.5× at marginal keys
(e.g. `pf_28_oop_cmm` 0.563 vs 0.160).

### Walkthrough (why one-sided flooring inflates)
A high-variance action gets a big win in one worker (+30) and a big loss in another (−30) the same
round; true net 0.
- **Buggy path:** worker 2 floors its −30 to 0 locally → master sums +30 and 0 → +30. The action
  keeps large positive regret → over-played.
- **Correct path:** workers report raw +30 and −30 → master sums to 0, then applies its single
  per-round floor → 0. Cancellation preserved.

### Root cause
The CFR+ write-floor belongs **once**, at the boundary where contributions are summed — not inside
every worker. `discount_enabled=False` (worker mode) already meant "skip the per-iteration discount;
the master applies a block discount"; the floor should have been bundled into that same split and
wasn't.

### The fix
In `cfr()`'s traverser regret update, apply the `max(0.0, prior+regret)` write-floor **only when
`discount_enabled`** (single-thread); in worker mode store the **raw signed sum**. The master's
`merge_round` keeps its existing per-round `max(0.0, decay·base + increment)` floor, so raw worker
deltas cancel *before* that single floor and the global cumulative still stays ≥ 0. Read-time
flooring in `get_strategy` is unchanged, so a worker's own within-chunk regret matching still floors
correctly (it just carries raw debt within a chunk → vanilla-CFR re-activation within the chunk,
bounded and reset every merge).

### Why it wasn't caught earlier
The parallel path was validated by info-set *overlap* and normalization, never by a *bias* or
*agreement-with-single-thread* measurement — and the floor lived in shared `cfr()` code that "looked
obviously correct" in single-thread. The earlier framing in the docs even had it backwards
(blamed the *master* floor); a first fix attempt wrongly deleted the master floor and was reverted
once `blueprint_trainer.py`'s line-251 write-floor was confirmed as the real CFR+ site.

### Retrain impact
**Retrain required to benefit.** The fix changes what parallel training stores; existing
parallel-trained blueprints (incl. the served `blueprint_par_*`) carry the bias and should be
retrained. Single-thread blueprints were never affected. Validated at scale (200k iters/arm,
`merge_every=2000`, 16 workers): jam bias (parallel − single) dropped to **+0.0014** (0.2941 vs
0.2955 over 10,804 jam-action keys) — statistically gone — at a 3.76× speedup. Bit-identity is not
expected (different RNG streams); the bias-relevant aggregate is what's validated.

### Lessons
An idempotent-looking op (flooring at 0) is **not** distributive over a sum when applied per-shard:
`Σ max(0, xᵢ) ≠ max(0, Σ xᵢ)` whenever any `xᵢ < 0`. Any reduce-from-workers design must apply such
clamps **once at the merge**, not inside each worker. And "it's a bias, not variance" is the tell
that more iterations won't save you — measure agreement-with-the-oracle, not just convergence.
Tests now lock both directions: `test_merge_cross_worker_cancellation` (raw deltas cancel) and
`test_floor_bias_direction_old_inflates_high_variance` (a millisecond regression re-creating the old
per-worker floor and asserting it inflates), so reintroducing the worker floor fails CI instantly.

---

## BUG-008 — LBR victim model drifted from the deployed bot: under-counts off-grid exploitability

| | |
|---|---|
| **Date** | 2026-05-29 |
| **Area** | Measurement (`evaluation/lbr.py`), action translation (`cfr/translation.py`) |
| **Severity** | High (directional bias in the scoreboard, not a play-time bug) |
| **Status** | Fixed |

**Summary.** The LBR harness's model of how the bot *responds* to an off-grid bet had
diverged from the actual deployed bot in two places — exactly the off-tree regime LBR
exists to measure — so LBR **understated** the exploitability of off-grid betting lines.
This is the "BotRange is the older sibling of RangeTracker, watch for drift" caution in
CLAUDE.md coming true. Found by a 3-agent review of the eval harnesses (the CFR trainer and
BR harness were clean).

### Symptom
No crash — a silently optimistic LBR number. An off-tree exploiter making big/odd bets
would in reality get **over-folded** by the bot, but LBR scored those lines as if the bot
**called**, so the measured lower bound on exploitability was too low precisely where the
exploiter should shine.

### Root cause (two drifts, same theme: victim ≠ deployed bot)
1. **Inverted untrained-bracket fold model (the critical one).** When LBR makes an off-grid
   bet, it translates onto the two bracketing grid sizes and blends the bot's per-hand fold
   probability. For an **untrained** bracket, `per_hand_action_prob` returned `0.0` fold
   (i.e. modelled the bot calling its whole range). But the deployed bot routes an untrained
   bracket's weight to **fold** (`translation.blend(missing_action='fold')` via
   `bot_strategy._blend_lookup`, shipped earlier this session). So LBR modelled a call where
   the bot folds → wrong-direction bias.
2. **Preflop wasn't translated at all.** Postflop LBR pseudo-harmonic-translated off-grid
   bets, but preflop used snap-to-nearest (`categorize_bet_size`), while the deployed bot
   translates **every** street. LBR makes off-grid preflop raises, so the preflop victim was
   a strawman.

### The fix
- `per_hand_action_prob` gained a `missing=` param (per-hand value when the bracket key is
  *untrained*, distinct from "trained but this action has 0 mass"). The fold query passes
  `missing=1.0`, mirroring the deployed bot's fold-routing.
- Preflop now translates through a shared `translation.preflop_grid(...)` helper — the SAME
  grid definition the live API (`strategy_api._preflop_grid`) uses — then blends like
  postflop. Centralized the size→char map as `sizing.SIZE_CHAR` (one copy, imported by both
  the API and LBR) and routed the API's `_preflop_grid` through the shared helper too.

### Why it wasn't caught earlier
The two siblings (`BotRange`/`RangeTracker`, LBR-victim/`bot_strategy`) were never merged, and
the fold-routing fallback was added to the *deployed* path this session without updating the
*measurement* path. No test compared the victim model to the deployed bot on an off-grid line.

### Retrain impact
None on training or the served bot — measurement-only. But any LBR number taken **before**
this fix understates off-grid exploitability; only post-fix LBR is trustworthy for the
upcoming retrain's before/after.

### Lessons
Every "the deployed bot does X off-tree" change must update the **victim model** in lockstep,
or the scoreboard lies in the one regime it's meant to police. Shared helpers (`preflop_grid`,
`SIZE_CHAR`) are the same anti-drift discipline as `keys.py`/`sizing.py` — extend them, don't
re-implement.

---

## BUG-007 — The new bot loses to the old one: a preflop action-grid coverage hole (not a training bug)

| | |
|---|---|
| **Date** | 2026-05-28 |
| **Area** | Action abstraction (preflop sizing), `cfr/translation.py`, measurement (`evaluation/cross_match.py`) |
| **Severity** | High as a *finding* (~200 mbb/hand head-to-head); the code defect inside it is medium |
| **Status** | Diagnosed; abstraction fix designed (rides next retrain); translation fold-fallback shipped as a stopgap |

**Summary.** A faithful head-to-head showed the *new*, GTO-correct small-open blueprint
**losing ~210–243 mbb/hand to the older big-open blueprint** — at near-equal iterations,
so not a training-gap artifact. This looked like the preflop-sizing redesign was a
regression. After a multi-round investigation (and one confidently-wrong hypothesis), the
real cause was a **coverage hole in the preflop action grid**: the bot has *zero* trained
response to a big open or any preflop all-in, so a specific opponent that opens off-grid
walks into a hole. It is **not** a training bug, **not** a postflop problem, and the
sizing redesign should **not** be reverted. The interesting part is *why* every
in-distribution metric missed it.

### Symptom
`cross_match.py` (old-vs-new, AIVAT-free, paired seed): OLD beats NEW **+242.9 ±42.5**
mbb/hand at the new run's 6.05M, **+212.1 ±42.7** at 7.5M. The gap barely closed with more
NEW training (~31 mbb over 1.45M iters) and NEW had already hit its BR floor (~14,300 since
~4M) — so the deficit was structural, not under-training.

### The wrong hypothesis (overturned by a controlled experiment)
First story: *small opens → ranges stay wider → more postflop volume → more exposure to the
leaky postflop abstraction.* Plausible, and **wrong**. The **new-vs-new mirror match is
~break-even OOP** — against an *equal-grid peer*, the OOP bleed vanishes. So the postflop
play is fine vs a peer; the loss is specifically about the *opponent's* sizes. Decompose,
don't theorize.

### Root cause (verified)
The blueprint has **0 `pf_*_*_a` keys** — no preflop info-set anywhere whose pattern
contains an all-in. Mechanism: `poker_game.py` only offers `allin` preflop when a sized
raise is *unaffordable*, and at 100 BB the 3-aggression cap (open/3-bet/4-bet) means even a
1.5×-pot 4-bet only commits ~56 BB — sized raises never exhaust the stack, so a preflop jam
is **never reachable in training**. The bot therefore never learns a response to a jam, and
— because a self-play bot never *opens* big — never learns a response to a big open either
(facing-large-3-bets *is* well trained: ~375 keys / 5.6M visits; the hole is specifically
the top of the open ladder and the all-in level).

The old bot opens up to 7 BB. The new BB facing a 6–7 BB open translates it onto its grid as
a blend like `{l: 0.6, a: 0.4}` — but the `a` bracket has **no trained key**, so that weight
is mishandled (see the code defect below) and the response collapses toward the pure `l`
strategy, which is calibrated to a **3.5 BB** open. So the new BB calls/commits far too
loosely into a 5–7 BB raise → that's where the chips go. (Separately, ~30% of the raw gap is
**intrinsic button-vs-BB positional skew** shared by both bots and should be netted out, not
charged to the redesign.)

### The code defect inside it — `translation.blend` dropped the untrained bracket
When a bracketing size has no trained strategy, the cross-match's `blend_dist` **dropped**
that weight (renormalising onto the nearer, too-small bracket); the live
`BlueprintStrategy._state_distribution` filled the unknown key with **uniform-over-legal**.
Both are wrong: neither folds to a bet bigger than anything the bot understands, so the bot
**under-folds to off-grid big opens** (and, live, to a human overbet). Fix: a
**missing-bracket → fold fallback** in the shared `translation.blend` (route an untrained
bracket's weight to `fold`, the conservative response to an off-grid bet), mirrored in
`cross_match.blend_dist` and fed by a `_blend_lookup` that returns `{}` (not uniform) for
untrained keys.

**But the fold-fallback is net-neutral for the head-to-head** (500k: −243 → −249.7, within
noise). It helped OOP big-open defense (−845 → −581) but the blunt "fold any untrained
overflow" also over-folds when NEW is IP facing the old bot's big **3-bets** (+117 → −2);
the two cancel. Lesson confirmed: **translation across a large grid gap is a stopgap, not a
fix** — exactly the Libratus/DeepStack reason for moving from translation to subgame solving.

### Why no in-distribution metric caught it
BR/LBR exploitability — the convergence scoreboard — never exposed this. A self-play
blueprint *never opens big*, so its own training and its self-referential metrics never
visit the "facing a big open" nodes; the hole is **invisible to in-distribution evaluation**
(you never face a size you never make). It only surfaced once we built a *deliberately
out-of-distribution* opponent (the old big-open bot) in a faithful head-to-head harness.
This is the general Libratus lesson: action-abstraction coverage holes are found by an
adversary using off-grid sizes, not by self-play exploitability.

### Fix / resolution
- **Not** a training bug; the harness is sound (the per-seat skew reproduces in the
  independent `match.py`; pattern/blend bookkeeping verified clean over 3000 hands).
- **Real fix = abstraction change, retrain:** add `allin` as an always-available preflop
  aggressive action at every node (open/3-bet/4-bet), and add a 4th open size (`x` = 5 BB,
  alphabet `s/m/l/x`) to anchor the mid/large-open zone; keep the 3-bet/4-bet multipliers.
  See [../../../docs/ROADMAP.md](../../../docs/ROADMAP.md) "Betting-abstraction redesign".
- **Translation fold-fallback kept** as cheap live insurance (it hardens the live bot against
  human overbets even though it doesn't move this head-to-head).
- **Deep raises (non-jam 5-bets+) → subgame solving**, not abstraction: train shallow
  (capped), play uncapped (relax the live `max_raises_per_street` gate), solve the deep tail.
  Rare lines abstracted would train thinly — the *same* failure mode as this hole.

### Retrain impact
Abstraction change → existing blueprints incompatible, fresh run required (no resume). Bundle
with the next **postflop-bucket** retrain (the real strength lever; BR floor ~14,300 is the
buckets, not convergence) and read the BR/LBR delta as a *bundle*.

### Lessons
1. **Equilibrium-optimal ≠ robust to a specific opponent.** Small opens are GTO-correct, yet
   the bot was fragile to off-grid opens because the abstraction had no response bracket
   there. "It's optimal" and "it loses this matchup" were both true and not in conflict.
2. **Action grids have coverage holes self-play can't see.** You never train a response to a
   size you never produce; in-distribution metrics (BR/LBR/self-play) are blind to it. Test
   with an out-of-distribution adversary.
3. **Decompose before theorising.** The first root cause (postflop volume) was plausible and
   wrong; the mirror-match control overturned it in one experiment.
4. **Translation is the leaky stopgap; solving is the fix** — the net-neutral fold-fallback
   re-derived the exact conclusion the Libratus/DeepStack literature reaches.
5. **Net out shared/positional variance** before attributing a head-to-head gap to a change
   (~30% here was button-vs-BB skew common to both bots).

---

## BUG-006 — River subgame solver build: three bugs across the layers (Phase 4)

| | |
|---|---|
| **Date** | 2026-05-27 |
| **Area** | River subgame solver (`src/subgame/river_tree.py`, `range_inputs.py`; design decisions) |
| **Severity** | Mixed — one manifested logic bug, one latent trap, one degenerate spec |
| **Status** | All three fixed/resolved during the build |

**Summary.** Building the Phase-4 river solver surfaced seven instructively-different
defects: (A) a betting-rule logic bug that a test caught immediately, (B) a latent
chip-accounting trap that *only* a manual review caught (tests structurally couldn't),
(C) a design decision that was internally degenerate (a no-op) once implemented,
(D) a Linear-CFR clock that reset per call, latent until step 5 would have leaned on it,
(E) a re-raise fraction that dropped a term -- and a regression test that was VACUOUS twice
over until a discriminating node was constructed, (F) a hero ~zero-reach silent uniform
read-off (the original uniform-fallback failure class, recurring on the hero side), and
(G) a solver all-in silently downgraded to a check at deep-stack nodes -- the most
consequential, since it would have biased the very scoring run meant to validate the solver.

### (A) Raising into an all-in — manifested, test-caught

**Symptom.** `test_facing_allin_only_fold_or_call` failed: facing an all-in, the other
player still had `allin` as a legal action.

**Root cause.** The tree's aggression generator added a shove whenever the actor's all-in
exceeded the current call (`allin_new_sc > max(sc)`). When the opponent was already all-in,
the actor's larger stack made that true — but there is nothing to raise into (the all-in
player cannot call more), so the only legal responses are fold/call.

**Fix.** Gate raises on the *opponent* still having chips behind:
`opp_behind = stacks[other] - sc[other]; if agg < max and opp_behind > 0: add raises`.
This also structurally blocks shoving over an all-in.

### (B) All-in-for-less doesn't return the uncalled chips — latent, review-caught

**Symptom.** None observed. Found by an independent review that probed showdown terminals
with *unequal* stacks and found e.g. `final_pot=120, contrib=(40,80)`.

**Root cause.** Showdown terminals feed the kernel `final_pot = c0 + c1` and
`hero_total = c_hero` directly. That is correct only when contributions are *matched*. With
unequal stacks a short call leaves the aggressor's excess uncalled; that excess is not
returned, so the showdown pot and hero contribution are inflated → wrong EV.

**Why tests couldn't catch it.** Chip-conservation asserts `final_pot == c0 + c1`, which holds
*by construction* whether or not the uncalled portion was returned. The defect is invisible to
the one invariant that looked closest. (Echoes BUG-003: a self-consistent-but-wrong quantity.)

**Why it can't fire today.** Both players start each hand at `STARTING_STACK` and must match all
prior-street action to reach a river betting node, so river-entry stacks are always equal — an
all-in-for-less cannot arise. The trap is purely latent: it would bite a future caller passing
unequal stacks to the standalone `build_river_tree`.

**Fix.** Enforce the invariant loudly — assert equal river-entry stacks in `RiverTree.__init__`,
documenting that unequal stacks need all-in-for-less handling (cap the aggressor's contribution to
the matched amount) before the assert is lifted. Plus a `test_raise_sizing_matches_engine` that
pins the tree's bet/raise-to chips against `PokerGame.calculate_bet/raise_amount` — the sizing
drift check chip-conservation structurally cannot provide. (Fold terminals were verified fine
unchanged: `final_pot − contrib` already returns the winner's own uncalled chips.)

### (C) Confidence-widening target was a no-op — design degeneracy

**Symptom.** A "locked" design decision said: when the range-tracker's confidence collapses, widen
the villain range toward "blueprint-reach-given-the-line."

**Root cause.** The range tracker *already* computes exactly that — its weights are
`uniform × ∏(blueprint action-probabilities along the line)`, which *is* the blueprint reach given
the line. So `blend(tracked, blueprint-reach-given-line, c) = tracked` for every confidence `c`. The
widening would do nothing.

**Fix.** Widen toward a genuinely *flatter* target: tempered reach `temper(tracked, β) ∝ tracked^β`
over the line-consistent support, with `β` a flattening knob (β=1 → untouched belief, β=0 → uniform
over the support = maximum flattening, the default). Guarded the `0**0` trap so card-removal /
line-impossible zeros stay zero. Only the villain range is blended; the hero range is blueprint
reach as-is.

### (D) Linear-CFR clock reset per run() — latent, review-caught

**Symptom.** None observed (steps 3/4 tests passed). Found by review reasoning about step 5.

**Root cause.** `RiverCFR.run()` set the strategy-sum weight `_t_weight = t` over a *local* `range(1,
iters+1)` each call. CFR+ averages the strategy weighted by the iteration number, so solving in
increments — `run(100); run(100)` — would restart the clock at 1 the second time, under-weighting the
later (more-converged, better) strategies relative to one `run(200)`. Harmless for a single `run()`,
but step 5's convergence-based early-stop / warm-start solves in increments, so it would have silently
skewed the average there.

**Fix.** Persist a cumulative `self._iter` across `run()` calls; `_t_weight = self._iter`. Now
`run(100)+run(100)` is bit-identical to `run(200)` (locked by `test_incremental_run_matches_single`).

### (E) Re-raise fraction dropped the actor's committed chips — review-caught

**Symptom.** None observed; found by a review probing a re-raise node.

**Root cause.** `blueprint_to_tree_dist` (the EV-gate baseline) recovered a tree raise's pot-fraction as
`(sized_chips - to_call)/(pot + to_call)`. A raise's true street total is `sc_actor + sized_chips`, so
the formula is correct ONLY when the raiser has no chips in the street yet (`sc_actor = 0`). At a
3rd-aggression node it skews low by `sc_actor/(pot+to_call)` — e.g. on sc=[30,10], pot=60, to_call=20
the ½-pot raise was recovered as 0.375 instead of 0.5, so a blueprint `raise_medium` mapped to the
wrong tree edge.

**Why tests missed it.** The only mapping test used a FIRST-raise node (`sc=0`), where the bug is
invisible. Impact is bounded: this feeds only the EV-gate baseline ("what would the blueprint do"), not
the solved strategy or the emitted size.

**Forward-insurance, not a live fix (verified, 2 review rounds).** A sweep of 332 sc>0 raise nodes ×
48 pot/stack configs against the production blueprint menu (0.33/0.66/1.0) found the offset crosses a
nearest-neighbour boundary at **0** nodes -- so the bug changed nothing observable today; it only
matters for a denser / overbet menu (Phase-4 widening). Keep the fix as correct insurance.

**Fix.** Include the actor's committed chips: `(node.sc[node.player] + sized_chips - to_call)/(pot +
to_call)`.

**Test-vacuity sub-lesson (the more important one).** The FIRST regression test added for this
(`raise_medium`/`raise_large` on an sc=10 node) was itself VACUOUS -- it passed on the buggy formula
too (the prior session made an arithmetic error claiming otherwise). A second review caught that. A
discriminating test needs a node where the dropped offset (delta = sc_actor/(pot+to_call)) STRADDLES a
nearest-neighbour boundary; the real test now uses sc=(54,34)/pot=80/to_call=20 (delta=0.34) where
`raise_medium` maps to `raise:61` (fix) vs `raise:86` (bug). LESSON: "covers the general case" is not
the same as "discriminates the fix" -- compute both branches on the test input and confirm they differ,
or the regression test is false confidence (twice over here).

### (F) Hero ~zero-reach -> silent uniform read-off — review-caught (same class as the original)

**Symptom.** None observed; found by review.

**Root cause.** The hero (bot) range was projected and used as-is with no positive-reach guard on the
bot's ACTUAL hand. If that hand has ~0 hero reach (the blueprint assigns it ~0 chance of taking this
line), its strategy-sum row never accumulates, so `average_strategy` returns uniform 1/A and the read-off
emits a near-RANDOM action -- a silent quality collapse, not a crash. This is the SAME failure-mode class
as the earlier seat/path uniform-fallback bug, now on the hero side; a board-collision guard existed but
not a zero-reach one. (In the live GameSession path it ~can't fire -- the bot plays the blueprint
pre-river, so its actual hand has positive through-turn reach -- but it's cheap, defensive, and matters
for other callers / float edge cases.)

**Fix.** In `solve_for_action`, after projecting hero, raise if `hero[actual_hand_row] <= 1e-12`, so
`decide()` falls back to the blueprint cleanly instead of reading off uniform. Also reordered: all
validation (board collision, hero reach, path navigation, seat) now happens BEFORE the costly solve.
Test: `test_hero_zero_reach_falls_back`.

**Also hardened (not bugs):** (i) `decide()`'s broad `except` (kept so a solve failure never crashes a
live hand) now LOGS rate-limited with a traceback before falling back -- so a genuine defect surfaces in
play instead of silently degrading to the blueprint (the failure mode that once hid the uniform-fallback
bug). (ii) the EV-gate baseline's no-analog redirect changed from allin-first to PASSIVE-first
(check>call>fold>allin) so it doesn't inflate the baseline EV. (iii) the final action sample uses a
seedable RNG (reproducible scoring/tests). Visibility/robustness, not behavior changes.

### (G) Solver all-in silently downgraded to a check at deep-stack nodes — review-caught, consequential

**Symptom.** None observed in passing tests, but verified by tracing: a chosen `'allin'` becomes `'check'`.

**Root cause.** The engine's `get_legal_actions` OMITS a discrete `'allin'` when every sized bet is
affordable (deep stacks) -- the legal list is e.g. `['check','bet_small','bet_medium','bet_large']`. But
the river tree ALWAYS offers `'allin'`. So when the solver chose to shove, `_pick_engine_action` found
`'allin'` not in `legal_actions`, `is_sized('allin')` is False, and it fell through to the
`('check','call','fold')` fallback -> emitted **check**. A GTO polarized shove silently became a check,
precisely at deep-stack nodes where shoving matters most.

**Consequence.** This is the worst of the six because it would have **biased the validation run itself**:
the head-to-head / LBR scoring meant to prove the solver would understate it every time it correctly
wanted to shove deep. (Also a live bug if/when the solver is ungated.)

**Fix.** When `choice == 'allin'` and `'allin'` not in `legal_actions`, emit a full-stack custom shove
`make_custom_action(is_raise=node.to_call>0, total=stacks[bot_seat])`. The engine's `custom_bet_bounds`
hi == the behind stack, so `_validate_custom` normalises the at-stack custom to `'allin'` (verified).
Test `test_allin_emits_shove_not_check_at_deep_stack` (deep-stack legal set without `'allin'`) asserts a
shove, not a check, both checking and facing a bet.

**Lesson.** The solver's action vocabulary (always-present all-in) and the engine's abstract legal set
(all-in only when sized bets don't fit) DIVERGE; any mapping between two action vocabularies must handle
every solver action that has no direct engine label, not just the sized ones. The earlier
"check/fold/call/allin map directly when legal" looked total but silently dropped the not-legal-allin case.

**Retrain impact.** None — all are inference/subgame-side (the trainer never builds a river tree,
tracker, or subgame CFR). No abstraction or key change.

**Lessons.**
- A chip-conservation invariant that holds *by construction* validates almost nothing about the
  quantities it sums; pin sizes against an independent oracle (here, the engine) too.
- "Latent, can't fire today" is worth a loud assert, not a comment — a standalone reusable
  component outlives the invariant its first caller happens to satisfy.
- A design spec can be self-referentially degenerate; implementing it (not just agreeing to it) is
  what exposed that "widen toward X" where the input already equals X is a no-op.

---

## BUG-005 — Potential-aware abstraction: silent-divergence risk class (review + hardening)

| | |
|---|---|
| **Date** | 2026-05-25 |
| **Area** | Postflop abstraction (`src/abstractions/postflop_v2.py`, `postflop_features.py`, `scripts/bake_postflop_table.py`) |
| **Severity** | Latent — no manifested defect; hardened proactively |
| **Status** | Hardened (C2/M3 fixed); related items filed / deferred |

**Summary.** A static review of the new distribution-aware (PostflopV2) abstraction
flagged a *class* of silent-divergence risks: the bucket for a hand is produced by
three code paths that don't have to agree, and a baked lookup table carried no link
to the centroids that produced it. No bug had manifested — but nothing in code
*enforced* consistency.

**Symptom.** None observed. Found by review, not by failure. (Contrast BUG-002/003/004,
which manifested as bad EV / negative stacks / wrong pots.)

### The risk class

1. **Three sampling regimes feed one nearest-centroid assignment.** Centroid fitting
   (`compute_postflop_buckets.py`, 60 runouts / 200 opp-samples), table baking
   (`bake_postflop_table.py`, 200 flop runouts / full 990 opp), and the runtime lazy
   fallback (`postflop_v2.py`, 120 runouts / 150 opp) estimate the equity distribution
   differently, so on boundary hands they can assign different buckets (~3% measured at
   bake time). The docs called the lazy path "correct but slow" but it is a *different
   function* than the table.
2. **No centroid→table version link.** A baked table stored only `ids`/`buckets`. If
   centroids were regenerated without re-baking, training/inference (table path) and the
   river/lazy paths would use *different* centroids — self-inconsistent, no error raised.

### Why it was not actively biting training

Verified, not assumed: instrumented a real 4,000-iteration run and counted **0** lazy
fallback calls. Training buckets every flop/turn hand via the deterministic baked table
and river via exact equity, so training and inference read the *same* buckets (the
"single bucketer everywhere" property — trainer, `best_response.py`, `lbr.py`,
`game_session.py`, `range_tracker.py` all route through `CardAbstraction.get_bucket`).
The ~3% disagreement is table-vs-lazy, and lazy only fires on a table miss (the table is
complete — 100% hit on random samples), so it never fires in the baked pipeline. The risk
is purely *latent* — it would bite a clone that skipped baking, or a future run on a
stale table.

### The fix (C2 + M3)

Stamp every baked table with a fingerprint of its centroids and assert on load:
- `postflop_features.centroid_hash(centroids, bins)` — sha1 over float64 centroids + bins.
- `bake_postflop_table._save` writes `centroid_hash` + `n_buckets` (K) + `bins` into the npz.
- `PostflopV2._verify_stamp` (in `_table`, on load): matching stamp → proceed; **stamp
  mismatch (stale table) or wrong K/bins → hard `ValueError`** naming the re-bake command;
  legacy stamp-less table → warn once and proceed (so pre-stamp tables keep working until
  the next re-bake). This makes "regenerate centroids, forget to re-bake" impossible to do
  silently.

Tests: `tests/test_postflop_table_stamp.py` (5) + real `np.savez`/`np.load` round-trip.

### Filed / deferred (not fixed now)

- **N1** — the 126 MB turn table reloads per `CardAbstraction` instance (→ per API request);
  training is unaffected (one instance). Fix at the next API/AWS perf pass (module-level cache).
- **M1** — river centroids fit on sampled equity but assigned on exact equity (consistent
  between train/infer; a mild calibration nit). Refit on exact equity if river play looks off.
- **M2** — EMD k-means uses Euclidean-mean updates rather than the 1-D Wasserstein barycenter,
  and doesn't reseed empty clusters (verified balanced occupancy, no dead buckets in practice).
  **Deferred decision:** revisit only if the v2 BR/LBR numbers disappoint vs baseline
  (11,256 / 3,636 mbb).
- **C1 lazy determinism** — the fallback's shared RNG makes lazy assignments order-dependent;
  moot in training (never fires). Cheap insurance for later.

### Lessons

- **"Different function, same purpose" is a latent train/inference hazard** even when the
  paths currently agree — enforce equality (one parameterization, or a stamp) rather than
  trusting that a fallback matches the primary path.
- **Artifacts that derive from inputs need a version link.** A baked table is a cache of the
  centroids; without a fingerprint, a stale cache corrupts silently. (Same spirit as the
  single-source-of-truth `cfr/keys.py`.)
- **Verify "is it actually happening" before rating severity.** Counting lazy calls during
  real training turned a "Critical silent divergence" into a measured "0 — latent only."

---

## BUG-004 — Calling an all-in cost 0 chips when the caller was already ahead this street

| | |
|---|---|
| **Date** | 2026-05-21 |
| **Area** | Game logic (`src/cfr/poker_game.py`) — `get_call_amount_from_history` |
| **Severity** | Critical — gameplay outcome + training correctness |
| **Status** | Fixed (existing blueprints stale w.r.t. all-in-called lines) |

**Summary.** When the last aggressive action was `allin`, the call-cost
calculation compared the all-in's **increment** against the caller's
**total street commitment** — apples to oranges. If the caller's
existing commitment exceeded the all-in's increment, the call cost was
clamped to 0 and the caller "called" for free.

**Symptom.** A user observed an all-in pot return `+86.9 BB` instead of
`+100 BB`. The bot's `Call` of an all-in showed no chip amount and the
bot's stack stayed positive (13.1 BB) after supposedly calling.

### Root cause

`get_call_amount_from_history` walks the history backwards to find the
last aggressive action and sets `last_bet_amt` from it. For sized
`bet_`/`raise_` actions it correctly sets it to the player's **total
street commitment** after the action (e.g.
`multiplier * pot_after_call + call_amount`, which is the raise-to
total). For `allin`, it instead set:

```python
last_bet_amt = self._allin_amount(history[:i], street, starting_pot,
                                  bet_player, p0_prev, p1_prev)
```

`_allin_amount` is the **chips the all-in put in this action** — an
increment, not a total. Then the final line does:

```python
result = max(0.0, last_bet_amt - player_contrib)
```

`player_contrib` is the caller's total street commitment. Subtracting
an increment from a total is meaningless: if the caller had already
committed more chips this street than the all-in's increment, the
result was negative and got clamped to 0.

### Concrete walkthrough (the user's hand, in chips)

- Through preflop/flop/turn both players committed 25 chips each.
- River, starting_pot = 50.
- P1 (user) bets medium → 33 chips committed this street.
- P0 (bot) raises large → 149 chips committed this street.
- P1 shoves all-in: increment = 142 chips (their entire remaining stack).
  P1's total street commitment after the shove = 33 + 142 = **175 chips**.
- P0 calls. Correct cost = 175 − 149 = **26 chips** (~13.1 BB).
- Buggy code: `last_bet_amt = _allin_amount = 142`,
  `player_contrib = 149`, `max(0, 142 − 149) = 0`. The bot called for **free**.
- Final pot was 186.9 BB (missing the 13.1) and the user won
  `186.9 − 100 = +86.9` BB instead of the expected `+100`.

### The fix

Make the all-in branch produce the same kind of value as the bet/raise
branch — the all-in player's total street commitment after the action.
An all-in player has no chips left, so their total committed for the
whole hand is `STARTING_STACK − prev_invested`; subtract `starting_pot`-
side contributions and the math falls out, but the simplest correct
formula is:

```python
bet_player_prev = p0_prev if bet_player == 0 else p1_prev
last_bet_amt = STARTING_STACK - bet_player_prev
```

This matches the semantics of the sized-bet branch and the subsequent
`max(0, last_bet_amt - player_contrib)` produces the right call cost.

### Why the existing tests missed it

- **Chip conservation was preserved.** `cost = 0` meant the bot's stack
  didn't change AND the pot didn't change — so
  `p0_stack + p1_stack + pot == 2 × STARTING_STACK` still held. The
  randomized chip-conservation fuzz could not see the defect.
- **The exact triggering sequence is narrow.** It needs the caller's
  this-street commitment to *exceed* the all-in's increment, which
  requires the caller to be the prior aggressor (e.g. bet then got
  shoved on for less, or raised big then got jammed on for less). Not
  every all-in-then-call hits it.
- **No targeted test covered an all-in following a bigger raise.**

### Detection / prevention

- Caught from a live game: the user noticed a `+86.9 BB` all-in pot
  that should have been `±100 BB`, and the action log showed
  `bot · river · Call` with no chips.
- Regression test added: `test_call_after_allin_costs_total_minus_caller_contrib`
  — reproduces the exact scenario and asserts `call_cost ≈ 26 chips`.
- Stronger fuzz invariant added in `test_random_session_playout_invariants`:
  *every `call` action with chips remaining must cost > 0*. This catches
  the entire bug class — including future regressions where some path
  produces a zero-cost call. Chip conservation alone could not.
- **Lesson:** internally consistent bugs (cost=0 in both the deduction
  and the pot) survive aggregate invariants like chip conservation.
  Each *primitive* — call, bet, raise, all-in — needs its own invariant
  tied to *what the action means in poker*, not just to chip totals.

### Impact

- **Gameplay.** All-in pots resolved through this path under-rewarded
  the winner (and under-cost the caller). The user's reported
  `Net +299.8 BB` over 170 hands was, on net, *less than it should have
  been* in some hands.
- **Training.** `get_call_amount_from_history` is used by
  `_action_cost('call')`, `calculate_current_pot`,
  `calculate_raise_amount`, and `get_player_contribution_this_round`.
  In CFR, every all-in-and-called line had a wrong call cost and a
  wrong pot, propagating into incorrect utilities and regrets for that
  whole class of lines. Blueprints trained before this fix
  (including the 6M-iteration `blueprint_20260520_003107.db`) are
  stale w.r.t. all-in lines and should be retrained.

---

## BUG-001 — Average-strategy readout silently dropped actions

| | |
|---|---|
| **Date** | 2026-05-19 |
| **Area** | Blueprint storage / inference (`src/storage/blueprint_db.py`) |
| **Severity** | Medium — inference correctness. No crash, no training corruption. |
| **Status** | Fixed (no retraining required) |

**Summary.** The blueprint's average strategy was normalized over a stale
"first-seen" action list, so any action a decision point picked up on *later*
visits (e.g. `allin`) was silently dropped from the strategy the bot reads back.

**Symptom.** A trained bot assigns 0 probability to going all-in in short-stack
spots, even though training had learned a shove there. It does not crash —
the strategy is just quietly incomplete. Found during code review.

### Background — how a strategy is stored

Each decision point (an *information set*) stores:

- `cumulative_strategy` — a **dict** `{action: accumulated probability mass}`,
  grown over training by repeatedly adding the current strategy into it.
- `legal_actions` — a **list**, set **once on the first visit** and never updated
  (`if not self.legal_actions: self.legal_actions = legal_actions.copy()`).

### Root cause

The postflop info-set key does **not** encode pot size (a known abstraction
limitation — see *Related: M1*). So two structurally different decisions — a
flop in a tiny limped pot and a flop in a huge 4-bet pot — collapse to the
**same key**. But their legal action sets differ: in a big pot relative to your
stack a sized bet costs more than you have and is replaced by `allin`; in a
small pot every sized bet is affordable and there is no `allin`.

`cumulative_strategy`, being a dict keyed by action name, merges both visits
correctly — it ends up with a key for every action ever played. But
`legal_actions` was frozen to whatever the **first** visit saw. If that visit
was a small-pot one, `legal_actions` never contains `allin`.

The readout (`BlueprintDB.get_average_strategy` / `get_record`) normalized
`cumulative_strategy` **over `legal_actions`**:

```python
total = sum(cumulative_strategy.get(a, 0.0) for a in legal_actions)   # stale list
return {a: cumulative_strategy.get(a, 0.0) / total for a in legal_actions}
```

Iterating the stale list excludes `allin` from both the sum and the result.

### Concrete walkthrough

1. **Visit 1** (small pot) — actions `[fold, call, raise_small, raise_medium, raise_large]`.
   `legal_actions` is frozen to these five.
2. **Visit 2** (big pot, same key) — actions `[fold, call, allin]`.
   `cumulative_strategy` now has six keys; `legal_actions` still has five.
3. **Export** — normalize over the five-action `legal_actions` → the returned
   strategy dict has **no `allin`** entry.
4. **Bot at a short-stack instance of this key** — looks up the strategy,
   `strategy.get('allin', 0.0)` → `0` → it never shoves, despite training for it.

### The fix

Normalize over `cumulative_strategy`'s own keys — every accumulated action:

```python
total = sum(cumulative_strategy.values())
return {a: v / total for a, v in cumulative_strategy.items()}
```

Applied to `BlueprintDB.get_average_strategy` and `get_record`. `legal_actions`
is now used only for the rare uniform fallback (a key with no accumulated mass).

**No retraining required** — the complete data was always in
`cumulative_strategy`; only the readout was wrong. Existing blueprint DBs
export correctly once read through the fixed code.

### Why it wasn't caught

- It is a **readout** bug. Training never reads the average strategy back
  (it uses regret-matching on `cumulative_regrets`), so nothing in training
  surfaces it.
- It only triggers for a key whose action set **varies across visits**, which
  only happens because of the pot-blind postflop key (M1) — a specific
  precondition.
- An earlier audit lumped it under M1 and reasoned "subgame solving will handle
  it." It is **not** an abstraction issue — it is a normalization bug. Subgame
  solving re-solves the *decision*; it does not fix how a stored average
  strategy is read.
- No test exercised a single info set accumulating two different action sets.

### Detection / prevention

- Found by re-reading the code carefully after a pointed question, then
  confirmed by tracing `accumulate_strategy` → `save_batch` → `get_average_strategy`.
- Regression test added: `test_db_average_strategy_includes_all_actions` —
  accumulates two different action sets into one info set, saves, reads back,
  asserts no action is dropped and the strategy sums to 1.
- **Lesson:** distinguish "the abstraction is coarse" (acceptable, by design)
  from "a plain bug rides on top of the coarse abstraction" (must fix). The
  pot-blind key (M1) was only the *precondition*; the dropped-action
  normalization was a separate, cheaply fixable bug.

**Related — M1 (open, deferred).** The postflop key omitting pot size is a real
abstraction limitation: a pot-merged key's strategy is a pot-*blind blend*.
That part genuinely is subgame solving's job (it re-solves with the true pot
and stacks) and is deliberately deferred. BUG-001 was the separable bug sitting
on top of it.

---

## BUG-002 — CFR sign convention mixed P0/P1 perspective

| | |
|---|---|
| **Date** | 2026-05-18 |
| **Area** | CFR training (`src/cfr/blueprint_trainer.py`) |
| **Severity** | Critical — training correctness |
| **Status** | Fixed (required discarding all earlier blueprints) |

**Summary.** `cfr()` propagated utilities with an inconsistent player
perspective, so a node could average a P0-perspective value with a
P1-perspective one.

**Symptom.** Self-play EV stabilized around **+17 chips/hand** — impossible in a
zero-sum game (it should approach ~0) — and the strongest and weakest starting
hands learned **near-identical strategies**.

**Root cause.** `get_utility()` always returns utility from P0's perspective,
but `cfr()` negated child values unconditionally on action recursion while
*not* negating at terminals or street transitions. The number of negations
between a terminal and a node depended on the path, so perspectives mixed.

**Fix.** Rewrote `cfr()` to always return a P0-perspective value; at a decision
node, convert to the acting player's perspective for the regret computation,
then convert back before returning.

**Detection.** Caught empirically: a 60k-iteration run showed EV +17 and
indistinguishable premium/trash strategies. Static review and the test suite
had missed it — no test checked that the *learned strategy was sensible*, only
structural invariants (regrets ≥ 0, probabilities sum to 1), which the broken
code still satisfied.

---

## BUG-003 — Contribution double-count produced negative stacks

| | |
|---|---|
| **Date** | 2026-05-19 |
| **Area** | Game logic (`src/cfr/poker_game.py`) |
| **Severity** | Critical — training and game-state correctness |
| **Status** | Fixed (affects blueprints trained before the fix) |

**Summary.** `get_player_contribution_this_round` accumulated a raise-*to*
total as if it were an increment, double-counting a player's earlier bet.

**Symptom.** A player's stack went **negative (−26)** during a "Play vs AI"
hand.

**Root cause.** The postflop bet/raise branch used `+=`, but the value
`BET_MULTIPLIERS[size] * pot_after_call + call_amount` is the raise-*to* total,
not an increment. A player who bet then re-raised the same street had the
earlier bet counted twice (e.g. contribution computed as 212 instead of 184).

**Fix.** One token: `+=` → `=` (assignment), matching the preflop branches,
which were already correct.

**Detection.** Caught by a **randomized `GameSession` playout** that asserts
`stack ≥ 0` and chip conservation after every action. Two static audits had
looked at this exact line and *rationalized the `+=` as correct* — they did not
consider a bet-then-reraise by the same player.

---

## Cross-cutting lessons

- **Property-based / randomized fuzz testing catches what example-based tests
  and static review miss.** BUG-003 was invisible to two audits but fell out of
  a random playout in seconds. Invariants worth fuzzing: no negative stacks,
  chip conservation (`stacks + pot == 2 × starting stack`), contributions sum
  to the pot, `|utility| ≤ stack`, regrets ≥ 0.
- **Verify the output is *sensible*, not just *structurally valid*.** BUG-002
  passed every "sums to 1 / non-negative" test; the giveaway was a nonsensical
  EV and premium hands playing like trash. Add tests that assert *quality*
  (e.g. self-play EV bounded near zero; strong and weak hands diverge).
- **Recurring footguns in poker chip arithmetic:** total vs increment,
  P0 vs P1 perspective, per-street vs cumulative, pre-bet vs post-bet pot.
  When reviewing such code, label every quantity with which convention it uses.
- **Separate "coarse by design" from "buggy."** A coarse abstraction can be an
  acceptable trade-off *and* still have a plain bug layered on top of it
  (BUG-001 on top of M1). Don't let "that's just the abstraction" hide a fixable
  defect.
