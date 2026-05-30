# Bug Log

A running record of notable bugs found in the AllIn poker-AI codebase, their
root causes, fixes, and the lessons learned. Kept for future debugging
reference and as a project narrative.

**Entry format** — each bug gets: ID, date, area, severity, status, a one-line
summary, the symptom, root cause, a concrete walkthrough, the fix, why it
wasn't caught earlier, retrain impact, and lessons. Append new bugs at the top.

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
