# Poker AI Roadmap

Last updated: 2026-05-28

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

## Phase 1 — Blueprint training ✅ COMPLETE

A heads-up blueprint is trained with Monte Carlo CFR+ and stored in SQLite.

| Component | File | Status |
|---|---|---|
| Hand evaluation | `src/abstractions/hand_evaluator.py` (phevaluator) | ✅ |
| Card abstraction | `src/abstractions/card_abstractions.py` — 15 preflop equity buckets (`pf_0..pf_14`) + **distribution-aware (potential-aware) postflop buckets: 12 flop / 12 turn / 10 river** (`PostflopV2`, EMD-clustered equity distributions) | ✅ |
| Preflop equity precompute | `scripts/compute_preflop_equity.py` | ✅ |
| Postflop bucket pipeline | `scripts/compute_postflop_buckets.py` (fit centroids) → `scripts/bake_postflop_table.py` (bake canonical→bucket tables, centroid-stamped) → `src/abstractions/{postflop_v2,postflop_features,canonical}.py` | ✅ |
| Action abstraction | `src/abstractions/action_abstractions.py` — small/medium/large + preflop ladders + all-in | ✅ |
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
| River endgame solver (v1, *unsafe*) | `src/subgame/river_subgame_solver.py` — small river tree, vectorized CFR+, ranges from `RangeTracker`, blueprint warm-start (`river_cfr.warm_start` + `blueprint_projection.py`), EV-gated; **served live** (`bot_public_state` feeds it river-entry pot/stacks/ranges/path) | ✅ built (~24× less river-exploitable than the blueprint); **already overbets** (menu includes 1.5× pot + all-in) |
| River nested off-grid sizing | inject the opponent's *exact* bet size as a real tree edge instead of snapping it to the nearest menu size (Libratus nested solving) | 📅 |
| Low-SPR deep-raise / 5-bet+ solving (any street) | reuse the river machinery for near-terminal deep nodes, incl. **preflop non-jam 5-bets+** and any beyond-cap reraise — the bot can re-raise off-abstraction here | 📅 |
| **Depth-limited turn/flop solving (leaf values)** | blueprint counterfactual values as **multi-valued** leaf states; **this is what lets the bot overbet/5-bet on the flop and turn** | 📅 (the hard lift, gates the rest) |
| Safe / nested subgame solving | adversarial root + opt-out values (gadget) — provably no-more-exploitable than the blueprint, all streets | 📅 |

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
  already happens); it's about *how the opponent's off-grid bet enters the tree*. Today
  `_navigate`/`_match_edge` **snaps** the realized bet to the nearest menu edge to locate
  the node; **nested sizing** injects the exact size as a real edge (Libratus). This is a
  separate layer from blueprint pseudo-harmonic translation (`cfr/translation.py`), which
  is the blueprint's off-grid handling — the snap is the *solver's* path reconstruction.
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
- *Lifting the live aggression cap is part of the deep-raise work, not a separate phase.*
  The engine caps everyone at 3 aggressions/street (`poker_game.py:42`
  `max_raises_per_street = 2`), which blocks **both** a human 5-bet **and the bot's own
  5-bet+**. "Train shallow (capped), play uncapped, solve the deep tail": relaxing the
  live cap is the mechanical prerequisite bundled into the deep-raise item — the blueprint
  stays capped (rare deep lines train thinly), the solver handles beyond-cap raises live
  for both players.

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
