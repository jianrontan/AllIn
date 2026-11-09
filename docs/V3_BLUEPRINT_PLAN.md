# v3 blueprint — PLAN, awaiting sign-off

Branch `rust-core`. Written 2026-11-09 after the BUG-027 investigation.
Follows [PHASE6_ABSTRACTION_PLAN.md](PHASE6_ABSTRACTION_PLAN.md) (SPR buckets built,
trained and scored) and [RUST_CORE_PLAN.md](RUST_CORE_PLAN.md) (Rust trainer).

## Why

**SPR buckets alone are a small, real gain.** SPR 100M scores 5.6% lower best
response than v2, and 8% lower as the BB. Head-to-head and LBR show no detectable
difference. On its own that doesn't justify a cutover, which costs a model re-fit,
new assets and a live A/B.

**BUG-027 has a different root cause than the bug log said.** Three
investigations (2026-11-09) found:
- **Not a code bug.** The live bot, the Python trainer, the Rust trainer and a
  from-scratch recompute all bucket J♥9♥ on Q♠9♣3♣ as 25. On that flop bucket 25
  is a clean second-pair bucket.
- **Not under-training.** The node is visited at the 95th percentile of flop keys.
  The 98% call has been stable from 10M to 200M iterations. Fold regret is about 0
  against call regret of about 20–44k.
- **The call is correct inside the trained game.** Against the blueprint villain's
  lead-shove range, J9 has 63–66% equity and calling gains about +100 chips over
  folding. That range is 51–57% no-pair hands and 5–8% top pair; AA/KK/QQ/AQ mostly
  check.
- **Why that range is so wide: the trained preflop tree has no 5-bet.** The cap is
  open → 3-bet → 4-bet. Facing a 4-bet the only legal actions are `['fold',
  'call']`, in both the capped and control menus (checked on the engine). So:
  - the 4-bettor never risks a jam, and 4-bets bluffs like J9s;
  - the 3-bettor can't jam, so it calls 4-bets with JTo 84%, KTo 85%, AA 98%;
  - 4-bet-pot flops start with wide, weak ranges.
  
  Against a realistic human shove range (Qx, sets, strong club draws), J9 has
  about 20% equity and the call loses about 74 chips.
- **Live play has the same hole.** The bot cannot 5-bet jam AA against a human
  4-bet. Live play is uncapped, but the bot only plays trained actions, so it
  calls.

**Two smaller findings feed in:**
- **v2's opponent model underrates flop shoves:** about 6% vs 25% with SPR keys,
  because v2's flop keys pool all SPRs. SPR keys fix this, which matters for
  exploitation.
- **The flop table is noisy.** It was baked from 200 of the 1,176 runouts per
  board. Against the exhaustive answer, 20.7% of hands land in a different bucket,
  and 10.7% land in one whose centroid equity differs by more than 0.05. 4 of the 12
  J9 combos in BUG-027 get bucket 24 instead of 25.

## What changes

| # | Change | Fixes | Retrain? |
|---|---|---|---|
| 1 | **Preflop jam at the raise cap.** Facing a 4-bet (the 3rd preflop aggression), offer `allin` beside fold/call. Facing the jam: fold/call. | BUG-027 at the root; the bot can 5-bet jam | yes (tree change) |
| 2 | **SPR buckets** on flop/turn (spr5, already built) | 5.6% BR; shove-frequency read | yes (already in spr5) |
| 3 | **Exhaustive flop bake** (`--flop-runouts 0`, same centroids) | ~20% bucket noise | yes (assignment change) |
| 4 | **River K 10 → 20** (re-fit river centroids, `--opp-samples 0`; river isn't baked) | Coarsest street; feeds flop/turn values and the gadget's blueprint CFV | yes |

**Change 1 design:**
- Preflop only.
- A jam is the only 5-bet: at 100 BB, after a 4-bet to ~22 BB every sized 5-bet
  commits more than half the stack.
- A 4-bet jam stays absent unless a 4-bet size clamps, as today. The capped menu
  removed free-standing jams on purpose (the over-jam fix), and v3 keeps that
  everywhere except this one node.
- Postflop cap nodes (bet → raise → raise) keep fold/call. First measure how much
  money is behind at those nodes with the 6.1 probe, and add a jam only if it's
  material.
- Pattern: the jam is `a` as the 4th preflop char, e.g. `lmma`. No new key
  characters, and `key_format` stays `spr5`.
- Tree identity: a new `menu_mode` value `capped_pj` (capped postflop menu, no
  voluntary all-in, preflop cap jam). It is stamped in the DB like `menu_mode` is
  today.
  - One factory, `sizing.game_kwargs(menu_mode)`, replaces the hand-built
    `PokerGame(postflop_menu=…, voluntary_allin=not is_capped_mode(…))` calls.
  - Every consumer in `src/`, the API and the eval harness moves to the factory.
  - Any `db_menu_mode` caller left unconverted (the shelved turn-solve scripts)
    must fail closed on `capped_pj`. Today `is_capped_mode('capped_pj')` would
    silently build the old tree.

**Deferred (not in v3):**
- **Finer flop/turn K.** The audit found 19% of second-pair combos share a bucket
  with top pair, and buckets 20–29 are 84% one-pair hands. Promising, but it needs
  a re-fit plus a turn re-bake (13.96M situations) and is a separate variable.
  Score v3 first.
- **River buckets on equity against several clusters of opponent hands (OCHS)**
  instead of a random hand. Targets the BUG-023 class. It's a new feature in both
  languages and touches river serving.
- **More preflop classes** in postflop keys. Multiplies every postflop key.

## Prerequisite: versioned abstractions (V3.0)

Today the centroids and baked tables live in one global directory
(`analysis/abstractions/`), and blueprints carry no abstraction fingerprint. v3
breaks that:
- A re-baked flop table has the **same centroid stamp** (the stamp hashes centroids
  + bins, not runouts). v2 would silently run on v3's table, with ~20% of hands
  re-bucketed.
- River K = 20 would give v2 river buckets 10–19 it never trained on.
- A live v2-vs-v3 A/B needs **both** abstractions loaded in one process.

So, first:
- A v3 abstraction directory, `analysis/abstractions/v3/`. v2's files stay in
  place.
- The table stamp also hashes the bake settings (`flop_runouts`).
- Every new blueprint gets an `abstraction_id` stamp: per-street centroid hashes
  plus table stamps.
- Readers resolve the abstraction dir from the blueprint and verify the stamp,
  failing closed on a mismatch.
- `PostflopV2` and the Rust `load_abstraction` take the dir per blueprint, not
  per process.
- An unstamped DB means "v2, global dir".

### V3.0 — done 2026-11-09

Implemented in `backend/bot/src/abstractions/abstraction_set.py`
(tests: `tests/test_abstraction_set.py`, 30 tests).

**The fingerprint.**
- `abstraction_id` = sha1 of the three centroid hashes plus the manifest's
  `flop_runouts`. It does not depend on whether the git-ignored tables are present.
- The root has no manifest; its default is `flop_runouts=200`.
- The root hashes to `LEGACY_ABSTRACTION_ID` (`742b8dd4…`). An unstamped blueprint
  means "the root as it was for v2", and is refused if the root ever stops hashing
  to that value.

**Where the stamp is written and checked.**
- `scripts/train_rust.py` stamps every new run with `abstraction_id`, and with the
  `abstraction` name for a named abstraction (`--abstraction NAME`).
- It refuses a resume across abstractions.
- It checks the tables' bake stamps before training (`verify_tables`); the Rust
  loader only checks centroids.
- The Python trainer doesn't stamp, which keeps its DBs byte-identical to the Rust
  parity fixtures. It refuses to resume a named abstraction.

**Bake and fit scripts.**
- `bake_postflop_table.py` takes `--abstraction`, stamps the flop table with
  `flop_runouts`, and takes the runout count from the manifest; a conflicting
  `--flop-runouts` is refused.
- `compute_postflop_buckets.py` takes `--abstraction`.
- `PostflopV2` and `CardAbstraction` take `abstraction_dir`. A table whose
  `flop_runouts` disagrees with the manifest is a hard error; a table from before
  the field existed counts as 200.

**Readers.**
- The evaluation tools resolve each blueprint's own directory: LBR, best response,
  `HeadToHeadMatch` and `CrossMatch` (per side, sharing one instance when the two
  agree), and the SPR probe.
- Serving fails closed on a named abstraction until V3.6: `BlueprintStrategy`
  (so also the river solver), the strategy API, and auto-resolution, which skips
  named-abstraction blueprints.

**Review fixes (2026-11-09).**
- **Named abstractions are strict.** A missing flop/turn table or an unstamped table
  is an error in `PostflopV2` and `verify_tables`. Before, a missing table fell back
  silently to 120-runout Monte Carlo bucketing, unlike the Rust trainer, which
  refuses.
- **One way to say "all runouts".** `flop_runouts` must be in [0, 1176), with 0
  meaning all.
- **`--flop-runouts` only binds flop bakes**, so turn and river bakes no longer
  reject it.
- **The training CLI's guards are tested end to end** through `train_rust.main()`.

**Carried forward from the review:**
- **V3.6: make the abstraction impossible to forget.** Abstraction safety is still
  opt-in per reader: `CardAbstraction()` defaults to the root. The blueprint should
  hand out its verified directory (e.g. `BlueprintDB.abstraction_dir`), and every
  serving reader (`GameSession`, `game_adapter`, the river solver's
  `_postflop_means`, the range tracker) should build its card abstraction from it.
- **Phase 7: the opponent-model guard compares only bucket counts** (read from the
  root). Models must record the `abstraction_id` they were fit on, and the guard
  must compare it. `fit_opponent_models.py` must bucket with the target blueprint's
  abstraction. Arm C keeps the flop/turn bucket counts but re-buckets ~20% of flop
  hands, so a count check can't catch it.
- **Known limit:** a root flop table baked exhaustively with the old CLI, before the
  `flop_runouts` field existed, would pass as a 200-runout v2 table. It can't be told
  apart from the file alone. Only matters if someone did that.

**Rust.** No change was needed: `load_abstraction(dir)` already takes a directory,
and its npz reader ignores the new `flop_runouts` member.

### V3.1 — done 2026-11-09

**The new arm.**
- `menu_mode='capped_pj'` is the capped arm plus a jam at the preflop raise cap.
  Facing the 4-bet, the legal actions are `['fold', 'call', 'allin']` when the shove
  commits more than a call. Facing the jam: fold/call.
- Postflop cap nodes, and live uncapped play, are unchanged.
- In the info-set key the jam is `a` as the 4th preflop char (e.g. `smma`).

**Python.**
- `PokerGame(preflop_cap_jam=...)` implements it on both legal-action paths (the
  history path and the state-threaded trainer path).
- `sizing.game_kwargs(menu_mode)` is now the one way to build an engine for an arm.
  The trainer, `GameSession`, LBR, best response and the SPR probe all use it.
- `postflop_menu_for` raises on an unknown arm; it used to fall back to control
  silently.
- `translation.postflop_grid_for('capped_pj')` gets the capped grid.

**Evaluation mirrors.** LBR's victim model and exploiter, and the head-to-head
match's legal actions, offer the cap jam too.

**Serving.** `BlueprintStrategy` and the strategy API refuse `capped_pj` until
V3.6 (`sizing.SERVING_MENU_MODES`).

**Rust.** `Engine.preflop_cap_jam` with the `"capped_pj"` mode. New parity
components:
- `engine_pj`: 3,000 walks, 1,558 cap-jam nodes;
- `mccfr_pj`: 600 iterations of capped_pj + spr5, a DB byte-identical to Python's.

Strict parity passes 42/42.

**Training CLIs.** `train_rust.py` and `run_blueprint_trainer.py` accept
`--menu-mode capped_pj`. A 400k-iteration smoke run (spr5, 4 workers) ran at about
22k it/s with a clean strategy shape.

**Tests:** `tests/test_preflop_cap_jam.py`, 47 tests.

### V3.2–V3.3 — done 2026-11-09

The v3 abstraction lives in `backend/bot/analysis/abstractions/v3/`,
`abstraction_id` `676871128fe5d97fe1acd32df732e273fb9ebf34`. Its contents:
- `abstraction.json`: `{"flop_runouts": 0}`.
- `postflop_centroids_flop.npz`, `postflop_centroids_turn.npz`: copies of the root
  (v2) centroids, K = 30 / 24.
- `postflop_centroids_river.npz`: new, K = 20.
- `postflop_table_flop.npz`: re-baked over all 1,176 runouts.
- `postflop_table_turn.npz`: copied from the root (already exhaustive).

The centroids and manifest are committed. The tables are git-ignored like the
root ones; `.gitignore` now covers `abstractions/*/postflop_table_*`.

**V3.2: exhaustive flop bake.** `bake_postflop_table.py --abstraction v3 --street
flop` took 33 min on one core, half the estimate.
- The table has 1,286,792 entries, the same situations as v2, and its stamp says
  `flop_runouts=0`. `verify_tables` accepts it.
- Against v2's 200-runout table, 21.1% of situations change bucket. In
  centroid-equity terms the mean move is 0.0115; 10.97% move more than 0.05, and
  the largest move is 0.232. This matches the audit's estimate of ~20%.
- An independent bucketing over all 1,176 runouts, on Q♠9♣3♣ plus 10 random
  flops (12,936 hands), agrees with the table on every hand, both through the
  Python lookup and through Rust's `Buckets(v3)`.

**V3.3: river K = 20.** Ran `compute_postflop_buckets.py --abstraction v3
--street river --buckets 20 --situations 10000 --opp-samples 0`.
- **The first fit left bucket 16 empty.** River features are single spikes on
  the 30 equity bins, so the random k-means start picked the same point twice.
  The copy never wins a hand, because ties go to the lower index.
  `kmeans_emd` now replaces repeated starting points (`_distinct_starts`). A
  start with no repeats is untouched and draws nothing from the RNG, so flop and
  turn fits are unchanged. The script also refuses to save a fit with an empty
  bucket. Test: `test_kmeans_starts_are_distinct`.
- All 20 buckets are occupied, holding 2–9% of hands each. The shares are lumpy
  because 30 bins give at most 30 distinct river values; more bins would be a
  separate change.
- 18 of the 20 buckets sit inside one v2 bucket; buckets 5 and 10 straddle two.
  The split is uneven: v2's buckets 8 and 9 (the strongest hands) split into four
  and three buckets.
- **Rust needs no change.** It reads K and bins from the centroid file and builds
  its numerator → bucket lookup from them. On 20,000 random river spots, Python
  and Rust agree on every bucket. No code anywhere assumes 10 river buckets.

**Smoke.** `train_rust.py --menu-mode capped_pj --key-format spr5 --abstraction
v3`, 400k iterations on 4 workers:
- ~25k it/s and 396k info sets;
- the DB is stamped `abstraction=v3` with the id above;
- river keys use all 20 buckets, and 12,168 cap-jam keys appear;
- the strategy-shape check is OK.

The full pytest suite passes (523).

## Steps

| Step | Work | Wall time | Gate |
|---|---|---|---|
| V3.0 | Versioned abstraction dir + `abstraction_id` stamp + reader checks (Python + Rust) | — | v2 serves byte-identically; a mismatched dir is refused |
| V3.1 | Preflop cap jam: `PokerGame` + Rust `engine.rs`, `menu_mode='capped_pj'`, `game_kwargs` factory, consumers moved over. New parity fixtures `engine_pj` + `mccfr_pj`; property tests (chip conservation, a jam closes betting, the cap jam only preflop) | — | parity strict passes; full pytest green |
| V3.2 | Exhaustive flop bake into `v3/` (same centroids). The turn table is already exhaustive (46 rivers) and is copied | ~1 h single-thread (measured ~2 s/board × 1,755 boards) | stamp records `flop_runouts=0`; spot-check vs the audit's exhaustive buckets |
| V3.3 | River centroids K = 20 (`--opp-samples 0`) into `v3/`; check the Rust river bucketing reads K from the file | minutes | occupancy sane; Python == Rust on random river spots |
| V3.4 | Train two arms, 150M each, snapshot every 10M, `--river-cache-boards 140000`: **A** = spr5 + cap jam on v2's abstraction (isolates change 1); **C** = all four changes (the candidate) | ~1–1.5 h per arm | strategy-shape probe clean (no "never fold" collapse; weak-hand preflop fold sane) |
| V3.5 | Evaluate (below) and pick the ship snapshot | ~4–5 h total, overnight | gates below |
| V3.6 | Serving path (Phase 6.2, extended), all behind the DB stamps; v2 behaviour unchanged: `BlueprintStrategy`, range tracker, river solver (projection, gadget CFV, guards), translation, `game_session`/`game_adapter` | — | tests incl. a BUG-027 replay through `GameSession`; v2 tests unchanged |

The V3.6 serving work covers:
- spr5 keys;
- the cap-jam node: a human 5-bet of any size maps to `a`, and the bot can choose
  `allin` there;
- the deep-raise guard covering only beyond-the-new-cap nodes;
- the v3 abstraction dir.

**Then Phase 7 (separate sign-off):** re-fit the opponent models on v3 keys
(exploit self-disables on a mismatch until then), `/api/abstractions` +
KeyExplorer, `assets-v3`, the Docker Rust stage, live A/B, cutover.

## Evaluation and gates

**Measures** (as in Phase 6 unless noted):
- **BUG-027 replay** (new, cheap). In the exact line:
  - BB's response to the 4-bet: the jam/fold split, and AA/KK no longer
    flat-calling ~98%;
  - the 4-bettor's range (does J9s still 4-bet?);
  - villain's flop lead-shove composition;
  - J9's equity and call EV against v3's own range.
  
  Same script as `scratchpad/bug027/range_ev/range_ev.py`, moved into
  `backend/bot/scripts/`.
- **Best response** (96 paired boards, seeds 42–45; ~45 min per blueprint).
  - Caveat: change 1 changes the game, so BR of v3 and BR of v2 measure different
    games. Compare **A vs SPR 100M** for the jam's effect and **C vs A** for the
    card changes (same tree). Treat v3 vs v2 as indicative only.
- **Head-to-head vs v2** (240k hands, AIVAT). The match engine uses v3's tree; v2,
  at nodes it never trained (facing a cap jam), plays the deployed live guard, as
  prod would.
- **Paired LBR** (10k hands).
- **Against a human-like opponent** (new). The fitted population human model
  (v2 keys) plays as a bot against v2 and against v3; compare v3 − v2. Look for an
  existing runner first (`scripts/measure_exploit.py`, `tests/run_maniac_live.py`)
  before writing one.

**Gates to recommend shipping C:**
1. **BUG-027 replay:**
   - BB jams or folds a meaningful share facing the 4-bet, and AA/KK mostly jam;
   - the flop shove range has materially more top pair+;
   - J9's call EV against v3's own range is ≤ 0, or the shove is no longer reached
     with J9's range.
2. **Card changes don't hurt:** BR of C ≤ BR of A, within noise.
3. **No regression:** head-to-head vs v2 and paired LBR not significantly worse
   (|t| < 2 or better).
4. **Against the human-like opponent:** v3 ≥ v2, within noise.

Gates 1 and 4 carry the decision. Gate 2 is the attribution check.

## Risks

- **The jam node lengthens every 4-bet line** and the 4-bet ranges change
  everywhere. Expect the preflop strategy to move more than SPR moved it. Watch
  the weak-hand preflop drift (74% → 40% fold over 200M in the spr5 run) with the
  strategy-shape probe and pick snapshots by evaluation, not iteration count.
- **Opponent-model re-fit (Phase 7):** human hands with a 5-bet were translated
  into beyond-cap keys under v2. Under v3 those map to trained `a` keys, so the
  re-fit has to re-derive keys from the raw recaps, not reuse stored v2 keys.
- **Two abstractions in one process** (V3.0) is the riskiest code change: the
  tables are ~126 MB for turn. The memory budget on the 0.25 vCPU Lightsail box
  needs checking before any live A/B (Phase 7).
- **Comparability.** Change 1 is an action-grid change, so BR/LBR against v2 are
  across different games. The A-vs-C split and the human-like-opponent measure
  exist to cover that.

## Decisions needed (sign-off)

1. The four changes as specified, with finer flop/turn K, OCHS and more preflop
   classes deferred?
2. V3.0 versioned abstractions first (served code, behind stamps, no change for
   v2)?
3. Two training arms (A, C), with the gates above deciding?
