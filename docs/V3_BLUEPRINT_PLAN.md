# v3 blueprint — plan and progress log (V3.0–V3.5 done; ship decision and V3.6 pending)

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

**Rebuilding the v3 tables** (git-ignored) on a fresh clone or another machine,
from `backend/bot/` with the root tables in place:

```bash
cp -L analysis/abstractions/postflop_table_turn.npz analysis/abstractions/v3/
python scripts/bake_postflop_table.py --abstraction v3 --street flop
python scripts/check_abstraction.py --abstraction v3
```

- The turn table is a plain copy because v3 keeps the root's turn centroids; its
  stamp checks against them. A turn re-bake takes hours.
- The flop bake takes ~33 min on one core.
- `check_abstraction.py` takes under a minute and must print `OK`.
- `train_on_cloud.sh` only builds the root abstraction. V3.4 trains on this
  machine; extend the script before training a named abstraction in the cloud.

**Review fixes (2026-11-09).**
- **k-means.** `kmeans_emd` now re-seeds a cluster that empties during the fit,
  on the point worst served by its own centroid (`_reseed`). Before, the cluster
  was left in place, empty.
  - The returned labels now come from a final assignment against the returned
    centroids. Before, they came from the last pass, which ran against the
    previous centroids, so the empty-bucket guard could pass a dead bucket.
  - Re-running the v3 river fit reproduces the committed centroids byte for
    byte.
- **Incomplete bakes.** The bake's periodic checkpoints are stamped `complete=0`,
  and the final save `complete=1`. Python's `check_table_stamp` and Rust's table
  loader refuse `complete=0`. Tables baked before the field existed count as
  complete. Each save also writes to `postflop_table_<street>.partial.npz` and
  renames it, so a crash mid-write can't leave a truncated table.
- **Parity.** New component `buckets_v3`: 20k flop, 20k turn and 2.4k river
  situations under v3, checked against Rust's `Buckets(v3)` (K = 30/24/20). The
  fixture records the abstraction id, so a re-baked dir shows up as stale.
  Strict parity passes 43/43, and the Python shim passes too.
- **Gate script.** `scripts/check_abstraction.py` (above) replaces the scratchpad
  checks.
- **Tests.** The distinct-starts test now forces a repeat. New tests cover
  re-seeding and final labels, and that a partial checkpoint is refused in
  Python and Rust.

### V3.4 — done 2026-11-10

Both arms ran with `train_rust.py --iterations 150000000 --workers 12
--merge-every 2000 --alpha 1.5 --gamma 1.0 --menu-mode capped_pj --key-format spr5
--snapshot-every 10000000 --river-cache-boards 140000 --seed 20261110`, arm C adding
`--abstraction v3`. `check_abstraction.py --abstraction v3` printed `OK` first.

| Arm | DB (`analysis/blueprints_rs/`) | Wall time | Info sets |
|---|---|---|---|
| A — spr5 + cap jam, v2 buckets | `blueprint_par_cappedpj_spr5_rs_20261110_004432_c58d21.db` | 52m39s (~47k it/s) | 416,453 |
| C — all four changes | `blueprint_par_cappedpj_spr5_v3_rs_20261110_013712_8d0598.db` | 53m27s (~47k it/s) | 443,759 |

Snapshots every 10M are in each DB's `_snaps/` dir. For scale, the SPR run had
404,066 info sets at 150M. The extra keys are the cap-jam nodes, plus the 20 river
buckets in C.

**Strategy shape (gate): clean at every checkpoint in both arms.** No "never fold"
collapse; BB folds 94% facing a 5BB open at 150M. The weak-hand open fold drifts
down with iterations, the same loosening the 100M and SPR runs showed:

| | 10M | 50M | 100M | 150M |
|---|---|---|---|---|
| A | 72% | 63% | 55% | 50% |
| C | 74% | 61% | 54% | 54% |
| SPR run (no jam) | — | — | 52% | 46% |

So V3.5 picks the ship snapshot by evaluation, not by iteration count.

**The cap jam is used.** A visit-weighted probe of every preflop node facing a
4-bet, and facing the 5-bet jam (scratchpad `v34_capjam_probe.py`; V3.5 moves the
full BUG-027 replay into `scripts/`):

| Facing a 4-bet: fold / call / jam | SPR 100M (no jam) | A 150M | C 150M |
|---|---|---|---|
| All hands | 40 / 60 / — | 55 / 26 / 18.5 | 54.5 / 27 / 18.5 |
| AA | 2 / 98 / — | 1 / 29 / 70 | 1 / 29 / 70 |
| KK | 3 / 97 / — | 2 / 15 / 83 | 2 / 15 / 83 |
| AKo | 4 / 96 / — | 4 / 7 / 90 | 4 / 7 / 90 |
| A5s | 17 / 83 / — | 27 / 39 / 34 | 26 / 40 / 35 |
| J9s | 18 / 82 / — | 27 / 54 / 19 | 28 / 53 / 19 |
| J9o | 28 / 72 / — | 53 / 39 / 8.5 | 54 / 37 / 9 |
| 72o | 77 / 23 / — | 87.5 / 7.5 / 5 | 87 / 8 / 5 |

- Facing the jam, premiums call 96–99%. J9s calls 54%, J9o 34%, 72o 13%
  (overall 60% fold).
- AA keeps a ~29% flat; the rest of the top range mostly jams.
- A and C are almost identical preflop. That is expected: the card changes are
  postflop.
- The numbers barely move from 50M to 150M (AA jam 76 → 70%, overall jam
  24 → 18.5%).

These are aggregates over every 4-bet line. Gate 1's per-line checks (the BUG-027
line, the flop lead-shove range, J9's call EV) are V3.5's replay.

### V3.5 — evaluated 2026-11-11, ship decision pending

Overnight run, 00:27–06:36. Scripts are in the session scratchpad `v35/`
(`run_v35.sh`, `h2h.py`, `lbr_paired.py`, `summary.py`); the BUG-027 replay moved
into the repo as `scripts/bug027_replay.py`.

**How the matches were played.**
- `h2h.py` is `run_match.py` (AIVAT) plus one rule. A capped-menu player (v2, SPR,
  or the human model) facing a capped_pj 5-bet jam has no trained key there, and
  `match.py` would coin-flip (folding AA half the time). Instead it plays the
  deployed guard's rule (`_facing_deep_raise_guard`, uninformed preflop read): call
  iff equity vs the top 20% of hands ≥ pot odds. That is what prod v2 does.
  Trained nodes are untouched.
- The **human model** is the fitted population model
  (`/Users/ron/Projects/AllIn/backend/bot/analysis/opponent_models`, v2 keys) as a
  player. Blueprint vs model only: no river solver, guards, or range tracker. Its
  play at cap-jam nodes is that guard rule, not observed behaviour, because v2-era
  humans never faced a cap jam.
- LBR and BR reuse the Phase 6 setups. A rerun of v2's paired LBR on today's code
  reproduced the Nov 9 per-hand results exactly, so v2's and SPR 100M's old BR
  runs (same boards) are reused.

**Results** (mbb/hand; A = spr5 + cap jam on v2 buckets, C = also the v3 buckets):

| | A 50M | A 100M | A 150M | C 50M | C 100M | C 150M | SPR 100M | v2 |
|---|---|---|---|---|---|---|---|---|
| H2H vs v2, 240k AIVAT (>0 = wins) | +67 ± 30 | +68 ± 30 | **+106 ± 30** | +45 ± 30 | **+128 ± 30** | **+105 ± 30** | −31 ± 33 | — |
| Vs human model, 240k AIVAT (bot's win) | — | +120 ± 38 | **+179 ± 38** | — | +142 ± 38 | **+186 ± 38** | −11 ± 41 | +135 ± 40 |
| … minus v2 | — | −15 ± 56 | +45 ± 55 | — | +7 ± 55 | +52 ± 55 | **−146 ± 57** | — |
| Paired LBR 10k, minus v2 (lower = better) | +125 ± 141 | −54 ± 175 | +3 ± 186 | +37 ± 146 | +25 ± 176 | +101 ± 184 | +185 ± 169 | — |
| BR, 96 paired boards (total) | — | **12,550** | 12,774 | — | 12,781 | 13,036 | 13,157 | 13,941 |

- **BR, paired.**
  - A 100M − SPR 100M = −607 ± 27 (−4.6%; as BB −7.2%): the jam's effect. BR of the
    capped_pj game vs the capped game measures slightly different games, so treat it
    as indicative.
  - C − A = +231 ± 42 at 100M and +262 ± 52 at 150M (+1.8 / +2.1%, t ≈ 5). C is
    better as SB (−2%) and worse as BB (+4.6 / +5.0%).
  - Versus v2: A 100M −10.0%, C 100M −8.3%, C 150M −6.5%.
  - 100M beats 150M in both arms (~+230), the same "later = looser = more
    exploitable" drift as the earlier runs.
- **C vs A head-to-head** (same tree): C wins +43 ± 30 at 100M and +75 ± 30 at
  150M (z 1.4 / 2.5).
- **Cap jams** happen in 1.3–1.7% of hands against v2. v2's guard called 55–60%
  of them.
- **SPR 100M (no jam) loses 146 ± 57 to v2 against the human model** (z −2.6),
  though it was neutral in H2H and LBR. Both jam arms recover that.

**BUG-027 replay** (`scripts/bug027_replay.py`; on v2 it reproduces the Nov 9
analysis: call EV +107.4 vs +106.9 chips):

| Exact line | v2 | SPR 100M | A 100M | A 150M | C 100M | C 150M |
|---|---|---|---|---|---|---|
| BB vs the 4-bet: fold / call / jam | 27 / 73 / – | 28 / 72 / – | 43 / 22 / 36 | 43 / 23 / 34 | 42 / 23 / 35 | 43 / 24 / 33 |
| AA jam / KK jam | – | – | 30 / 81 | 42 / 82 | 61 / 78 | 50 / 83 |
| Bot 4-bets here with J9s / J9o | 10 / 8 | 19 / 8 | 5 / 2 | 4 / 3 | 4 / 2 | 3 / 2 |
| BB flop lead-shove frequency | 6% | 25% | 25% | 25% | 26% | 27% |
| Top pair or better in the shove range | 6% | 10.5% | 18% | 23.5% | 14% | 17% |
| J9hh equity vs the shove (needs 0.39) | 0.66 | 0.63 | 0.60 | 0.57 | 0.64 | 0.60 |
| J9hh call EV, chips | +107 | +96 | +85 | +70 | +99 | +83 |

J9hh's equity against human-style shove ranges is unchanged: 0.20 against top
pair, sets and nut flush draws.

**Gates.**
1. **BUG-027 replay: partly.**
   - ✔ The BB now jams 33–36% and folds 42% facing the 4-bet, and KK+ mostly jams.
     AA keeps a 40–70% flat, a slow-play mix.
   - ✔ The shove range has 2–4× more top pair+ than v2.
   - ✘ J9's call is still +EV against the blueprint's own shove range, so the bot
     still calls (~99%). But J9 reaches the spot 2–6× less often, because it
     4-bets 2–5% here instead of 8–19%.
   - The remaining gap is humans shoving far narrower than equilibrium. That is the
     exploitation layer's job (Phase 7 re-fit); no blueprint fixes it.
2. **Card changes don't hurt BR: ✘.** C is ~2% more exploitable than A at equal
   iterations (worse as BB, better as SB), although C beats A head-to-head. The v3
   buckets are not a clear improvement.
3. **No regression vs v2: ✔.** Every candidate wins H2H (A 150M and C 100M/150M at
   z > 3.5), and LBR is neutral everywhere (|z| < 1).
4. **Vs the human model, v3 ≥ v2 within noise: ✔.** The best are C 150M (+52 ± 55)
   and A 150M (+45 ± 55); the 100M snapshots are level with v2.

**Reading.**
- The jam (change 1) carries the measurable gain: BR −4.6% vs SPR, H2H +100 over
  SPR's result, and SPR's human-model regression reversed.
- The v3 card changes (2 and 3) win H2H against A but lose BR. They also cost a
  second abstraction dir and a 33-min bake, and need the bucket-cache fix below.
- Snapshot: 100M wins BR in both arms, while 150M scores higher against the human
  model (+45 to +60, about 1σ) and against v2 for A. The two signals disagree, and
  neither is decisive.

**Review (2026-11-11, three reviewers: abstraction mixing, eval harness,
training path) — the H2H, human-model and LBR numbers above are SUPERSEDED**
by a re-run on the fixed tools (results to follow). The best-response and
BUG-027 numbers stand. Fixed in 07f73ad:
- **`match.py` drifted from the engine.** It kept sized bets the stack can't
  cover instead of the engine's single `allin` (~11% of decisions lost mass on
  the trained jam, e.g. `..._turn_r0_k` offered `[check, …]` not
  `[check, allin]`), and it skipped the min-bet / min-raise filters, so v2 bet 1
  chip into limped pots, putting spr5 players on keys their engine can't
  produce (2.9% of A-vs-v2 hands, 8.5% vs the human model). Matches now run on
  each player's own `PokerGame` state machine (`state_legal_actions` /
  `state_action_cost` / `advance_node_state`), and an untrained key falls back
  to prod's passive fold/call (not uniform). New test `test_match_engine.py`
  replays every offered action against `get_legal_actions`.
- **LBR was invalid (every value negative).** The exploiter jammed into faced-jam
  keys the blueprint never trained, priced them as "the bot always folds", and
  the simulated bot then played a 50/50 coin flip there. Now:
  - LBR jams only where the blueprint's own tree has an all-in;
  - the simulated bot's legal actions apply the engine's stack clamp and
    min-bet filter;
  - the bot answers an off-grid LBR bet with prod's bracket blend, as LBR prices
    it (an untrained bracket folds; an untrained faced jam is the coin flip).
  - 2k-hand smoke: v2 +2,144, SPR 100M +564, A 150M +547 (all positive now).
- **AIVAT** replayed an spr5 B's range with v2-format keys (missing every
  flop/turn lookup). The point estimates were unbiased; only C-vs-A variance
  suffered.
- **The range tracker's bucket caches** are now keyed by abstraction dir (they
  mixed v2/v3 buckets in one process; the replay no longer clears them by hand).
- **Smaller latent fixes:**
  - `RiverSubgameSolver` buckets and ranks with the blueprint's own abstraction.
  - The PyPokerEngine `Player` refuses spr5 / named-abstraction DBs.
- **Left for V3.6 / Phase 7** (behind the serving guards, which fail closed):
  - threading spr5 keys through `GameSession`, the range tracker's
    `_action_matrix`, `tilt.py` and `replay.py`;
  - the `/api/abstractions` bucket counts;
  - the opponent-model fit and build scripts using root buckets. Its consistency
    check stays on the root centroids on purpose: the models were fit on them,
    so it should keep rejecting v3 until the models carry an abstraction id.
- **Training path: no bug.** Key packing fits K=20, all 20 river buckets are
  distinct keys, Rust == Python bucketing under v3, and the cap jam is correct.

**Fixed in 07f73ad (was "found for V3.6"):** `range_tracker`'s postflop bucket cache
was process-global and keyed by board only, so one process tracking ranges under two
abstractions would reuse the first one's buckets. It is now keyed by
(abstraction dir, board); the replay no longer clears it by hand.

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
