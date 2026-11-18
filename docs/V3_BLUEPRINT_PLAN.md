# v3 blueprint — plan and progress log (V3.0–V3.6, V3.10, V3.11 and V3.15 done, base arm C300; V3.7–V3.9 built, gates deferred; next: train arm D (V3.16 history keys), then control C-long when there is time; V3.12 pruning built but not trained; then V3.13 DBR, V3.14; see "Next steps")

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
- Serving failed closed on a named abstraction until V3.6, which lifted it (below).

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

**Serving.** `BlueprintStrategy` and the strategy API refused `capped_pj` until
V3.6, which added it to `sizing.SERVING_MENU_MODES`.

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

### V3.5 re-run on the fixed harness (2026-11-12/13) — complete

Overnight run from 01:30 (`analysis/blueprints_rs/eval_runs/20261112/run_all.sh`,
git-ignored; `resume.sh` there reruns only the jobs without a result). BR finished;
the queue was paused at 10:44 with 31 of 104 match/LBR jobs left and resumed on
11-13 01:30 with A300/C300 vs v2 added (43 jobs, done 03:16). Both arms were resumed
from 150M to 300M before this run.

**Best response, the iteration test** (`scripts/br_attr.py`, 24 boards per seed, so a
seed is one set of boards and comparisons are paired by seed). v2 seed 42 gave
14,027.7, reproducing the Nov 9 run (14,028) on today's code.

| C − A (mbb/hand) | seed 42 | seed 43 | seed 44 | seed 45 |
|---|---|---|---|---|
| 150M | +106 | +333 | | |
| 200M | +107 | +483 | | |
| 300M | +80 | +555 | +345 | +430 |

| Total BR, seed 42 / 43 | 150M | 200M | 300M |
|---|---|---|---|
| A | 12,947 / 11,819 | 13,187 / 11,971 | 13,553 / 12,246 |
| C | 13,053 / 12,152 | 13,294 / 12,454 | 13,633 / 12,801 |

- **The "C needs more iterations" theory is refuted.** The gap does not shrink with
  training (mean +353 at 300M over four seeds). It sits in the BB seat
  (+250 to +720 per seed) and in preflop gain (street 0: +115 to +470); C is
  slightly better as SB (−70 to −175). The preflop buckets are identical, so this is
  the BB's preflop defence becoming easier to exploit given C's postflop play: the
  abstraction-pathology pattern (a finer abstraction need not be less exploitable in
  the real game).
- **Both arms get more exploitable from 150M to 300M** (+430 to +650 per seed). The
  rise is on the turn and river (A seed 42: turn +236, river +192); preflop is flat.
  Same "BR bottoms out, then rises" as the 90M Python retrain. Pick the snapshot by
  play, not BR.

**Matches** (AIVAT, 40k hands per seed, 6 seeds = 240k hands; mbb/hand, >0 = the
first-named bot wins; ± is the pooled standard error). These replace the superseded
V3.5 table; the human-model scale changed because the simulated human now jams at
measured human rates.

| | v2 | SPR 100M | A100 | A150 | A300 | C100 | C150 | C300 |
|---|---|---|---|---|---|---|---|---|
| vs human model | +481 ± 41 | +592 ± 39 | +601 ± 37 | +631 ± 37 | +659 ± 37 | +611 ± 37 | +658 ± 37 | +680 ± 37 |
| vs v2 | — | +186 ± 34 | +209 ± 32 | +245 ± 32 | +280 ± 32 | +233 ± 32 | +268 ± 32 | **+381 ± 31** |
| C vs A, same iterations | | | | | | +46 ± 31 | +24 ± 30 | **+79 ± 30** |

**Paired LBR** (`lbr_paired.py`, seed 777, 10k hands, the same deals for every
blueprint; mbb/hand, lower = less exploitable):

| | v2 | SPR 100M | A100 | A150 | A300 | C100 | C150 | C300 |
|---|---|---|---|---|---|---|---|---|
| LBR | 1,164 | 453 | 337 | 673 | 710 | 383 | 666 | 599 |

Each value is ±~300 and a paired difference ±~200, so only v3 vs v2 is resolved
(A100 −827 ± 288, C300 −565 ± 297, A300 −454 ± 297, SPR 100M −710 ± 255). C − A is
+47 / −7 / −111 (± ~200) at 100M / 150M / 300M: no difference.

- Every new blueprint beats v2 clearly: +186 to +381 head-to-head, +110 to +200 more
  than v2 against the human model, and roughly half v2's LBR.
- Play keeps improving to 300M in both arms (vs v2: A +209 → +280, C +233 → +381;
  vs the human model +60 to +70) even as BR rises.
- **At 300M C beats A**: +79 ± 30 head-to-head (z = 2.6, five of six seeds positive)
  and +101 ± 45 more than A against v2 (z = 2.3). Against the human model C300 leads
  A300 by +21 ± 52 (not significant). At 100M and 150M the edge is +24 to +46, not
  significant.

**Reading.** Against the gates (gates 1 and 4 carry the decision, gate 2 is the
attribution check):
- Gate 1 (BUG-027 replay) and gate 4 (vs the human model ≥ v2) pass for both arms;
  gate 3 (no regression vs v2, H2H and LBR) passes for both.
- Gate 2 fails for C: its BR is ~2.7% worse than A's (+353 mean at 300M), in the BB
  seat and preflop.
- The head-to-head edge C gained by 300M is the first significant difference between
  the arms, and it shows up twice (directly and against v2). Against the human-like
  opponent, the measure closest to prod, the arms are level.
- C's cost: a second abstraction dir and its baked tables in the assets, the
  two-abstraction memory risk, and opponent models fit on v3 buckets (V3.10 refits
  them on v3 keys either way).

**Recommendation: C at 300M**, pending sign-off — the stronger player wherever the
measures resolve, at the price of a worse BR lower bound. A at 300M is the fallback
if the BR gap or the extra assets weigh more. Snapshot: 300M in either arm (the best
play measured; BR is not used to pick it).

## Next steps (decided 2026-11-16, revised after a three-agent review)

The user chose three training changes: history tags in the keys (V3.16, from V3.15),
pruning (V3.12) and a population-biased blueprint (V3.13). A three-agent review the same
day found blockers in the first draft (a pruning rule that biased training, no room in
the Rust packed key, a silent opponent-model mis-parse, a DBR design that reused the
bot's own table) and an under-sized training budget; this section is the revised plan.

### The arms, and why there are several

Each arm changes one thing against another arm, so any difference in play can be
attributed to that change, and a change that hurts can be dropped without losing the
others.

**Decided 2026-11-17 (after the step-3 calibration):** train **D first**, then
**C-long when there is time**; **E is dropped**. The user did not want ~36 h of
laptop training mostly for experiments. Pruning measured only 7–15% faster and carries
a slow-comeback risk for actions it skips, so a whole 14 h arm is not worth it. The
pruning code stays in the trainer, off by default, for a future run that needs the
speed.

| Arm | What | Compared with | Answers |
|---|---|---|---|
| **D** (first) | C + the V3.16 keys, trained from scratch | C300 first (interim), then C-long at matching snapshots | Do the new keys help? |
| **C-long** (later, when there is time) | C300 resumed (same config, same keys) to D's iteration count | — (the control) | Is D's gain just more training? C was still improving at 300M (vs v2 +233 → +381 from 150M to 300M), so C-long is also a ship candidate. |
| ~~E~~ (dropped 2026-11-17) | D + pruning | D at equal wall time | Does pruning help? Not trained: see the decision above. |
| **DBR** | forks of the base winner (D, or C300/C-long if D loses), one per P_max | that winner, on held-out players | Does baking in the population model help? |

### Order

| # | Step | Machine cost |
|---|---|---|
| 1 | V3.16 code: key format `spr5a`, Python + Rust + parity | light |
| 2 | V3.12 code: pruning in the Rust trainer, off by default, bit-identical when off | light |
| 3 | Short `spr5a` runs: measure it/s on the bigger table; calibrate the prune threshold; read the pruning telemetry — **done 2026-11-17** (c=0.01, n_min=100; ~30k it/s) | ~1 h, plugged in |
| 4a | **Done 2026-11-18.** Train **D** to ~1.6B iterations (~15 h at the measured ~30k it/s; see "Step 3 — calibration results"), recipe below. Resumable: run whenever the laptop is idle. Optional early look: at a mid snapshot (e.g. 800M, ~7–8 h in) screen D against C300; if D is clearly behind, stop and rethink before spending the rest | heavy, idle laptop |
| 4b | Later, when there is time: train **C-long** from 300M to D's final iteration count (~7.7 h at ~47k it/s) | heavy, idle laptop |
| 5 | Evaluate (see "Evaluation for these arms"): first D vs C300 (interim; confounded by D's extra iterations), then D vs C-long once 4b exists; screen, pick, confirm | heavy, a few hours per comparison |
| 6 | Refit and restamp the opponent models and the jam read on `spr5a` (needed before any exploit-on evaluation of D) | light |
| 7 | V3.13 code (two tables, fork mode), then DBR forks per P_max | code days; each fork a few hours |
| 8 | V3.14: final pick among C300 / C-long / D / DBR, hardening, then Phase 7 (separate sign-off) | — |

Disk: snapshots every 100M at ~430 MB each are ~7 GB for D (C-long's are ~130 MB
each); 731 GB were free on 2026-11-17.

**Run recipes (step 4).** Both arms use C's settings so that only the named change
differs: `--workers 12 --merge-every 2000 --alpha 1.5 --gamma 1.0 --menu-mode capped_pj
--abstraction v3 --river-cache-boards 140000 --snapshot-every 100000000 --seed 20261110`
(V3.4's recipe; per-iteration deals come from `iteration_rng(seed, iteration)`, so arms
with the same seed see the same deal at the same iteration, and pruning's coin uses its
own RNG).
- **D (4a, first):** a new run, `--key-format spr5a`, `--iterations 1600000000
  --checkpoint-every 10000000` (as C300's run, so D's snapshots land on the same
  iteration counts as C300's and C-long's; the trainer's partial-round warning is
  expected, as it was for C), no `--prune`. To pause, stop it with Ctrl-C (it checkpoints first) and later rerun the same
  command with `--resume <its training DB name>` and the remaining iterations.
- **C-long (4b, later):** copy C's training DB (`blueprint_par_cappedpj_spr5_v3_rs_20261110_013712_8d0598.db`,
  not a snapshot) and `--resume` the copy with `--key-format spr5` up to D's iteration
  count, so C300 itself stays untouched.
- **E (dropped):** would have been D's command plus `--prune --prune-c 0.01
  --prune-min-visits 100 --prune-probability 0.95` (step 3's pick); kept here only in
  case pruning is revived.
- Stopping and resuming any arm with the same command (plus `--resume`) continues the
  same deal stream. It is not guaranteed bit-identical to an uninterrupted run (merge
  rounds can fall on different iterations after a resume), which does not matter for the
  comparison.

**Step 4a progress.** D night 1 (2026-11-17) ran 0→800M in 7 h 36 min (~29k it/s);
the table holds 1,448,370 info sets at 800M, and the strategy-shape probe is clean (weak
opens fold 54%, BB folds 96% to a 5BB open). The optional 800M screen (2026-11-18, six
seeds × 40k hands each, AIVAT; seeds 801–806):

| Match | mbb/hand | Per-seed |
|---|---|---|
| D800 vs C300 head-to-head | **+80 ± 30** | 44, 89, 54, 77, 152, 63 |
| D800 vs the human model | +664 ± 36 | 754, 570, 577, 712, 621, 751 |
| C300 vs the human model (same seeds) | +733 ± 36 | 837, 705, 742, 627, 756, 731 |
| D minus C300 vs the human model | −69 ± 51 (unpaired SE) | — |

D800 beats C300 head-to-head (all six seeds positive), and its edge against the human
model is lower by an amount that is not significant (~1.4 SE). Both readings are
confounded by D800 having 2.7× C300's iterations, so they only say D is not clearly
behind; the clean comparison is D vs C-long (4b). The screen passed the pre-set gate
(stop only if D trails by more than 2 SE on either reading), and night 2 resumed D from
800M toward 1.6B at 02:21 on 2026-11-18. Note for step 5: the human-model edge is the
reading to watch; if D at 1.6B still trails C there, check whether the `spr5a` keys cost
exploit-free value against human-like play before picking D.

**Step 4a done (2026-11-18).** D reached 1.6B at 15:51 (paused once at 1.452B and
resumed; shape probe unchanged: weak opens fold 54%, BB folds 96% to a 5BB open). The
1.6B screen (same design, fresh seeds 1601–1606):

| Match | mbb/hand | Per-seed |
|---|---|---|
| D1.6B vs C300 head-to-head | **+114 ± 29** | 212, 59, 83, 80, 87, 161 |
| D1.6B vs the human model | +717 ± 35 | 676, 720, 629, 697, 802, 775 |
| C300 vs the human model (same seeds) | +709 ± 37 | 829, 752, 566, 728, 636, 746 |
| D minus C300 vs the human model | +7 ± 51 (unpaired SE) | — |

D still wins head-to-head on every seed. Against the human model the 800M gap (−69 ± 51)
is gone, and part of it was noise in C300's own number (+733 on the 800M seeds, +709
here). So D is not behind on either reading. This is still the interim comparison (D has
5.3× C300's iterations); the clean test of the `spr5a` keys is D vs C-long (4b).

### V3.16 — history tags in the keys (`spr5a`)

- **Keys:**
  - flop: + preflop's last aggressor (none/ip/oop);
  - turn: + the flop's and preflop's last aggressors (9 values);
  - river: + the SPR bucket (river keys have none today), no tag;
  - preflop: unchanged.
  The initiative token on the turn is the fallback if step 3's it/s makes the 5.4× turn
  split impractical (it halves the turn split for ~85% of the gain; V3.15).
- **Why not river tags:** V3.15. SPR alone fits the turn's budget; any river tag would set
  the run length (8.9–22×) for a street the live solver re-solves with full history.
  Revisit after D is measured.
- **Size:** effective splits 1.75 (flop), 5.4 (turn), 3.5 (river) → ~1.45M info sets
  (flop ~340k, turn ~890k, river ~200k, preflop 30k), ~3.3× C300; the DB ~430 MB vs
  130 MB.
- **Training budget:** the largest split sets it: 5.4 × 300M ≈ 1.6B iterations to give
  turn keys C300's coverage. 47k it/s was measured on a 444k-key table; measure on
  `spr5a` before quoting hours.
- **Key string:** tags sit before the pattern, `{c}_{s}_{pos}_{street}_r{n}_t{tags}_{pattern}`
  (e.g. `pf_3_12_oop_turn_r2_tio_km`, tags as n/i/o per street). Off-grid translation
  (`key[:-1]+c`: `bot_strategy.py`, `lbr.py`, `game_session.py`) and the tilt's
  `rsplit('_', 1)` keep working because the pattern stays last.
- **"Last aggressor"** is defined once, on pattern characters (`smlxo2a`; blinds are not
  aggression; limped/checked = none), and shared by Python and Rust (Rust already has the
  pattern in `Node.pattern`). An `allin` is only offered when it costs more than a call
  and stacks are equal, so there is no call-for-less all-in. The tags are constant within
  a street, like SPR: compute one street-context value at street start and pass it
  wherever `spr` goes today.
- **Python:**
  - `make_info_set_key(..., key_format=, tags=)` raises when an `spr5a` key is built
    without tags (today the format is inferred from `spr is None`, so a caller that forgot
    the tags would silently build spr5-shaped keys, miss every lookup and play the
    passive fallback);
  - `GameSession` keeps each street's pattern (`d['history']` resets per street), with a
    `from_dict` default rebuilt from `action_log` for in-flight sessions; fix the
    `action_log` fallback that can log fold/check for a custom bet, and make `replay.py`
    count aggression the same way (it drops unmapped actions but counts `custom`);
  - every consumer: `GameSession`/`game_adapter`, `public_state['spr']` (read by
    `tilt.py`), the range tracker's `_action_matrix`, `tilt.py`, `replay.py`, AIVAT
    (`aivat.py`), the BR walker and its key cache (cache key must include the tags),
    `lbr.py`, `match.py` (`act`), `cross_match.py` (`key_for`), `h2h_eval.py`,
    `bug027_replay.py`, `export_rust_fixtures.py`, `train_rust.py` (choices),
    `measure_street_history.py`, the Python trainer recursion; the hard-coded
    `('v2', 'spr5')` gates in `lbr.py`, `match.py`, `cross_match.py`, `best_response.py`;
    the generic gates (`bot_strategy.py`, `strategy_api.py`, `config.py`); the dead turn
    projection gets a `require_key_format` guard.
- **Opponent model:** `_levels` (`opponent_model.py`) strips only the 2-character `rN`
  token and would read a tag as the betting pattern, silently (the exploit read, live
  last-N and DBR's target would all be wrong). Parse the pattern as the last token, with a
  test per key format; consider a facing+tags rung between `+facing` and the full key;
  re-tune alpha (turn full keys split 5.4× on ~17.5k human decisions).
- **Jam read:** restamp (its stamp includes `key_format`); its context keeps
  `spr_bucket('spr5')`.
- **Rust:** the packed u64 key is full (pattern 40 + strength 10 + bucket 8 + seat 1 +
  street 2 + SPR 3 bits). Cut the strength field to 6 bits (largest K is 30) and use the
  freed bits for the tags and river SPR; packed keys are never persisted (`db.rs` stores
  the key string), so the layout can change. Update `parse_key`, `unpack_key`/
  `key_string`, `KeyFormat::parse`. Parity fixtures for `spr5a` including a limped
  check-down, a checked-through flop, the cap jam and (Python only) uncapped live
  re-raises; v2 and spr5 string parity unchanged.

### V3.12 — pruning (design revised after review)

- **Why:** with merged keys, training explores hopeless lines and lets them update shared
  downstream keys at full weight. Pruning mostly stops exploring lines that have been
  hopeless for a long time. Pluribus credits it with roughly halving training time,
  which matters with ~3.3× the keys. Honest expectation: the main gain is probably speed.
  A low-frequency mixed action (J9o 4-betting 2% of the time) still has positive regret,
  so it is never pruned and still updates downstream keys at full weight; the tags help
  that case more.
- **Rule:** at a traverser node, action a is skipped only if all hold:
  1. its CFR+ regret is 0 (current probability 0) and the node's positive-regret sum is
     > 0. Skipping it leaves this visit's node value and every sibling's regret update
     exactly as CFR+ computes them (a zero-probability action contributes nothing to the
     node value). It is NOT equivalent to unpruned CFR+ overall: the skipped action's own
     increment is lost, and that increment can be positive (an action at zero regret can
     turn profitable), so its re-activation waits for a probe visit. The shadow
     threshold and the 5% probes make this an approximate experiment, which is why E
     keeps its own quality gates (review, 2026-11-16);
  2. its shadow score S(a), divided by the key's discounted raw-visit count N, is below
     −c × pot at that node, with N ≥ n_min (per-key warm-up instead of a global one);
  3. at least one action is still explored;
  4. a coin from a separate RNG stream lands on "prune" (95%), so pruning off stays
     bit-identical and D and E see the same deals.
  Never on the river, never fold, never a call that goes straight to showdown (an all-in
  call). Calls that close a non-final street can be pruned (flatting 3-bets and 4-bets
  with trash is a main route into pooled downstream keys).
- **Shadow score:** the same increments and the same discount as the regrets (applied to
  pruned actions too, so R ≥ S holds), floored just below the prune threshold (as
  Pluribus: −310M against −300M) so recovery time is bounded. Opponent nodes are never
  pruned.
- **Parallel and resume:** workers keep base S plus raw increments; the master applies
  the decay and floor once per round in the same `merged()` path (the floored regret path
  is unchanged, so the reverted "Fix #2" failure cannot recur). The discount clock ticks
  once per merge round, hence the per-key N above. Checkpoint/resume carry S and N; the
  resume guard refuses a pruning mismatch; the prune parameters are stamped in
  `training_metadata`; S and N are stripped from served snapshots.
- **Tests:** pruning off is bit-identical to today; the Python twin and parity cover
  pruning off only.
- **Calibration and telemetry (step 3, on `spr5a`):** choose c and n_min from short runs;
  log dormant (key, action) pairs and re-activations (zero re-activations is an alarm);
  carry the traverser's own reach in `Node` and report the share of postflop regret-update
  mass from paths with own reach below 1e-3, pruning off vs on (the direct test of the
  motivation).
- **Gates:** strategy-shape probe (with the postflop checks below); BUG-027 replay; it/s
  vs D; H2H vs D at equal wall time (primary) and no worse at equal iterations; LBR no
  worse.

### V3.16 / V3.12 — implementation 2026-11-16 (calibration and training pending)

The code for steps 1–2 is implemented; this does **not** mark the training/evaluation
steps or either arm's quality gates passed. C300 remains the selected base; production
and its exploitation settings are unchanged.

- `spr5a` is explicit in Python and Rust. Flop keys carry `t{preflop aggressor}`;
  turn keys carry `t{flop aggressor}{preflop aggressor}`; river keys carry SPR with
  no tag. n/i/o means no aggressor / IP / OOP. Packed Rust keys use 6 strength bits
  and 4 history bits. Existing v2/spr5 key strings and training fixtures are unchanged.
- Live sessions persist completed street patterns and reconstruct them from the action
  log for older sessions. The range trackers, tilt, recap fitter, match/LBR/AIVAT readers,
  river blueprint projection, history-measurement script and Python oracle thread the
  same context. Missing `spr5a` tags fail loudly. Legacy turn-solving paths refuse formats
  they cannot project. Opponent-model backoff reads the final pattern token, not the tag.
- Rust pruning is opt-in (`--prune`). Defaults `--prune-c 0.01`,
  `--prune-min-visits 100`, `--prune-probability 0.95` are **uncalibrated starting
  values**, not a selected E configuration. Only zero-probability traverser actions may
  be skipped; never river, fold, or a showdown call. A separate RNG leaves deal generation
  untouched. Shadow scores and raw visits use the regret discount; workers carry raw
  increments and the master floors once per merge. The recovery floor uses the largest
  pot observed at a shared key, conservatively retaining its negative evidence.
- Resumable checkpoints include an atomic `pruning_state` table and exact pruning
  parameter/version metadata. A changed configuration or missing state is refused.
  Served snapshots strip the table and are marked as such; resume an E run from its
  training DB, not its served snapshot.
- Progress reports skipped/explored actions, dormant/reactivation events, and absolute
  postflop regret-update mass on own-reach < 1e-3 paths (also collected with pruning off).
  Checkpoints print active dormant-pair counts and a bounded key/action/score/visit sample.
  Counters describe the current process; the per-key shadow scores and visits persist.
- Lightweight verification covers new-format Python/Rust fixture parity, packed-key
  collisions, live roundtrips and recap replay, mixed-format matches/LBR, disabled-pruning
  value identity, actual skips/reactivations, shadow bounds, parallel merging, exact
  checkpoint continuation and stripped snapshots. The expensive BR smoke is excluded
  from the broad backend check because BR work remains parked.

Verification: 623 backend tests passed, one skipped, and the expensive BR smoke was
excluded; subsequent focused history/projection checks passed (8 tests). Rust: 40
unit tests and 51 parity/robustness tests passed. A 1,000-iteration, 2-worker `spr5a`
run on the real v3 tables, with deliberately aggressive **test-only** pruning settings
(c=1e-8, n_min=1), completed checkpointing and exercised skips and reactivations. This
is a functional smoke, not a throughput calibration or a strategy-quality result.

Next is step 3, with approval before the ~hour-long local calibration. Do not start the
C-long/D/E overnight runs or quote their speed until that calibration is measured.
The new models and jam read still need refitting/restamping before exploit-on D/E
quality evaluation (step 6). No production deployment, base/menu switch, commit or push
is part of this implementation.

### Step 3 — calibration results (2026-11-16/17)

Four sequential 24M-iteration runs on the laptop (AC power), each with the step-4 recipe
(`spr5a`, seed 20261110, 12 workers, merge-every 2000), checkpoint every 12M. Output in
`backend/bot/analysis/blueprints_rs/calib_20261116/` (git-ignored, 2.3 GB, deletable).

| Run | Wall time | it/s (late, 20–24M) | Info sets | Skipped / explored actions | Dormant / re-activated events | Postflop update mass at own reach < 1e-3 | Shape probe |
|---|---|---|---|---|---|---|---|
| D-probe (no pruning) | 13m21s | ~29–30k | 1,428,474 | 0 / 7.88B | 0 / 0 | 71.1% | OK |
| E, c=0.01, n_min=100 | 12m27s | ~33–35k | 1,423,317 | 468M / 4.16B | 63.3M / 14.4M | 59.9% | OK |
| E, c=0.05, n_min=100 | 13m25s | ~32k | 1,424,523 | 394M / 4.85B | 69.3M / 14.8M | 62.4% | OK |
| E, c=0.2, n_min=100 | 16m35s | ~24–25k | 1,427,152 | 216M / 6.38B | 61.9M / 15.3M | 66.9% | OK |

- **Speed on the bigger table:** ~30k it/s without pruning (C300's 47k was on a 444k-key
  table). 1.43M info sets, as predicted (~1.45M). A 3M rerun right after the batch, on
  the warm machine, ran at ~31k, and `pmset` recorded no thermal or CPU-power limit, so
  the runs are comparable and heat did not cause c=0.2's slowdown.
- **Pruning overhead is real:** c=0.2 explores fewer actions than no pruning but runs
  ~20% slower. The separate per-key prune hash table and shadow updates cost more than
  its few skips save. Only c=0.01 skips enough to come out ahead: 7% faster over the
  whole 24M, ~15% faster in the last stretch. Its skip share is still rising (5% of
  actions at 3M, 13% in the last 3M interval), so the gain should grow over a long run.
  That is still far from Pluribus's ~2×.
- **Telemetry is healthy:** re-activations are nonzero in every run (no alarm). Dormant
  pairs sit on clearly losing raises and calls (shadow per visit is about −1 to −30
  chips). The shape probe passes in all four runs.
- **The motivation, measured:** without pruning, 71% of postflop regret-update mass comes
  from paths where the traverser's own reach is below 1e-3. c=0.01 cuts that to 60%. The
  rest is low-frequency mixed actions, which pruning cannot touch (expected; see "Why"
  above).
- **Selected E configuration (E later dropped, see "The arms"):** `--prune-c 0.01
  --prune-min-visits 100 --prune-probability 0.95`. n_min was not varied, and a c below 0.01 was not tried. This
  is a speed calibration only; E's quality gates still decide.
- **Budget at measured speed:** 1.6B iterations → D ≈ 15 h, E ≈ 13–14 h (less if
  the skip share keeps growing), C-long (spr5, 444k keys, ~47k it/s) ≈ 7.7 h from 300M.
  That is ~36 h of idle laptop in total, resumable. Both aggressor tags stay in;
  the initiative-token fallback is not needed at 15 h per arm.
- **Possible speed-up if pruning is revived (not needed now E is dropped):** keep each key's prune state inside its
  info-set node instead of a second `FxHashMap`. That removes a hash lookup per traverser
  node and should recover most of the overhead seen at c=0.2. It matters because E is
  judged against D at equal wall time.

### V3.13 — population-biased (DBR) blueprint (design revised after review)

- **Two tables.** `pack_key` holds the seat but no role, and one table serves both seats,
  so "the opponent's own CFR strategy" would be the bot's served strategy. Add a role bit
  (bot / restricted opponent): four traversal combinations (bot as SB or BB × who
  traverses). At an opponent visit the action probabilities are
  P_conf·model + (1 − P_conf)·free; regret matching updates only the free part; the
  mixture's value passes up the tree; only the bot table's average strategy is served.
  About 2× training memory.
- **The model inside training:** P_conf = P_max·n/(s + n) with n = human decisions at the
  full key; drop the blueprint rung from the model so P_conf alone shrinks toward the free
  strategy (otherwise thin keys double-shrink toward the frozen base); a key with no model
  mass gets P_conf = 0, never the uniform fallback; the model's prior is the frozen base,
  never the table being trained; precompute the model per packed key and backoff level in
  Rust.
- **Fork, not resume:** at the fork zero the strategy sums and `strategy_visit_count` and
  reset the regret clocks, or a few hours of DBR barely moves a ~1.6B-iteration average;
  an explicit fork mode (the resume guard checks the clock) with the fork stamped in the
  DB.
- **Held-out evaluation:** leave-one-player-out or a time split (15 players); rebuild the
  population model, jam read and tilt from the fit set only; the held-out evaluation model
  uses the frozen base as its prior for every arm, never the DBR blueprint.
- **Cost:** paired LBR plus H2H against the base, not the repo's BR (board-clairvoyant,
  not valid exploitability). Evaluate with the live layers (tilt, exploit, river solve)
  both on and off, since the tilt assumes a GTO-ish base and may exploit twice. The safe
  gadget's "no worse than the blueprint" weakens to "no worse than DBR": anchor it to the
  base blueprint, or re-validate.
- **Sweep:** P_max 0.25 / 0.5 / 0.75, each fork a few hours (set from step 3's it/s).

### Evaluation for these arms

- **Power:** paired AIVAT H2H has SE ≈ ±30 mbb/hand at 240k hands, so |t| > 2 only
  detects gains of ~60+. For 80% power, n ≈ 240k·(84/δ)²: ~680k hands for δ = 50, ~1.9M
  for δ = 30. LBR: ±200 at 10k paired hands is a standard error, so for a non-inferiority
  margin of 200 at true difference 0, 30k hands gives only ~53% power; 80% power needs
  ~62k paired hands with a one-sided 95% test, ~79k with a two-sided 95% interval
  (n = 10k·((z_α + z_0.8)·200/200)²; review 2026-11-16).
- **Margins, set before running:** every "no worse" gate states its non-inferiority
  margin and sample size up front.
- **Snapshot choice without selection bias:** picking the best of ~16 snapshots on the
  same data inflates the result by ~2 SE. Screen snapshots on one seed set at modest
  size, pick, then confirm the pick on fresh seeds at full size, and report only the
  confirmation. Pre-register each arm's primary comparison (interim: D vs C300; then D vs
  C-long at matching iterations).
- **Probes that actually test the new keys:** the strategy-shape probe checks preflop only
  (`strategy_shape.py`); add postflop checks on tagged keys (the preflop aggressor c-bets
  more than the caller, sane lead frequency, no tagged turn key collapses to never-fold)
  and a visit-tail check (`probe_rare_line_convergence.py`). The human-model opponent in
  `h2h_eval.py` keys on v2 and remembers nothing from earlier streets, so H2H against it
  barely rewards line-awareness; add a served-stack test (`GameSession` + the river
  solver) and an API smoke in the V3.6 style that checks every live bot key on D is
  trained.

### Added to Phase 7

- The image and release asset grow ~300 MB; re-run `measure_serving_memory.py` plus a
  latency smoke on the bigger DB (cold SQLite reads on 0.25 vCPU).
- `/api/abstractions` and `KeyExplorer.jsx` already mis-parse spr5 keys (they read
  `r3_km` as the pattern); fix them for the tags too. The Hand Explorer returns 501 for
  non-v2 blueprints; building tags would need the full betting line as input.
- The CI refit (`refit-opponent-models.yml`) still fits v2 and needs `--db`; CI does not
  run `core-rs` parity.

## Steps

| Step | Work | Wall time | Gate |
|---|---|---|---|
| V3.0 | Versioned abstraction dir + `abstraction_id` stamp + reader checks (Python + Rust) | — | v2 serves byte-identically; a mismatched dir is refused |
| V3.1 | Preflop cap jam: `PokerGame` + Rust `engine.rs`, `menu_mode='capped_pj'`, `game_kwargs` factory, consumers moved over. New parity fixtures `engine_pj` + `mccfr_pj`; property tests (chip conservation, a jam closes betting, the cap jam only preflop) | — | parity strict passes; full pytest green |
| V3.2 | Exhaustive flop bake into `v3/` (same centroids). The turn table is already exhaustive (46 rivers) and is copied | ~1 h single-thread (measured ~2 s/board × 1,755 boards) | stamp records `flop_runouts=0`; spot-check vs the audit's exhaustive buckets |
| V3.3 | River centroids K = 20 (`--opp-samples 0`) into `v3/`; check the Rust river bucketing reads K from the file | minutes | occupancy sane; Python == Rust on random river spots |
| V3.4 | Train two arms, 150M each, snapshot every 10M, `--river-cache-boards 140000`: **A** = spr5 + cap jam on v2's abstraction (isolates change 1); **C** = all four changes (the candidate) | ~1–1.5 h per arm | strategy-shape probe clean (no "never fold" collapse; weak-hand preflop fold sane) |
| V3.5 | Evaluate (below) and pick the ship snapshot | ~4–5 h total, overnight | gates below |
| V3.5b | ✅ results in 2026-11-13 (re-run finished, A300/C300 vs v2 added); recommendation C at 300M; the pick itself moved to V3.10 (decided 2026-11-13) | ~2 h | gates below; snapshot by play, not BR |
| V3.6 | ✅ 2026-11-13 (see "V3.6 — done"). Serving path (Phase 6.2, extended), all behind the DB stamps; v2 behaviour unchanged: `BlueprintStrategy`, range tracker, river solver (projection, gadget CFV, guards), translation, `game_session`/`game_adapter`. Built for **both** arms (A300 and C300), with the serving memory of each measured | — | tests incl. a BUG-027 replay through `GameSession`, on A300 and C300; v2 tests unchanged |
| V3.7 | Rust serving adapter, Docker wheel and DCFR experiment implemented; correctness checks pass; latency gate deferred to V3.14 / Phase 7 by user decision (2026-11-13) | ~1 week | Rust == Python solve on fixed spots; p95 latency ≤ 5 s on a 0.25 vCPU profile; gadget still ≤ blueprint |
| V3.8 | Candidate anchor implemented, OFF by default; evaluator/range bugs fixed; powered statistical gates deferred to final hardening by user decision (2026-11-13) | days | LBR at 0.1×/0.25× pot falls (z > 2); other fractions and H2H vs v2 within \|t\| < 2 |
| V3.9 | F / F_EXTENDED implemented; local structure/parity/safety checks pass; sustained solver-on H2H and latency gates pending | days, after V3.7 | p95 ≤ 5 s on Rust; H2H with the solver on ≥ menu A (\|t\| < 2 or better) |
| V3.10 | ✅ 2026-11-15 (see "V3.10 — done"): base arm **C at 300M**. **Pick the base arm** (A or C, from V3.5b's results and V3.6's memory numbers; V3.10–V3.13 build on it), then re-fit the opponent models on its v3 keys, stamped with `abstraction_id` (moved here from Phase 7: V3.11 and V3.13 need it) | ~1 day | the model guard compares abstraction ids; exploit no longer self-disables on the v3 blueprint |
| V3.11 | ✅ 2026-11-15 (see "V3.11 — done"): humans jam wider than assumed, so the read mostly calls more. Both gates pass after the review fixes (gate 1 +7.6 / +6.1 chips/spot; gate 2 +33.3 ± 4.8 mbb/hand). Pooled facing-jam population read + logged-spot evaluator (targets BUG-027) | days | logged-spot EV(call) − EV(fold) improves over the current guard; no regression vs the human model |
| V3.12 | ~~Pruning arm **E**~~ (code built and calibrated; arm dropped 2026-11-17, pruning stays off by default) (Rust trainer), design revised after review 2026-11-16 (see "Next steps"): prune only zero-probability actions with a per-key shadow-score test; trained on V3.16's keys | days + one training run | strategy-shape probe (+ postflop checks); BUG-027 replay; it/s; H2H vs D at equal wall time, no worse at equal iterations; LBR no worse |
| V3.13 | Data-biased-response (DBR) blueprint against the population model: a second (opponent) table, forked from the base winner (D, or C300/C-long if D loses) with fresh averages, models refit on its keys, P_max swept, held out by player/time (see "Next steps") | 1–2 weeks, after V3.16 and V3.12 | held-out human-model gain vs the cost (paired LBR + H2H vs the base), live layers on and off |
| V3.16 | History tags in the keys (`spr5a`): flop + preflop's last aggressor; turn + the flop's and preflop's; river + SPR. Python + Rust + parity; arm **D** from scratch and control **C-long** (C300 resumed), both to ~1.6B (see "Next steps") | days + two training runs | D vs C-long at matching snapshots, screened then confirmed on fresh seeds; human model and LBR within pre-set margins |
| V3.14 | Final pick among C300 / C-long / D / DBR blueprints; hardening, including deferred V3.7 latency and V3.8 statistical validation (below) | — | the gates, re-run on the final candidates |
| V3.15 | ✅ measured 2026-11-16, corrected after review: today's keys mis-state equity by more than 5 points for 12/9% of flop, 42/33% of turn and 76/72% of river hand mass; the chosen tags cut the turn to 13/17% and SPR cuts the river to 58/47% (→ V3.16) | minutes to measure | — |

The V3.6 serving work covered (all done; see "V3.6 — done" below):
- spr5 keys, threaded through `GameSession`, the range tracker's `_action_matrix`,
  `tilt.py` and `replay.py`;
- the cap-jam node: a human 5-bet of any size maps to `a`, and the bot can choose
  `allin` there;
- the deep-raise guard covering only beyond-the-new-cap nodes;
- `capped_pj` in `SERVING_MENU_MODES`, and auto-resolution and the fail-closed
  refusals lifted for stamped v3 DBs;
- the v3 abstraction dir (built now, so either arm can be picked at V3.10): every
  serving reader builds its card abstraction from the blueprint's verified directory;
  measure resident memory serving A300 vs C300 (C loads a second set of tables), for
  the pick and the Phase 7 memory budget;
- the four open river-solver review items (they affect v2 too):
  1. the safe gadget's fallback solve is starved of time in `auto` mode: reserve a
     fixed share of the shared deadline for it;
  2. the blueprint projection treats untrained deep nodes as uniform, inflating
     `e_bp` (~3.9% in one spot): project call/check there instead;
  3. custom amounts are rounded with `:g`, so a solver raise at exactly min-raise can
     round below it and be rejected: format with `.10g` or clamp to the minimum;
  4. a re-solve that gives the bot's hand zero reach silently falls back to the
     blueprint: count it and fall back explicitly;
- the remaining coin-flip fallbacks (e.g. an untrained faced jam) replaced by the
  equity-vs-belief guard.

### V3.15 — street history (measured 2026-11-16; added 2026-11-15)

Postflop keys carry only the current street's betting (`make_info_set_key`'s
`bet_pattern`); earlier streets survive only through the SPR bucket (how big the pot got,
not how) and, outside the blueprint, through the range tracker, which the river solver
and the guards use. So preflop/flop/turn play averages over every line that reaches the
same key: a check-raised flop and a check-called flop share turn keys. C300 has 444,217
info sets (192k flop, 165k turn, 57k river, 30k preflop) and 288 within-street patterns
per postflop street; full action history would multiply turn and river keys many times.
Libratus and Pluribus, as published, keep the full betting history and abstract only cards.

- **Measure first:** for common postflop keys, compare the opponent ranges the tracker
  holds across the histories that share each key (e.g. total-variation distance by
  history, weighted by reach). Offline, minutes.
- **If large:** a small history token (preflop pot type srp/3bp/4bp plus the previous
  street's last aggressor, ~6× postflop keys) as a training arm judged like A and C;
  full action history is a v4 question.

**Measured 2026-11-16, corrected after review (C300, 48 boards per street,
`scripts/measure_street_history.py`):**

The script walks C300's public betting tree on each board, carrying both players'
per-hand reach under the blueprint. At the first decision of the flop, turn and river it
groups the earlier-street lines that a candidate key would merge. Training updates a key
from every line that reaches it, weighted by the opponent's reach only (the traverser
explores all its own actions; `trainer.rs` `cfr()`). So the opponent range a merged key
is trained against is the sum of the opponent's reach over its lines. For each hand of
the player deciding and each line, the error is the gap between the hand's equity against
the line's real opponent range and against that pooled range, weighted by how often the
hand really stands there (joint reach). Turn and river equities are exact; flop equities
use 40 sampled runouts. Pruned line mass: 0.18% (flop/turn), 1.1% (river).

The first version of these tables (committed earlier the same day) pooled lines by joint
reach and grouped hands by fine preflop bucket. A review found both wrong; the corrected
numbers are higher (turn today 28/24% → 42/33%) and the conclusions sharper.

"Effective split" is how many ways a key's training visits really spread (exp of the
entropy of the visit mass over the new sub-keys, averaged by visit mass), relative to
today's keys. A street that splits k ways needs about k× the iterations for its keys to
keep C300's coverage: C300's keys have similar coverage on every postflop street (median
~1,700–2,200 of 12,600 merge rounds), so the largest split sets the run length.

Share of hand mass whose equity is off by more than 5 points (ip / oop):

| Street | Key | Off by > 5 points | Effective split |
|---|---|---|---|
| flop | street only (reference) | 52.4 / 17.1% | — |
| flop | **today: + SPR** | 11.5 / 8.6% | 1 |
| flop | + preflop's last aggressor | 2.4 / 2.6% | 1.75 |
| flop | finer SPR (11 edges), no tag | 3.9 / 7.3% | — |
| flop | random split, 3 ways (baseline) | 2.4 / 4.8% | 2.6 |
| turn | street only (reference) | 70.8 / 68.8% | — |
| turn | **today: + SPR** | 41.9 / 32.6% | 1 |
| turn | + flop's last aggressor | 22.1 / 21.9% | 2.4 |
| turn | + initiative (see below) | 16.5 / 20.0% | 2.6 |
| turn | + flop's and preflop's last aggressors | 13.0 / 16.7% | 5.4 |
| turn | + both aggressors + the flop's last bet size | 10.2 / 12.4% | 11.9 |
| turn | finer SPR, no tag | 30.6 / 28.7% | — |
| turn | finer SPR + both aggressors | 8.6 / 11.9% | — |
| turn | random split, 9 ways (baseline) | 28.3 / 25.4% | 8.3 |
| turn | random split, 27 ways (baseline) | 21.6 / 19.3% | 21.5 |
| river | **today: street only (river keys carry no SPR)** | 76.0 / 71.5% | 1 |
| river | + SPR | 57.9 / 46.5% | 3.5 |
| river | + SPR + turn's last aggressor | 48.0 / 39.8% | 8.9 |
| river | + SPR + initiative | 45.3 / 39.6% | 10.6 |
| river | + SPR + turn's and flop's last aggressors | 33.2 / 29.8% | 22.2 |
| river | + SPR + turn's, flop's and preflop's | 24.3 / 23.8% | — |
| river | + SPR, then a random 9-way split (baseline) | 55.4 / 45.0% | — |

The **initiative** token is 5-valued: who made the last bet or raise of the most recent
street that had one (ip/oop), and whether that street was the previous one or the
previous street checked through; "none" if nobody has bet since the blinds. It merges
lines that the two-aggressor tag keeps apart, e.g. "OOP raised preflop, IP bet the flop"
and "IP raised preflop, IP bet the flop" are both "IP bet last street".

- **Real structure, not just more keys:** at matching size the tags beat a random split
  by a wide margin on the turn (13/17% at a 5.4× split vs 28/25% at 8.3×) and the river.
  On the flop a random split does about as well as the tag, which only says the flop
  error is small either way.
- **Flop:** one tag (preflop's last aggressor) takes 11.5/8.6% to 2.4/2.6% for a 1.75×
  split.
- **Turn:** the biggest blueprint-served error. Both aggressors: 42/33% → 13/17% for 5.4×;
  initiative gets ~85% of that gain for 2.6×; bet size on top costs another 2.2× for 3
  points.
- **River:** the worst error, but the live bot re-solves the river with the full history.
  The blueprint's river still matters in three ways: the solver's answer is used only if
  it beats the blueprint's river action by an EV margin
  (`river_subgame_solver.py` `ev_gate`), the safe gadget anchors on the blueprint's river
  values, and during training the turn (served straight from the blueprint) learns its
  continuation values from the blueprint's river. SPR alone (76/72% → 58/47%) fits inside
  the turn's 5.4× budget; any river tag sets the whole run's length (8.9–22×) for a street
  the solver already handles, so river tags are deferred.
- **Caveats:**
  - Equity error is not EV: polarised and condensed ranges with equal equity play
    differently, and the key's strategy is a compromise, not a wrong guess.
  - Lines are weighted by the blueprint's own self-play. Humans visit low-reach lines
    more, where a merged key's pool (dominated by common lines) is most wrong, so the gain
    for the human seat is probably understated.
  - Measured at street starts only.
  - Re-running this on a tagged blueprint is circular (the tags split exactly these
    groups), so it is not a gate for V3.16.
- **Decision (2026-11-16):** see V3.16 in "Next steps".

### V3.11 — done 2026-11-15

**The data contradicts BUG-027's premise.** The plan assumed humans jam narrower than
equilibrium. The logged jams say the opposite. The server deals the cards, so every recap
holds both hole cards, even on folds. Across the 6,664 logged hands:

| Human jams the bot faced | flop (92, 13 players) | turn (50, 11 players) |
|---|---|---|
| Jammer's strength percentile on the board, median (1 = the strongest hand) | 0.66 | 0.75 |
| Share from the top 20% of hands (the guard's fixed assumption) | 32% | 44% |
| Pot odds the bot needed, median | 0.20 | 0.23 |
| Calling was right (EV(call) > EV(fold)) | 75 of 92 | 32 of 50 |

80 of the 92 flop jams are in 4-bet pots, and those are the widest (28% from the top 20%).
On this data, the leak is folding too much, not calling too much.

**Logged-spot evaluator** (`scripts/eval_facing_jams.py`). Each logged human jam before the
river that the bot answered is replayed through a `GameSession` bound to the blueprint:
real cards, logged actions, the blueprint's range tracker. The served bot then decides, and
its P(call) is read off the path its decision took. Each spot scores
P(call) × (equity × final pot − call cost), exact from both real hands (flop/turn:
every runout). The estimate is unbiased, with no response variance, and runs in seconds.

**The read** (`src/exploitation/jam_read.py`, fit by `scripts/fit_jam_read.py`). It
models where human flop/turn jams fall in strength. Each jam is ranked among all hands
on its board with the blueprint's own ordering (`_combo_strength`), and counted in 10
percentile bins. The counts back off from all jams, to street, to pot type
(srp/3bp/4bp), to SPR bucket, each rung a Dirichlet whose prior is the coarser rung. At
a faced flop/turn jam, `_facing_allin_guard` asks the read first:
- it computes the bot's equity against each bin, then EV(call) under 400 Dirichlet draws;
- it acts only when P(EV > 0) or P(EV < 0) is at least 0.9 and the mean edge clears
  the 1-chip margin;
- when the tracker's read is trusted, the read's EV is mixed with the tracker's,
  weighted p_max·n/(s + n) (p_max 0.9, s 20); an untrusted tracker is never mixed in;
- it stays silent until its street has 20 logged jams;
- otherwise the guard continues exactly as before.

It is behind `ALLIN_JAM_READ` (default off) and applies in both A/B arms. It is stamped
like the opponent models and refused for another blueprint (healthz `jamRead`). C300's
read: `backend/bot/analysis/opponent_models/c300/jam_read.json` (142 jams, git-ignored).

**Review (2026-11-15, three agents in Codex) — three findings, all confirmed and fixed:**
1. **An untrusted tracker could decide through the blend.** The tracker's equity was
   mixed in whatever its trust; with little jam data its weight is near 1, so an
   untrusted tracker decided the stack-off. Now it is mixed in only when `_trust_read`
   passes. Writing the regression test exposed a second sparse-data hole: with no jams
   the read's uniform root prior ("humans jam any hand") is tight enough to look
   confident, so the read now stays silent below 20 jams on the street (`min_jams`).
2. **Bin equities mis-weighted runouts.** Per-runout ratios were averaged, but card
   removal leaves runouts with different numbers of live hands; wins and pairs are now
   pooled over all (hand, runout) pairs (test: exact on an exhaustive turn).
   `RangeTracker.hero_equity` averages per runout the same way (not changed here).
3. **The evaluator replayed custom bets at grid size.** A custom bet is logged under its
   snapped grid name with the real chips; it is now replayed as a custom bet to the real
   total, and every replayed spot is checked against the logged pot and call cost
   (within 0.05 chips: the log rounds chips to 2 decimals).

**Gate 1: logged-spot EV improves over the current guard. ✔** (after the review fixes)
C300, 158 spots (2 skipped: a bot 5-bet that `capped_pj` plays as its jam); every
replayed spot matches the logged pot and call cost. Leave-one-player-out: each player's
spots are scored with a read fit on the other players. Chips summed over spots;
always-fold = 0:

| Policy | control arm | exploit arm (C300 models) |
|---|---|---|
| the bot that played (v2) | +18,076 | +18,076 |
| C300 served today | +18,707 | +19,114 |
| C300 + jam read | **+19,912** | **+20,069** |
| always call | +19,785 | +19,785 |
| perfect decisions | +21,942 | +21,942 |
| read − served, chips/spot | +7.6 (SE 3.6; by player 4.7) | +6.1 (SE 3.3; by player 3.7) |

- **Control arm:** the read calls more in 69 spots (calling averaged +109 chips there) and
  less in 6 (+16).
- **Before the fixes** it was +8.0 / +6.9 chips per spot.
- **Sensitivity** (before the fixes): p_max 1.0 → +7.6; conf 0.95 → +8.0; alpha 40 → +8.3.
  The defaults stay.

**Gate 2: no regression against the human model. ✔, rerun after the review fixes
(2026-11-16).** `h2h_eval.py --jam-read-a`
re-scores C300's faced flop/turn jams against the human population model. Calling a jam
that commits the stack ends the decisions, so the counterfactual is exact per spot, with
no second run or pairing. Over 4 seeds × 40k hands:
- **The read gains +33.3 ± 4.8 mbb/hand after the fixes.** Every seed is positive
  (+26.6, +43.6, +28.7, +34.1), and it changed 1,965 of 4,586 spots. Before the fixes it
  was +32.8 ± 4.7 (1,963 changed): the bin-equity fix moved two decisions.
- **C300 vs the model:** +618 ± 45.
- **Caveats:** the human model comes from the same export as the read, so this is not
  independent evidence. The match has no tracker, so the read runs unblended (p_max
  applies only when a tracker is present).

**BUG-027 itself:** the read calls J9hh on Q♠9♣3♣ (flop|4bp|r0). Its equity against the
logged human 4-bet-pot flop jams is ~0.65, against 0.39 needed. On today's data, that
call was right and the human had it that hand.

**Open.**
- 142 jams from 13 players, three of them over half the spots. Refit as data grows.
- Preflop jams are untouched (16 spots, the guard's top-20% path).
- River bets that close the action are not scored yet; the evaluator covers preflop,
  flop and turn.
- The CI refit doesn't fit the read; Phase 7 needs that, like the opponent models.

### V3.10 — done 2026-11-15

**Base arm: C at 300M** (C300, `blueprint_par_cappedpj_spr5_v3_rs_20261110_013712_8d0598_snaps/
snap_300000000.db`), picked 2026-11-15. C300 beats A300 head-to-head (+79 ± 30 mbb/hand) and
beats v2 by more (+381 vs +280); LBR is level within noise (599 vs 710, each ±300). It costs
~12 MB more to serve (V3.6). The BR gap that favoured A (+353) was not used: the repo's BR
lets the exploiter see the whole board at every street (verified 2026-11-15), so it is
neither exploitability nor a lower bound. V3.11–V3.13 build on C300.

**Stamped models.** `opponent_model.blueprint_model_stamp(db)` is what a model's keys
depend on: the key format, the abstraction id (plus its name, for reading) and `cap_jam`
(a preflop 5-bet into a 4-bet is the trained jam). Every model file (per-player counts,
built model, population) carries it. `HumanModel` and `build_opponent_model.py` refuse a
file whose stamp differs from the blueprint's (`check_model_stamp`). An unstamped file is
the legacy v2 setup (v2 keys, root abstraction, no cap jam), so the v2 models load
unchanged. This replaces the old check that guessed bucket counts from the blueprint's
keys (and its full key scan at load).

**Refit.** `fit_opponent_models.py --db <blueprint>` re-derives keys from the raw recaps
with that blueprint's key format, card abstraction and cap jam. Without `--db` it fits
the v2 setup, so the CI refit workflow is unchanged. `replay_hand(..., cap_jam=True)`
records any preflop 5-bet into a 4-bet, by either player, as `a`, as `GameSession` does
live. Live last-N refits replay the same way (`HumanModel.cap_jam`).

| Fit on 6,664 recaps (15 players) | v2 keys | C300 keys |
|---|---|---|
| human decisions | 17,534 | 17,534 |
| on keys the blueprint trained | 100% | 100% |
| river buckets | 0–9 | 0–19 |
| flop/turn keys with an SPR bucket | — | all (r3 the most common) |
| human 5-bets into a 4-bet recorded as `a` | 16 of 104 | 104 of 104 |

The C300 models are in `backend/bot/analysis/opponent_models/c300/` (git-ignored; rebuild
with `fit_opponent_models.py --db <C300> --out-dir ...` then `build_opponent_model.py
--dir ... --db <C300>`).

**API check.**
- C300, `ALLIN_EXPLOIT=1`, C300 models: healthz reports 15 player models. A 12-hand smoke
  played 34 bot decisions in `exploit_tilt` mode, and all 43 bot keys were trained.
- C300 with the old v2 models: refused with a stamp mismatch, so exploitation stays off.
- v2 with the old models: unchanged (15 player models, the Hand Explorer answers).
- Full backend suite: 605 passed, 1 skipped.

**Not done here.**
- The offline exploit scoreboard (`measure_exploit.py`) and the alpha tuning were not
  re-run on v3 keys; alpha 10 / 160 is carried over.
- The CI population refit still fits v2. Serving v3 in prod (Phase 7) needs that workflow
  to pass `--db` with the served blueprint.

### V3.6 — done 2026-11-13

**Binding a session to its blueprint.** `game_session.session_kwargs(db)` returns the
blueprint's menu mode, key format and verified abstraction dir; the API builds every
`GameSession` with it (live game, explorer river solve, abandoned-hand sweep). The
session then:
- builds flop/turn keys with the street-start SPR bucket (`_spr()`, from
  `d['starting_pot']`), and passes the same bucket to both range trackers' `observe`
  (`_action_matrix(spr=)`) and to the bot (`public_state['spr']`, read by the exploit
  tilt);
- buckets with `card_abstractions.shared_card_abstraction(dir)`: one CardAbstraction
  per directory per process, shared with the river solver. Before, every request built
  its own and reloaded the baked tables (the turn table is 126 MB).

`exploitation/replay.py` takes `key_format` and rebuilds the street-start pot from the
logged chips; `opponent_model._levels` skips the `rN` token.

**The preflop cap jam, live.** Live play uncaps re-raises, so facing the 4-bet the
engine offers sized 5-bets the blueprint never trained. At that node
(`_at_preflop_cap`: preflop, `capped_pj`, exactly 3 aggressions, the last one just
made):
- the bot is offered exactly the trained `fold / call / allin`;
- the human keeps the live sizes plus `allin`, and any 5-bet (sized, custom or a jam)
  is recorded as `a`. The trackers observe it as `allin` on the trained legal set, and
  the bot answers from the trained facing-jam key (fold/call).

Beyond the cap the deep-raise guard still owns untrained nodes; the cap node and the
facing-jam node are trained, so it defers there.

**Refusals lifted.** `BlueprintStrategy` (so the river solver), the API and
auto-resolution take any known key format, a verified abstraction dir and a serving
menu mode (`capped_pj` added). Auto-resolution only checks that the dir exists; the
loader verifies the abstraction id. Still v2-only, failing closed:
- the PyPokerEngine `Player` (its `GameAdapter` builds v2 keys from PyPokerEngine
  state; not a serving path);
- `HumanModel`: the fitted models are v2 keys on root buckets, so it refuses a v3
  blueprint and exploitation self-disables until V3.10;
- the Hand Explorer: `/api/strategy/from-hand` returns 501 for a v3 blueprint (its key
  builder has no street-start pot; Phase 7).

**River solver.**
1. The safe gadget keeps 40% of the budget (`_GADGET_RESERVE`) when the exploit solve
   may fall back to it (an untrusted read); a trusted read still gets the whole budget.
2. `blueprint_to_tree_dist` projects an untrained node as the served passive play
   (check, or call facing a bet) instead of uniform.
3. `make_custom_action` writes the full repr when `%g` would round, so a raise to
   exactly the minimum is no longer rejected.
4. Zero hero reach raises `UnsolvableSpot`: counted in `stats['zero_reach']`, logged
   without a traceback, and played by the explicit fallback.

**Coin flips.** The deep-raise guard no longer needs a tracker (it judges against the
top-fraction range without one), and a failed or unsolvable river spot facing an
untrained bet goes through that guard before the safe call.

**Tests.** `tests/test_v3_serving.py` (18), including the BUG-027 hand through
`GameSession` on A300 and C300: every bot node is a trained key, the flop lead-jam is
`_flop_r0_a`, and the cap node is played as trained. The refusal tests in
`test_abstraction_set.py` now pin the new behaviour. API smoke (scratch script, 20
hands per blueprint through `/game/new`, `/action`, `/bot-action`, `/next-hand`): A300
63/63 and C300 50/50 bot keys trained, every flop/turn key SPR-tagged, river solves
ran; v2 unchanged (no SPR tokens, Hand Explorer 200, 53/53 trained). With
`ALLIN_EXPLOIT=1`, v2 loads the 15 player models and A300 self-disables.

**Serving memory** (`scripts/measure_serving_memory.py`: the served bot plus 60 hands
against a sticky caller, 13 river solves, one blueprint per process):

| | v2 | A300 | C300 |
|---|---|---|---|
| RSS after the hands | 270 MB | 272 MB | 284 MB |

One process serves one blueprint, so C does not load a second set of tables: the v3
turn table is a copy of the root's (126 MB each) and the exhaustive flop table is the
same size as the root's (11.6 MB). C costs ~12 MB more than A, from its river buckets
and caches. The "two-abstraction memory risk" applies only if one process loaded both.

### Review follow-up — verified and fixed (2026-11-14)

Two independent-review findings were reproduced and corrected:

1. Checkpoint guard: changing the tiny-bet policy flag previously passed resume
   validation. The new versioned `policy_identity` includes policy sources, loaded
   native binary identity, committed blueprint contents (WAL included), abstraction
   assets, active mmap cache contents and effective settings/runtime versions.
   Legacy checkpoints without the full identity are refused, not silently upgraded.
   Seed aggregation rejects mixed complete identities; historical records without
   them are explicitly unverified. Original completed artifacts are unchanged.
2. LBR singleton fallback: with a missing sized key, live play folded/called 50/50
   while LBR priced and played a certain fold; zero legal mass also diverged.
   Pricing, execution and belief updates now share one response helper. Singletons
   use the live passive fallback; genuine blends retain fold-on-missing-bracket.
   Regression tests cover missing, empty and zero-legal-mass rows plus real blends.

The recorded V3.9 live-session H2H is not an LBR measurement. No evidence was
found that its policy changed during the run, but the old four-file fingerprints
cannot retrospectively supply the stronger new provenance. Keep that limitation
with the results. Prior LBR diagnostics do not establish the tiny-bet gate; use
the corrected evaluator for the deferred V3.8 comparison. No menu/blueprint choice,
candidate activation, deployment or new sustained run is implied by these fixes.

Validation after fixes: all 89 affected regression tests pass. A direct CLI check
reused ten checkpointed hands and reproduced all thirty hand records of an
uninterrupted run exactly. SQLite WAL/asset/native-binary/flag identity changes,
legacy-checkpoint refusal and both singleton/multi-bracket fallback paths are
covered. No long evaluation was restarted, and historical result files were not
rewritten.

### V3.9 sustained results — complete (2026-11-14)

All 24 jobs / 240,000 hands completed: three independent seeds of 10k hands for
both candidate menus, both blueprints, and intended-live/equal-cap comparisons.
Source fingerprints agree. No recorded solve was time-budget truncated.

Intended live policies (candidate cap 3 vs existing A cap 5), AIVAT chance-control
estimate in mbb/hand; positive favors candidate; ± is one standard error:

| Candidate | A300 | C300 |
|---|---:|---:|
| F vs A | +40 ± 87 | +8 ± 87 |
| F_EXTENDED vs A | +33 ± 88 | +19 ± 87 |

Equal-cap comparisons are similarly unresolved. These runs show no statistically
clear strength winner; they do NOT establish equivalence or formal non-inferiority.
One resumed group's historical fallback counters are unavailable and explicitly
marked incomplete; its hand-level payoffs and solve-budget records were preserved.

Full results: `analysis/blueprints_rs/eval_runs/v39/sustained/summary.json` and the
per-seed JSONs. The queue, report helper and keep-awake process have exited.
No menu is selected; defaults remain the existing configuration. The user's
selection decision and deferred production-profile latency gate remain open.

### V3.9 — candidate menus implemented; strength/latency gates pending (2026-11-13)

- Added solver-only A, F and F_EXTENDED profiles with separate bet/raise menus.
  F_EXTENDED adds 0.25 and 3.0 opening bets to F. `ALLIN_RIVER_MENU` selects a
  profile; default A retains the existing five-aggression live tree. F variants
  default to the planned three. Explicit custom menus retain their old behavior.
  No blueprint action abstraction change or retraining. Health shows actual menus.
- Important baseline correction: the earlier review counted A at THREE
  aggressions, but live A uses FIVE. Compare at equal caps to attribute menu effects;
  also measure the intended complete live policies (A5 vs F3) separately.
- Exact human edges are still injected under the configured aggression budget;
  F3 cannot represent a fourth aggression and uses the existing fallback path.
- Fixed an existing zero-cap edge case: Python and Rust previously still offered
  an opening bet at aggression cap zero. Both now check down. Named production
  profiles use positive caps; this fix does not change their default trees.

Decision-node counts, same sizes as the plan's original review:

| Pot / stack | A3 | F3 | F_EXTENDED3 | A5 | F5 | F_EXTENDED5 |
|---|---:|---:|---:|---:|---:|---:|
| 8 / 192 | 210 | 124 | 162 | 1086 | 394 | 506 |
| 40 / 180 | 72 | 74 | 96 | 80 | 88 | 120 |
| 80 / 96 | 16 | 24 | 32 | 16 | 24 | 32 |

Local fixed-board Rust CFR+275, two repetitions (mean seconds, NOT production p95):
- Deep spot: A3 1.06, F3 0.59, extended3 0.79; A5 5.21, F5 1.87, extended5 2.43.
- Middle spot: A3 0.34 vs F3 0.35; short spot: A3 0.07 vs F3 0.11.
- Each reported convergence gap is computed on its OWN tree. Gaps across different
  menus do not establish relative playing strength. No winner/default change.
- Full JSONL in `analysis/blueprints_rs/eval_runs/v39/fixed_spot_menu_comparison.jsonl`;
  reproduce with `scripts/compare_river_menus.py` (includes equal-cap comparisons).

Validation / match preparation:
- 78 affected tree/CFR/control/river-serving/v3 tests pass; 35 Rust unit tests pass.
  Checks include F/extended tree accounting, injected edges, native/reference parity
  and the three-board correct/wrong-belief safe-gadget battery. These finite tests
  compare against the projected blueprint, not a universal full-game safety proof.
- `scripts/river_menu_match.py` drives TWO synchronized live GameSession views, one
  per solver, including each player's own range chaining. It asserts matching
  engine state and zero-sum payouts. Alternating seats, per-hand seeds and source
  fingerprints support repeatable comparisons; depth overrides isolate size/depth.
- Uses existing AIVAT chance controls c1/c3. Model-dependent c2 is disabled because
  the opponent is another solver, not the blueprint assumed by the older harness.
  Raw estimates are also retained. Smoke-test win rates are not strength evidence.
- Initial A300 F-vs-A 200-hand smoke: both solvers fired six times each, zero
  fallbacks. C300 extended-vs-A: thirteen solves per side, zero fallbacks.
  Both F and F_EXTENDED completed 200-hand smokes on both A300 and C300:
  800 hands total, 76 river solves, zero solver fallbacks. The small-sample scores
  are not a strength verdict; F/extended even chose the same realized lines here.
- All three profiles return healthy API responses with the expected menu/cap.

Still required for promotion: sustained 30–50k-hand solver-on comparisons across
seeds (plus equal-cap attribution) and representative p95 latency. The user approved the sustained local run on 2026-11-13; it is queued below in
Decisions. A remains
the default; V3.9 is NOT declared complete merely from these smoke tests.

### V3.8 — candidate implemented; statistical gates pending (2026-11-13)

The existing pseudo-harmonic translator mitigates interior size gaps. Its lower
boundary clamps to the smallest grid point: this boundary behavior is confirmed
in code, but a statistically resolved practical leak has NOT yet been established.

- Added experimental `ALLIN_SUBMINIMUM_ANCHOR=1` (default OFF), for custom
  postflop opening bets only. Below the smallest legal sized bracket, blend toward
  a synthetic call-only zero-size response using the same pseudo-harmonic weight.
  Preflop, raises, on-grid and larger bets retain their existing decision mapping.
  The synthetic point never becomes a blueprint key or a human-action observation.
- The live response, bot self-range and LBR victim use the same anchor. Health
  exposes `subminimumAnchor`. No training/abstraction change or retraining.
- Verification fixes found during review:
  1. LBR candidate generation now enforces the minimum full bet/raise increment;
     previously tiny probes could be illegal and manufacture an apparent leak.
  2. LBR translation grids now filter unavailable sized tiers and exclude all-in
     for sub-stack custom bets, matching GameSession's grid construction.
  3. Both live self-range and LBR beliefs now condition on the response blend
     actually played, rather than only its nearest-size blueprint key. This fixes
     an existing between-size range inconsistency as well as the new anchor.
- `lbr_paired.py --fractions` configures POSTFLOP probes only; preflop keeps its
  original menu. Results include flag state, source hashes, paired per-hand
  payoffs and elapsed time so superseded runs can be distinguished.
- Earlier 1,000-hand pilot comparisons were inconclusive, then superseded by
  the evaluator corrections above. They are diagnostics, not evidence for passing
  a gate. Results live under `analysis/blueprints_rs/eval_runs/v38/` (git-ignored).
  Files named `*_final_anchor*_smoke100.json` exercise both A300 and C300 on the
  corrected model. These are smoke tests, not statistically powered EV evidence.
- Unit/regression coverage includes weight boundaries, serialization, no synthetic
  DB lookups, no phantom human shove, actual session self-range callback, and legal
  LBR grids/probes. All 85 affected session/range/v3/LBR tests pass with the
  anchor both OFF and ON. Both health flag states return 200. All four final
  A300/C300 anchor-off/on 100-hand smoke runs completed (~26–27 seconds each);
  they establish execution coverage only, not a statistically resolved gain.

Still required before enabling: powered paired LBR by fraction on both arms,
regression checks at other fractions and H2H vs v2. A larger local run may take
hours; the user decided on 2026-11-13 to defer it to final hardening
(V3.14 / Phase 7). Do not start the multi-hour run now. No claim that
V3.8 is complete or the practical leak is resolved; keep the flag OFF.

### V3.7 — implementation complete; latency gate deferred (2026-11-13)

Serving integration:
- `ALLIN_RIVER_BACKEND=python|rust` (default Python); `ALLIN_RIVER_ALGORITHM=cfrplus|dcfr`
  (default CFR+, DCFR requires Rust). Startup checks the extension/configuration;
  health exposes the backend and algorithm. No deployment or default switch.
- Rust imports the exact serving tree, including injected human bets and appended
  decision ids. Graph validation rejects cycles, shared children, disconnected
  nodes, invalid ids, excessive depth and non-finite terminal values.
- Warm starts, persistent iteration clocks, ordinary solves and safe gadgets are
  supported. Read-only EV/safety walks reuse Python with cached Rust averages.
- Docker builds a pinned-toolchain Python wheel in a separate stage. The runtime
  gets the wheel without the compiler. Build context excludes Rust targets,
  local virtual environments, Claude worktrees and training runs.

Algorithm experiment:
- Alternating DCFR(1.5,0,2), including the gadget's opt-out decision, with signed
  regrets discounted after each player's update and quadratic average weights.
  Reference: [Brown/Sandholm 2019](https://arxiv.org/abs/1809.04040).
- Independent Python experiment in `backend/core-rs/bench/compare_dcfr.py`;
  native/reference checks cover incremental runs and both gadget seats.
- On one fixed board and seeded nonuniform ranges, DCFR100 gaps vs CFR+275 were
  0.116 vs 0.157 (pot 12), 0.211 vs 0.380 (pot 30), and 0.258 vs 1.577 (pot 80).
  DCFR uses TWO traversals per iteration: 100 iterations = 200 traversals.
  This is preliminary convergence evidence, not a broad playing-strength result.
- Corrected `test_safe_river_gadget` to actually use the selected serving backend
  for its gadget (previously it constructed Python RiverCFR directly). The safety
  battery passes with native CFR+, DCFR300 and DCFR100. Three boards / correct and
  deliberately wrong beliefs; finite-iteration tests are not a universal proof.

Latency, local ARM Linux container with `--cpus=.25`, 275 CFR+ iterations,
10 repetitions per spot (includes construction and convergence checks):

| Entry pot / stack | Ordinary p95 (s) | Gadget p95 (s) |
|---|---:|---:|
| 12 / 194 | 4.79 | 4.51 |
| 30 / 185 | 2.74 | 3.05 |
| 80 / 160 | 0.58 | 1.20 |

These INITIAL six synthetic cases were below 5 seconds, but subsequent runs
were NOT stable: do not treat this table as a passed deployment gate. This is a
CPU-quota proxy, not Lightsail hardware or full API request latency; an auto-anchor
request can perform more than one solve. Script: `scripts/benchmark_river_backend.py`.

Both A300 and C300 completed 20-hand native DCFR100 serving probes (3 and 1 river
solves respectively), ending around 199 / 195 MB RSS. These short probes do not
replace the Phase-7 full frontend/API/local-DynamoDB play-through.

Validation:
- Backend suite: 562 passed, 1 skipped (567.5 seconds); excluded
  `test_best_response_vectorized.py`, but INCLUDED the slow two-board BR/API smoke.
- 34 native Rust unit tests; 35 focused adapter/binding tests; 7 existing river
  parity tests. Native DCFR and CFR+ both pass the corrected gadget-safety battery.
- Final Linux wheel: 7 adapter/reference/configuration tests pass. Non-root prod
  image health returns 200 with Rust/DCFR and no Rust compiler installed.
- The worktree blueprint/table symlinks point outside the Docker context. The
  startup smoke therefore mounted a standalone read-only local blueprint copy;
  container tests lazily bucketed missing tables. Phase 7 must ship real release
  assets: this local image is NOT a deployment-ready asset bundle.
- Historical fixture manifest is stale; new serving parity tests compare against
  the current Python implementation directly. The existing fixture parity also
  passes. No tests claim universal finite-iteration gadget safety.
- Subsequent sequential timing on the final image failed the target: deepest
  CFR+275 ordinary p95 11.71 s; that run was stopped after the failure was clear.
  Optimized DCFR100 measured 4.26 s ordinary / 5.91 s gadget p95 on the deepest
  spot, so DCFR has NOT cleared the latency gate either.
- Diagnostic on the SAME final Linux image: at `--cpus=1`, deepest CFR+275 took
  0.93 s wall / 0.91 s CPU (gadget 0.94 / 0.94); at `--cpus=.25`, it took 9.30 s
  wall / 2.33 s CPU (gadget 10.10 / 2.52). Identical strategies/gaps. The quota
  changes measured CPU cost too, consistent with host/VM scheduling or clock
  effects, not just a fourfold wall-time multiplier. The precise host cause is
  unproven. Do NOT tune strategy quality down merely to pass this unstable proxy.
- Decision: retain Python/CFR+ defaults and keep Rust/DCFR opt-in. Implementation
  and local correctness validation are complete; V3.7's latency gate remains OPEN
  pending a stable, production-representative Linux benchmark environment. No
  cloud resources were provisioned and nothing was committed, pushed or deployed.

### Scope added 2026-11-12: the next improvements are part of v3

Sources: the 2026-11-11 research and bet-menu reviews (summarised here; their scratch
scripts were not kept).

**V3.7 — serve the Rust river solver, with a faster CFR variant.**
- Today the live solver runs 275 iterations of CFR+ with simultaneous updates and
  linear averaging in Python: ~15 s for a single-raised pot on the 0.25 vCPU box. The
  Rust port measured 7× faster but is not served.
- Work:
  - a `ALLIN_RIVER_BACKEND=python|rust` adapter;
  - alternating updates with DCFR(1.5, 0, 2) or DCFR+, for an expected 2–3× fewer
    iterations (DCFR beats CFR+ and PCFR+ on HUNL subgames);
  - re-check the safe gadget under the new weighting;
  - a Rust stage in the Docker image.
- Expected: ~15 s → 1–2 s per solve. The freed latency pays for V3.9.

**V3.8 — sub-minimum bets at inference.**
- Humans almost never bet off the grid: 7 of 4,285 postflop opening bets (0.16%), 0
  of 280 first raises, 0 of 1,458 preflop opens (6,664 hands).
- In a clairvoyance toy game, pseudo-harmonic translation gives an exploiter 0 in
  every interior gap, including 2.0× to all-in. The one hole is below the smallest
  size: such a bet is clamped to the 0.33 response, so the bot over-folds. Exploiter
  gain is 100 / 79 / 41 / 24 milli-pot per bet at 0.05 / 0.10 / 0.20 / 0.25× pot.
- Fix at inference: anchor a sub-minimum bet between "size 0 = never fold" and the
  0.33 response, pseudo-harmonically. `bot_strategy` and `translation` have no
  sub-minimum handling today.
- Measure first (M0): LBR probing {0.1, 0.25, 0.5, 0.8, 1.25, 2.5, 3.0}× pot, reported
  per fraction (`LBR_BET_FRACTIONS` is only 0.5 / 1.0 / 1.5 today). A fraction counts
  as a leak only if its gain over the on-grid fraction has z > 2.
- No new trained blueprint sizes: no human volume sits in a gap, and each added tier
  costs 1.2–1.35× the info sets plus a retrain. If the grid is touched again, the
  evidence favours cutting raise tiers to ~2 (×0.28 the patterns; each raise size gets
  only 2–6% of the mass at raise nodes, all-in 6–8%). That is a separate one-variable
  arm, only if M0 finds an interior leak with real volume.

**V3.9 — river solver bet menu "F".**
- Bets 0.33 / 0.5 / 0.75 / 1 / 1.5 / 2 + all-in; raises 0.5 / 1 + all-in; 3
  aggressions. Today's menu A uses 0.5 / 0.75 / 1 / 1.5 for both bets and raises.
- Tree size (nodes): single-raised pot SPR 4.5: A 72, F 74; 3-bet pot SPR 1.2: A 16,
  F 24; limped SPR 24: A 210, F 124. F costs the same as A in single-raised pots and
  40% less in the deep worst case, and adds the block bet and a 2× overbet.
- The solver needs no raise sizes for the human's sake: a human's exact raise is
  spliced into the tree (`inject_realized_edge`). F also closes the documented
  missing-0.33 gap (`river_subgame_solver.py`).
- Compare A vs F vs F + 0.25 / 3.0: node counts, Rust p95 latency, solve gap at 275
  iterations, then a paired AIVAT H2H (30–50k hands) with the solver on in both arms.

**V3.10 — opponent models on v3 keys.**
- Re-derive keys from the raw recaps; don't reuse stored v2 keys (5-bet lines mapped
  to beyond-cap keys under v2 and map to trained `a` keys under v3).
- Models record the `abstraction_id` (and key format) they were fit on, and the
  guard compares it; `fit_opponent_models.py` buckets with the target blueprint's
  abstraction.

**V3.11 — pooled read on how strong human jams are (BUG-027).**
- BUG-027 is an opponent-model problem: J9 has 0.57–0.66 equity against the
  blueprint's own shove range but ~0.20 against a human one.
- At flop/turn facing-jam nodes (`_facing_allin_guard`), take equity against a
  low-dimensional human model, P(jam | strength percentile, SPR, street, pot type),
  pooled across all big-bet nodes (per-key counts in 4-bet pots are tiny).
- Gate on that model's credible interval, not the per-hand tracker confidence (in
  BUG-027 a read of 0.046 < 0.2 sent the bot back to the wide blueprint range).
- Safety: weight the model by its data the way DBR does (P_max·n/(s+n)).
- Our data is unusual: the server deals the cards, so recaps have both hole cards
  even on folds, with no showdown selection bias.
- **Logged-spot evaluator:** for every logged human jam the bot faced (and every river
  bet that closes the action), compute exact EV(call) − EV(fold) from the human's real
  cards by enumerating the remaining board. Unbiased, no response variance; scores
  this guard (or any guard) on real data in seconds.

**V3.12 — negative-regret pruning arm.**
- Pluribus-style: on 95% of iterations, skip the traversing player's actions with
  regret below a large negative threshold, except on the last street or one step from
  a terminal; floor regret at a slightly lower bound.
- Why: pruning makes a bucket's strategy fit the hands that actually reach it, which
  acts like a finer abstraction with no new keys. Today a hand gets full regret
  weight at a bucketed postflop node it almost never reaches (J9o in a 4-bet pot when
  it 4-bets 2%): the BUG-023/027 class. Pluribus credits part of a ~2× speed-up to it.
- Conflict: our trainer floors regret at 0 (CFR+), so this needs an unfloored arm or
  a per-action dormant counter, and the parallel merge must be designed for
  unfloored regret from the start (BUG-014 and the reverted "Fix #2").
- Train on the base arm's config, one variable changed; gate on the strategy-shape
  probe first.

**V3.13 — safe exploitative (DBR) blueprint.**
- At the human seat's nodes the opponent plays the fitted population model with
  probability P_conf(n) and its own CFR strategy otherwise (data-biased response,
  Johanson & Bowling 2009). Superseded by the revised design in "Next steps" (a second table, a fork per P_max of a few hours).
- Targets pre-river exploitation, which is only the bounded tilt today.
- Measure BR (the cost) against a held-out human-model match (the gain), sweeping
  P_max. Evaluating against the same model is in-sample, so hold out players or time.

**Not included:** the research's "add the sizes humans actually use" (Libratus-style
self-improvement). The recap data shows humans press the size buttons (0.16%
off-grid), so there is no volume to add.

**Then Phase 7 (deployment, separate sign-off; nothing ships until it is all tested):**
- `/api/abstractions` + KeyExplorer for spr5 keys and the cap jam;
- `assets-v3` (the blueprint, plus the v3 tables if C);
- the memory budget on the 0.25 vCPU Lightsail box (V3.6 measured ~270–285 MB per process for one blueprint; budget the gunicorn worker count against it);
- test hardening:
  - the ~26 silent print-and-return skips become real skips or failures;
  - CI builds `allin_core` and runs parity and `cargo test`;
  - mark the 501 s BR smoke test slow;
  - fix the two env leaks and the global-random seeding;
  - the missing tests: match payoffs, AIVAT on engine records, a real v2 + v3
    tracker cache, `check_abstraction.py`, a `bug027_replay.py` smoke, a parity
    freshness gate;
- a full local play-through (frontend + API + local DynamoDB, exploit on);
- merge rust-core into main (main has AGENTS.md and an IDEAS.md entry; apply the
  CLAUDE.md edits to AGENTS.md too);
- staged deploy: v3 with exploit off, live A/B vs v2, cutover, then exploit on (the
  June announcement promises it).

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

## Decisions

V3.9 sustained queue RESUMED at the user's request (2026-11-14), superseding the
pause below. All three partial checkpoints validated against policy source hashes
and configuration; 57,066 saved hands are reused. Complete-hand resumption avoids
counting pause time against solves. Legacy partials did not store cumulative
solver counters: affected results explicitly mark `stats_complete=false` instead
of inventing historical fallback counts. Hand-level payoffs and solve-budget
records remain intact. Queue concurrency remains three, and defaults are unchanged.

V3.9 sustained queue PAUSED at the user's request (2026-11-13): 3 completed
10k-hand jobs plus 27066 checkpointed hands in interrupted jobs
(57066 hands retained). Workers, controller, report helper and keep-awake
process are stopped. See sustained/paused.json; do not restart without the user's
request. Interrupted timed hands are excluded from checkpoints.

Decided 2026-11-13: user approved running V3.9's sustained local comparisons now.
Queue: `analysis/blueprints_rs/eval_runs/v39/sustained/`. Three independent seeds
(101/102/103), 10k hands per seed, 275 iterations, Rust CFR+, three concurrent jobs.
Both A300/C300, F/F_EXTENDED vs A, at intended live caps AND equal cap 3:
24 jobs / 240k hands total. V3.8 anchor forced OFF to isolate the menu experiment.
Progress checkpoints, per-solve budget diagnostics and source fingerprints are
retained. `summary.json` is regenerated after completion/failure; inspect its
`complete`, `source_consistent` and budget-truncation fields before interpreting.
This records a STARTED run, not a passed gate; no menu default or deploy change.

Decided 2026-11-13: defer the larger corrected V3.8 statistical comparison to
final hardening (V3.14 / Phase 7). Keep the sub-minimum anchor OFF until its
gates pass. Continue subsequent implementation without starting that heavy run.

Decided 2026-11-13 (Codex continuation): defer V3.7 production-profile latency
validation to final hardening (V3.14 / Phase 7). Do not provision a cloud benchmark
host now. This does not waive the latency gate; proceed with V3.8 and subsequent
implementation while retaining the measured failures and experimental status.

Signed off 2026-11-09:
1. The four changes as specified, with finer flop/turn K, OCHS and more preflop
   classes deferred.
2. V3.0 versioned abstractions first (served code, behind stamps, no change for v2).
3. Two training arms (A, C), with the gates above deciding.

Decided 2026-11-12: V3.7–V3.13 (Rust river solver, sub-minimum anchor, river menu F,
the jam read, the pruning arm and the DBR blueprint) are part of v3, not later work.

Decided 2026-11-13: the base-arm pick waits until V3.10, the first step that needs
it. V3.6–V3.9 are built and tested for both arms, and V3.6 measures C's extra serving
memory first.

Decided 2026-11-15: the base arm is C at 300M (see "V3.10 — done").

Still open:
- Phase 7 deployment, which needs its own sign-off.
