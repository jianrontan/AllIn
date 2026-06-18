# Turn Safe-Gadget Plan (Phase 5b)

**Status (updated 2026-06-18):** Component #1 (safe gadget) BUILT + wired + gated — it
**halved** the real-game regression (−170 → −76 mbb vs maxbet) but is not ≥0; 2-agent review
found no correctness bug. Research (Modicum / DeepStack / DecisionHoldem) then **re-ordered the
remaining plan** — see §7. The big EV lever is the **multi-valued leaf**, not the gadget.
Siblings: [DEPTH_LIMITED_SOLVER_PLAN.md](DEPTH_LIMITED_SOLVER_PLAN.md),
[NN_LEAF_PLAN.md](NN_LEAF_PLAN.md), [../backend/bot/docs/SAFE_RIVER_SOLVING_PLAN.md](../backend/bot/docs/SAFE_RIVER_SOLVING_PLAN.md).

---

## 1. Why the naïve turn revival failed (the gate, 2026-06-17)

Paired CRN gate on the retrain snapshot `snap_37500000.db` (lossless-169 / 30-flop-24-turn),
ARGMAX, fixed-iters; null test (turn disabled both arms) = `+0.0 ± 0.0`, so the instrument is clean:

| Opponent | PAIRED (turn − river) | Read |
|---|---|---|
| `maxbet` (off-model maniac) | **−170.1 ± 52.5 mbb** (−3.2σ) | turn solver **HURTS** |
| `blueprint` (on-model) | **−72.5 ± 98.2 mbb** (−0.7σ) | **≈ neutral** |

- Heavier solve (n30 / rivers16 / it200) gave the **same −170** vs maxbet → **structural, not
  under-resolution**. The internal EV gate got *more* confident (+44.8 chips/deviation) while the
  realized result got *more* negative → the leaf gate **systematically over-predicts** deviations.
- **Dominant failure = off-model fragility:** neutral vs the opponent it models, badly negative vs
  off-model aggression. The solve overfits the assumed blueprint range.
- **Root cause of the over-prediction:** the turn leaf value (`cfv.turn_leaf_value_exact` /
  `cfv.turn_leaf_matrix_both`) assumes **both players play the *blueprint* on the river** — but the
  served bot plays the **river *solver*** on the river. The leaf is *self-inconsistent* with deployment.
- **Latency:** p50 7.3 s, max 32 s vs the blueprint opponent (high-SPR turns build big trees) → also
  blows the ~5 s live budget. Speed is a co-equal requirement, not an afterthought.

This is exactly what **Libratus (safe re-solving)** and **Pluribus (multi-continuation leaves)**
engineer around — so it's a fixable design gap, not a dead end.

## 2. The core asymmetry (why river is easy, turn is hard)

- **River subgame** ends at a **terminal showdown** → leaf value is *exact*. `solve_river_gadget`
  (`solve_control.py:78`) therefore gives **safety** (opt-out at the exact blueprint river-entry CFV)
  *and* **+EV** (optimises against an exactly-valued game). That's why it ships.
- **Turn subgame** depth-limits at "turn betting closed → river to come." `TurnCFR._terminal`
  (`turn_cfr.py:62`) collapses the river into a **leaf-value matrix** `M0/M1`. The leaf value is now
  an *estimate*, and §1 shows the current estimate (blueprint-river continuation) is wrong for how the
  bot actually plays. **So the turn needs (a) a safety floor that doesn't trust the leaf, and (b) a
  leaf value consistent with deployment.**

## 3. Design — three components

### Component 1 — Turn safe-gadget (safety floor / off-model hedge) — DO FIRST
Generalise the shipped river gadget to the turn root. The opt-out machinery is street-agnostic:

| River (exists) | Turn (build) |
|---|---|
| `RiverCFR.run_gadget(hero_reach, villain_reach, optout, villain_seat)` (`river_cfr.py:140`) | `TurnCFR.run_gadget` — **mostly inherited** (`TurnCFR(RiverCFR)`, `turn_cfr.py:33`); the gadget node logic is identical, it sits at the turn root |
| `blueprint_cfv(tree, ba, raw_strategy, reach0, reach1, villain_seat)` = blueprint **river-entry** CFV (`blueprint_projection.py:83`) | `blueprint_cfv_turn` = blueprint **turn-entry** CFV: project the blueprint's *turn+river* play onto the turn tree and eval the villain's turn-entry value. Reuses `cfv.turn_leaf_value_exact` for the river leg |
| `solve_control.solve_river_gadget` (`solve_control.py:78`) | `solve_control.solve_turn_gadget` — same structure |

Villain opt-out at the blueprint turn-entry value **bounds exploitability ≈ ≤ blueprint** (degraded
only by leaf-value error in the opt-out, which is computed exactly). This is the smallest lift, reuses
shipped code, **cannot regress serving**, and directly attacks the −170. Wire it through
`TurnSubgameSolver` exactly like the river path opts into `safe_gadget=True, gadget_anchor='auto'`.

**Gate after #1:** re-run `measure_turn_match.py --paired --opponent maxbet`. Expect −170 → ≥ 0 (safe).

### Component 2 — Nested river-solve leaf value (the +EV fix) — DO SECOND
Replace the blueprint-river leaf with the value of an actual **nested `solve_river_gadget`** at each
turn leaf (Libratus-style nested safe solving), so the leaf is *self-consistent* with how the bot plays
the river:
- In `TurnCFR._terminal` / the `leaf_matrix_fn`, source `M0/M1` from a nested river-gadget solve at the
  leaf's `(pot, leaf_stacks, board5)` instead of `turn_leaf_value_exact` (blueprint-river).
- **Cache aggressively:** many turn leaves share a river-entry `(pot, stacks, board5)`. Extend the
  existing `TurnCFR._leaf_cache` (`turn_cfr.py:52`, keyed by pot/stacks) to memoise the *solved* value.
- Optional (Pluribus robustness): give the villain a small **menu of continuation strategies** at the
  leaf, not just one — further hardens against off-model play.

**Gate after #2:** re-run the paired gate **offline** (generous `--turn-budget`, n≥48/rivers≥16).
Expect it to go **+EV** if the leaf-consistency theory holds. *This is the real test of the redesign.*

### Component 3 — NN leaf (live speed) — DO LAST, only if #2 is +EV offline
Nested solving per leaf is too slow live. Replace it with a learned net regressing the `M0/M1` leaf
matrices ([NN_LEAF_PLAN.md](NN_LEAF_PLAN.md)) — range-independent, ~ms inference. Trained offline against
the nested-solve (or rollout) reference from #2.

## 4. The <10 s live-budget plan (parallel track)

Latency is gated regardless of correctness. Levers, cheapest first:
1. **Turn SPR cap** (`TURN_MAX_SPR`, currently 10) — high-SPR turns build the biggest trees and time
   out; the cap bounds tree size. Tune against the latency/coverage tradeoff on an **idle** core
   (current measurements were under training contention → inflated).
2. **Leaf caching** (#2) — amortises the dominant cost (the per-leaf river pass) across a hand/session.
3. **NN leaf** (#3) — the real answer for live: collapses the per-leaf solve to a net eval.
4. **Bake** for the deployment box (few vCPUs) if real-time stays infeasible.
The hard wall-clock abort (`TURN_TIME_BUDGET`, eager deadline-checked leaf pre-build) already degrades
over-budget solves cleanly to the blueprint — keep that as the safety net.

## 5. Build order & exit criteria

1. **#1 turn gadget** → paired gate vs maxbet ≥ 0 (safe, no regression). *Ship-safe milestone.*
2. **#2 nested leaf** → offline paired gate **> 0 beyond ~2σ** vs maxbet AND ≥ blueprint vs blueprint.
   *This is the "turn solving is actually useful" gate.*
3. **#3 NN leaf** + SPR/latency tuning → live solve p90 < ~10 s with #2's EV preserved.
4. Final serve gate: paired turn-vs-river **and** beats the **river-only** baseline head-to-head
   (the bar the naïve revival missed), on the shipped blueprint.

## 6. Risks / open questions
- **Safety with approximate leaves:** the gadget bound degrades with opt-out leaf-value error. The
  opt-out is computed exactly (project real blueprint play), so the bound should hold; verify on the
  wrong-belief tail as the river gadget did (`test_safe_river_gadget.py` analogue).
- **Nested-solve cost** may make even the *offline* #2 gate slow (a river solve per leaf × many leaves);
  caching + sampling rivers mitigates, but watch wall-time.
- **Does #2 actually flip to +EV?** If the leaf-consistency fix still leaves it ≈0 vs the modeled
  opponent, the marginal value over the river-only stack may not justify the complexity — measure
  before building #3.

---

## 7. Component #1 RESULT + research-driven re-order (2026-06-18)

**Component #1 (safe gadget) shipped + gated.** `blueprint_cfv_turn` + `solve_turn_gadget` +
inherited `TurnCFR.run_gadget`, wired into `solve_turn_for_action` behind `self.safe_gadget`; safety
test `test_safe_turn_gadget.py` PASS. Paired gate vs maxbet on snap_52500000 (8000 hands):
**unsafe −170±52 → gadget −76±41.** Halved the regression, not ≥0. 2-agent review: NO correctness
bug; residual is (a) the live gadget hard-codes the **'belief' anchor** (no `auto` → only clamps,
never exploits) and (b) the **single-continuation leaf** (below).

### What the working bots actually do (research)
- **Slumbot does NOT solve** — it's a giant precomputed blueprint (≈250k core-hours, 2 TB RAM),
  lookup table, no search. "Perform like Slumbot" = a much bigger blueprint (out of our compute reach).
- **Modicum (Brown–Sandholm 2018)** — depth-limited turn solving with **multi-valued states**: at the
  depth limit the opponent picks among **4 bias continuations** (blueprint, fold-biased=×10 fold-prob
  then renorm, call-biased, raise-biased). Leaf evaluated by **rollout OR a DNN**. 700 core-hours
  offline, 4-core real-time, beat bigger bots. THE cheap technique.
- **DeepStack** — continual re-solving + a learned **CFV network** at the leaf (the expensive path).
- **DecisionHoldem (2022, open-source)** — modern recipe = **safe depth-limited solving + diverse
  opponents** (= our gadget **plus** multi-valued, composed). Turn = 10k real-time iters.

### The flaw in the original §3/§5 ordering (corrected)
The original plan said "bake a high-fidelity single-continuation leaf" closes part of the −76. **It
does not.** The 1c EV gate already values deviations with `turn_leaf_value_exact` (the EXACT,
all-rivers, fine SINGLE-continuation leaf) and it STILL over-predicts (+40 internal vs −76 realized).
So the EV gap is **not coarseness — it's the single-continuation assumption**. Two orthogonal axes:
- **EV gap ⟸ single-continuation** → only **multi-valued** fixes it (not fidelity, not the gadget).
- **Latency wall ⟸ the live rollout** → only **precompute (bake or DNN)** fixes it.
They fight: multi-valued = ~4× the rollout → worse latency. So "max EV under 10s" is impossible from
live rollouts; the multi-valued leaf MUST be precomputed. (That's why Modicum also built a DNN.)

### Re-ordered build (supersedes §5)
1. **DE-RISK FIRST (cheap, offline): the multi-valued leaf.** Build the 4-bias-continuation leaf
   (reweight the OPPONENT's projected blueprint river strategy: fold×10 / call×10 / raise×10 /
   baseline; combine worst-for-hero) and run the paired gate **offline, generous budget (ignore the
   10s cap)** vs DIVERSE opponents (maxbet/jam/widejam + a passive style). **Make-or-break: does it
   flip −76 toward ≥0/positive?** If not, turn solving isn't worth it for us — STOP. If yes, we know
   the ceiling. Cheap to build (wrap `cfv.turn_leaf_matrix_both` ×4 + combine).
2. **THEN latency:** make the multi-valued leaf fast — **bake** the 4-continuation leaf values offline
   (Modicum's preflop table was 240 MB; our coarse abstraction is likely tabulatable), else **DNN**
   ([NN_LEAF_PLAN.md](NN_LEAF_PLAN.md)). A *latency* decision made AFTER the EV is proven.
3. **Compose with the gadget + add the `auto` anchor** (mirror river `_safe_solve`) — DecisionHoldem
   does both (safe + diverse). Gadget = safety wrapper; multi-valued = EV engine.
4. **MED latency fix** (independent): `blueprint_cfv_turn` runs a tree-walk OUTSIDE `turn_time_budget`
   — bound it before any live-serving.

**Strategic note:** done with the SINGLE-continuation leaf, turn solving is marginal (the −76 polish).
Done with **multi-valued** (the proven-cheap technique), it's how a small blueprint outperforms bigger
ones (Modicum) — so it's the real lever, gated on the §7.1 de-risk experiment paying off.
