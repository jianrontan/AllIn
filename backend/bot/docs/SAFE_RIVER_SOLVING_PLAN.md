# Safe river subgame solving (Phase 5a) — scope

**Goal.** Make the river solver *provably no more exploitable than the blueprint*, instead of
the current "unsafe v1" that solves against the tracked ranges directly. This is the
principled fix for the known **narrow-but-confidently-wrong belief** danger (L2/Inference #2):
today a sharp wrong villain range produces a sharp best-response to a phantom, which can be
*more* exploitable than just playing the blueprint.

**Why the river is the easy case.** The river is the last street, so a re-solved river subgame
runs to **showdown** — there is **no leaf-value function** to learn (the thing that makes
turn/flop safe solving hard). All we need are the blueprint's *river-entry counterfactual
values*, which are a one-street rollout.

## The technique: the re-solving gadget (reach / max-margin)
At the subgame root, give the **opponent** an explicit "opt-out" alternative for each of their
hands, paying the blueprint's counterfactual value (CFV) that hand would get by *not* entering
this subgame. Re-solving the *gadget* game then cannot make the opponent worse off than the
blueprint already guaranteed → the resulting strategy is no-more-exploitable than the blueprint
(Burch/Moravčík/Brown, "safe and nested subgame solving"). Reference: roadmap Phase 5a.

## Pieces to build

1. **Blueprint river-entry CFVs (the new piece).** For the villain's range at river entry,
   compute `v_blueprint(h)` = the counterfactual value of villain hand `h` under the blueprint
   from this node to showdown, against the bot's blueprint range. This is a single-street
   expectation over the blueprint's river strategy (which we can read via
   `blueprint_strategy_on_tree`, already in `subgame/blueprint_projection.py`) and the showdown
   kernel (`evaluation/showdown_kernel.py`). Output: a length-H vector of opt-out values.
   - Cost: one pass over the river tree with the blueprint strategy + the cached board winrates
     (`postflop_v2._RIVER_BOARD_CACHE` already memoizes the per-board equity). Cheap.
   - Subtlety: the CFVs must be on the SAME (hero, villain) reach the solve uses (river-entry
     snapshots), so they're consistent with the gadget constraint.

2. **The gadget tree.** Add a synthetic root layer where the villain chooses, per hand, between
   "enter the subgame" and "take `v_blueprint(h)`". Implement as a thin wrapper over
   `river_tree` (a new terminal per villain hand paying the opt-out value), or as a constraint
   in `RiverCFR` (a per-hand value floor for the villain). The latter is lighter: add an
   opt-out value vector to `RiverCFR` and, each iteration, give the villain `max(subgame_value,
   v_blueprint)` per hand at the root (the reach-gadget formulation).

3. **Re-solve + read-off.** Run `RiverCFR` on the gadget game (same kernel), read the bot's
   action as today (`read_action_strategy`). The EV-gate against the blueprint baseline stays
   (it's complementary), and the just-added non-converged-margin (H1) still applies.

4. **Validation (the gate to ship).** The safety claim is testable directly:
   - On a battery of river spots, measure exploitability of (a) the blueprint's river strategy
     and (b) the gadget-solved strategy via `RiverCFR.exploitability` — the gadget must be
     `<=` the blueprint on EVERY spot (the no-more-exploitable guarantee).
   - Confirm the narrow-wrong-belief case: feed a deliberately wrong sharp villain range; the
     unsafe v1 should over-exploit (and be punishable), the gadget should clamp to ~blueprint.
   - Maniac/LBR regression: no loss vs the current served bot on the live harness.

## Effort + sequencing
- Piece 1 (CFVs) is the only genuinely new code; pieces 2–3 reuse `river_tree`/`RiverCFR`.
- Bounded multi-session task, not a rewrite. Order: CFVs → gadget value floor in RiverCFR →
  wire into `solve_for_action` (behind a flag) → exploitability validation → flip the flag.
- It SUPERSEDES the L2/#2 heuristic (which I deliberately did not ship because no cheap
  heuristic addresses the narrow-wrong case — the gadget is the real fix).

## Status — SHIPPED. Served `safe_gadget=True, gadget_anchor='auto'` (policy B) since 2026-06-10; the
## "default OFF" notes below predate the flip and are historical. `ALLIN_GADGET_ANCHOR` defaults to 'auto'.

All four pieces are implemented and tested:
- **Piece 1 (CFVs):** `blueprint_projection.blueprint_cfv` — per-villain-hand opt-out CFV
  at the river-entry root under both players on the blueprint (reuses
  `blueprint_strategy_on_tree` + `RiverCFR._eval`). Zero-sum tested.
- **Piece 2 (gadget floor):** `RiverCFR.run_gadget` — the re-solve (reach) gadget: the
  villain regret-matches per hand between Follow and Terminate(opt-out). Safety-property
  tested (villain BR vs gadget hero ≤ vs blueprint).
- **Piece 3 (wiring):** `solve_control.solve_river_gadget` + `RiverSubgameSolver(safe_gadget=,
  gadget_anchor=)`. Two anchors built (the fork from the design review):
  - `gadget_anchor='blueprint'` — UNIFORM card-removal villain range → the provable
    guarantee (safe vs ANY villain hand, robust to a wrong belief).
  - `gadget_anchor='belief'` — the tracked river-entry belief (read-aware, weaker).
- **Piece 4 (validation):** `tests/test_safe_river_gadget.py` — the exploitability battery.

**MEASURED (3 spots × correct/wrong belief, villain BR over the TRUE range, chips/matchup):**

| scenario | blueprint | unsafe-v1 | gad(belief) | gad(blueprint) |
|----------|-----------|-----------|-------------|----------------|
| belief CORRECT | 5.5 / 4.4 / 9.0 | **0.8 / 1.1 / −0.9** | 1.6 / 1.9 / −0.6 | 1.6 / 1.9 / −0.6 |
| belief WRONG   | 5.5 / 4.4 / 9.0 | 8.1 / 8.2 / 16.1 | 9.3 / 10.3 / 16.3 | **1.6 / 1.9 / −0.6** |

**Conclusion — ship `gadget_anchor='blueprint'` for safety.** It is the ONLY option
bounded ≤ blueprint in EVERY case (hard gate passes, worst excess +0.0000), and it is
belief-INDEPENDENT (correct/wrong rows identical → the guarantee). unsafe-v1 AND
gad(belief) both become *more* exploitable than the blueprint under a wrong belief
(the narrow-wrong leak — confirmed in 3/3 spots); gad(belief) gives NO safety benefit
over unsafe (worst of both). Cost of the blueprint anchor: when the read is right,
unsafe-v1 exploits a touch more (0.8 vs 1.6) — but both crush the blueprint baseline.

**Default stays `safe_gadget=False` (unsafe-v1 served) until the user flips it** — that
choice is the safety-vs-exploitation tradeoff: a robust guarantee vs maximal read
exploitation against a mostly-on-model human. The recommended flip is
`safe_gadget=True, gadget_anchor='blueprint'`.

### Adaptive anchors + how to benchmark them (2026-06-10)

Two adaptive `gadget_anchor` modes auto-switch between exploit and safe per spot:
- `'confidence'` (A) — exploit (unsafe-v1) on a trusted+informative read, else clamp to
  the blueprint anchor. Cheap; clamps even where the exploit was safe.
- `'auto'` (B) — like A, but on an untrusted read runs a per-spot SELF-CHECK (villain BR
  to the unsafe strategy vs to the blueprint, over the uniform range) and STILL exploits
  when that is within blueprint. Best-of-both; pays the self-check's 2 BR walks.

The validation battery (`test_safe_river_gadget.py`) adds an `auto` column and confirms it
picks unsafe-v1 on correct beliefs and gad(blueprint) on wrong ones (≤ blueprint always).

**Two measurement axes** (`tests/compare_gadget_policies.py` runs off/A/B live vs a maniac):
- SAFETY (worst-case exploitability) — the battery's exact single-board BR. The clean
  discriminator; isolates the river decision.
- EV (live BB/hand vs an opponent) — variance-DOMINATED by preflop stack-offs and only
  ~6% of hands reach a river solve, so the BB/hand gap needs a huge N and is a weak
  instrument for a river-only change. Use the harness's **anchor-decision breakdown** +
  **latency** instead (both clean at low N). Smoke (200 hands, maxbet maniac): A clamps
  ALL river solves to safe (over-conservative); B's self-check finds the exploit safe and
  KEEPS exploiting; latency off≈403ms < B≈428ms < A≈609ms (A always runs the full gadget
  solve on an untrusted read; B usually just adds 2 BR walks to the unsafe solve).

## Out of scope (still needs leaf values → Phase 5b)
Turn/flop safe solving needs blueprint CFVs as a leaf-value function (the depth-limited
continual-resolving path). This plan is river-only and complete on its own.
