# Poker AI Roadmap

Last updated: 2026-05-25

Status legend: ✅ done · 🚧 in progress · 📅 planned

This roadmap tracks the arc from a static blueprint to online, subgame-solving
play. For how the current system works see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md);
for commands see [../USER_GUIDE.md](../USER_GUIDE.md).

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
| CFR+ trainer | `src/cfr/blueprint_trainer.py` — external-sampling MCCFR+, DCFR discounting | ✅ |
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

## Phase 3 — Hand-level range tracking 🚧 IN PROGRESS

The prerequisite for subgame solving: a hand-level Bayesian belief over the
opponent's hole cards, which a river solver consumes as its input range.

| Component | File | Status |
|---|---|---|
| Hand-level Bayesian range tracker | `src/game/range_tracker.py` (`RangeTracker`) — per-hand weights, card removal, blueprint-model Bayesian updates, confidence score, equity-vs-range | ✅ |
| GameSession integration | `game_session.py` — per-hand tracker, `observe` on human actions, `reveal` on streets, persisted in session JSON | ✅ |
| Confidence-aware consumer | `bot_strategy.py` (`ConfidenceAwareStrategy`) — blueprint while confident, equity-vs-range fallback when confidence collapses | ✅ |
| "Bot's read" UI | `public_view().botRead` + `AiGame.jsx` panel (confidence + top hands) | ✅ |
| Off-tree / confidence scaffolding (older) | `src/subgame/off_tree_detector.py`, `confidence_detector.py` | 🚧 partial |

## Phase 4 — Subgame solving 📅 NEXT

Improve on the blueprint at runtime by re-solving the current spot with full
information (real pot, real stacks, the Phase-3 range), fixing the M1
abstraction loss.

| Component | File | Status |
|---|---|---|
| River endgame solver (unsafe v1) | `src/subgame/` — small river tree, vectorized CFR+, ranges from `RangeTracker` | 📅 to build |
| Depth-limited turn/flop solving | blueprint counterfactual values as leaf values | 📅 to build |
| Safe / nested subgame solving | adversarial root + opt-out values (gadget) | 📅 to build |

Approach: when the spot warrants it (and the compute budget allows), solve a
**depth-limited** subgame with a **finer action abstraction and real stacks**
instead of reading the blended blueprint. The bot's hole cards already flow
through `public_state`, so the solver drops in via the existing `BotStrategy`
interface. Background: [1], [2].

---

## Phase 5 — Online 1v1 play on AWS 📅 PLANNED

Deploy for real-time online heads-up play.

- Swap `InMemorySessionStore` for a Redis/DynamoDB-backed store (multi-process).
- Consider a WebSocket transport for live play (the `game/` engine is already
  transport-agnostic).
- **Unrestricted human bet sizing**: widen the thin `{action, size}` contract to
  `{action, amount}` — a localized change in `GameSession` and the API.

---

## References

[1] [Depth-Limited Solving for Imperfect-Information Games](https://dl.acm.org/doi/10.5555/3327757.3327865)
[2] [Safe and Nested Subgame Solving for Imperfect-Information Games](https://proceedings.neurips.cc/paper_files/paper/2017/file/7fe1f8abaad094e0b5cb1b01d712f708-Paper.pdf)
