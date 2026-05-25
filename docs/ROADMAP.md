# Poker AI Roadmap

Last updated: 2026-05-22

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
| Card abstraction | `src/abstractions/card_abstractions.py` — 15 preflop equity buckets (`pf_0..pf_14`) + 8 postflop texture buckets (`0–7`) | ✅ |
| Preflop equity precompute | `scripts/compute_preflop_equity.py` | ✅ |
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
| Exploitability evaluator | `src/evaluation/best_response.py` + `tests/run_evaluation.py` | ✅ |

**Design intent:** the `game/` engine has **no Flask imports** and the
`BotStrategy` / `SessionStore` interfaces are deliberately thin so the later
phases (subgame solving, online play, AWS) are additive, not rewrites. The
`BotStrategy` interface already receives full public state, not just the bucketed
key, so a subgame solver is a drop-in replacement.

---

## Phase 3 — Subgame solving 🚧 NEXT

Improve on the blueprint at runtime by re-solving the current spot with full
information (real pot, real stacks), fixing the M1 abstraction loss.

| Component | File | Status |
|---|---|---|
| Off-tree / deviation detection | `src/subgame/off_tree_detector.py`, `subgame_detector.py` | 🚧 scaffolding |
| Blueprint confidence detection | `src/subgame/confidence_detector.py` | 🚧 scaffolding |
| Blueprint adapter for solving | `src/subgame/player_blueprint_adapter.py` | 🚧 scaffolding |
| Depth-limited solver | `src/subgame/subgame_solver.py` | 📅 to build |
| Safe / nested subgame solving | adversarial root + opt-out values | 📅 to build |

Approach: when the spot warrants it (and the compute budget allows), solve a
**depth-limited** subgame with a **finer action abstraction and real stacks**
instead of reading the blended blueprint. Drop it in via the existing
`BotStrategy` interface. Background: [1], [2].

---

## Phase 4 — Online 1v1 play on AWS 📅 PLANNED

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
