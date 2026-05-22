# Bug Log

A running record of notable bugs found in the AllIn poker-AI codebase, their
root causes, fixes, and the lessons learned. Kept for future debugging
reference and as a project narrative.

**Entry format** — each bug gets: ID, date, area, severity, status, a one-line
summary, the symptom, root cause, a concrete walkthrough, the fix, why it
wasn't caught earlier, retrain impact, and lessons. Append new bugs at the top.

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
