# backend/bot/tests/test_lever_a_oracle.py
"""
Lever A oracle fuzz: validate a STATE-THREADED derivation of (legal actions,
per-action cost, pot, contributions, to_call, min_raise) against the existing
HISTORY-BASED engine functions, at every node of many random games.

The point of Lever A is to stop replaying `history` to re-derive chip math in the
CFR hot path; instead we thread the within-street state down the recursion. This
test pins that the threaded state reproduces the engine EXACTLY (bit-identical) —
once it does, the same transitions move into poker_game.py + cfr, gated again by
the seed-compare harness.

The state derivation here reuses the engine's SIZING helpers (get_preflop_bet_amounts
/ BET_MULTIPLIERS), so this isolates the risk to the STATE TRACKING (pot /
contributions / to_call / min_raise / legal-action selection), not the sizing
formulas (which are unchanged).

Run: python tests/test_lever_a_oracle.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.poker_game import (
    PokerGame, STARTING_STACK, _is_custom, _custom_total, make_custom_action)

_passed = _failed = 0
_fail_examples = []


def check(cond, msg):
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        if len(_fail_examples) < 25:
            _fail_examples.append(msg)


# The state-threaded logic under test now lives in poker_game.py as real
# methods (init_node_state / state_legal_actions / state_action_cost /
# advance_node_state); this fuzz drives THOSE against the history-based
# engine functions at every node of many random games.


# ---------------------------------------------------------------------------
# Fuzz walk: compare state vs engine at every node
# ---------------------------------------------------------------------------

def test_state_matches_engine():
    g = PokerGame()
    rng = random.Random(20260527)
    cov = {'allin_reopen': 0}        # explicit coverage markers (see end of fn)
    for _ in range(4000):
        g._calc_cache.clear()
        starting_pot = 3.0
        p_inv = [0.0, 0.0]
        street = 0
        history = []
        st = g.init_node_state(0, starting_pot)
        guard = 0
        while street <= 3 and guard < 60:
            guard += 1
            p0_stack = STARTING_STACK - p_inv[0] - st['c'][0]
            p1_stack = STARTING_STACK - p_inv[1] - st['c'][1]
            cp = g._acting_player(len(history), street)
            stack_cp = p0_stack if cp == 0 else p1_stack
            stack_other = p1_stack if cp == 0 else p0_stack

            eng_legal = g.get_legal_actions(street, history, starting_pot, cp,
                                            p0_stack, p1_stack, p_inv[0], p_inv[1])
            st_legal = g.state_legal_actions(street, st, cp, stack_cp)
            check(eng_legal == st_legal,
                  f"legal mismatch st={street} hist={history} eng={eng_legal} st={st_legal}")

            # also pin pot / contributions / to_call against the engine
            eng_pot = g.calculate_current_pot(starting_pot, history, street, p_inv[0], p_inv[1])
            check(eng_pot == st["pot"],
                  f"pot mismatch st={street} hist={history} eng={eng_pot} st={st['pot']}")
            for pl in (0, 1):
                eng_c = g.get_player_contribution_this_round(
                    history, street, starting_pot, pl, p_inv[0], p_inv[1])
                check(eng_c == st["c"][pl],
                      f"contrib p{pl} mismatch hist={history} eng={eng_c} st={st['c'][pl]}")

            if not eng_legal:
                # street transition (engine path), then re-init state for next street
                p0_this = g.get_player_contribution_this_round(
                    history, street, starting_pot, 0, p_inv[0], p_inv[1])
                p1_this = g.get_player_contribution_this_round(
                    history, street, starting_pot, 1, p_inv[0], p_inv[1])
                new_pot = g.calculate_current_pot(starting_pot, history, street, p_inv[0], p_inv[1])
                p_inv = [p_inv[0] + p0_this, p_inv[1] + p1_this]
                starting_pot = new_pot
                street += 1
                history = []
                st = g.init_node_state(street, starting_pot)
                continue

            # per-action cost agreement
            for a in eng_legal:
                eng_cost = g._action_cost(a, street, history, starting_pot, cp, p_inv[0], p_inv[1])
                st_cost = g.state_action_cost(a, street, st, cp, stack_cp)
                check(eng_cost == st_cost,
                      f"cost {a} mismatch hist={history} eng={eng_cost} st={st_cost}")

            # Custom (off-grid) action consistency. The state methods support
            # bet_custom_/raise_custom_ (a future solver / unrestricted sizing),
            # which the engine-legal walk never generates -- so validate that
            # branch (cost + advance: pot & contribution) directly vs the engine.
            bounds = g.custom_bet_bounds(street, history, starting_pot, cp,
                                         p0_stack, p1_stack, p_inv[0], p_inv[1])
            if bounds is not None:
                lo, hi = bounds
                is_raise = (st['bet_to'] - st['c'][cp]) > 0
                for frac in (0.0, 0.37, 0.99):
                    total = min(lo + frac * (hi - lo), hi - 1e-6)
                    cact = make_custom_action(is_raise, total)
                    e_cost = g._action_cost(cact, street, history, starting_pot,
                                            cp, p_inv[0], p_inv[1])
                    s_cost = g.state_action_cost(cact, street, st, cp, stack_cp)
                    check(e_cost == s_cost,
                          f"custom cost {cact} hist={history} eng={e_cost} st={s_cost}")
                    nst = g.advance_node_state(st, cact, street, cp, stack_cp, p_inv)
                    e_pot = g.calculate_current_pot(
                        starting_pot, history + [cact], street, p_inv[0], p_inv[1])
                    check(e_pot == nst['pot'],
                          f"custom pot {cact} hist={history} eng={e_pot} st={nst['pot']}")
                    e_c = g.get_player_contribution_this_round(
                        history + [cact], street, starting_pot, cp, p_inv[0], p_inv[1])
                    check(e_c == nst['c'][cp],
                          f"custom c {cact} hist={history} eng={e_c} st={nst['c'][cp]}")

            action = rng.choice(eng_legal)
            if action == 'allin':
                tc = max(0.0, st['bet_to'] - st['c'][cp])
                # PIN the proven invariant: equal stacks + matched-contribution
                # street advance => a shove always covers the call, so a SHORT
                # all-in (stack < to_call) never occurs. If it ever does, the
                # max(bet_to, match_level) defensive code becomes load-bearing
                # and would need its own equivalence proof.
                check(not (tc > 0 and stack_cp < tc),
                      f"UNEXPECTED short all-in hist={history} stack={stack_cp} to_call={tc}")
                if stack_cp > tc:        # over-the-top shove (reopens) -- the live branch
                    cov['allin_reopen'] += 1
            st = g.advance_node_state(st, action, street, cp, stack_cp, p_inv)
            history = history + [action]
            if g.is_terminal(history, street):
                # Validate the TERMINAL state fed to get_utility (the fuzz's old
                # blind spot): threaded pot / contributions must equal the engine
                # at the terminal history, or get_utility (hence regrets) diverges.
                e_pot = g.calculate_current_pot(starting_pot, history, street, p_inv[0], p_inv[1])
                check(e_pot == st['pot'],
                      f"TERMINAL pot hist={history} eng={e_pot} st={st['pot']}")
                for pl in (0, 1):
                    e_c = g.get_player_contribution_this_round(
                        history, street, starting_pot, pl, p_inv[0], p_inv[1])
                    check(e_c == st['c'][pl],
                          f"TERMINAL c{pl} hist={history} eng={e_c} st={st['c'][pl]}")
                break

    # Coverage marker: the over-the-top (reopening) all-in IS the load-bearing
    # all-in branch, so make sure the fuzz actually exercised it rather than
    # silently never reaching it.
    check(cov['allin_reopen'] > 0,
          f"over-the-top all-in branch never exercised (coverage gap); "
          f"reopen count={cov['allin_reopen']}")


if __name__ == '__main__':
    test_state_matches_engine()
    print(f"\nResults: {_passed} checks passed, {_failed} failed")
    if _fail_examples:
        print("First mismatches:")
        for m in _fail_examples:
            print("  " + m)
    sys.exit(1 if _failed else 0)
