import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.cfr.poker_game import PokerGame, STARTING_STACK

game = PokerGame()

# Scenario: P1 (postflop acts first) has a SHORT stack and goes all-in.
# P0 has full stack -> calling is fine. Reverse: P0 made a big bet, then P1
# all-in for LESS, P0 "calls" the smaller amount. Check call cost vs stack.

# Postflop street=1. P1 acts first. P1 bets big, P0 goes all-in for MORE than P1
# can cover. Then P1 must call but has limited stack.
# history before P1's call: ['bet_large', 'allin']
sp = 50.0
# P0 went all-in. P1 prev invested heavily.
for p1_prev in [0, 50, 100, 150, 190, 197]:
    p0_prev = 0.0
    hist = ['bet_large', 'allin']
    cur = game._acting_player(len(hist), 1)  # who calls
    call_cost = game._action_cost('call', 1, hist, sp, cur, p0_prev, p1_prev)
    rem = STARTING_STACK - (p1_prev if cur==1 else p0_prev) - \
        game.get_player_contribution_this_round(hist, 1, sp, cur, p0_prev, p1_prev)
    legal = game.get_legal_actions(1, hist, sp, cur, STARTING_STACK-p0_prev, STARTING_STACK-p1_prev, p0_prev, p1_prev)
    flag = "  <-- CALL COST EXCEEDS STACK" if call_cost > rem + 1e-6 else ""
    print(f"p1_prev={p1_prev} cur={cur} call_cost={call_cost:.2f} remaining={rem:.2f} legal={legal}{flag}")

# Also: does an all-in by the SHORT stack get capped at their actual chips?
print("\n-- short-stack all-in amount capping --")
for prev in [0, 100, 195, 199, 205]:
    amt = game._allin_amount([], 1, sp, 0, prev, 0.0)
    print(f"p0_prev={prev} allin_amount={amt} (should be {max(0,STARTING_STACK-prev)})")
