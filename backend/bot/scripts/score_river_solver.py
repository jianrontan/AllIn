# backend/bot/scripts/score_river_solver.py
"""
Head-to-head scoring: the RiverSubgameSolver vs the plain blueprint, played
through GameSession (which already wires the solver in as the bot). The solver
sits in the bot seat; the blueprint drives the other seat. Reports the solver's
net in bb and bb/100. Seats alternate each hand, so position averages out.

IMPORTANT: this is the go/no-go HARNESS, not a verdict. Heads-up poker variance
is enormous -- a meaningful number needs thousands of hands (ideally with AIVAT
variance reduction, like run_match.py). The solve is also ~1s+ per river node, so
a big run is a long offline job (run it like training, not inline). A lower-
variance alternative is to add the solver to the LBR victim model (lbr.py) and
re-measure exploitability -- the rigorous Phase-4 scoreboard -- which is the next
measurement step.

Run from backend/bot/:
    python scripts/score_river_solver.py --hands 50 --max-iters 200
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import resolve_blueprint_path
from src.storage.blueprint_db import BlueprintDB
from src.game.game_session import GameSession, advance_bot_turns
from src.game.bot_strategy import BlueprintStrategy
from src.subgame.river_subgame_solver import RiverSubgameSolver


def run(hands, max_iters, time_budget):
    db = BlueprintDB(resolve_blueprint_path(), read_only=True)
    strat_fn = BlueprintStrategy(db).range_model_fn()   # opponent model for the trackers
    blueprint = BlueprintStrategy(db)                   # drives the non-solver seat
    solver = RiverSubgameSolver(db, max_iters=max_iters, time_budget=time_budget)

    session = GameSession.new('score', 'p', strategy_fn=strat_fn)
    rivers = 0
    t0 = time.time()
    for h in range(hands):
        guard = 0
        while session.data['status'] == 'in_hand' and guard < 120:
            if session.is_human_turn():
                seat = session.current_player()
                a = blueprint.decide(session.info_set_key(seat),
                                     session.legal_actions(), {})
                session.apply_action(a)
            else:
                advance_bot_turns(session, solver)
            guard += 1
        if session.data.get('river_entry_bot') is not None:
            rivers += 1
        if session.data['status'] == 'hand_over' and h < hands - 1:
            session.start_next_hand()

    # human_net is the BLUEPRINT seat's cumulative net; the solver's is its neg.
    solver_net_chips = -session.data['human_net']
    solver_bb = solver_net_chips / 2.0
    db.close()
    secs = time.time() - t0
    print(f"\nhands={hands}  rivers_reached={rivers}  "
          f"solver_net={solver_bb:+.1f} bb  ({solver_bb / hands * 100:+.1f} bb/100)  "
          f"in {secs:.0f}s ({secs / max(1, hands):.1f}s/hand)")
    print("NOTE: variance-dominated at this sample; not a verdict. See module docstring.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Score RiverSubgameSolver vs blueprint (head-to-head).")
    p.add_argument('--hands', type=int, default=50)
    p.add_argument('--max-iters', type=int, default=200)
    p.add_argument('--time-budget', type=float, default=8.0)
    args = p.parse_args()
    run(args.hands, args.max_iters, args.time_budget)
