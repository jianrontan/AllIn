# backend/bot/src/evaluation/lbr_solver.py
"""
LBR exploitability of the RiverSubgameSolver (the rigorous Phase-4 scoreboard).

LBREvaluator measures how exploitable the BLUEPRINT is (an off-tree greedy
exploiter plays it). This subclass swaps the VICTIM on the river: instead of the
bot sampling its blueprint, it plays the RiverSubgameSolver. The number it
returns is directly comparable to the blueprint's LBR mbb/hand -- a LOWER on the
solver going DOWN vs the blueprint is the go/no-go signal.

WHAT CHANGES vs the base LBR loop
---------------------------------
The base only tracks the exploiter's belief about the bot (`BotRange`, used by the
exploiter to best-respond). The solver-as-victim needs the MIRROR beliefs the bot
itself would hold:
  * opp_belief -- the bot's Bayesian belief about the EXPLOITER (RangeTracker with
    the bot's cards removed), updated from the exploiter's PRE-RIVER actions under
    the blueprint model. Its confidence falls when the exploiter plays off-model
    (the regime the widening is for).
  * hero       -- the bot's own blueprint-reach range (RangeTracker spanning all
    hands), updated from the bot's pre-river actions.
Both are SNAPSHOT at river entry (frozen before river betting), exactly as
GameSession does, and handed to the solver. On the river the bot plays
solver.decide(...) (full solve -> EV gate -> emit), and its engine action is
mapped back into LBR's (char, add, aggr) accounting.

This subclass REIMPLEMENTS play_hand (the base stays untouched / trusted) but
reuses _lbr_decide / _bot_act / equity_vs_range / restricted_probs / _resolve /
BotRange unchanged. Slow (~1 solve per river bot decision) -> an offline job.
"""
from .lbr import (LBREvaluator, BotRange, SB, BB, MAX_AGGR_PER_STREET, _FULL_DECK)
from ..cfr.poker_game import STARTING_STACK
from ..cfr.keys import make_info_set_key
from ..cfr import translation
from ..game.range_tracker import RangeTracker


def _grid_action_for_char(char, to_call):
    """Map a pattern char to the blueprint GRID action the bot models the opponent
    as having taken (for the opponent-belief update)."""
    if char == 'k':
        return 'check'
    if char == 'c':
        return 'call'
    if char == 'f':
        return 'fold'
    if char == 'a':
        return 'allin'
    size = {'s': 'small', 'm': 'medium', 'l': 'large'}.get(char, 'medium')
    return ('raise_' if to_call > 0 else 'bet_') + size


def _grid_legal(to_call, can_aggr):
    """The blueprint-grid legal set the bot assumes (for both belief updates and
    the solver's EV-gate baseline lookup)."""
    if to_call > 0:
        legal = ['fold', 'call']
        if can_aggr:
            legal += ['raise_small', 'raise_medium', 'raise_large', 'allin']
    else:
        legal = ['check']
        if can_aggr:
            legal += ['bet_small', 'bet_medium', 'bet_large', 'allin']
    return legal


class SolverLBREvaluator(LBREvaluator):
    def __init__(self, blueprint_db, solver, seed=None, flop_runout_samples=120):
        super().__init__(blueprint_db, seed=seed, flop_runout_samples=flop_runout_samples)
        self.solver = solver
        self.diag = {'reached_river': 0, 'bot_river_decision': 0}

    def _river_char(self, add, to_call, pot, stack_actor):
        """Pattern char for a custom river bet/raise of `add` chips (mirror the
        translating victim: nearest bracket on the pseudo-harmonic grid)."""
        eff = translation.eff_fraction(add, to_call, pot)
        grid = list(translation.POSTFLOP_GRID)
        allin_frac = translation.eff_fraction(stack_actor, to_call, pot)
        if allin_frac > 1.0:
            grid.append(('a', allin_frac))
        return translation.nearest_char(eff, grid)

    def _solver_bot_act(self, bot_seat, bot_hand, vis, invested, stack, committed,
                        to_call, num_aggr, pattern, river_villain, river_hero,
                        river_entry_pot, river_specs):
        """The bot's RIVER action via the solver. Returns (char, add, aggr) +
        the snapped grid action (for the exploiter's belief update)."""
        pot = sum(invested)
        can_aggr = num_aggr < MAX_AGGR_PER_STREET and stack[bot_seat] > max(0, to_call)
        legal = _grid_legal(to_call, can_aggr)
        pos = self._pos(bot_seat)
        key = make_info_set_key(
            3, pos, self.cards.get_bucket(list(bot_hand), None),
            self.cards.get_bucket(list(bot_hand), vis), pattern)
        entry_stack = STARTING_STACK - invested[bot_seat] + committed[bot_seat]  # river-entry behind
        ps = {
            'street': 'river', 'community': vis, 'hole_cards': list(bot_hand),
            'riverEntryPot': river_entry_pot,
            'riverEntryStacks': (entry_stack, entry_stack),
            'botSeat': bot_seat, 'opp_range': river_villain, 'hero_range': river_hero,
            'riverPath': list(river_specs),
        }
        action = self.solver.decide(key, legal, ps)

        if action == 'check':
            return 'k', 0, False, 'check'
        if action == 'fold':
            return 'f', 0, False, 'fold'
        if action == 'call':
            return 'c', to_call, False, 'call'
        if action == 'allin':
            return 'a', stack[bot_seat], True, 'allin'
        # custom bet/raise -> additional chips this street
        total = float(action.rsplit('_', 1)[1])
        add = max(0.0, total - committed[bot_seat])
        if add >= stack[bot_seat]:
            return 'a', stack[bot_seat], True, 'allin'
        char = self._river_char(add, to_call, pot, stack[bot_seat])
        return char, add, True, _grid_action_for_char(char, to_call)

    def play_hand(self, lbr_seat, lbr_hand, bot_hand, board):
        bot_seat = 1 - lbr_seat
        invested = [SB, BB]
        stack = [STARTING_STACK - SB, STARTING_STACK - BB]
        botrange = BotRange(lbr_hand, self.cards)              # exploiter's belief (base)
        opp_belief = RangeTracker(bot_hand, self.cards)        # bot's belief about exploiter
        hero = RangeTracker((), self.cards)                    # bot's own range
        river_villain = river_hero = None
        river_entry_pot = 0.0
        river_specs = []
        folded = None

        street = 0
        while street <= 3:
            vis = self._visible_board(board, street)
            if street > 0:
                botrange.reveal(vis)
                if street < 3:
                    opp_belief.reveal(vis)
                    hero.reveal(vis)
            if street == 3:                                    # river entry: freeze beliefs
                opp_belief.reveal(vis)
                hero.reveal(vis)
                river_villain = opp_belief
                river_hero = hero
                river_entry_pot = sum(invested)
                river_specs = []
                self.diag['reached_river'] += 1
            committed = [SB, BB] if street == 0 else [0, 0]
            num_aggr = 0
            pattern = ''
            actor = 0 if street == 0 else 1
            need = {0, 1}
            guard = 0
            while need and folded is None:
                guard += 1
                if guard > 16:
                    break
                other = 1 - actor
                if stack[actor] <= 0:
                    need.discard(actor)
                    actor = other
                    continue
                to_call = max(0, committed[other] - committed[actor])
                # Real aggression availability at this node (raise cap / stack), so
                # the belief updates use the legal set the bot actually models the
                # opponent as having -- not a phantom one that always includes raises.
                can_aggr = num_aggr < MAX_AGGR_PER_STREET and stack[actor] > max(0, to_call)

                if actor == lbr_seat:
                    char, add, aggr = self._lbr_decide(
                        lbr_seat, lbr_hand, vis, street, invested, stack, committed,
                        to_call, num_aggr, pattern, botrange, bot_seat)
                    # The bot's belief about the (pre-river) exploiter action.
                    if street < 3:
                        snapped = _grid_action_for_char(char, to_call)
                        glegal = _grid_legal(to_call, can_aggr)
                        if snapped in glegal:
                            opp_belief.observe(self.restricted_probs, snapped, street,
                                               self._pos(lbr_seat), pattern, glegal, vis)
                elif street < 3:
                    char, add, aggr = self._bot_act(
                        bot_seat, bot_hand, vis, street, invested, stack, committed,
                        to_call, num_aggr, pattern, botrange, lbr_seat)
                    # The bot's own pre-river range reach.
                    snapped = _grid_action_for_char(char, to_call)
                    glegal = _grid_legal(to_call, can_aggr)
                    if snapped in glegal:
                        hero.observe(self.restricted_probs, snapped, street,
                                     self._pos(bot_seat), pattern, glegal, vis)
                else:
                    # RIVER: the bot plays the solver.
                    self.diag['bot_river_decision'] += 1
                    char, add, aggr, snapped = self._solver_bot_act(
                        bot_seat, bot_hand, vis, invested, stack, committed,
                        to_call, num_aggr, pattern, river_villain, river_hero,
                        river_entry_pot, river_specs)
                    # Keep the exploiter's belief consistent with the bot's river action.
                    glegal = _grid_legal(to_call, can_aggr)
                    if snapped in glegal:
                        botrange.observe(self.restricted_probs, snapped, street,
                                         self._pos(bot_seat), pattern, glegal, vis)

                if char == 'f':
                    folded = actor
                    break
                add = min(add, stack[actor])
                invested[actor] += add
                committed[actor] += add
                stack[actor] -= add
                pattern += char
                if street == 3:                                # record the river path
                    if char == 'k':
                        river_specs.append('check')
                    elif char == 'c':
                        river_specs.append('call')
                    elif char == 'a':
                        river_specs.append('allin')
                    else:
                        river_specs.append(('raise' if to_call > 0 else 'bet', add))
                if aggr:
                    num_aggr += 1
                    need = {other}
                else:
                    need.discard(actor)
                actor = other

            if folded is not None:
                break
            if min(stack) <= 0:
                break
            street += 1

        return self._resolve(lbr_seat, lbr_hand, bot_hand, board, invested, folded)
