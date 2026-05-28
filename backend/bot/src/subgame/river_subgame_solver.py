# backend/bot/src/subgame/river_subgame_solver.py
"""
RiverSubgameSolver (Phase-4, step 6a) -- the assembly that turns a river
situation into an action, wiring together the kernel, tree, CFR+, ranges, and
solve-control built in steps 1-5.

It is a BotStrategy: off the river (or whenever the solver inputs are missing) it
delegates to the blueprint (BlueprintStrategy.decide); on the river it solves the
actual subgame and reads off the bot's action for its real hand.

ARCHITECTURE (decided step 6): the solver CONSUMES the ranges from public_state
(the villain's RangeTracker + a hero RangeTracker) rather than reconstructing the
betting history itself. That decouples it from history replay and works the same
in the GameSession and LBR contexts (each supplies the ranges its own way).

SCOPE of 6a: the core `solve_for_action` (explicit inputs -> action distribution)
and a fall-back-safe `decide`. NOT yet here (step 6c): the GameSession fields
(river-entry pot/stacks, realized path, the bot's range tracker, river-entry
villain snapshot), exact custom-size action emission, and the LBR victim wiring +
scoring. The EV-gate blueprint baseline and warm-start prior need the
blueprint<->tree bridge (step 6b); until then the solver returns the solved
strategy when the solve converged and otherwise falls back to the blueprint.
"""
import logging

import numpy as np

from ..game.bot_strategy import BlueprintStrategy
from ..abstractions.hand_evaluator import HandEvaluator
from ..abstractions.card_abstractions import CardAbstraction
from ..abstractions.sizing import POSTFLOP_BET_MULT
from ..cfr.poker_game import make_custom_action
from ..evaluation.showdown_kernel import build_board_arrays
from .river_tree import build_river_tree, is_sized, sized_chips, DEFAULT_MENU
from .river_cfr import RiverCFR  # noqa: F401  (re-exported convenience)
from .solve_control import solve_river, ev_gate, hand_action_evs
from .range_inputs import (
    project_tracker, blend_villain, hand_index_map, hand_row,
    read_action_strategy, DEFAULT_TEMPER_BETA)

_LOG = logging.getLogger(__name__)


def blueprint_to_tree_dist(bp_dist, node):
    """Redistribute a blueprint action distribution (over ENGINE actions) onto the
    tree node's action menu -- the EV-gate baseline ('what the blueprint would do
    here'). check/fold/call/allin map directly; a bet_*/raise_* maps to the tree
    sized edge whose SIZE FRACTION is nearest the blueprint size's fraction
    (POSTFLOP_BET_MULT). Mass with no analogous tree action falls back to
    allin/call/check. Renormalised over the node's actions."""
    out = {a: 0.0 for a in node.actions}
    pot, tc = node.pot_mid, node.to_call
    sc_actor = node.sc[node.player]               # actor's chips already in this street
    tree_bets = [(a, sized_chips(a) / pot) for a in node.actions if a.startswith('bet:')]
    # A raise's true pot-fraction is (new_street_total - to_call)/(pot + to_call),
    # where new_street_total = sc_actor + the action's additional chips. Omitting
    # sc_actor (the actor's existing street commitment) skews re-raise fractions
    # low; include it so 3rd-aggression nodes map to the right blueprint size.
    tree_raises = ([(a, (sc_actor + sized_chips(a) - tc) / (pot + tc)) for a in node.actions
                    if a.startswith('raise:')] if (pot + tc) > 0 else [])
    total = 0.0
    for bp_a, p in (bp_dist or {}).items():
        if p <= 0:
            continue
        dest = None
        if bp_a in ('check', 'fold', 'call', 'allin'):
            dest = bp_a if bp_a in out else None
        elif bp_a.startswith('bet_') or bp_a.startswith('raise_'):
            kind = 'bet' if bp_a.startswith('bet_') else 'raise'
            frac = POSTFLOP_BET_MULT.get(bp_a.split('_')[1])   # None for *_custom_*
            pool = tree_bets if kind == 'bet' else tree_raises
            if frac is not None and pool:
                dest = min(pool, key=lambda t: abs(t[1] - frac))[0]
        if dest is None:                          # no analogous tree action
            # Effectively DEAD on real input: the blueprint only stores grid
            # actions (check/fold/call/allin/bet_*/raise_*), every one of which has
            # a direct or fraction analog above; only *_custom_* would land here and
            # _state_distribution never emits those. Kept as defensive routing:
            # aggressive blueprint mass with no tree analog -> the nearest in spirit
            # (all-in), matching the blueprint's intent. (Note: the EV-gate margin,
            # not this branch, governs how eager the gate is to deviate.)
            dest = next((alt for alt in ('allin', 'call', 'check') if alt in out), None)
        if dest is not None:
            out[dest] += p
            total += p
    if total > 0:
        return {a: v / total for a, v in out.items()}
    u = 1.0 / len(out)
    return {a: u for a in out}


class RiverSubgameSolver(BlueprintStrategy):
    def __init__(self, blueprint_db, *, max_iters=400, check_every=40,
                 time_budget=8.0, gap_threshold=None, temper_beta=DEFAULT_TEMPER_BETA,
                 ev_margin=1.0, menu=DEFAULT_MENU, rng=None):
        super().__init__(blueprint_db)
        self.max_iters = max_iters
        self.check_every = check_every
        self.time_budget = time_budget
        self.gap_threshold = gap_threshold
        self.temper_beta = temper_beta
        # Min chip-EV advantage (per dealt matchup) the solved strategy must show
        # over the blueprint baseline before we deviate from the blueprint.
        self.ev_margin = ev_margin
        self.menu = tuple(menu)
        # Seedable RNG for the final action sample, so scoring runs / tests are
        # reproducible (the mix itself is correct either way).
        self._rng = rng if rng is not None else np.random.default_rng()
        self._evaluator = HandEvaluator()
        self._cards = CardAbstraction()
        self._fallback_count = 0       # solve failures that degraded to the blueprint
        # Diagnostics: why does / doesn't the solver change the bot's action?
        self.stats = {'river_calls': 0, 'solved': 0, 'fallback': 0,
                      'deviated': 0, 'kept_blueprint': 0}

    # -- BotStrategy interface -------------------------------------------------
    def decide(self, info_set_key, legal_actions, public_state):
        ps = public_state or {}
        spec = self._solver_inputs(ps)
        if spec is None:
            return super().decide(info_set_key, legal_actions, public_state)
        self.stats['river_calls'] += 1
        try:
            dist, node, info = self.solve_for_action(**spec)
            # EV gate: deviate to the solved strategy only if it beats the
            # blueprint baseline (mapped onto the tree) by self.ev_margin chips.
            bp_engine = self._state_distribution(info_set_key, legal_actions, ps)
            baseline = blueprint_to_tree_dist(bp_engine, node)
            row = hand_row(info['ba'], spec['hole'], info['idx'])
            evs = hand_action_evs(info['cfr'], node, row, info['reach0'], info['reach1'])
            chosen, _gate = ev_gate(node.actions, dist, baseline, evs, self.ev_margin)
            self.stats['solved'] += 1
            self.stats['deviated' if _gate['used'] == 'solved' else 'kept_blueprint'] += 1
            return self._pick_engine_action(chosen, legal_actions, spec, node)
        except Exception:
            # Never crash a live hand -- but LOG (rate-limited, with traceback) so a
            # genuine defect surfaces instead of silently degrading to the blueprint
            # (the failure mode that once hid the uniform-fallback bug).
            self._fallback_count += 1
            self.stats['fallback'] += 1
            n = self._fallback_count
            if n <= 5 or n % 100 == 0:
                _LOG.warning("RiverSubgameSolver fell back to blueprint (#%d)",
                             n, exc_info=True)
            return super().decide(info_set_key, legal_actions, public_state)

    def _solver_inputs(self, ps):
        """Extract the river-solve inputs from public_state, or None to signal
        'fall back to the blueprint'. The river-entry fields are added to
        bot_public_state in step 6c; absent them, this returns None."""
        if ps.get('street') != 'river':
            return None
        required = ('riverEntryPot', 'riverEntryStacks', 'botSeat', 'hole_cards',
                    'opp_range', 'hero_range', 'riverPath')
        if any(ps.get(k) is None for k in required):
            return None
        tracker = ps['opp_range']
        return {
            'board': ps['community'],
            'pot_entry': ps['riverEntryPot'],
            'stacks': ps['riverEntryStacks'],
            'bot_seat': ps['botSeat'],
            'hole': ps['hole_cards'],
            'villain_tracker': tracker,
            'hero_tracker': ps['hero_range'],
            'confidence': getattr(tracker, 'confidence', 1.0),
            'river_path': ps['riverPath'],
        }

    # -- core: solve + read off ------------------------------------------------
    def solve_for_action(self, *, board, pot_entry, stacks, bot_seat, hole,
                         villain_tracker, hero_tracker, confidence, river_path):
        """Solve the river subgame and return (action_dist, node, info), where
        action_dist is {tree_action: prob} for the bot's actual hand at its
        decision node along `river_path`.

        board: 5 SuitRank cards. pot_entry/stacks: river-entry pot + (equal) behind
        stacks. bot_seat: 0/1. hole: bot's two cards. villain_tracker/hero_tracker:
        RangeTrackers (villain = bot's-cards-removed belief; hero = bot's blueprint
        reach). river_path: realized river actions before this decision, as labels
        ('check'/'call'/'fold'/'allin') or ('bet'|'raise', chips) for sized."""
        ba = build_board_arrays(board, self._evaluator, self._cards)
        idx = hand_index_map(ba)
        tree = build_river_tree(pot_entry, stacks, menu=self.menu)

        villain = blend_villain(project_tracker(villain_tracker, ba, idx),
                                confidence, self.temper_beta)
        hero = project_tracker(hero_tracker, ba, idx)
        if bot_seat == 0:
            reach0, reach1 = hero, villain
        else:
            reach0, reach1 = villain, hero

        # Validate everything BEFORE the (costly) solve, so a spot we can't
        # represent falls back to the blueprint without wasting a solve.
        row = hand_row(ba, hole, idx)
        if row is None:
            raise ValueError("bot hole cards collide with the board")
        # Guard against a silent uniform read-off: if the bot's ACTUAL hand has
        # ~zero hero reach (the blueprint gives it ~0 chance of taking this line),
        # its strat_sum row never moves, so average_strategy would return uniform
        # 1/A and the bot would emit a near-random action. Treat as unsolvable and
        # let decide() fall back to the blueprint cleanly.
        if hero[row] <= 1e-12:
            raise ValueError("bot hand has ~zero hero reach; solve can't represent it")
        node = self._navigate(tree, river_path)
        if node is None or node.terminal:
            raise ValueError("river path did not land on a decision node")
        if node.player != bot_seat:
            raise ValueError(f"path landed on seat {node.player}, not the bot ({bot_seat})")

        cfr, info = solve_river(
            tree, ba, reach0, reach1, max_iters=self.max_iters,
            check_every=self.check_every, gap_threshold=self.gap_threshold,
            time_budget=self.time_budget)

        dist = read_action_strategy(cfr, node, hole, ba, idx)
        # Carry the solve context so decide() can run the EV gate without re-solving.
        info.update({'cfr': cfr, 'ba': ba, 'idx': idx,
                     'reach0': reach0, 'reach1': reach1})
        return dist, node, info

    # -- navigation along the realized river path ------------------------------
    def _navigate(self, tree, river_path):
        """Walk from the root following `river_path`, snapping sized actions to the
        nearest tree edge (off-grid human bets map to the closest menu size)."""
        node = tree.root
        for spec in river_path:
            if node.terminal:
                return None
            i = self._match_edge(node, spec)
            if i is None:
                return None
            node = node.children[i]
        return node

    @staticmethod
    def _match_edge(node, spec):
        """Index of the child edge matching `spec`. Plain labels match exactly;
        ('bet'|'raise', chips) snaps to the nearest sized edge of that kind, or to
        all-in if that is closer / no sized edge exists."""
        if isinstance(spec, str):
            return node.actions.index(spec) if spec in node.actions else None
        kind, chips = spec                       # ('bet'|'raise', chips)
        best_i, best_d = None, None
        allin_i = node.actions.index('allin') if 'allin' in node.actions else None
        for i, a in enumerate(node.actions):
            if is_sized(a) and a.startswith(kind + ':'):
                d = abs(sized_chips(a) - chips)
                if best_d is None or d < best_d:
                    best_i, best_d = i, d
        if allin_i is not None:
            # If the realized size is larger than every sized edge, it's an all-in.
            allin_better = best_i is None or chips > max(
                (sized_chips(a) for a in node.actions if is_sized(a)), default=0.0)
            if allin_better:
                return allin_i
        return best_i

    # -- map the chosen tree action to an engine action ------------------------
    def _pick_engine_action(self, dist, legal_actions, spec, node):
        """Sample a tree action from `dist` and map it to an engine action,
        emitting the solver's EXACT size for sized bets/raises via a custom action.

        check/fold/call/allin map directly when legal. A sized bet/raise becomes
        `make_custom_action(is_raise, total)` where total = the bot's river street
        total after the action = node.sc[bot] + the action's additional chips. The
        engine accepts bet_custom_/raise_custom_ (Phase-1a unrestricted sizing), so
        the bot now wagers the precise size the solve chose -- the solver's edge."""
        labels = list(dist.keys())
        weights = np.array([max(0.0, dist[a]) for a in labels], dtype=float)
        if weights.sum() <= 0:
            choice = labels[0]
        else:
            choice = str(self._rng.choice(labels, p=weights / weights.sum()))

        bot_seat = spec['bot_seat']
        if choice in ('check', 'fold', 'call') and choice in legal_actions:
            return choice
        if choice == 'allin':
            if 'allin' in legal_actions:
                return 'allin'
            # Deep-stack node: the engine omits a discrete 'allin' (every sized bet
            # is affordable), but the solver still wants to shove. Emit a full-stack
            # custom bet/raise (raise-to total = the bot's entire river-entry stack);
            # the engine normalises an at/above-stack custom to all-in. Without this
            # a GTO shove silently degrades to a check -- biasing scoring against the
            # solver exactly where shoving matters.
            return make_custom_action(node.to_call > 0, spec['stacks'][bot_seat])
        if is_sized(choice):
            is_raise = choice.startswith('raise:')
            total = node.sc[bot_seat] + sized_chips(choice)   # raise-to street total
            return make_custom_action(is_raise, total)
        # Fallbacks that are always sensible.
        for a in ('check', 'call', 'fold'):
            if a in legal_actions:
                return a
        return legal_actions[0]
