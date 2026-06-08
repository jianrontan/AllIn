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

# The LIVE river solver builds a deeper tree than the blueprint's 3-aggression cap so
# it can represent (and solve) an uncapped human re-raise war on the river -- the bot
# can 5-bet/6-bet+ and the human can too (GameSession uncaps re-raises). This is a
# RUNTIME solve, NOT the blueprint, so it needs no retrain. It does NOT blow up the
# tree: river_tree only adds aggressive children while a player still has money behind
# (`opp_behind > 1e-9`), so once stacks commit the node collapses to jam/call/fold on
# its own -- a deep river spot is near-terminal and cheap. (Nodes past aggression 3
# have no blueprint warm-start, so they solve from scratch, which is fast at that depth;
# ranges come from the tracker, not the blueprint.) See BUG-014-era roadmap / #1.
LIVE_RIVER_MAX_AGGRESSIONS = 5

# Skip the river solve on a HIGH-SPR river (small pot, deep stacks). The river tree's
# raise depth -- and thus the CFR solve cost -- grows with SPR, because the river_tree
# `opp_behind` prune only collapses a node once stacks commit; at a small-pot/deep-stack
# entry many raises fit before commit, so the (depth-5) tree explodes (measured: pot 6 /
# stacks 197 -> ~1500 nodes, ~20s, which blows the ~5s live budget and returns an
# UNCONVERGED strategy). These are also the lowest-stakes spots (a tiny pot), so the
# blueprint is the right call. The all-in guard still fires (it runs before the solver in
# decide), and a deep river raise-war commits stacks fast -> becomes near-terminal ->
# guard-covered. SPR = effective_stack / river_entry_pot. Tunable: raise it (toward ~6)
# to solve more medium pots once depth-5 timing at higher SPR is characterized; lower it
# for a stricter no-hang guarantee. ~4 keeps the solved tree small enough to converge in
# budget. (Found by the 2026-06-04 agent audit; matches the user's "skip small pots".)
#
# Value chosen from a node-count measurement (depth-5 river tree at 100bb): the live
# budget only checks time AFTER a full check_every(=40)-iteration block, so the tree must
# be small enough that one block fits ~5s (~<=275 decision nodes at ~0.45ms/node/iter under
# training contention). Measured nodes by pot: 14BB(SPR 6.6)=164 / ~3s (fits); 10BB(SPR 9.5)
# =284 / ~5.2s (OVER -- back to a laggy, unconverged solve); 6BB(SPR 16)=652 (way over).
# Also, fewer iterations fit as the tree grows, so high-SPR spots barely converge (likely no
# better than the blueprint). 6.0 = the largest that fits one block in budget (~14BB pots);
# the solver's value is on bigger/low-SPR pots anyway, and the blueprint handles small ones.
SOLVER_MAX_SPR = 6.0


def blueprint_to_tree_dist(bp_dist, node, postflop_menu=None):
    """Redistribute a blueprint action distribution (over ENGINE actions) onto the
    tree node's action menu -- the EV-gate baseline ('what the blueprint would do
    here'). check/fold/call/allin map directly; a bet_*/raise_* maps to the tree
    sized edge whose SIZE FRACTION is nearest the blueprint size's fraction
    (`postflop_menu`, default POSTFLOP_BET_MULT). Mass with no analogous tree action
    falls back to allin/call/check. Renormalised over the node's actions.

    `postflop_menu` MUST match the arm the blueprint was trained under: a capped
    blueprint stores `overbet2`, which is only in POSTFLOP_BET_MULT_CAPPED -- passing
    the default control menu would drop that mass to the wrong nearest size."""
    menu = postflop_menu if postflop_menu is not None else POSTFLOP_BET_MULT
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
            frac = menu.get(bp_a.split('_')[1])   # None for *_custom_*
            pool = tree_bets if kind == 'bet' else tree_raises
            if frac is not None and pool:
                dest = min(pool, key=lambda t: abs(t[1] - frac))[0]
        if dest is None:                          # no analogous tree action
            # When this fires: the blueprint wants a sized bet/raise but this tree
            # node has NO bet/raise edge of that kind -- they all collapsed to all-in
            # at low SPR (tree_bets/tree_raises empty). All-in is then the ONLY
            # aggressive action, so routing the blueprint's aggression to it is the
            # faithful mapping, not a fallback hack. (It also catches *_custom_* sizes,
            # which carry no POSTFLOP_BET_MULT fraction -- though _state_distribution
            # never emits those today.) If the node has no aggression at all (facing
            # an all-in: only fold/call), the chain drops to call then check.
            # KNOWN EDGE: when the solver menu's SMALLEST bet collapses to all-in but a
            # smaller blueprint bet was still affordable, this over-commits to all-in
            # vs the blueprint's small bet -- a menu-granularity artifact (the solver
            # menu lacks the blueprint's 0.33/0.66 sizes). Narrow + low-magnitude at
            # the SPRs where it occurs; fixing it would mean widening the live solver
            # menu (slower solves), so it's accepted. (The EV-gate margin, not this
            # branch, governs how eager the gate is to deviate.)
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
                 ev_margin=1.0, menu=DEFAULT_MENU, rng=None,
                 guard_confidence=0.2, guard_margin=1.0):
        super().__init__(blueprint_db)
        # Postflop size menu the served blueprint was trained under (control vs
        # capped), so the EV-gate baseline projection maps the blueprint's stored
        # sizes (incl. capped's overbet2) onto the river tree correctly.
        from ..abstractions.sizing import db_menu_mode, postflop_menu_for
        self._postflop_menu = postflop_menu_for(db_menu_mode(blueprint_db))
        self.max_iters = max_iters
        self.check_every = check_every
        self.time_budget = time_budget
        self.gap_threshold = gap_threshold
        self.temper_beta = temper_beta
        # Min chip-EV advantage (per dealt matchup) the solved strategy must show
        # over the blueprint baseline before we deviate from the blueprint.
        self.ev_margin = ev_margin
        self.menu = tuple(menu)
        # Facing-all-in guard (flop/turn): only override the blueprint when the
        # range belief is at least this confident, and only when the call/fold EV
        # edge clears this chip margin (avoid flipping on a knife-edge / MC noise).
        self.guard_confidence = guard_confidence
        self.guard_margin = guard_margin
        # Seedable RNG for the final action sample, so scoring runs / tests are
        # reproducible (the mix itself is correct either way).
        self._rng = rng if rng is not None else np.random.default_rng()
        self._evaluator = HandEvaluator()
        self._cards = CardAbstraction()
        self._fallback_count = 0       # solve failures that degraded to the blueprint
        # Diagnostics: why does / doesn't the solver change the bot's action?
        self.stats = {'river_calls': 0, 'solved': 0, 'fallback': 0,
                      'deviated': 0, 'kept_blueprint': 0, 'allin_guard': 0}

    # -- BotStrategy interface -------------------------------------------------
    def decide(self, info_set_key, legal_actions, public_state):
        ps = public_state or {}
        self.last_debug = None
        # Near-terminal all-in guard runs FIRST (facing a jam that commits the whole
        # stack is a pure equity decision). _run_guard is shared with the turn solver.
        guard = self._run_guard(legal_actions, ps)
        if guard is not None:
            return guard
        spec = self._solver_inputs(ps)
        if spec is None:
            self.last_debug = {'mode': 'blueprint', 'street': ps.get('street')}
            return super().decide(info_set_key, legal_actions, public_state)
        self.stats['river_calls'] += 1
        try:
            dist, node, info = self.solve_for_action(**spec)
            return self._gate_and_pick(dist, node, info, info_set_key,
                                       legal_actions, ps, spec, 'river_solver', 'river')
        except Exception:
            # Never crash a live hand -- but LOG (rate-limited, with traceback) so a
            # genuine defect surfaces instead of silently degrading to the blueprint
            # (the failure mode that once hid the uniform-fallback bug).
            self._fallback_count += 1
            self.stats['fallback'] += 1
            self.last_debug = {'mode': 'fallback', 'street': ps.get('street'),
                               'solved': False}
            n = self._fallback_count
            if n <= 5 or n % 100 == 0:
                _LOG.warning("RiverSubgameSolver fell back to blueprint (#%d)",
                             n, exc_info=True)
            return super().decide(info_set_key, legal_actions, public_state)

    # -- shared guard + gate/pick (reused by TurnSubgameSolver) ----------------
    def _run_guard(self, legal_actions, ps):
        """Run the near-terminal all-in guard, swallowing any defect into a DEFER
        (never crash a live hand: advance_bot_turns only catches GameError, so an
        unguarded raise here would 500 the hand). Returns the guard action or None,
        and records guard stats / debug. Shared by river decide() and the turn solver."""
        try:
            guard = self._facing_allin_guard(legal_actions, ps)
        except Exception:
            self._fallback_count += 1
            self.stats['fallback'] += 1
            self.last_debug = {'mode': 'guard_error', 'street': ps.get('street')}
            return None
        if guard is not None:
            self.stats['allin_guard'] += 1
            self.last_debug = {'mode': 'allin_guard', 'street': ps.get('street'),
                               'action': guard}
        return guard

    def _gate_and_pick(self, dist, node, info, info_set_key, legal_actions, ps, spec,
                       mode, street):
        """EV-gate the solved strategy against the blueprint baseline (mapped onto the
        tree), record diagnostics, and emit the engine action. Shared by the river and
        turn solve paths -- both produce (dist, node, info) of the same shape."""
        bp_engine = self._state_distribution(info_set_key, legal_actions, ps)
        baseline = blueprint_to_tree_dist(bp_engine, node, self._postflop_menu)
        row = hand_row(info['ba'], spec['hole'], info['idx'])
        evs = hand_action_evs(info['cfr'], node, row, info['reach0'], info['reach1'])
        chosen, gate = ev_gate(node.actions, dist, baseline, evs, self.ev_margin)
        self.stats['solved'] += 1
        deviated = gate['used'] == 'solved'
        self.stats['deviated' if deviated else 'kept_blueprint'] += 1
        self.last_debug = {
            'mode': mode, 'street': street, 'solved': True, 'deviated': deviated,
            'nodeActions': list(node.actions),
            'solvedStrategy': {a: round(float(p), 4) for a, p in dist.items()},
            'baseline': {a: round(float(p), 4) for a, p in baseline.items()},
            'evSolved': round(float(gate['ev_solved']), 3),
            'evBaseline': round(float(gate['ev_baseline']), 3),
            'evDelta': round(float(gate['delta']), 3),
            'evMargin': float(self.ev_margin),
            'iters': int(info.get('iters', 0)),
            'gap': (round(float(info['gap']), 4) if info.get('gap') is not None else None),
            'converged': bool(info.get('converged', False)),
        }
        return self._pick_engine_action(chosen, legal_actions, spec, node)

    # -- facing-an-all-in terminal guard (preflop / flop / turn) ---------------
    def _facing_allin_guard(self, legal_actions, ps):
        """Guard the FACING-an-all-in decision on PREFLOP / flop / turn.

        When the bot faces a bet whose call would commit its ENTIRE remaining
        stack, the hand is near-terminal: calling leaves no money behind either
        player, so the board simply runs out to showdown -- there is no further
        decision and no continuation to value (no leaf-value function needed, the
        thing that makes turn/flop solving hard). The call/fold choice is therefore
        a pure equity-vs-pot-odds comparison against the live opponent belief, which
        is sharper than a coarse/undertrained blueprint bucket and directly guards
        the bot from calling off (or wrongly folding) vs a jam.

        WHY ALL STREETS incl. PREFLOP (fix #1, 2026-05-31): the blueprint plays a
        FIXED, opponent-agnostic GTO call frequency facing a jam -- it ignores the
        range tracker's belief about THIS opponent. Vs a human who jams too strong
        (the observed leak), GTO over-calls. Routing every facing-jam decision (any
        street) through the tracked-range equity makes the bot ADAPT: it folds the
        marginal hands the un-adapted blueprint calls off with. A called preflop jam
        runs all 5 board cards out -> still pure equity, no solve. The river is
        handled by the full solver downstream, so this guard stays preflop/flop/turn.
        (`river` deliberately excluded -> falls through to `_solver_inputs`.)

        ON SPR: the trigger is `to_call >= bot_stack` (the call IS all-in), NOT an
        SPR threshold -- and that is deliberate. A massive OVERBET jam from a
        high-SPR start (small pot, deep stacks) satisfies it and SHOULD: once jammed
        and called, betting is closed and the board runs out, so it is near-terminal
        regardless of the pre-jam SPR, and the pot-odds call is exactly right. The
        real high-SPR risk is not the trigger but the range QUALITY: a giant overbet
        is an off-model action, and here the call risks the whole stack, so a bad
        belief is expensive. That is what `guard_confidence` covers -- an off-model
        jam decays RangeTracker confidence, so the guard DEFERS to the blueprint
        rather than act on an untrusted belief exactly in that case.

        Returns the engine action ('call'/'fold') to play, or None to DEFER to the
        normal path (blueprint / river solver) when the guard doesn't apply:
          * river (-> the full solver handles it);
          * not facing a bet (no fold+call), or calling does NOT commit the whole
            stack (money behind -> NOT near-terminal: that is the deep-raise case,
            which needs leaf values and is out of scope here);
          * range tracking is off / the belief is below `guard_confidence` (don't
            override on a belief we don't trust -- defer to the blueprint);
          * the call/fold EV edge is within `guard_margin` chips (a knife-edge /
            Monte-Carlo-noise spot -> don't override).

        NOTE this guards FACING a jam only. Guarding the bot PROPOSING a jam needs
        the opponent's calling model and a value for the non-jam alternative (a
        leaf value), so it is deliberately left to a later slice.
        """
        street = ps.get('street')
        if street not in ('preflop', 'flop', 'turn'):
            return None                              # river -> full solver downstream
        if 'call' not in legal_actions or 'fold' not in legal_actions:
            return None                              # not facing a bet
        tracker = ps.get('opp_range')
        hole = ps.get('hole_cards')
        board = ps.get('community')                  # [] preflop -- that's fine
        seat = ps.get('seat')
        if tracker is None or not hole or board is None or seat is None:
            return None
        to_call = float(ps.get('to_call') or 0.0)
        bot_stack = float(ps['p0_stack'] if seat == 0 else ps['p1_stack'])
        if to_call <= 0.0 or to_call < bot_stack - 1e-6:
            return None                              # call leaves money behind -> not near-terminal
        # Default missing confidence to 0.0 (DEFER), not 1.0: a tracker lacking a
        # confidence reading is an untrusted belief, and acting on it risks the
        # whole stack. The real RangeTracker always sets .confidence; this only
        # bites a malformed/foreign tracker, where defer is the safe direction.
        if getattr(tracker, 'confidence', 0.0) < self.guard_confidence:
            return None                              # belief untrusted -> defer to blueprint

        # More board cards still to come (preflop=5, flop=2) -> Monte-Carlo runout
        # equity, which is noisy; use more samples the further from the river so the
        # estimate's std stays well under guard_margin and can't flip call<->fold at
        # the pot-odds boundary. (turn=1, river=0 are exact -> few/no runouts.)
        n_runouts = 600 if street == 'preflop' else 200
        eq = float(tracker.hero_equity(list(hole), list(board), n_runouts=n_runouts))
        pot_mid = float(ps.get('pot') or 0.0)        # chips in the middle now (incl. the jam)
        call_cost = min(to_call, bot_stack)          # all-in-for-less caps the bot's risk
        if to_call <= bot_stack + 1e-9:
            final_pot = pot_mid + to_call            # bot fully calls
        else:
            # All-in-for-less: the opponent's unmatched excess returns, leaving the
            # matched pot. (Unreachable while stacks reset equal each street, but
            # correct if that ever changes.)
            final_pot = pot_mid - to_call + 2.0 * bot_stack
        ev_call = eq * final_pot - call_cost         # EV(fold) = 0 (forfeit, no further loss)
        if ev_call > self.guard_margin:
            return 'call'
        if ev_call < -self.guard_margin:
            return 'fold'
        return None                                  # too close -> defer to blueprint

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
        # High-SPR (small-pot, deep-stack) river -> skip the solve (too slow + lowest
        # stakes); blueprint handles it and the all-in guard already ran. See SOLVER_MAX_SPR.
        pot_entry = ps['riverEntryPot']
        stacks = ps['riverEntryStacks']
        eff = stacks[0] if isinstance(stacks, (list, tuple)) else stacks
        if pot_entry <= 0 or eff / pot_entry > SOLVER_MAX_SPR:
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
        tree = build_river_tree(pot_entry, stacks, menu=self.menu,
                                max_aggressions=LIVE_RIVER_MAX_AGGRESSIONS)

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
        node, edge_path = self._navigate(tree, river_path)
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
        # The EV gate needs the reaches INTO this node (root reaches conditioned on
        # the realized river betting), not the root reaches -- otherwise the villain
        # range at a non-root node still includes hands that wouldn't have taken the
        # line, biasing the deviate/keep decision.
        node_reach0, node_reach1 = cfr.reach_into(edge_path, reach0, reach1)
        info.update({'cfr': cfr, 'ba': ba, 'idx': idx,
                     'reach0': node_reach0, 'reach1': node_reach1})
        return dist, node, info

    # -- navigation along the realized river path ------------------------------
    def _navigate(self, tree, river_path):
        """Walk from the root following `river_path`. A realized sized bet/raise is
        handled by NESTED SOLVING (Libratus): if its exact size is off the solver's
        menu, inject it into the tree as a real edge so the downstream solve sees the
        TRUE pot/stacks, instead of snapping it to the nearest menu size (which would
        solve the bot's response against the wrong pot). If the exact size can't be a
        distinct legal edge (it's really an all-in, below min-raise, or already
        on-grid), fall back to nearest-edge snapping.

        Must run BEFORE the CFR tables are sized (solve_river), since an injected
        edge adds a node_id. Returns (node, edge_indices) — child indices from the
        root, used to condition reaches into the node for the EV gate. Returns
        (None, _) if the path cannot be followed."""
        node = tree.root
        edges = []
        for spec in river_path:
            if node.terminal:
                return None, edges
            i = None
            if not isinstance(spec, str):
                # Sized bet/raise: try to splice the EXACT size in as a new edge
                # (nested solving). inject_realized_edge returns None when the size
                # is already on-grid / really an all-in / below min-raise; in those
                # cases _match_edge below snaps it (correct — it IS representable).
                kind, chips = spec
                if tree.inject_realized_edge(node, kind, chips) is not None:
                    i = len(node.children) - 1          # the just-appended edge
            if i is None:
                i = self._match_edge(node, spec)
            if i is None:
                return None, edges
            edges.append(i)
            node = node.children[i]
        return node, edges

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
