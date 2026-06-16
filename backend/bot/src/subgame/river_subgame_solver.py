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
import threading
from collections import OrderedDict

import numpy as np

from ..game.bot_strategy import BlueprintStrategy
from ..abstractions.hand_evaluator import HandEvaluator
from ..abstractions.card_abstractions import CardAbstraction
from ..abstractions.sizing import POSTFLOP_BET_MULT
from ..cfr.poker_game import make_custom_action
from ..evaluation.showdown_kernel import build_board_arrays
from .river_tree import build_river_tree, is_sized, sized_chips, DEFAULT_MENU
from .river_cfr import RiverCFR  # noqa: F401  (re-exported convenience)
from .solve_control import solve_river, solve_river_gadget, ev_gate, hand_action_evs
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
    # C1: value-jam threshold -- at an UNTRAINED beyond-cap node, a hand whose
    # uniform-floor equity clears this bar is a monster that could re-jam for value /
    # deny equity rather than flatly call (the deep guard's default). Gated behind
    # `value_jam` and DEFAULT OFF: the maniac A/B (2026-06-09) showed it fires ~never
    # under the capped engine (~1/60 deep nodes) because a standalone 'allin' is only
    # legal when the bot is already near-committed (voluntary_allin=False), and we
    # deliberately never CONSTRUCT a custom jam from an untrained node (the BUG-011
    # risk). Kept as a documented opt-in (it would fire under a control-menu / deeper
    # or folding-opponent setting); not worth enabling vs a presser that stacks off
    # via calls anyway.
    VALUE_JAM_EQ = 0.80

    # Assumed opponent JAM range (top fraction of preflop hands) for a FACED ALL-IN when
    # the read is uninformed. A real 100BB+ jam over a min-open is a SELECTED range, not
    # uniform, so equity-vs-uniform over-estimates and stacks off dominated hands
    # (KQo/A5s/55). Judging vs the top ~20% folds those while never folding premiums.
    # See BUG-022. (Preflop only; the postflop EMD strength buckets aren't equity-ordered.)
    JAM_RANGE_FRACTION = 0.20

    # Inference #1: at an UNTRAINED first-to-act postflop node the passive fallback checks
    # 100%; value-bet a trained size only when equity clears this high bar (so it's a thin/
    # no-bluff value bet -- hard to exploit -- never a stray aggressive size).
    FIRST_ACT_VALUE_EQ = 0.65

    # H1: how much larger an EV edge an UNDER-converged river solve must show before it
    # is allowed to deviate from the (trained) blueprint baseline -- a partial solve is
    # least trustworthy exactly where it most wants to move, so distrust it.
    _NONCONVERGED_MARGIN_MULT = 4.0

    # 'auto' gadget anchor: how much MORE exploitable (relative slack on the blueprint's
    # river best-response) the unsafe exploit strategy may be on a spot before the
    # self-check falls back to the safe blueprint-anchored gadget. A small positive slack
    # tolerates finite-iteration solve noise so the check doesn't flap on a near-tie.
    _AUTO_SAFE_MARGIN = 0.05

    # A belief is treated as UNINFORMED (don't trust the tracked range for a stack-off)
    # only when its effective-hand-count / live-hand-count is AT/ABOVE this -- i.e. the
    # belief is essentially still uniform (no real read). Set very near 1.0 on purpose:
    # the canonical BUG-022 case (a uniform belief) sits at ratio = 1.0 AND already has its
    # confidence collapsed by the off-menu decay, so this gate is belt-and-suspenders for
    # the exact-uniform-at-high-confidence edge. A LOWER value would wrongly discard a
    # genuine MILD read (a 1.5x tilt is ratio ~0.97) and revert to un-adapted blueprint
    # play at trained all-in nodes -- a regression. 0.99 rejects only ~uniform beliefs.
    INFORMATIVE_RATIO = 0.99

    def __init__(self, blueprint_db, *, max_iters=400, check_every=40,
                 time_budget=8.0, gap_threshold=None, temper_beta=DEFAULT_TEMPER_BETA,
                 ev_margin=1.0, menu=DEFAULT_MENU, rng=None,
                 guard_confidence=0.2, guard_margin=1.0, value_jam=False,
                 safe_gadget=False, gadget_anchor='belief', purify_threshold=0.0):
        super().__init__(blueprint_db, purify_threshold=purify_threshold)
        self.value_jam = value_jam
        # Phase 5a: solve the river under the re-solving GADGET (provably
        # no-more-exploitable than the blueprint) instead of the unsafe-v1 direct
        # solve against the tracked ranges. Default OFF until the exploitability
        # validation (SAFE_RIVER_SOLVING_PLAN.md piece 4) clears it to flip on.
        self.safe_gadget = safe_gadget
        # Which villain range anchors the gadget's opt-out / reach (the safety vs
        # read-exploitation fork):
        #   'blueprint' -- UNIFORM (card-removal) villain range: the no-more-exploitable
        #     guarantee then holds vs ANY villain hand (robust to a narrow-WRONG belief),
        #     trading away read-based exploitation. The provable anchor.
        #   'belief' -- the tracked river-entry belief: safe RELATIVE to that belief and
        #     still read-aware, but a narrow-wrong belief isn't fully clamped to blueprint.
        #   'confidence' (A) -- exploit (unsafe-v1) on a trusted+informative read, else
        #     clamp to the 'blueprint' anchor. Cheap (one BR-walk-free gate), but clamps
        #     even where the exploit was safe.
        #   'auto' (B) -- like 'confidence' but, on an untrusted read, runs a per-spot
        #     safety self-check and STILL exploits when that is within the blueprint's
        #     river best-response. Gets unsafe-v1 exploitation when the read is good AND
        #     the safe clamp when it isn't, at the cost of the self-check's BR walks.
        # All built so the anchor can be chosen from measured exploitability / win-rate.
        if gadget_anchor not in ('belief', 'blueprint', 'confidence', 'auto'):
            raise ValueError("gadget_anchor must be 'belief', 'blueprint', 'confidence' "
                             f"or 'auto', got {gadget_anchor!r}")
        self.gadget_anchor = gadget_anchor
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
        # The served BOT is a MODULE-LEVEL SINGLETON shared across concurrent gunicorn
        # threads (--threads 4). The two LRU equity caches below and the numpy RNG are
        # mutable shared state: an unsynchronised OrderedDict move_to_end/popitem can
        # corrupt its linked list, and numpy Generator.choice is not thread-safe (could
        # silently mis-sample the solved action). This lock guards those small critical
        # sections (NOT the expensive equity compute, which stays outside it). last_debug
        # is already thread-local; per-request solve state (tree/CFR/ba) is not shared.
        self._lock = threading.Lock()
        # Seedable RNG for the final action sample, so scoring runs / tests are
        # reproducible (the mix itself is correct either way).
        self._rng = rng if rng is not None else np.random.default_rng()
        self._evaluator = HandEvaluator()
        self._cards = CardAbstraction()
        self._fallback_count = 0       # solve failures that degraded to the blueprint
        # Uniform-range equity floor cache (the latency fix): equity vs a uniform
        # opponent range depends only on (hole, board), not game state, so it is safe
        # to memoize process-wide. Preflop (board == []) keys on the hole alone, so an
        # aggressive presser that fires this path every decision reuses <=1326 values.
        # OrderedDict = LRU (evict oldest at the cap) so a long-lived shared process
        # never pays the full-flush latency cliff a plain dict's clear() would.
        self._uniform_eq_cache = OrderedDict()
        self._uniform_eq_cap = 50000
        # Same idea for the top-fraction jam-range floor (faced all-in, uninformed read).
        self._jam_eq_cache = OrderedDict()
        # Per-street postflop bucket -> centroid MEAN equity (derived once from the
        # committed EMD centroids; robust to a re-bake that reorders bucket indices).
        self._pf_means_cache = {}
        # Diagnostics: why does / doesn't the solver change the bot's action?
        self.stats = {'river_calls': 0, 'solved': 0, 'fallback': 0,
                      'deviated': 0, 'kept_blueprint': 0, 'allin_guard': 0,
                      'deep_raise_guard': 0, 'premium_no_fold': 0,
                      # Safe-gadget anchor outcome (Phase 5a): how often the gadget
                      # EXPLOITED (unsafe-v1 cleared by pre-filter/self-check) vs CLAMPED
                      # to a safe gadget solve. 0/0 when safe_gadget is off.
                      'gadget_exploit': 0, 'gadget_clamp': 0}

    # -- BotStrategy interface -------------------------------------------------
    def _pre_solve_guards(self, info_set_key, legal_actions, ps):
        """The all-in / deep-raise / first-act guards that run before ANY solve, on every
        street. Returns an action to PLAY, or None to proceed to the solver/blueprint.
        SHARED between the river decide and the TURN decide so both run the IDENTICAL guard
        set the deployed bot uses -- otherwise the turn solver acts on nodes production
        intercepts with a guard (a faithfulness gap caught in the 2026-06-16 gate review)."""
        # Near-terminal all-in guard FIRST (facing a stack-committing jam is pure equity).
        guard = self._run_guard(legal_actions, ps)
        if guard is not None:
            return guard
        # Deep-raise guard: a raise/jam into an UNTRAINED beyond-cap node (e.g. a human
        # 5-bet -> pf_*_ip_slll). The blueprint's passive fallback would FOLD THE NUTS half
        # the time; decide by equity vs the range instead (call/fold only, no stray raise).
        try:
            deep = self._facing_deep_raise_guard(info_set_key, legal_actions, ps)
        except Exception:
            # Never crash a live hand. Count + record so a genuine defect surfaces; prefer
            # CALL over the blueprint's 50/50 coin-flip so an ERROR never folds the nuts.
            self._fallback_count += 1
            self.stats['fallback'] += 1
            self.last_debug = {'mode': 'deep_guard_error', 'street': ps.get('street')}
            deep = self._safe_untrained_call(info_set_key, legal_actions, ps)
        if deep is not None:
            return deep
        # First-act value guard: value-bet a trained size at an untrained first-to-act node
        # instead of the passive check that surrenders value with a strong made hand.
        try:
            first_act = self._first_act_value_guard(info_set_key, legal_actions, ps)
        except Exception:
            first_act = None
        return first_act

    def decide(self, info_set_key, legal_actions, public_state):
        ps = public_state or {}
        self.last_debug = None
        guarded = self._pre_solve_guards(info_set_key, legal_actions, ps)
        if guarded is not None:
            return guarded
        spec = self._solver_inputs(ps)
        if spec is None:
            self.last_debug = {'mode': 'blueprint', 'street': ps.get('street')}
            action = super().decide(info_set_key, legal_actions, public_state)
            return self._premium_no_fold(action, legal_actions, ps)
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
            safe = self._safe_untrained_call(info_set_key, legal_actions, ps)
            if safe is not None:
                return safe          # untrained faced bet -> never coin-flip on an error
            action = super().decide(info_set_key, legal_actions, public_state)
            return self._premium_no_fold(action, legal_actions, ps)

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
        # M2: use the raw-key blueprint distribution (NOT _state_distribution's
        # translation blend) so the baseline is consistent with the _is_untrained check
        # below -- they must agree on "trained?". On the river the solver injects the
        # exact realized size via _navigate, so the off-grid translation blend is
        # redundant here and only introduces a baseline/untrained-check mismatch.
        bp_engine = self._distribution(info_set_key, legal_actions)
        baseline = blueprint_to_tree_dist(bp_engine, node, self._postflop_menu)
        row = hand_row(info['ba'], spec['hole'], info['idx'])
        evs = self._hand_action_evs(info, node, row)
        untrained = self._is_untrained(info_set_key, legal_actions)
        # C3: at an UNTRAINED key the baseline is the passive coin-flip fallback -- gating
        # the solve against that garbage can wrongly KEEP it, so trust the solve outright.
        # Trained keys gate against a real baseline.
        if evs is None:
            # M1: no compatible villain mass for the bot's exact hand -> EVs undefined,
            # can't gate. Untrained -> trust the solve (no real baseline); trained -> keep
            # the real blueprint baseline rather than gate on garbage.
            chosen = dist if untrained else baseline
            gate = {'ev_solved': 0.0, 'ev_baseline': 0.0, 'delta': 0.0,
                    'used': 'solved_untrained' if untrained else 'baseline'}
        elif untrained:
            ev_s = float(sum(dist.get(a, 0.0) * v for a, v in zip(node.actions, evs)))
            chosen = dist
            gate = {'ev_solved': ev_s, 'ev_baseline': 0.0, 'delta': 0.0,
                    'used': 'solved_untrained'}
        else:
            # H1: an UNDER-converged solve is least trustworthy exactly where it most
            # wants to deviate. Require a larger EV edge to leave the (real) baseline
            # when the solve did not converge -- the warm-start prior is unwired, so this
            # is the graceful-degradation safety net for a partial solve.
            margin = self.ev_margin
            if not info.get('converged', False):
                margin = self.ev_margin * self._NONCONVERGED_MARGIN_MULT
            chosen, gate = ev_gate(node.actions, dist, baseline, evs, margin)
        self.stats['solved'] += 1
        deviated = gate['used'] != 'baseline'
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
            'anchor': info.get('anchor'),            # safe-gadget path taken (None = unsafe-v1)
        }
        # Telemetry: tally exploit vs clamp for the safe-gadget anchors. 'exploit' in the
        # label = unsafe-v1 was served (pre-filter or self-check cleared it); everything
        # else ('*_safe_fallback', 'belief', 'blueprint') ran a gadget solve = clamp.
        anchor = info.get('anchor')
        if anchor is not None:
            self.stats['gadget_exploit' if 'exploit' in anchor else 'gadget_clamp'] += 1
        return self._pick_engine_action(chosen, legal_actions, spec, node)

    def _hand_action_evs(self, info, node, row):
        """Per-action chip EVs for the bot's hand, used by the EV gate. River: the solved
        average-strategy values (the river leaf is an exact showdown). Overridden on the
        TURN (1c) to value the depth-limit leaf with the EXACT rollout, not the bucketed
        matrix the solve used (so the gate can't self-grade a losing deviation)."""
        return hand_action_evs(info['cfr'], node, row, info['reach0'], info['reach1'])

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
        # Trust the tracked range only when CONFIDENT *and* INFORMATIVE. A uniform belief
        # held at high confidence (e.g. an on-menu-but-unmodeled jam that left the range
        # unmoved) must NOT be acted on for a whole-stack decision -- that is the BUG-022
        # mechanism one branch over. _trust_read folds both checks; defer otherwise (the
        # deep guard then judges vs the jam range / uniform floor). Missing confidence ->
        # 0.0 (defer), the safe direction for a malformed/foreign tracker.
        if not self._trust_read(tracker):
            return None                              # untrusted/uninformed -> defer

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

    # -- facing-a-deep-raise guard (preflop / flop / turn, money behind) --------
    def _facing_deep_raise_guard(self, info_set_key, legal_actions, ps):
        """Guard a FACING-a-bet decision (deep raise OR jam) when the blueprint has
        nothing usable at this node -- i.e. a beyond-cap / untrained key.

        WHY: training caps aggression at 3/street (max_raises_per_street=2), but LIVE
        play uncaps re-raises so a human can 5-bet/6-bet+. Those nodes (e.g.
        pf_*_ip_slll) are never visited in training, so BlueprintStrategy._distribution
        returns its PASSIVE fallback: uniform over {call, fold}. That folds premium
        hands (AA to a 5-bet) ~half the time -- the exact leak observed live. The
        _facing_allin_guard runs FIRST and owns the TRUSTED jam (it acts on the tracked
        range when confident); it DEFERS on a collapsed read -- and at an untrained node
        deferring is a 50/50 coin-flip, so we cover both the jam it punted on AND every
        non-all-in deep raise (which never commits the whole stack, so it slips past the
        all-in guard's to_call >= bot_stack trigger entirely).

        Fix: when (and only when) the blueprint key is untrained, decide call/fold by
        the tracked-range equity vs pot odds, never a raise (a stray raise from an
        untrained node is the BUG-011 class). This is an unsafe v1 stopgap (it ignores
        future-street play / implied odds), but beyond-cap nodes are DEEP -- a called
        5-bet leaves low residual SPR, so a runout-equity call is a good approximation
        right where this fires, and it is strictly better than folding the nuts. The
        principled replacement is the Phase-4 deep-raise subgame solver.

        A collapsed-confidence read does NOT defer here (unlike _facing_allin_guard):
        there is no trained blueprint to defer to at an untrained node, so deferring
        would just be a coin flip. It instead judges against a uniform range, which
        still never folds a premium. See the equity branch below.

        Returns 'call'/'fold', or None to DEFER (river -> solver; trained key ->
        blueprint; no faced bet; or missing tracker inputs)."""
        street = ps.get('street')
        if street not in ('preflop', 'flop', 'turn'):
            return None                              # river -> full solver downstream
        if 'call' not in legal_actions or 'fold' not in legal_actions:
            return None                              # not facing a bet
        # Only override when the blueprint has NO usable mass on the legal actions
        # here -- mirrors exactly when _distribution would fall into its passive
        # fallback. A trained key keeps its learned (balanced) mixed strategy.
        if not self._is_untrained(info_set_key, legal_actions):
            return None                              # trained -> defer to blueprint
        tracker = ps.get('opp_range')
        hole = ps.get('hole_cards')
        board = ps.get('community')                  # [] preflop is fine
        seat = ps.get('seat')
        if tracker is None or not hole or board is None or seat is None:
            return None
        to_call = float(ps.get('to_call') or 0.0)
        bot_stack = float(ps['p0_stack'] if seat == 0 else ps['p1_stack'])
        if to_call <= 0.0:
            return None                              # not facing a bet
        # NB: we do NOT exclude the faced-all-in case (to_call >= bot_stack). The
        # all-in guard runs first and ACTS on a trusted jam; it only reaches here when
        # it DEFERRED (collapsed read), where deferring at an untrained node is a
        # coin-flip. A called jam just runs the board out, so an equity call/fold is
        # even better-founded for it than for the money-behind deep-raise case.
        # Runout equity vs the opponent range (more samples the further from showdown
        # so MC noise can't flip a premium at the pot-odds boundary). When the read is
        # TRUSTED, use the tracked belief (adapts: folds marginal hands the read says
        # are behind). When it is NOT -- a maniac's off-model spam collapses confidence
        # but `observe` KEEPS the prior range, so the belief is near-uniform anyway --
        # do NOT defer to the blueprint's 50/50 coin-flip at this untrained node:
        # judge against a UNIFORM range (card removal only). A premium dominates every
        # possible range (AA >= ~0.78 equity vs anything > any preflop pot odds), so it
        # NEVER folds; only genuine trash folds. This makes "never fold the nuts to a
        # deep raise" a hard guarantee, independent of the tracker's confidence state.
        if self._trust_read(tracker):
            n_runouts = 600 if street == 'preflop' else 200
            eq = float(tracker.hero_equity(list(hole), list(board), n_runouts=n_runouts))
        elif street == 'preflop':
            # Uninformed read at this untrained (beyond-cap) PREFLOP node -- a 5-bet+ jam
            # OR a money-behind 5-bet+ both face a SELECTED range, not uniform (nobody
            # gets here any-two). Vs-uniform over-estimates and stacks off dominated hands
            # (T8o/KQo/A5s/55) -- BUG-022 + B1. Judge vs the top-fraction jam range.
            eq = self._jam_range_equity(hole, [], self.JAM_RANGE_FRACTION)
        else:
            # FLOP / TURN beyond-cap node, uninformed read -- a FACED ALL-IN *or* a
            # money-behind deep raise both face a SELECTED range (4+ aggressions to get
            # here), so judge vs the top-fraction range ranked by the EMD bucket's
            # centroid-MEAN equity (not vs uniform, which over-values -- BUG-023 class).
            # Centroid means are equity-ordered + draw-aware. (Inference #5 extended the
            # earlier turn-all-in-only fix to the money-behind case; same implied-odds
            # trade as B1. River is solver-owned (_solver_inputs returns early above).)
            eq = self._jam_range_equity(hole, board, self.JAM_RANGE_FRACTION)
        # EV(call) vs EV(fold)=0, with all-in-for-less handled (mirror the all-in
        # guard's chip math). For the money-behind deep-raise case this is the eq >=
        # pot-odds rule but scored as if the call runs out -- the documented v1
        # approximation; for a jam it is exact (the board does run out).
        pot = float(ps.get('pot') or 0.0)            # chips in the middle now (incl. the raise/jam)
        call_cost = min(to_call, bot_stack)
        if to_call <= bot_stack:
            final_pot = pot + to_call
        else:
            final_pot = pot - to_call + 2.0 * bot_stack   # opp's unmatched excess returns
        ev_call = eq * final_pot - call_cost
        action = 'call' if ev_call >= 0.0 else 'fold'
        # C1 (opt-in): a monster at this untrained beyond-cap node should re-jam for
        # value / deny equity rather than flatly call. Gated tightly so it is NOT the
        # BUG-011 stray raise: only upgrades a CALL (never a fold), only on dominating
        # uniform-floor equity, only with money behind, and only emits a LEGAL all-in
        # (never a constructed/off-grid size).
        if (self.value_jam and action == 'call' and eq >= self.VALUE_JAM_EQ
                and to_call < bot_stack - 1e-6 and 'allin' in legal_actions):
            action = 'allin'
        self.stats['deep_raise_guard'] += 1
        self.last_debug = {'mode': 'deep_raise_guard', 'street': street,
                           'action': action, 'eq': round(eq, 3),
                           'evCall': round(ev_call, 2), 'toCall': to_call,
                           'botStack': bot_stack}
        return action

    def _uniform_floor_equity(self, hole, board):
        """Equity of `hole` vs a UNIFORM opponent range (card removal only), used by the
        deep-raise guard when the tracker's read has collapsed. Depends only on (hole,
        board), so it is memoized process-wide -- the latency fix: this path fires on
        nearly every faced bet vs an aggressive presser, and the 600-runout MC it
        replaced was ~0.7s each. Preflop (board == []) keys on the hole alone (<=1326
        values). 200 runouts is ample for a coarse floor where a premium sits far from
        any pot-odds boundary; turn/river are exact regardless of the count."""
        ck = (frozenset(hole), frozenset(board))   # order-free: a board is a set of cards
        with self._lock:                            # OrderedDict isn't thread-safe
            cached = self._uniform_eq_cache.get(ck)
            if cached is not None:
                self._uniform_eq_cache.move_to_end(ck)   # mark most-recently-used
                return cached
        from ..game.range_tracker import RangeTracker
        eq = float(RangeTracker(list(hole), self._cards).hero_equity(   # outside the lock
            list(hole), list(board), n_runouts=200))
        with self._lock:
            self._uniform_eq_cache[ck] = eq
            if len(self._uniform_eq_cache) > self._uniform_eq_cap:
                self._uniform_eq_cache.popitem(last=False)   # evict LRU; no flush cliff
        return eq

    def _postflop_means(self, n_board):
        """Per-bucket MEAN equity for the street with `n_board` cards, derived from the
        committed EMD centroids (each is an equity-distribution histogram, so its mean is
        `centroid @ bin_centers`). The bucket INDEX isn't guaranteed equity-ordered after
        a re-bake, but the centroid mean always is the bucket's strength -- so ranking by
        this is robust. Cached per street."""
        cached = self._pf_means_cache.get(n_board)
        if cached is not None:
            return cached
        from ..abstractions.postflop_features import load_centroids
        street = {3: 'flop', 4: 'turn', 5: 'river'}[n_board]
        centroids, bins = load_centroids(street)
        centroids = np.asarray(centroids, dtype=float)
        centers = (np.arange(bins) + 0.5) / bins
        sums = centroids.sum(axis=1)
        means = (centroids @ centers) / np.where(sums > 0.0, sums, 1.0)
        self._pf_means_cache[n_board] = means
        return means

    def _combo_strength(self, combo, board):
        """A scalar 'how strong is this hand here' for ranking the opponent's range.
        Preflop: the equity-quantile bucket INDEX (equity-ordered by construction).
        Postflop: the EMD strength bucket's centroid MEAN equity (equity-ordered + draw-
        aware -- a strong draw's distribution has a high mean)."""
        if not board:
            b = self._cards.get_bucket(list(combo), None)
            return float(int(b.split('_')[1]) if isinstance(b, str) else int(b))
        sb = self._cards.get_bucket(list(combo), list(board))
        idx = int(sb.split('_')[1]) if isinstance(sb, str) else int(sb)
        means = self._postflop_means(len(board))
        return float(means[idx]) if 0 <= idx < len(means) else 0.0

    def _jam_range_equity(self, hole, board, frac):
        """Hero equity vs the top-`frac` of opponent hands (a realistic SELECTED jam /
        deep-raise range), for an uninformed faced all-in / deep raise. Ranks live combos
        by `_combo_strength` (preflop bucket index; postflop centroid-mean equity), keeps
        the strongest `frac`, and returns hero equity vs that range. Folds dominated hands
        the uniform floor would stack off; premiums still clear pot odds. Cached on
        (hole, board, frac); 200 MC runouts is ample for a coarse range floor."""
        ck = (frozenset(hole), frozenset(board), round(frac, 3))
        with self._lock:                            # OrderedDict isn't thread-safe
            cached = self._jam_eq_cache.get(ck)
            if cached is not None:
                self._jam_eq_cache.move_to_end(ck)
                return cached
        from ..game.range_tracker import RangeTracker
        t = RangeTracker(list(hole), self._cards)
        if board:
            t.reveal(list(board))                         # card removal before ranking
        scored = []
        for i, h in enumerate(t.hands):
            if t.w[i] <= 0.0:
                continue
            scored.append((self._combo_strength(h, board), i))
        scored.sort(reverse=True)                         # strongest first
        keep = {i for _, i in scored[:max(1, int(round(frac * len(scored))))]}
        for i in range(len(t.w)):
            if i not in keep:
                t.w[i] = 0.0
        eq = float(t.hero_equity(list(hole), list(board), n_runouts=200))   # outside lock
        with self._lock:
            self._jam_eq_cache[ck] = eq
            if len(self._jam_eq_cache) > self._uniform_eq_cap:
                self._jam_eq_cache.popitem(last=False)
        return eq

    def _belief_is_informative(self, tracker):
        """True iff the tracked range has meaningfully concentrated off uniform (a real
        read). A uniform belief = NO information and must not be trusted for a stack-off
        even at high confidence (BUG-022 / A6). Measures effective-hands (inverse Simpson)
        vs the live-hand count. A stub/foreign tracker without weights returns True (let
        the confidence gate decide)."""
        w = getattr(tracker, 'w', None)
        if w is None:
            return True
        s = float(w.sum())
        if s <= 0.0:
            return True
        p = w / s
        n_live = int((w > 0.0).sum())
        if n_live <= 1:
            return True
        eff_n = 1.0 / float((p * p).sum())
        return (eff_n / n_live) < self.INFORMATIVE_RATIO

    def _trust_read(self, tracker):
        """Use the tracked range for a stack-off only when CONFIDENT and INFORMATIVE."""
        return (getattr(tracker, 'confidence', 0.0) >= self.guard_confidence
                and self._belief_is_informative(tracker))

    def _is_untrained(self, info_set_key, legal_actions):
        """True iff the blueprint has NO usable mass on `legal_actions` at this key --
        i.e. BlueprintStrategy._distribution would fall into its passive coin-flip
        fallback (a beyond-cap / zero-mass node). Single source of truth for the deep
        guard, the safe-call degradation, and the river EV-gate skip (C3). A failed
        lookup returns False (treat as trained -> defer to the blueprint, the safe
        direction)."""
        try:
            stored = self.db.get_average_strategy(info_set_key) if self.db else None
        except Exception:
            return False
        if not stored:
            return True
        return sum(max(0.0, stored.get(a, 0.0)) for a in legal_actions) <= 1e-12

    def _safe_untrained_call(self, info_set_key, legal_actions, ps):
        """Safe degradation on an ERROR path: at an UNTRAINED node FACING A BET, return
        'call' rather than letting decide() fall through to BlueprintStrategy's 50/50
        coin-flip (which folds the nuts half the time -- the leak the guards remove).
        Never throws. Returns 'call', or None (no override -> the normal fallback)."""
        if 'call' not in legal_actions or 'fold' not in legal_actions:
            return None
        if not self._is_untrained(info_set_key, legal_actions):
            return None           # trained -> let the blueprint play its real strategy
        return 'call'             # untrained + facing a bet -> never fold on an error

    def _premium_no_fold(self, action, legal_actions, ps):
        """AA/KK never fold PREFLOP at 100BB heads-up -- they dominate every range at any
        depth. The blueprint mixes a tiny fold mass at some trained preflop nodes (CFR
        noise); sampling it folds the nuts (observed: pf_29_ip_sl folding AA to a 3-bet).
        If the chosen action is a preflop fold with pocket AA/KK, upgrade it to call
        (never raise -- keep the blueprint's aggression when it already raises). This is
        belt-and-suspenders to the guards, which only cover UNTRAINED nodes; this fires
        at TRAINED ones. Scoped to AA/KK by RANK (not the equity bucket, which also holds
        QQ/AKs -- legitimately foldable to deep aggression)."""
        if action != 'fold' or ps.get('street') != 'preflop':
            return action
        hole = ps.get('hole_cards') or ()
        if len(hole) == 2 and hole[0][1] == hole[1][1] and hole[0][1] in ('A', 'K'):
            if 'call' in legal_actions:
                self.stats['premium_no_fold'] += 1
                self.last_debug = {'mode': 'premium_no_fold', 'street': 'preflop',
                                   'hand': ''.join(hole), 'overrode': 'fold->call'}
                return 'call'
        return action

    def _first_act_value_guard(self, info_set_key, legal_actions, ps):
        """Inference #1: at an UNTRAINED FIRST-TO-ACT postflop node (no bet faced) the
        blueprint passive fallback checks 100%, surrendering value with a strong made
        hand. If the hand's equity vs the range clears FIRST_ACT_VALUE_EQ, value-bet ONE
        conservative TRAINED size (a `bet_*` that's already in legal -- never a constructed
        / off-grid size, so no BUG-011 stray-raise); otherwise defer (-> check). Scoped to
        flop/turn (preflop opens are well-trained; the river is solver-owned). Returns the
        bet action or None."""
        if 'call' in legal_actions or 'fold' in legal_actions or 'check' not in legal_actions:
            return None                                  # facing a bet (the guards own it)
        if ps.get('street') not in ('flop', 'turn'):
            return None                                  # preflop trained / river -> solver
        sized = next((a for a in ('bet_medium', 'bet_small', 'bet_large')
                      if a in legal_actions), None)
        if sized is None:
            return None                                  # no trained sized bet available
        if not self._is_untrained(info_set_key, legal_actions):
            return None                                  # trained -> the blueprint plays
        tracker = ps.get('opp_range')
        hole = ps.get('hole_cards')
        board = ps.get('community')
        if tracker is None or not hole or board is None or ps.get('seat') is None:
            return None
        if self._trust_read(tracker):
            eq = float(tracker.hero_equity(list(hole), list(board), n_runouts=200))
        else:
            eq = self._uniform_floor_equity(hole, board)
        if eq >= self.FIRST_ACT_VALUE_EQ:
            self.last_debug = {'mode': 'first_act_value', 'street': ps.get('street'),
                               'action': sized, 'eq': round(eq, 3)}
            return sized
        return None                                      # not strong enough -> check (defer)

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
            # 1b: if the turn solver deviated, the river must honor the turn's promise
            # (gadget-clamp to the turn-solved reaches) rather than auto-exploit.
            'turn_deviated': bool(ps.get('turnDeviated')),
        }

    # -- core: solve + read off ------------------------------------------------
    def solve_for_action(self, *, board, pot_entry, stacks, bot_seat, hole,
                         villain_tracker, hero_tracker, confidence, river_path,
                         turn_deviated=False):
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

        if self.safe_gadget:
            # 1b: after a TURN deviation, force the gadget to CLAMP to the turn's promise
            # (the 'belief' anchor opt-out = blueprint_cfv at the turn-solved hero reach
            # [via 1a] + the tracked villain) rather than auto-exploit -- otherwise the
            # river plays a strategy the turn leaf didn't plan for (NN_LEAF_PLAN 6d).
            anchor = 'belief' if turn_deviated else self.gadget_anchor
            cfr, info = self._safe_solve(tree, ba, reach0, reach1, hero, villain,
                                         bot_seat, villain_tracker, idx, anchor=anchor)
        else:
            cfr, info = solve_river(
                tree, ba, reach0, reach1, max_iters=self.max_iters,
                check_every=self.check_every, gap_threshold=self.gap_threshold,
                time_budget=self.time_budget)

        # Reaches INTO this node (root reaches conditioned on the realized river
        # betting), needed for the EV gate AND the deep-node zero-reach guard below.
        node_reach0, node_reach1 = cfr.reach_into(edge_path, reach0, reach1)
        # Guard a silent UNIFORM read-off at the DECISION node, not just the root: if the
        # bot's actual hand has ~zero reach INTO this realized node, its strat_sum row
        # there never moved, so average_strategy returns uniform 1/A -> a near-random
        # action read off as the solve's recommendation. The root-reach guard above
        # doesn't catch a hand that reaches the root but rarely takes the realized line.
        # Treat as unsolvable -> decide() falls back to the blueprint cleanly.
        hero_node_reach = node_reach0 if bot_seat == 0 else node_reach1
        if hero_node_reach[row] <= 1e-12:
            raise ValueError("bot hand has ~zero reach into the realized node; "
                             "read-off would be uniform")
        dist = read_action_strategy(cfr, node, hole, ba, idx)
        # Carry the solve context so decide() can run the EV gate without re-solving.
        info.update({'cfr': cfr, 'ba': ba, 'idx': idx,
                     'reach0': node_reach0, 'reach1': node_reach1})
        return dist, node, info

    # -- safe-gadget solve policy (belief / blueprint / auto) ------------------
    def _safe_solve(self, tree, ba, reach0, reach1, hero, villain, bot_seat,
                    villain_tracker, idx, anchor=None):
        """Run the river solve under the safe-gadget policy (`anchor`, default
        self.gadget_anchor; 1b passes 'belief' to clamp to the turn promise) and
        return (cfr, info). 'belief'/'blueprint' anchor the re-solving gadget directly
        (tracked reach vs uniform card-removal range). 'auto' EXPLOITS (unsafe-v1) when
        that is provably within the blueprint's river best-response on THIS spot, and
        otherwise falls back to the blueprint-anchored gadget -- the Phase-5a per-spot
        safety self-check (mirrors tests/test_safe_river_gadget.py). A confidence
        PRE-FILTER skips the check (and its best-response walks) when the read is already
        trusted+informative -- the narrow-WRONG belief the self-check guards against is
        unlikely there.

        Latency note: 'auto' runs TWO solves (unsafe + gadget) ONLY on the rare spot where
        the read is untrusted AND the self-check trips; the pre-filter avoids the double
        solve in the common (trusted) case, and 'belief'/'blueprint' always run one."""
        from .blueprint_projection import blueprint_cfv, blueprint_strategy_on_tree
        anchor = anchor or self.gadget_anchor       # 1b may force 'belief' (turn-promise clamp)
        villain_seat = 1 - bot_seat
        raw = self.db.get_average_strategy if self.db else (lambda k: None)
        # Uniform card-removal villain range -- the robust ('blueprint') anchor + the
        # range the self-check measures exploitability over.
        uniform_villain = (project_tracker(villain_tracker, ba, idx) > 0).astype(float)

        def gadget(anchor_villain, optout=None):
            g0, g1 = (hero, anchor_villain) if bot_seat == 0 else (anchor_villain, hero)
            if optout is None:
                # Opt-out CFVs must be on the SAME reaches the gadget weights with.
                optout = blueprint_cfv(tree, ba, raw, g0, g1, villain_seat,
                                       self._postflop_menu)
            return solve_river_gadget(
                tree, ba, hero, anchor_villain, optout, villain_seat,
                max_iters=self.max_iters, check_every=self.check_every,
                time_budget=self.time_budget)

        if anchor == 'belief':
            cfr, info = gadget(villain)
            info['anchor'] = 'belief'
            return cfr, info
        if anchor == 'blueprint':
            cfr, info = gadget(uniform_villain)
            info['anchor'] = 'blueprint'
            return cfr, info

        # --- 'confidence' (A) and 'auto' (B): solve unsafe, exploit on a trusted read ---
        cfr, info = solve_river(
            tree, ba, reach0, reach1, max_iters=self.max_iters,
            check_every=self.check_every, gap_threshold=self.gap_threshold,
            time_budget=self.time_budget)
        if self._trust_read(villain_tracker):
            info['anchor'] = ('confidence_exploit' if anchor == 'confidence'
                              else 'auto_exploit_trusted')   # pre-filter: skip the BR walks
            return cfr, info
        if anchor == 'confidence':
            # A: untrusted read -> straight to the safe blueprint-anchored gadget. Cheaper
            # than 'auto' (no self-check / BR walks), but clamps even on the spots where the
            # exploit would have been safe -- it gives up that exploitation.
            cfr_g, info_g = gadget(uniform_villain)
            info_g['anchor'] = 'confidence_safe'
            return cfr_g, info_g
        # B 'auto': self-check -- villain best-response (over the robust UNIFORM range) to
        # the unsafe strategy vs to the blueprint -- the exact comparison the battery makes.
        bp_strat = blueprint_strategy_on_tree(tree, ba, raw, self._postflop_menu)
        e_unsafe = float((uniform_villain * cfr._br_value(
            tree.root, villain_seat, hero, cfr.average_strategy)).sum())
        e_bp = float((uniform_villain * cfr._br_value(
            tree.root, villain_seat, hero, lambda nid: bp_strat[nid])).sum())
        diag = {'eUnsafe': round(e_unsafe, 3), 'eBlueprint': round(e_bp, 3)}
        if e_unsafe <= e_bp + self._AUTO_SAFE_MARGIN * (abs(e_bp) + 1.0):
            info['anchor'] = 'auto_exploit_safe'         # exploit is within blueprint
            info['autoSelfCheck'] = diag
            return cfr, info
        cfr_g, info_g = gadget(uniform_villain)          # over-exploiting -> safe gadget
        info_g['anchor'] = 'auto_safe_fallback'
        info_g['autoSelfCheck'] = diag
        return cfr_g, info_g

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
            with self._lock:                        # numpy Generator isn't thread-safe
                choice = str(self._rng.choice(labels, p=weights / weights.sum()))
        # Record the sampled TREE action so a turn solve can chain the bot's own range
        # off the SOLVED play (continual re-solving 1a). Harmless to the river path.
        if self.last_debug is not None:
            self.last_debug['chosenTreeAction'] = choice

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
