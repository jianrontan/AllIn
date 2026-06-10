# backend/bot/src/subgame/blueprint_projection.py
"""
Project the blueprint's river strategy onto the full river subgame tree -- the
per-node / per-hand piece deferred in step 6b. Needed to MEASURE how exploitable
the blueprint's river play is (run it through RiverCFR.exploitability), which is
the direct scoreboard for the river solver: blueprint river-exploitability minus
the solver's (~0) = the leak the solver removes.

For each tree decision node and each hand:
  * the blueprint key = make_info_set_key(3, node-owner position, hand's preflop +
    river-strength buckets, the river bet PATTERN from the root to this node);
  * the river pattern is the tree path's actions categorised to blueprint chars
    (sized bets/raises -> nearest blueprint size by pot fraction);
  * the blueprint's stored distribution for that key is mapped onto the node's
    tree action menu (blueprint_to_tree_dist).
Hands sharing a (preflop, strength) bucket share a key, so the blueprint is
queried once per bucket group per node (reusing ba['groups'][3]).
"""
import numpy as np

from ..abstractions.sizing import POSTFLOP_BET_MULT
from ..cfr.keys import make_info_set_key, action_char
from .river_tree import is_sized, sized_chips
from .river_subgame_solver import blueprint_to_tree_dist


def tree_action_char(action, node, postflop_menu=None):
    """A tree action -> the blueprint pattern char it maps to. check/call/fold/
    allin are direct; a sized bet/raise -> the nearest blueprint size char by pot
    fraction (`postflop_menu`, default POSTFLOP_BET_MULT), matching how the engine
    would categorise it. The menu MUST match the blueprint's arm (capped adds the
    2.0x 'overbet2'/'2' size) or the projected river pattern keys won't match the
    blueprint's stored keys."""
    menu = postflop_menu if postflop_menu is not None else POSTFLOP_BET_MULT
    if action == 'check':
        return 'k'
    if action == 'call':
        return 'c'
    if action == 'fold':
        return 'f'
    if action == 'allin':
        return 'a'
    if action.startswith('bet:'):
        frac = sized_chips(action) / node.pot_mid
    else:  # raise:
        tc = node.to_call
        frac = (node.sc[node.player] + sized_chips(action) - tc) / (node.pot_mid + tc)
    size = min(menu.items(), key=lambda kv: abs(kv[1] - frac))[0]
    return action_char(f'bet_{size}')


def _node_patterns(tree, postflop_menu=None):
    """node_id -> blueprint river pattern (chars from the root to that node)."""
    pat = {}

    def walk(node, p):
        if node.terminal:
            return
        pat[node.node_id] = p
        for a, child in zip(node.actions, node.children):
            walk(child, p + tree_action_char(a, node, postflop_menu))

    walk(tree.root, '')
    return pat


def blueprint_strategy_on_tree(tree, ba, raw_strategy, postflop_menu=None):
    """Per-node blueprint strategy on the tree.

    raw_strategy(key) -> {action: prob} or None (e.g. BlueprintDB.get_average_strategy).
    Returns a list indexed by node_id; each entry is an [H, A] row-stochastic
    array (A = that node's tree actions). Suitable as a strat_fn for
    RiverCFR.exploitability via `lambda nid: out[nid]`.

    `postflop_menu` selects the blueprint's arm (control vs capped) so both the
    river PATTERN chars and the action-distribution projection use the right size
    set -- pass postflop_menu_for(db_menu_mode(db)). Default = control menu.
    """
    return _project(tree, ba, raw_strategy, 3, ba['strg'][3], ba['groups'][3],
                    postflop_menu)


def blueprint_cfv(tree, ba, raw_strategy, reach0, reach1, villain_seat,
                  postflop_menu=None):
    """Per-villain-hand counterfactual value at the river-entry root, with BOTH
    players playing the blueprint from the root to showdown -- the opt-out values
    for the safe-solving gadget (Phase 5a, SAFE_RIVER_SOLVING_PLAN.md piece 1).

    `v_blueprint(h)` is what villain hand h is already guaranteed by NOT entering
    the re-solve: the value of staying on the blueprint. Giving the villain this as
    a per-hand floor in the gadget game is what makes the re-solved strategy
    no-more-exploitable than the blueprint (Burch/Moravcik/Brown).

    reach0/reach1 are the river-ENTRY reaches (same snapshot the solve uses, so the
    opt-out is consistent with the gadget constraint). `villain_seat` is the
    non-hero seat. Returns a length-H vector in MEASURE units (weighted by the
    HERO's reach -- the opponent of the villain), matching node_action_values /
    the gadget value floor. `postflop_menu` selects the blueprint's arm (see
    blueprint_strategy_on_tree).
    """
    from .river_cfr import RiverCFR

    cfr = RiverCFR(tree, ba)
    strat = blueprint_strategy_on_tree(tree, ba, raw_strategy, postflop_menu)
    v0, v1 = cfr._eval(tree.root, np.asarray(reach0, float),
                       np.asarray(reach1, float), lambda nid: strat[nid])
    return v1 if villain_seat == 1 else v0


def blueprint_turn_strategy_on_tree(tree, ba, raw_strategy, postflop_menu=None):
    """Turn analogue of blueprint_strategy_on_tree (Stage 3): the blueprint's TURN
    strategy projected onto a TurnTree, for measuring how exploitable the blueprint's
    turn play is vs the solved turn strategy. `ba` MUST be
    build_turn_board_arrays(board4, cards) so it carries pf/strg2/groups2. Keys are
    street-2 turn keys; the turn pattern resets at the tree root (current-street only),
    matching the blueprint's turn key format."""
    return _project(tree, ba, raw_strategy, 2, ba['strg2'], ba['groups2'], postflop_menu)


def _project(tree, ba, raw_strategy, street, strg, groups, postflop_menu):
    """Shared projection: street 3 (river, strg[3]/groups[3]) or 2 (turn,
    strg2/groups2). _node_patterns + tree_action_char + blueprint_to_tree_dist are
    street-agnostic (they read node.pot_mid/to_call/actions), so only the key street +
    bucket arrays differ."""
    patterns = _node_patterns(tree, postflop_menu)
    pf = ba['pf']
    out = [None] * len(tree.decision_nodes)
    for node in tree.decision_nodes:
        nid = node.node_id
        pos = 'oop' if node.player == 1 else 'ip'
        pattern = patterns[nid]
        mat = np.zeros((ba['H'], len(node.actions)))
        for mask, rep in groups:
            key = make_info_set_key(street, pos, pf[rep], strg[rep], pattern)
            tree_dist = blueprint_to_tree_dist(raw_strategy(key) or {}, node, postflop_menu)
            mat[mask] = np.array([tree_dist[a] for a in node.actions])
        out[nid] = mat
    return out
