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
    patterns = _node_patterns(tree, postflop_menu)
    pf = ba['pf']
    strg = ba['strg'][3]                 # river strength bucket per hand
    groups = ba['groups'][3]             # [(mask, rep_idx), ...] by (preflop, strength)
    out = [None] * len(tree.decision_nodes)

    for node in tree.decision_nodes:
        nid = node.node_id
        pos = 'oop' if node.player == 1 else 'ip'
        pattern = patterns[nid]
        mat = np.zeros((ba['H'], len(node.actions)))
        for mask, rep in groups:
            key = make_info_set_key(3, pos, pf[rep], strg[rep], pattern)
            tree_dist = blueprint_to_tree_dist(raw_strategy(key) or {}, node, postflop_menu)
            mat[mask] = np.array([tree_dist[a] for a in node.actions])
        out[nid] = mat
    return out
