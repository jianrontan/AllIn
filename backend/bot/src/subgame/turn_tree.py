# backend/bot/src/subgame/turn_tree.py
"""
Turn betting tree for the depth-limited turn subgame solver (M1).

Mirrors river_tree.py's one-street betting scaffold and chip accounting EXACTLY
(same OOP-first order, engine-matching bet/raise sizing, aggression cap, all-in
dedup, min-raise legality, equal-stack invariant) -- with ONE difference: the turn
is NOT the last street. When turn betting CLOSES without a fold (check-check, or a
call), the river is still to come, so that terminal is a DEPTH-LIMIT LEAF, not a
showdown. Its value is the blueprint's river-continuation value (src/subgame/cfv.py:
the reach-conditioned leaf matrix), supplied at solve time by M2. Only FOLD
terminals are true terminals (pot awarded, no river).

Leaf node fields: `final_pot` (chips in the middle entering the river) and
`leaf_stacks` (each seat's behind stack entering the river) are exactly the
(pot_entry, stacks) the leaf value function feeds to its inner river evaluation.
An all-in-and-called turn line closes with leaf_stacks == (0, 0): the leaf value
function then builds a trivial inner river tree (no chips behind -> no betting) and
returns pure equity-to-river, so an all-in on the turn needs NO special case here.

CONVENTIONS: identical to river_tree.py (see its docstring). The sized-edge label
format (bet:/raise:/allin) and the size menu / caps are imported from river_tree so
the two trees can NEVER drift on how an edge is spelled (path/projection code parses
these) or on the bet sizes (single source -> abstractions/sizing.py).
"""
from .river_tree import (_sized, is_sized, sized_chips, DEFAULT_MENU,
                         MAX_AGGRESSIONS, MIN_BET, OOP_SEAT, IP_SEAT)

__all__ = ['TurnTree', 'TurnNode', 'build_turn_tree', 'is_leaf', 'is_sized',
           'sized_chips', 'DEFAULT_MENU', 'MAX_AGGRESSIONS', 'MIN_BET',
           'OOP_SEAT', 'IP_SEAT']


class TurnNode:
    """A node in the turn tree. Decision nodes carry parallel `actions`/`children`
    lists; terminals carry the pot/contribution descriptor (fold) plus, for a
    depth-limit LEAF, the (final_pot, leaf_stacks) the leaf value function needs."""
    __slots__ = ('terminal', 'player', 'to_call', 'pot_mid', 'sc', 'actions',
                 'children', 'node_id', 'agg', 'last_inc', 'final_pot', 'contrib',
                 'folder', 'leaf_stacks')

    def __init__(self, terminal):
        self.terminal = terminal
        # decision-node fields
        self.player = None
        self.to_call = 0.0
        self.pot_mid = 0.0
        self.sc = (0.0, 0.0)
        self.actions = []
        self.children = []
        self.node_id = -1
        self.agg = 0
        self.last_inc = 0.0
        # terminal-node fields
        self.final_pot = 0.0        # total chips in the middle entering the river
        self.contrib = (0.0, 0.0)   # each seat's TOTAL invested (P/2 + turn chips)
        self.folder = None          # seat that folded (fold terminal); None => LEAF
        self.leaf_stacks = (0.0, 0.0)   # behind per seat entering the river (LEAF only)


def is_leaf(node):
    """A depth-limit leaf: a non-fold terminal (turn betting closed -> river to come)."""
    return node.terminal and node.folder is None


class TurnTree:
    """Builds + holds the turn betting tree. `decision_nodes` is the dense list the
    CFR+ solver attaches regret/strategy tables to (indexed by node_id). Non-fold
    terminals are LEAVES carrying (final_pot, leaf_stacks) for the leaf value fn."""

    def __init__(self, pot_entry, stacks, menu=DEFAULT_MENU,
                 max_aggressions=MAX_AGGRESSIONS):
        if pot_entry <= 0:
            raise ValueError("pot_entry must be positive")
        self.pot_entry = float(pot_entry)
        self.stacks = (float(stacks[0]), float(stacks[1]))
        # EQUAL-STACK INVARIANT (mirrors RiverTree). Holds at turn entry: both seats
        # start each hand at STARTING_STACK and must match all prior-street action to
        # reach a turn betting node, so behind stacks are equal. This guarantees every
        # closing call is full (no all-in-for-less) and therefore every LEAF has EQUAL
        # behind stacks -- exactly the equal-stack precondition the inner river
        # evaluation (RiverTree) itself asserts. If unequal turn-entry stacks are ever
        # needed, handle all-in-for-less before lifting this (see RiverTree docstring).
        if abs(self.stacks[0] - self.stacks[1]) > 1e-6:
            raise ValueError(
                f"turn tree assumes equal turn-entry stacks (got {self.stacks}); "
                "unequal stacks need all-in-for-less handling (see RiverTree docstring)")
        self.menu = tuple(menu)
        self.max_aggressions = int(max_aggressions)
        self.entry_contrib = self.pot_entry / 2.0   # each seat's prior-street investment
        self.decision_nodes = []
        # Turn betting starts with OOP to act, both even (sc = 0), no aggression yet.
        self.root = self._build(player=OOP_SEAT, sc=[0.0, 0.0], agg=0,
                                last_inc=MIN_BET, prev_check=False)

    # -- terminals -------------------------------------------------------------
    def _terminal(self, sc, folder):
        n = TurnNode(terminal=True)
        n.final_pot = self.pot_entry + sc[0] + sc[1]
        n.contrib = (self.entry_contrib + sc[0], self.entry_contrib + sc[1])
        n.folder = folder
        if folder is None:
            # Depth-limit LEAF: the river is to come. behind = turn-entry stack minus
            # turn chips. A non-fold close (check-check / call) leaves sc matched, so
            # leaf_stacks are equal -- the precondition the inner river eval asserts.
            ls = (self.stacks[0] - sc[0], self.stacks[1] - sc[1])
            assert abs(ls[0] - ls[1]) < 1e-6, ("leaf stacks must be equal", sc, ls)
            n.leaf_stacks = ls
        return n

    # -- recursive builder (identical betting logic to RiverTree) ---------------
    def _build(self, player, sc, agg, last_inc, prev_check):
        node = TurnNode(terminal=False)
        node.player = player
        to_call = max(sc) - sc[player]
        pot_mid = self.pot_entry + sc[0] + sc[1]
        node.to_call = to_call
        node.pot_mid = pot_mid
        node.sc = (sc[0], sc[1])
        node.agg = agg
        node.last_inc = last_inc
        node.node_id = len(self.decision_nodes)
        self.decision_nodes.append(node)

        stack_p = self.stacks[player] - sc[player]
        other = 1 - player

        def add(label, child):
            node.actions.append(label)
            node.children.append(child)

        if to_call <= 1e-9:
            # ---- no bet to call: check or open a bet ----
            if prev_check:
                # second consecutive check closes the turn -> LEAF (river to come).
                add('check', self._terminal(sc, folder=None))
            else:
                add('check', self._build(other, sc, agg, last_inc, prev_check=True))
            self._add_aggressive(node, add, player, other, sc, agg, last_inc,
                                  pot_mid, to_call, stack_p, opening=True)
        else:
            # ---- facing a bet/raise: fold / call / (raise) ----
            add('fold', self._terminal(sc, folder=player))
            call_amt = min(to_call, stack_p)        # all-in-for-less allowed (won't occur, equal stacks)
            sc_call = list(sc)
            sc_call[player] += call_amt
            add('call', self._terminal(sc_call, folder=None))   # call closes -> LEAF
            opp_behind = self.stacks[other] - sc[other]
            if agg < self.max_aggressions and opp_behind > 1e-9:
                self._add_aggressive(node, add, player, other, sc, agg, last_inc,
                                     pot_mid, to_call, stack_p, opening=False)

        return node

    def _add_aggressive(self, node, add, player, other, sc, agg, last_inc,
                         pot_mid, to_call, stack_p, opening):
        """Append the sized bet/raise menu actions + a single all-in (deduped).
        Identical to RiverTree._add_aggressive."""
        if stack_p <= 1e-9:
            return
        need_allin = False
        seen_costs = set()
        for frac in self.menu:
            if opening:
                amount = frac * pot_mid
                new_sc_p = sc[player] + amount
                inc = amount
                legal_min = amount >= MIN_BET - 1e-9
            else:
                new_sc_p = frac * (pot_mid + to_call) + to_call
                amount = new_sc_p - sc[player]
                inc = new_sc_p - max(sc)
                legal_min = inc >= last_inc - 1e-9 and amount > 0
            if not legal_min:
                continue
            if amount >= stack_p - 1e-9:
                need_allin = True
                continue
            cost_key = round(amount, 6)
            if cost_key in seen_costs:
                continue
            seen_costs.add(cost_key)
            child_sc = list(sc)
            child_sc[player] = new_sc_p
            kind = 'bet' if opening else 'raise'
            add(_sized(kind, amount),
                self._build(other, child_sc, agg + 1, inc, prev_check=False))

        allin_new_sc = sc[player] + stack_p
        if need_allin or allin_new_sc > max(sc) + 1e-9:
            child_sc = list(sc)
            child_sc[player] = allin_new_sc
            inc = allin_new_sc - max(sc)
            add('allin', self._build(other, child_sc, agg + 1, max(inc, last_inc),
                                     prev_check=False))


def build_turn_tree(pot_entry, stacks, menu=DEFAULT_MENU,
                    max_aggressions=MAX_AGGRESSIONS):
    """Convenience: return a built TurnTree. `stacks` = (seat0_behind, seat1_behind)
    remaining at TURN entry; `pot_entry` = chips already in the middle."""
    return TurnTree(pot_entry, stacks, menu=menu, max_aggressions=max_aggressions)
