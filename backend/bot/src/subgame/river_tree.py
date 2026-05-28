# backend/bot/src/subgame/river_tree.py
"""
River betting tree for the Phase-4 subgame solver (step 2 of the build).

Builds the full one-street betting tree for a heads-up river, in EXACT chips,
with real pot/stacks and the solver's own (finer) size menu. This is the
scaffold the two-sided CFR+ (step 3) runs on; it carries NO ranges and does NO
showdown math -- terminals just describe the pot/contributions so the showdown
kernel can be applied per hero hand at solve time.

CONVENTIONS (match cfr/poker_game.py so the tree is consistent with the game the
blueprint was trained on -- only the size *menu* differs):
  * Heads-up, OOP acts first on the river. Postflop OOP = BB = seat 1 (per
    PokerGame._acting_player's +1 offset), IP = SB/button = seat 0.
  * pot_entry P = chips already in the middle at river start. Because the hand
    reached the river with all prior bets called, BOTH players contributed P/2 --
    so each player's total invested at a terminal is P/2 + (their river chips).
  * bet  = frac * pot_in_middle.
  * raise: new street-total for the raiser = frac * (pot_in_middle + to_call)
    + to_call  (i.e. a raise is `frac` of the pot-after-call on top of the call),
    exactly PokerGame.calculate_raise_amount's postflop branch.
  * <= 3 aggressions per street (1 bet + 2 raises), matching
    PokerGame.max_raises_per_street (=2) + 1.
  * A sized bet/raise whose cost meets-or-exceeds the actor's remaining stack is
    not a distinct action -- it collapses to all-in (deduped). After an all-in,
    the other player has only fold/call.
  * Min legal bet = 1 BB; a raise's increment over the current call must be >=
    the previous bet/raise increment (standard no-limit min-raise).

The default menu is deliberately SMALL ({0.5,0.75,1.0,1.5}x pot + all-in) per the
locked design decision: start lean for solve speed + convergence, widen only
where LBR reveals a leak.
"""
from ..abstractions.sizing import BIG_BLIND
from ..cfr.poker_game import STARTING_STACK

# Solver's river bet/raise size menu, as fractions of the pot (bet) or the
# pot-after-call (raise). All-in is added separately. Start small; grow on LBR.
DEFAULT_MENU = (0.5, 0.75, 1.0, 1.5)
MAX_AGGRESSIONS = 3          # 1 bet + 2 raises (PokerGame.max_raises_per_street + 1)
MIN_BET = float(BIG_BLIND)   # 1 BB
OOP_SEAT = 1                 # BB acts first postflop
IP_SEAT = 0


class RiverNode:
    """A node in the river tree. Decision nodes carry parallel `actions`/`children`
    lists; terminals carry the pot/contribution descriptor the kernel needs."""
    __slots__ = ('terminal', 'player', 'to_call', 'pot_mid', 'sc', 'actions',
                 'children', 'node_id', 'final_pot', 'contrib', 'folder')

    def __init__(self, terminal):
        self.terminal = terminal
        # decision-node fields
        self.player = None          # seat to act (0/1)
        self.to_call = 0.0          # chips the actor must add to continue
        self.pot_mid = 0.0          # chips already in the middle at this node
        self.sc = (0.0, 0.0)        # river chips committed per seat at this node
        self.actions = []           # list of action labels (str)
        self.children = []          # parallel list of RiverNode
        self.node_id = -1           # dense index among decision nodes (for CFR tables)
        # terminal-node fields
        self.final_pot = 0.0        # total chips contested at showdown/fold
        self.contrib = (0.0, 0.0)   # each seat's TOTAL invested (P/2 + river chips)
        self.folder = None          # seat that folded (fold terminal), else None


# -- action labels -------------------------------------------------------------
# 'check', 'fold', 'call', 'allin', or 'bet:<chips>' / 'raise:<chips>' where
# <chips> is the EXACT additional cost to the actor (so the label is unambiguous
# and the CFR/replay code never re-derives a size).
def _sized(kind, chips):
    return f"{kind}:{chips:.6g}"


def is_sized(label):
    return label.startswith('bet:') or label.startswith('raise:')


def sized_chips(label):
    """Additional chips for a 'bet:'/'raise:' label."""
    return float(label.split(':', 1)[1])


class RiverTree:
    """Builds + holds the river betting tree. `decision_nodes` is the dense list
    the CFR+ solver attaches regret/strategy tables to (indexed by node_id)."""

    def __init__(self, pot_entry, stacks, menu=DEFAULT_MENU,
                 max_aggressions=MAX_AGGRESSIONS):
        if pot_entry <= 0:
            raise ValueError("pot_entry must be positive")
        self.pot_entry = float(pot_entry)
        self.stacks = (float(stacks[0]), float(stacks[1]))
        # EQUAL-STACK INVARIANT. Showdown terminals assume both players' total
        # contributions are matched (the kernel uses final_pot = c0 + c1 and
        # hero_total = c_hero directly). That is correct ONLY when no all-in-for-
        # less occurs -- i.e. when river-entry stacks are equal, which they always
        # are in this game (both start each hand at STARTING_STACK and must match
        # all prior-street action to reach a river betting node). With UNEQUAL
        # stacks a short call leaves the aggressor's excess uncalled; that excess
        # is not returned here, so the showdown pot/contribution would be inflated
        # and the EV wrong. chip-conservation can't catch this (contrib sums to
        # final_pot by construction either way), so enforce the invariant loudly.
        # If unequal stacks are ever needed, cap the aggressor's contribution to
        # the matched amount at all-in-for-less showdown terminals before lifting
        # this assert.
        if abs(self.stacks[0] - self.stacks[1]) > 1e-6:
            raise ValueError(
                f"river tree assumes equal river-entry stacks (got {self.stacks}); "
                "unequal stacks need all-in-for-less handling (see RiverTree docstring)")
        self.menu = tuple(menu)
        self.max_aggressions = int(max_aggressions)
        self.entry_contrib = self.pot_entry / 2.0   # each seat's prior-street investment
        self.decision_nodes = []
        # River betting starts with OOP to act, both even (sc = 0), no aggression yet.
        self.root = self._build(player=OOP_SEAT, sc=[0.0, 0.0], agg=0,
                                last_inc=MIN_BET, prev_check=False)

    # -- terminals -------------------------------------------------------------
    def _terminal(self, sc, folder):
        n = RiverNode(terminal=True)
        n.final_pot = self.pot_entry + sc[0] + sc[1]
        n.contrib = (self.entry_contrib + sc[0], self.entry_contrib + sc[1])
        n.folder = folder
        return n

    # -- recursive builder -----------------------------------------------------
    def _build(self, player, sc, agg, last_inc, prev_check):
        """player to act; sc = river chips committed per seat; agg = aggressions
        so far; last_inc = last bet/raise increment (for min-raise); prev_check =
        the immediately preceding action was a check (so a check here closes)."""
        node = RiverNode(terminal=False)
        node.player = player
        to_call = max(sc) - sc[player]
        pot_mid = self.pot_entry + sc[0] + sc[1]
        node.to_call = to_call
        node.pot_mid = pot_mid
        node.sc = (sc[0], sc[1])
        node.node_id = len(self.decision_nodes)
        self.decision_nodes.append(node)

        stack_p = self.stacks[player] - sc[player]   # chips the actor has left
        other = 1 - player

        def add(label, child):
            node.actions.append(label)
            node.children.append(child)

        if to_call <= 1e-9:
            # ---- no bet to call: check or open a bet ----
            if prev_check:
                # second consecutive check closes the street -> showdown.
                add('check', self._terminal(sc, folder=None))
            else:
                add('check', self._build(other, sc, agg, last_inc, prev_check=True))
            self._add_aggressive(node, add, player, other, sc, agg, last_inc,
                                  pot_mid, to_call, stack_p, opening=True)
        else:
            # ---- facing a bet/raise: fold / call / (raise) ----
            add('fold', self._terminal(sc, folder=player))
            call_amt = min(to_call, stack_p)        # all-in-for-less allowed
            sc_call = list(sc)
            sc_call[player] += call_amt
            add('call', self._terminal(sc_call, folder=None))   # call closes -> showdown
            # A raise is only possible if the OPPONENT still has chips behind to
            # call it; once they are all-in there is nothing to raise into, so the
            # actor has only fold/call (this also blocks shoving over an all-in).
            opp_behind = self.stacks[other] - sc[other]
            if agg < self.max_aggressions and opp_behind > 1e-9:
                self._add_aggressive(node, add, player, other, sc, agg, last_inc,
                                     pot_mid, to_call, stack_p, opening=False)

        return node

    def _add_aggressive(self, node, add, player, other, sc, agg, last_inc,
                         pot_mid, to_call, stack_p, opening):
        """Append the sized bet/raise menu actions + a single all-in (deduped)."""
        if stack_p <= 1e-9:
            return
        need_allin = False
        seen_costs = set()
        for frac in self.menu:
            if opening:
                amount = frac * pot_mid                       # bet = frac * pot
                new_sc_p = sc[player] + amount
                inc = amount                                  # increment over 0
                legal_min = amount >= MIN_BET - 1e-9
            else:
                new_sc_p = frac * (pot_mid + to_call) + to_call   # engine raise-to
                amount = new_sc_p - sc[player]                # additional cost
                inc = new_sc_p - max(sc)                      # raise over the call
                legal_min = inc >= last_inc - 1e-9 and amount > 0
            if not legal_min:
                continue
            if amount >= stack_p - 1e-9:
                need_allin = True                             # collapses to all-in
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

        # All-in as a genuine aggression: shove the whole remaining stack, but only
        # if it strictly exceeds a call (otherwise it's just a call/all-in-for-less).
        allin_new_sc = sc[player] + stack_p
        if need_allin or allin_new_sc > max(sc) + 1e-9:
            child_sc = list(sc)
            child_sc[player] = allin_new_sc
            inc = allin_new_sc - max(sc)
            # After an all-in the other player faces it with only fold/call: that
            # falls out naturally (their stack can't afford a raise over the shove).
            add('allin', self._build(other, child_sc, agg + 1, max(inc, last_inc),
                                     prev_check=False))


def build_river_tree(pot_entry, stacks, menu=DEFAULT_MENU,
                     max_aggressions=MAX_AGGRESSIONS):
    """Convenience: return a built RiverTree. `stacks` = (seat0_behind, seat1_behind)
    remaining at river entry; `pot_entry` = chips already in the middle."""
    return RiverTree(pot_entry, stacks, menu=menu, max_aggressions=max_aggressions)
