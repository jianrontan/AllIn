# backend/bot/src/cfr/translation.py
"""
Pseudo-harmonic action translation (Ganzfried & Sandholm 2013).

The blueprint is trained on a discrete grid of bet sizes (postflop 0.33 / 0.66 /
1.0x pot, plus all-in). A real opponent can bet ANY size. Snapping an off-grid
bet to the single nearest grid size is exploitable: a 0.95x-pot bet and a 1.0x
bet are treated identically, and a 1.5x overbet collapses to "1.0x" so the bot
under-folds. Action translation instead maps the off-grid bet onto the TWO
bracketing grid sizes and blends the blueprint's responses, weighting the nearer
size more. The pseudo-harmonic mapping is the standard low-exploitability choice.

This module is pure arithmetic over bet *fractions* (bet / pot-after-call, the
same axis the blueprint keys raises on, see cfr/keys.py + the M-B note in
lbr.py). Both the live bot (bot_strategy.py) and the LBR victim model (lbr.py)
translate through here so the deployed bot and the exploitability measurement
never drift.
"""

# Postflop multipliers, for reference / the convenience default grid. The
# *actual* grid used to translate is built per node by the caller, because
# preflop sizes are absolute ladders (open 3/5/7 BB), not pot fractions — so a
# node's grid must come from that node's real legal sizes. A grid is a list of
# (char, frac) sorted ascending by frac, where frac is the size on the same axis
# as eff_fraction() (bet / pot-after-call).
POSTFLOP_GRID = [('s', 0.33), ('m', 0.66), ('l', 1.0)]


def eff_fraction(total_add, to_call, pot):
    """Bet size as a fraction of the pot-after-call (the blueprint's raise axis).

    total_add : chips the actor ADDS (the increment, incl. any call portion).
    to_call   : chips needed to call before this action (0 for a bet).
    pot       : pot BEFORE this action.
    """
    if to_call > 0:                                  # a raise
        pot_after_call = pot + to_call
        return (total_add - to_call) / pot_after_call if pot_after_call > 0 else 0.0
    return total_add / pot if pot > 0 else 0.0       # a bet


def pseudo_harmonic_weight(x, a, b):
    """Weight placed on the LOWER size `a` for an actual fraction x in [a, b]
    (Ganzfried-Sandholm). The remainder 1-w goes on the higher size `b`."""
    if b <= a:
        return 1.0
    x = min(max(x, a), b)
    return ((b - x) * (1.0 + a)) / ((b - a) * (1.0 + x))


def translate_bet(eff_frac, grid):
    """Map an off-grid bet fraction onto the node's trained grid.

    grid : sorted [(char, frac), ...] (ascending) of the sizes available at this
           node, e.g. [('s',0.33),('m',0.66),('l',1.0),('a',4.2)].
    Returns [(char, weight), ...] (1 or 2 entries). A fraction on a grid point
    returns that single char; one between two returns both pseudo-harmonically
    weighted; one at/below the smallest or at/above the largest clamps.
    """
    if not grid:
        return []
    chars = [c for c, _ in grid]
    fracs = [f for _, f in grid]

    if eff_frac <= fracs[0]:
        return [(chars[0], 1.0)]
    if eff_frac >= fracs[-1]:
        return [(chars[-1], 1.0)]
    for i in range(len(fracs) - 1):
        a, b = fracs[i], fracs[i + 1]
        if a <= eff_frac <= b:
            wa = pseudo_harmonic_weight(eff_frac, a, b)
            if wa >= 1.0 - 1e-9:
                return [(chars[i], 1.0)]
            if wa <= 1e-9:
                return [(chars[i + 1], 1.0)]
            return [(chars[i], wa), (chars[i + 1], 1.0 - wa)]
    return [(chars[-1], 1.0)]


def nearest_char(eff_frac, grid):
    """The single grid char closest to `eff_frac` — used where a betting line
    must be summarised by one pattern character (the stored bet_pattern, the
    range-tracker's observe). The principled blend (translate_bet) is reserved
    for the actual decision."""
    if not grid:
        return 'x'
    return min(grid, key=lambda cf: abs(cf[1] - eff_frac))[0]


def blend(translated, strat_for_char):
    """Blend per-bracket strategies. `translated` is translate_bet()'s output;
    `strat_for_char(char) -> {action: prob}` looks up the blueprint response for
    a bracketing size. Returns the weight-mixed, renormalised {action: prob}."""
    out = {}
    for char, w in translated:
        if w <= 0.0:
            continue
        for a, p in strat_for_char(char).items():
            out[a] = out.get(a, 0.0) + w * p
    s = sum(out.values())
    return {a: p / s for a, p in out.items()} if s > 1e-12 else out
