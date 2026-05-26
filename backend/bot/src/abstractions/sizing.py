# backend/bot/src/abstractions/sizing.py
"""
Single source of truth for the BETTING-SIZE abstraction (preflop + postflop).

The sizing numbers were previously duplicated across the trainer engine
(`cfr/poker_game.py`), the exploitability harness (`evaluation/lbr.py`), and the
PyPokerEngine path (`abstractions/action_abstractions.py`). That is a drift
hazard — the same class of bug `cfr/keys.py` was created to kill for info-set
keys. Define the sizes ONCE here; every consumer imports them, and
`tests/test_action_abstraction_roundtrip.py` / the sizing consistency test assert
they agree.

IMPORTANT: changing any number here is an ABSTRACTION change — the blueprint must
be retrained (existing blueprints become incompatible, exactly like a bucket
change). It is never a hot-swappable/resumable edit.

Scheme (heads-up, 100 BB, BB = 2 chips):
  * Preflop OPEN (first-in raise): absolute, BB-anchored. The preflop pot is tiny
    and fixed, so pot-relative is the wrong unit; BB is natural. Small opens are
    GTO-optimal in heads-up (position + wide range); bigger human opens are
    handled at inference by action translation, not by the bot's own ladder.
  * Preflop 3-BET and 4-BET+: pot-relative (fraction of pot-after-call). UNIFIED
    into one rule (the old absolute 3-bet ladder collapsed below the min-raise
    versus a large open). Larger multipliers than postflop because preflop raises
    define ranges and deny equity.
  * Postflop bet/raise: fraction of the pot (a raise is a fraction of
    pot-after-call). Overbets (>pot) are intentionally omitted from the blueprint
    grid and left to the Phase-4 subgame solver.
"""

BIG_BLIND = 2

# Preflop open: raise-TO totals in big blinds.
PREFLOP_OPEN_BB = {'small': 2.0, 'medium': 2.5, 'large': 3.5}

# Preflop 3-bet / 4-bet+: fraction of pot-after-call.
PREFLOP_RAISE_MULT = {'small': 0.66, 'medium': 1.0, 'large': 1.5}

# Postflop bet/raise: fraction of pot (raise = fraction of pot-after-call).
POSTFLOP_BET_MULT = {'small': 0.33, 'medium': 0.66, 'large': 1.0}

SIZES = ('small', 'medium', 'large')


def preflop_open_chips():
    """Preflop open raise-TO totals in CHIPS, e.g. {'small': 4, ...}."""
    return {k: v * BIG_BLIND for k, v in PREFLOP_OPEN_BB.items()}
