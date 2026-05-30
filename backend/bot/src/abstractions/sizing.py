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
    pot-after-call). ONE overbet tier (1.5x = 'overbet') is included; larger
    overbets (2x+) are left to the subgame solver's own menu.

Voluntary all-in: every betting node also offers 'allin' as a voluntary
aggressive action (not just when a sized bet exhausts the stack). All-in has no
multiplier here — its size is the remaining stack, computed by the engine.

NOTE on size sets: the three nodes no longer share one size list —
  * preflop OPEN has FOUR sizes (small/medium/large/xlarge), 'xlarge' is open-only;
  * preflop 3-bet/4-bet has THREE (small/medium/large) — SIZES below;
  * postflop has FOUR (small/medium/large/overbet).
So iterate the relevant dict's keys, not SIZES, outside the 3-bet/4-bet path.
"""

BIG_BLIND = 2

# Preflop open: raise-TO totals in big blinds. 'xlarge' (5 BB, char 'x') is the
# 4th, open-only anchor so big human opens translate against a trained bracket.
PREFLOP_OPEN_BB = {'small': 2.0, 'medium': 2.5, 'large': 3.5, 'xlarge': 5.0}

# Preflop 3-bet / 4-bet+: fraction of pot-after-call. (No xlarge: 3-bet/4-bet stay
# 3 sizes + voluntary all-in; deeper raises are handled by subgame solving.)
PREFLOP_RAISE_MULT = {'small': 0.66, 'medium': 1.0, 'large': 1.5}

# Postflop bet/raise: fraction of pot (raise = fraction of pot-after-call).
# 'overbet' (1.5x pot, char 'o') is the one blueprint overbet tier.
POSTFLOP_BET_MULT = {'small': 0.33, 'medium': 0.66, 'large': 1.0, 'overbet': 1.5}

# The 3-bet/4-bet core sizes (preflop pot-relative). Open + postflop have their own
# 4-size sets above; do not use this tuple for those.
SIZES = ('small', 'medium', 'large')

# size-name -> betting-pattern char. Canonical here (the size SOT) so the API's
# grid builder and the LBR victim model share ONE mapping instead of each keeping a
# private copy. 'xlarge' is the open-only 4th size (char 'x'); 'overbet' is postflop
# (char 'o'). Mirrors the relevant entries of cfr/keys.ACTION_CHARS.
SIZE_CHAR = {'small': 's', 'medium': 'm', 'large': 'l', 'xlarge': 'x', 'overbet': 'o'}


def preflop_open_chips():
    """Preflop open raise-TO totals in CHIPS, e.g. {'small': 4, ...}."""
    return {k: v * BIG_BLIND for k, v in PREFLOP_OPEN_BB.items()}
