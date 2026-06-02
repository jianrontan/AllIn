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

# CAPPED postflop menu (Fix #4, the proposal/response redesign): adds a 2.0x tier
# ('overbet2', char '2') AND is meant to be paired with voluntary_allin=False on the
# engine, so the bot's PROPOSAL menu tops out at 2x pot and all-in only EMERGES when
# a sized tier clamps to the stack (low SPR) -- it is no longer a free-standing
# voluntary action at every node. This is the over-jam fix (Measurement A: the
# over-jamming is the menu offering a degenerate all-in, not SPR-blind keys). It is
# NOT the default: it is selected per-PokerGame so the control arm stays identical
# for the C measurement. Larger overbets (>2x) stay the subgame solver's job.
# Changing it is an abstraction change -> retrain (a blueprint trained under it is
# incompatible with one trained under POSTFLOP_BET_MULT).
POSTFLOP_BET_MULT_CAPPED = {'small': 0.33, 'medium': 0.66, 'large': 1.0,
                            'overbet': 1.5, 'overbet2': 2.0}

# CAPPED-NO-2.0x menu: the capped arm WITHOUT the 2.0x tier (tops out at 1.5x),
# still paired with voluntary_allin=False. This is the clean one-variable test arm
# for "is the 2.0x tier worth it?" -- it differs from POSTFLOP_BET_MULT_CAPPED ONLY
# by the overbet2 entry, so an LBR/BR comparison of capped vs capped_no2 (on
# CONVERGED arms) isolates the 2.0x tier's value. NOTE: vs this menu, the SPR 1.5-2.0
# band loses both the 2.0x bet AND (no voluntary jam) any all-in -- the documented
# jam gap the 2.0x tier was added to close; that's exactly what the test measures.
POSTFLOP_BET_MULT_CAPPED_NO2 = {'small': 0.33, 'medium': 0.66, 'large': 1.0,
                                'overbet': 1.5}

# The 3-bet/4-bet core sizes (preflop pot-relative). Open + postflop have their own
# 4-size sets above; do not use this tuple for those.
SIZES = ('small', 'medium', 'large')

# size-name -> betting-pattern char. Canonical here (the size SOT) so the API's
# grid builder and the LBR victim model share ONE mapping instead of each keeping a
# private copy. 'xlarge' is the open-only 4th size (char 'x'); 'overbet' is postflop
# (char 'o'). Mirrors the relevant entries of cfr/keys.ACTION_CHARS.
SIZE_CHAR = {'small': 's', 'medium': 'm', 'large': 'l', 'xlarge': 'x', 'overbet': 'o',
             'overbet2': '2'}


def postflop_menu_for(menu_mode):
    """The postflop bet/raise size dict for a menu_mode
    ('control' | 'capped' | 'capped_no2'). One place to resolve the arm so every
    consumer (engine, BR, LBR, the API grid) selects the same dict instead of
    hard-coding POSTFLOP_BET_MULT."""
    if menu_mode == 'capped':
        return POSTFLOP_BET_MULT_CAPPED
    if menu_mode == 'capped_no2':
        return POSTFLOP_BET_MULT_CAPPED_NO2
    return POSTFLOP_BET_MULT


# menu_modes that drop the free-standing voluntary all-in (Fix #4 family). Both
# capped variants do; 'control' keeps the voluntary jam. One predicate so the
# engine builders (trainer, GameSession, BR, LBR) agree on which arms are
# voluntary_allin=False.
def is_capped_mode(menu_mode):
    return menu_mode in ('capped', 'capped_no2')


def db_menu_mode(blueprint_db):
    """Read the action-abstraction arm a blueprint was trained under from its DB
    metadata. A pre-stamp DB (trained before the menu_mode flag existed) is
    'control' -- the only arm that existed then. So an eval harness auto-matches
    the blueprint without the caller having to know which arm it is."""
    if blueprint_db is None:
        return 'control'
    try:
        return blueprint_db.get_metadata('menu_mode', 'control') or 'control'
    except Exception:
        return 'control'


def preflop_open_chips():
    """Preflop open raise-TO totals in CHIPS, e.g. {'small': 4, ...}."""
    return {k: v * BIG_BLIND for k, v in PREFLOP_OPEN_BB.items()}
