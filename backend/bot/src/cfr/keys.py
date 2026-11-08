# backend/bot/src/cfr/keys.py
"""
Single source of truth for information-set key construction and the
action -> betting-pattern-character mapping.

Both the trainer (cfr/blueprint_trainer.py) and any consumer that needs to
look a situation up in the blueprint (the evaluation harness, a future subgame
solver) MUST build keys through make_info_set_key so the two can never drift.
If the key format ever changes, it changes here and everywhere stays in sync.
"""

STREET_NAMES = ['preflop', 'flop', 'turn', 'river']

# Community cards visible per street, indexed by street index (0=preflop..3=river).
# Single source of truth: the game core, the LBR/eval harness, and the API all map
# street<->board-count through this (previously duplicated in three places).
BOARD_COUNT = [0, 3, 4, 5]
_BOARD_COUNT_TO_STREET = {n: i for i, n in enumerate(BOARD_COUNT)}


# --- Key formats -----------------------------------------------------------------
# 'v2'  : the served v2 format (no SPR on the key).
# 'spr5': v2 + a stack-to-pot bucket on FLOP and TURN keys (Phase 6, see
#         docs/PHASE6_ABSTRACTION_PLAN.md). The bucket is the SPR at the START of
#         the street, aligned to preflop pot types: >40 limped, 12-40 single-raised,
#         5-12 3-bet, 2-5 4-bet, <2 jam zone. Preflop and river keys are unchanged.
# A blueprint DB stamps its format in training_metadata['key_format'] (absent =
# 'v2'); every reader must build keys in the format of the blueprint it opened.
KEY_FORMATS = ('v2', 'spr5')
SPR_EDGES = (2.0, 5.0, 12.0, 40.0)        # bucket b = number of edges <= SPR
SPR_STREETS = (1, 2)                       # flop, turn
_KEY_STARTING_STACK = 200                  # == poker_game.STARTING_STACK (test-guarded)


def spr_bucket(key_format, street, street_pot):
    """The SPR bucket (0..4, 0 = jam zone) an info-set key carries, or None when the
    format/street has none (v2, preflop, river).

    `street_pot` is the pot (chips) when the street BEGAN. Both players start every
    hand with the same stack and a street only closes once contributions match, so
    at a street start both have invested pot/2 and the effective stack is
    STARTING_STACK - pot/2: the SPR is a pure function of the street-start pot,
    and therefore of the public betting history."""
    if key_format == 'v2' or street not in SPR_STREETS:
        if key_format not in KEY_FORMATS:
            raise ValueError(f"unknown key_format {key_format!r}")
        return None
    if key_format != 'spr5':
        raise ValueError(f"unknown key_format {key_format!r}")
    if not (street_pot > 0) or street_pot == float('inf'):
        raise ValueError(f"street_pot must be finite and > 0, got {street_pot!r}")
    spr = (_KEY_STARTING_STACK - street_pot / 2.0) / street_pot
    b = 0
    for e in SPR_EDGES:
        if spr >= e:
            b += 1
    return b


def street_from_board_count(n):
    """Street index (0=preflop .. 3=river) from the number of community cards.
    The inverse of BOARD_COUNT. Raises ValueError for a count that is not a legal
    board size (0, 3, 4, or 5)."""
    try:
        return _BOARD_COUNT_TO_STREET[n]
    except KeyError:
        raise ValueError(f"not a legal board size: {n} (expected 0, 3, 4, or 5)")

# Fine preflop bucket -> coarse preflop class, for the postflop startBucket collapse
# (imperfect recall). Imported lazily inside make_info_set_key to avoid any import
# order coupling (card_abstractions imports nothing from cfr). See
# card_abstractions.FINE_TO_COARSE / _build_fine_to_coarse for the rationale.
_FINE_TO_COARSE = None
# Memo: fine bucket id ('pf_<n>') -> coarse class id ('pf_<m>'). _coarse_class is a
# PURE function of its string arg but was a measured BR hotspot -- profiled at 122M
# calls / ~116s in a 2-sample best-response walk, each redoing split('_') + an int()
# + an f-string. There are only NUM_PREFLOP_BUCKETS (169, lossless) distinct inputs, so a dict
# memo collapses it to ~free. Bit-identical (same output, just cached).
_COARSE_CACHE = {}


def _coarse_class(fine_bucket):
    """Collapse a fine preflop bucket id ('pf_<n>') to its coarse class id
    ('pf_<m>', m in 0..NUM_PREFLOP_COARSE-1) for postflop keys. Idempotent on an
    already-coarse id only if it happens to be a valid fine index; callers must pass
    the FINE bucket (they all do -- card_abstractions.preflop_bucket returns fine).
    Memoized (see _COARSE_CACHE) -- it's a hot path in the best-response walk."""
    cached = _COARSE_CACHE.get(fine_bucket)
    if cached is not None:
        return cached
    global _FINE_TO_COARSE
    if _FINE_TO_COARSE is None:
        from ..abstractions.card_abstractions import FINE_TO_COARSE
        _FINE_TO_COARSE = FINE_TO_COARSE
    n = int(fine_bucket.split('_')[1])
    result = f"pf_{_FINE_TO_COARSE[n]}"
    _COARSE_CACHE[fine_bucket] = result
    return result

# CFR action name -> single betting-pattern character.
# 'x' = 4th preflop OPEN size (5 BB, open-only). 'o' = postflop overbet (1.5x pot,
# bet or raise). '2' = postflop overbet2 (2.0x pot, capped-menu Fix #4 only, bet or
# raise). 'a' = all-in (voluntary in the default menu; emergent-only under the
# capped menu, where a sized tier that clamps to the stack maps here).
ACTION_CHARS = {
    'check': 'k', 'call': 'c', 'fold': 'f',
    'bet_small': 's', 'bet_medium': 'm', 'bet_large': 'l',
    'raise_small': 's', 'raise_medium': 'm', 'raise_large': 'l',
    'bet_xlarge': 'x',
    'bet_overbet': 'o', 'raise_overbet': 'o',
    'bet_overbet2': '2', 'raise_overbet2': '2',
    'allin': 'a',
}


def action_char(action):
    """Map a CFR action name to its single betting-pattern character.

    Raises ValueError on an unmapped action. Every legal grid action MUST be in
    ACTION_CHARS; silently defaulting (this used to return 'x') would now alias the
    real `bet_xlarge` char `'x'` and corrupt info-set keys. Off-grid / custom human
    bets are mapped to a grid char by cfr/translation.py — never routed here."""
    try:
        return ACTION_CHARS[action]
    except KeyError:
        raise ValueError(f"action_char: unmapped action {action!r} (not a grid action)")


def make_info_set_key(street, position, preflop_bucket, postflop_strength, bet_pattern,
                      spr=None):
    """
    Build the canonical info-set key.

    street            : 0=preflop, 1=flop, 2=turn, 3=river
    position          : 'ip' (button/SB) or 'oop' (BB)
    preflop_bucket    : the acting player's FINE preflop bucket (e.g. 'pf_9',
                        from card_abstractions.preflop_bucket). For POSTFLOP keys it
                        is collapsed here to the coarse class (imperfect recall), so
                        callers always pass the fine id and never choose.
    postflop_strength : the acting player's postflop strength bucket for this
                        street (ignored preflop, may be None)
    bet_pattern       : current-street betting pattern (e.g. 'm', 'km')
    spr               : SPR bucket from spr_bucket() (spr5 format, flop/turn), or
                        None for a v2 key. Passing one on preflop/river is an error.

    Preflop : {fine_bucket}_{position}_{bet_pattern}
    Postflop: {coarse_class}_{postflop_strength}_{position}_{street}_{bet_pattern}
    spr5    : {coarse_class}_{postflop_strength}_{position}_{street}_r{spr}_{bet_pattern}
    """
    if street == 0:
        if spr is not None:
            raise ValueError("preflop keys carry no SPR bucket")
        return f"{preflop_bucket}_{position}_{bet_pattern}"
    coarse = _coarse_class(preflop_bucket)
    if spr is None:
        return (f"{coarse}_{postflop_strength}_{position}_"
                f"{STREET_NAMES[street]}_{bet_pattern}")
    if street not in SPR_STREETS:
        raise ValueError(f"street {street} keys carry no SPR bucket")
    if not (isinstance(spr, int) and 0 <= spr <= len(SPR_EDGES)):
        raise ValueError(f"SPR bucket must be an int in 0..{len(SPR_EDGES)}, got {spr!r}")
    return (f"{coarse}_{postflop_strength}_{position}_"
            f"{STREET_NAMES[street]}_r{spr}_{bet_pattern}")
