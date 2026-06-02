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

# Fine preflop bucket -> coarse preflop class, for the postflop startBucket collapse
# (imperfect recall). Imported lazily inside make_info_set_key to avoid any import
# order coupling (card_abstractions imports nothing from cfr). See
# card_abstractions.FINE_TO_COARSE / _build_fine_to_coarse for the rationale.
_FINE_TO_COARSE = None
# Memo: fine bucket id ('pf_<n>') -> coarse class id ('pf_<m>'). _coarse_class is a
# PURE function of its string arg but was a measured BR hotspot -- profiled at 122M
# calls / ~116s in a 2-sample best-response walk, each redoing split('_') + an int()
# + an f-string. There are only NUM_PREFLOP_BUCKETS (30) distinct inputs, so a dict
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


def make_info_set_key(street, position, preflop_bucket, postflop_strength, bet_pattern):
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

    Preflop : {fine_bucket}_{position}_{bet_pattern}
    Postflop: {coarse_class}_{postflop_strength}_{position}_{street}_{bet_pattern}
    """
    if street == 0:
        return f"{preflop_bucket}_{position}_{bet_pattern}"
    coarse = _coarse_class(preflop_bucket)
    return (f"{coarse}_{postflop_strength}_{position}_"
            f"{STREET_NAMES[street]}_{bet_pattern}")
