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

# CFR action name -> single betting-pattern character.
ACTION_CHARS = {
    'check': 'k', 'call': 'c', 'fold': 'f',
    'bet_small': 's', 'bet_medium': 'm', 'bet_large': 'l',
    'raise_small': 's', 'raise_medium': 'm', 'raise_large': 'l',
    'allin': 'a',
}


def action_char(action):
    """Map a CFR action name to its single betting-pattern character."""
    return ACTION_CHARS.get(action, 'x')


def make_info_set_key(street, position, preflop_bucket, postflop_strength, bet_pattern):
    """
    Build the canonical info-set key.

    street            : 0=preflop, 1=flop, 2=turn, 3=river
    position          : 'ip' (button/SB) or 'oop' (BB)
    preflop_bucket    : the acting player's preflop bucket (e.g. 'pf_9')
    postflop_strength : the acting player's postflop strength bucket for this
                        street (ignored preflop, may be None)
    bet_pattern       : current-street betting pattern (e.g. 'm', 'km')

    Preflop : {preflop_bucket}_{position}_{bet_pattern}
    Postflop: {preflop_bucket}_{postflop_strength}_{position}_{street}_{bet_pattern}
    """
    if street == 0:
        return f"{preflop_bucket}_{position}_{bet_pattern}"
    return (f"{preflop_bucket}_{postflop_strength}_{position}_"
            f"{STREET_NAMES[street]}_{bet_pattern}")
