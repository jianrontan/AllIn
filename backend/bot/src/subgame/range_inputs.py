# backend/bot/src/subgame/range_inputs.py
"""
Range inputs for the river subgame solver (Phase-4, step 4).

Turns the live game state into the two reach vectors the CFR+ solver consumes,
on the kernel's H-hand board basis (build_board_arrays), and reads the bot's
action back out after the solve:

  * project_tracker(tracker, ba)  -- project a RangeTracker belief (villain's, or
    the bot's own blueprint-reach tracker) onto the H-hand board basis. Hands not
    present in the tracker (e.g. using its removed hole cards) land at 0.
  * blend_villain(tracked, confidence, beta)  -- the confidence-weighted widening:
        villain_reach = c * tracked + (1 - c) * temper(tracked, beta)
    c = tracker confidence (how much to trust the sharp read), temper() = the
    widening target. beta is the flattening knob: beta=1 -> untouched belief,
    beta=0 (DEFAULT) -> uniform over the line-consistent hands (maximum
    flattening). Only the VILLAIN range is blended; the hero (bot) range is its
    blueprint reach as-is (no uncertainty about how the bot itself plays).
  * read_action_strategy(cfr, node, hand, ba)  -- the bot's solved action
    distribution for its ACTUAL hand at a decision node.

The hero range is built the SAME way as the villain range -- a RangeTracker fed
the BOT's actions under the blueprint (hero_hole=() so it spans all hands) -- so
no separate replay code is needed; the caller (GameSession wiring, step 6)
constructs that tracker and passes it to project_tracker.
"""
import numpy as np

# Default flattening exponent for the widening target. 0.0 == maximum flattening
# (uniform over the line-consistent hands). Exposed so a future testing framework
# can tune it (and potentially vary it with confidence). See blend_villain.
DEFAULT_TEMPER_BETA = 0.0


def hand_index_map(ba):
    """frozenset(hand) -> row index, for the ba H-hand basis."""
    return {frozenset(h): i for i, h in enumerate(ba['hands'])}


def project_tracker(tracker, ba, idx=None, uniform=False):
    """Project a RangeTracker's weight vector onto the ba H-hand board basis.

    A tracker hand maps to its row by card identity (order-independent). Hands the
    tracker doesn't carry -- those using its removed hole cards, or board-colliding
    ones it has already zeroed -- get weight 0, which is correct: the opponent
    cannot hold the bot's cards or a board card.

    uniform=True returns a PRESENCE mask (1.0 for every card-possible tracked hand,
    ignoring the belief weights) -- i.e. the true board+hero card-removal UNIFORM range.
    This is what the safe-gadget's robust ('blueprint') anchor and the auto self-check
    must measure exploitability over: a belief-weighted (or >0-support) range would drop
    a hand that observe() drove to a hard zero, weakening the "<= blueprint vs ANY villain
    hand" guarantee. (Any over-inclusion is harmless -- the showdown kernel removes
    hero-card collisions per matchup anyway.)
    """
    idx = idx if idx is not None else hand_index_map(ba)
    out = np.zeros(ba['H'])
    for h, w in zip(tracker.hands, tracker.w):
        if not uniform and w <= 0.0:
            continue
        row = idx.get(frozenset(h))
        if row is not None:
            out[row] = 1.0 if uniform else float(w)
    return out


def temper(reach, beta):
    """Widening target: reach^beta over the SUPPORT (hands with reach>0),
    renormalised to sum to 1. beta=0 -> uniform over the support (max flattening);
    beta=1 -> the (normalised) input reach. Hands at 0 stay at 0 (so card-removal
    and line-impossible hands are preserved; guards the 0**0 trap)."""
    out = np.zeros_like(reach, dtype=float)
    pos = reach > 0.0
    if beta == 0.0:
        out[pos] = 1.0
    else:
        out[pos] = reach[pos] ** beta
    s = out.sum()
    return out / s if s > 0.0 else out


def blend_villain(tracked, confidence, beta=DEFAULT_TEMPER_BETA):
    """Confidence-weighted villain range (sums to 1):
        c * normalize(tracked) + (1 - c) * temper(tracked, beta).
    c=1 -> trust the sharp belief; c=0 -> the widening target. Returns the tracked
    range unchanged (just normalised) if it has no positive mass."""
    tracked = np.asarray(tracked, dtype=float)
    s = tracked.sum()
    if s <= 0.0:
        return tracked
    t = tracked / s
    prior = temper(tracked, beta)
    c = float(np.clip(confidence, 0.0, 1.0))
    return c * t + (1.0 - c) * prior


def hand_row(ba, hand, idx=None):
    """Row index of `hand` (two SuitRank cards) in the ba basis, or None."""
    idx = idx if idx is not None else hand_index_map(ba)
    return idx.get(frozenset(hand))


def read_action_strategy(cfr, node, hand, ba, idx=None):
    """The bot's solved {action: probability} for its ACTUAL `hand` at `node`
    (a decision node in the solved tree)."""
    row = hand_row(ba, hand, idx)
    if row is None:
        raise ValueError(f"hand {hand} not on the board basis (uses a board card?)")
    avg = cfr.average_strategy(node.node_id)   # [H, A]
    return {a: float(p) for a, p in zip(node.actions, avg[row])}
