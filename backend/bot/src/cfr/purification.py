# backend/bot/src/cfr/purification.py
"""
Strategy purification (thresholding) -- an inference-only transform of a stored
average strategy.

A CFR average strategy carries tiny probability mass on dominated actions (pure
sampling/convergence noise). Purification zeroes any action below a probability
THRESHOLD and renormalises, which can REDUCE exploitability in an abstracted game
(Ganzfried & Sandholm, "Strategy Purification"): the noise mass is a small leak a
best-responder picks off, and removing it sharpens the strategy toward the
abstraction's intended play. It also cuts action variance.

Single source of truth so the live bot (game/bot_strategy.py) and the
exploitability scoreboard (evaluation/best_response.py) purify IDENTICALLY -- an
A/B that purifies one but not the other would be meaningless.

THRESHOLD SEMANTICS (a single knob in [0, 1]):
  * 0.0            -> identity (purification OFF; the default everywhere).
  * 0 < t <= max p -> drop every action with prob < t, renormalise the rest
                      (light "thresholding": t=0.01 removes <1% dust).
  * t  >  max p    -> nothing clears the bar, so keep the ARGMAX action(s) only
                      (FULL purification). t=1.0 is therefore always full purify.
Ties at the max are kept together (a genuine 50/50 stays mixed under full purify).
"""
import numpy as np


def purify_probs(probs, threshold):
    """Purify a probability vector (np.array, sums to 1) at `threshold`. Returns a
    renormalised vector; see the module docstring for threshold semantics."""
    probs = np.asarray(probs, dtype=float)
    if threshold <= 0.0 or probs.size == 0:
        return probs
    keep = probs >= threshold
    if not keep.any():                       # threshold above every entry -> argmax only
        keep = probs >= probs.max()
    out = np.where(keep, probs, 0.0)
    s = out.sum()
    return out / s if s > 0 else np.ones_like(probs) / probs.size


def purify_dist(dist, threshold):
    """Purify a {action: prob} dict at `threshold`. Returns a new dict (same keys);
    identity when threshold <= 0 or the dict is empty."""
    if threshold <= 0.0 or not dist:
        return dist
    keys = list(dist.keys())
    p = purify_probs(np.array([dist[k] for k in keys], dtype=float), threshold)
    return {k: float(v) for k, v in zip(keys, p)}
