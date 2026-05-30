# backend/bot/tests/test_lbr_range.py
"""
Validate BotRange (the exploiter's belief over the bot's hand) with an INJECTED
fake blueprint, so the Bayesian update is deterministic and checkable:

  fake bot: preflop buckets pf_10+ always raise_large; everything else folds.
  (Fine scheme is 30 buckets pf_0..pf_29; 10 is just an arbitrary in-range threshold.)

After observing the bot raise_large preflop, every surviving hand in the belief
must be a pf_10+ hand (weak hands had P(raise_large)=0 -> zeroed). We also check
card removal and that strategy rows are proper distributions.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.lbr import LBREvaluator, BotRange
from src.abstractions.card_abstractions import CardAbstraction


class FakeDB:
    """pf_10+ -> raise_large; else fold. Postflop: strength>=4 bet, else check."""
    def get_average_strategy(self, key):
        parts = key.split('_')
        pf_num = int(parts[1])
        if len(parts) == 4:   # preflop: pf_{num}_{pos}_{pattern}
            return {'raise_large': 1.0} if pf_num >= 10 else {'fold': 1.0}
        strength = int(parts[2])   # postflop: pf_{num}_{strength}_{pos}_{street}_{pat}
        return {'bet_large': 1.0} if strength >= 4 else {'check': 1.0}


def main():
    ev = LBREvaluator(blueprint_db=FakeDB(), seed=0)
    cards = CardAbstraction()
    lbr_hand = ('SA', 'SK')

    rng = BotRange(lbr_hand, cards)
    assert all(lbr_hand[0] not in h and lbr_hand[1] not in h for h in rng.hands)
    n0 = len(rng.hands)
    print(f"initial bot hands: {n0} (expect C(50,2)=1225)")
    assert n0 == 1225

    legal = ['fold', 'call', 'raise_large']
    # Strategy rows must be valid distributions over legal for active hands.
    mat = rng.action_probs(ev.restricted_probs, 0, 'ip', '', legal, [])
    rowsums = mat.sum(axis=1)
    assert np.allclose(rowsums, 1.0), f"rows not normalised: {rowsums.min()}..{rowsums.max()}"
    print("action_probs rows sum to 1.0 OK")

    # Observe the bot raise large preflop -> belief concentrates on pf_10+.
    rng.observe(ev.restricted_probs, 'raise_large', 0, 'ip', '', legal, [])
    survivors = [(h, w) for h, w in zip(rng.hands, rng.w) if w > 0]
    assert survivors, "all mass removed -- update bug"
    bad = [h for h, w in survivors
           if int(cards.get_bucket(list(h), None).split('_')[1]) < 10]
    print(f"after raise_large: {len(survivors)} hands survive, "
          f"{len(bad)} of them are pf<10 (expect 0)")
    assert not bad, f"weak hands wrongly survived: {bad[:5]}"
    assert abs(rng.w.sum() - 1.0) < 1e-9, "weights should renormalise to 1"

    # Reveal a flop -> any surviving combo colliding with the board is zeroed.
    flop = ['HA', 'D4', 'C2']   # note HA collides with some surviving hands
    rng.reveal(flop)
    collide = [h for h, w in zip(rng.hands, rng.w)
               if w > 0 and (h[0] in set(flop) or h[1] in set(flop))]
    print(f"after reveal {flop}: {len(collide)} colliding hands remain (expect 0)")
    assert not collide

    # Postflop strategy rows still valid distributions for active hands.
    legal2 = ['check', 'bet_large']
    mat2 = rng.action_probs(ev.restricted_probs, 1, 'oop', '', legal2, flop)
    active = rng.w > 0
    assert np.allclose(mat2[active].sum(axis=1), 1.0)
    print("postflop action_probs rows sum to 1.0 OK")

    print("\nPASS: BotRange card-removal + Bayesian update behave correctly.")


# pytest entry point (the assertions live in main()).
def test_lbr_range():
    main()


if __name__ == '__main__':
    main()
