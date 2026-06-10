# backend/bot/tests/h1_kuhn_check.py
"""
H1 verification: does the trainer's average-strategy SITING bias the served strategy?

A review agent flagged that blueprint_trainer accumulates the average strategy ONLY at
OPPONENT nodes (unweighted, once per visit) during an external-sampling traversal -- and
claimed this weights by the opponent's reach rather than the traverser's own reach,
biasing the average. The trainer's docstring asserts the opposite (the standard ES-MCCFR
justification: the opponent samples, so reaching its node already happens ~prop. to the
player's own reach -> add unweighted).

This settles it on Kuhn poker, whose Nash equilibrium is known in closed form. We replicate
the trainer's EXACT scheme (external sampling, alternating traverser t%2, CFR+ regret-match
at the traverser's nodes, unweighted strategy-sum at the opponent's nodes) on Kuhn and check
the converged AVERAGE strategy against alpha-INDEPENDENT equilibrium invariants:

  * P0 never bets Q                          (bet(Q | first to act) = 0)
  * P0 bets K exactly 3x as often as J       (bet(K) = 3 * bet(J))
  * P1 facing a bet calls Q with prob 1/3    (call(Q | bet) = 1/3)
  * P1 facing a check bets J with prob 1/3   (bet(J | check) = 1/3)
  * game value to P0 = -1/18                 (computed under the average strategy)

If the trainer's siting matched these, H1 is a false positive. If the average is
systematically off, H1 is a real bias.

Run: python tests/h1_kuhn_check.py [iterations]
"""
import random
import sys

import numpy as np

# Cards: 0=J, 1=Q, 2=K. P0 acts first. Both ante 1 (pot starts 2). Actions: 'p'=pass
# (check or fold), 'b'=bet (bet or call). Decision histories and who acts:
#   ''  -> P0 ;  'p' -> P1 (faced check) ;  'b' -> P1 (faced bet) ;  'pb' -> P0 (faced bet).
_TERMINALS = {'pp', 'bp', 'bb', 'pbp', 'pbb'}
_ACTIONS = ['p', 'b']

regret = {}        # infoset key -> np.array[2] cumulative regret (CFR+ floored on write)
strat_sum = {}     # infoset key -> np.array[2] cumulative strategy (the served average)


def _strategy(key):
    """Current strategy via CFR+ regret matching (mirrors InformationSet.get_strategy)."""
    r = np.maximum(regret.get(key, np.zeros(2)), 0.0)
    s = r.sum()
    return r / s if s > 0 else np.array([0.5, 0.5])


def _term_util_p0(h, cards):
    """P0's utility at a terminal history."""
    p0c, p1c = cards
    if h == 'bp':              # P0 bet, P1 folded -> P0 wins P1's ante
        return 1.0
    if h == 'pbp':             # P0 checked, P1 bet, P0 folded -> P0 loses its ante
        return -1.0
    win = 1.0 if p0c > p1c else -1.0
    amt = 2.0 if h in ('bb', 'pbb') else 1.0   # showdown stakes: bet+call=2, else ante=1
    return win * amt


def _cfr(cards, h, updating):
    """One external-sampling traversal step, P0-perspective return -- EXACTLY the trainer's
    structure: explore-all + CFR+ regret at the updating player's node; unweighted
    strategy-sum + SAMPLE one at the opponent's node."""
    if h in _TERMINALS:
        return _term_util_p0(h, cards)
    player = len(h) % 2
    key = f"{cards[player]}{h}"
    strategy = _strategy(key)
    if player == updating:
        util = np.array([_cfr(cards, h + a, updating) for a in _ACTIONS])   # P0 perspective
        sign = 1.0 if player == 0 else -1.0
        own = sign * util
        node_val = float((strategy * own).sum())
        prior = regret.get(key, np.zeros(2))
        regret[key] = np.maximum(prior + (own - node_val), 0.0)            # CFR+ write-floor
        return sign * node_val
    # opponent node: accumulate the OPPONENT's strategy unweighted, once per visit, sample one
    strat_sum[key] = strat_sum.get(key, np.zeros(2)) + strategy
    a = random.choices(_ACTIONS, weights=strategy)[0]
    return _cfr(cards, h + a, updating)


def _avg(key):
    s = strat_sum.get(key, np.zeros(2))
    t = s.sum()
    return s / t if t > 0 else np.array([0.5, 0.5])


def _tree_value_p0(cards, h):
    """Expected P0 value at history h under the AVERAGE strategy (full-tree expectation)."""
    if h in _TERMINALS:
        return _term_util_p0(h, cards)
    player = len(h) % 2
    st = _avg(f"{cards[player]}{h}")
    return sum(st[i] * _tree_value_p0(cards, h + a) for i, a in enumerate(_ACTIONS))


def run(iters, seed=0):
    random.seed(seed)
    deck = [0, 1, 2]
    for t in range(iters):
        random.shuffle(deck)
        _cfr((deck[0], deck[1]), '', t % 2)

    bet = lambda key: float(_avg(key)[1])     # P('b') = bet/call
    # equilibrium invariants (alpha-independent)
    checks = [
        ("P0 bet(Q)=0",            bet('1'),      0.0,   0.03),
        ("P0 bet(K)=3*bet(J)",     bet('2'),      3 * bet('0'), 0.04),
        ("P1 call(Q|bet)=1/3",     bet('1b'),     1 / 3, 0.04),
        ("P1 bet(J|check)=1/3",    bet('0p'),     1 / 3, 0.04),
    ]
    deals = [(a, b) for a in range(3) for b in range(3) if a != b]
    value = sum(_tree_value_p0(c, '') for c in deals) / len(deals)
    checks.append(("P0 game value=-1/18", value, -1 / 18, 0.01))

    print(f"\nKuhn MCCFR (trainer's opponent-node averaging), {iters:,} iters\n")
    print("  P0 first-to-act bets:  J={:.3f}  Q={:.3f}  K={:.3f}"
          .format(bet('0'), bet('1'), bet('2')))
    print("  P1 facing a check bets: J={:.3f}  Q={:.3f}  K={:.3f}"
          .format(bet('0p'), bet('1p'), bet('2p')))
    print("  P1 facing a bet calls:  J={:.3f}  Q={:.3f}  K={:.3f}"
          .format(bet('0b'), bet('1b'), bet('2b')))
    print("  P0 facing check-bet calls: J={:.3f} Q={:.3f} K={:.3f}\n"
          .format(bet('0pb'), bet('1pb'), bet('2pb')))
    ok = True
    for name, got, want, tol in checks:
        good = abs(got - want) <= tol
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:24s} got={got:+.4f} want={want:+.4f} (tol {tol})")
    print(f"\n  VERDICT: {'average matches equilibrium -> H1 FALSE POSITIVE (siting is correct)' if ok else 'average is OFF -> H1 is a REAL bias'}")
    return ok


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    sys.exit(0 if run(n) else 1)
