# backend/bot/src/cfr/strategy_shape.py
"""
Strategy-shape sanity probe -- a cheap, per-info-set health check that catches a
COLLAPSED blueprint on the first checkpoint, instead of waiting for a human to
eyeball the bot-debug overlay (the way BUG-014's "open xlarge with 100% of hands,
never fold" went unnoticed for a week).

WHY this exists. Every aggregate metric is blind to a *balanced-but-degenerate*
strategy: EV(cum/round) reads the current iterate; EV(served) only checks seat
balance; LBR under-measures a balanced wide preflop open; AIVAT is for head-to-head
variance reduction. None of them scream when the strategy collapses onto one action.
Only inspecting the actual decisions does -- so this probe inspects the actual
preflop decisions and emits a verdict (OK / WARN / COLLAPSE).

It is pure dict math over the stored AVERAGE strategy (no numpy, no engine, no DB
dependency) so it can run from the trainer's in-memory info_sets at every checkpoint
AND from a DB via scripts/check_strategy_shape.py. Pass a `strat_fn(key) -> {action:
prob} | None`.

The collapse signatures it detects (the BUG-014 fingerprints):
  - weak preflop hands that NEVER fold the button AND open one size ~always
    (open node `pf_<weak>_ip_`); and
  - weak hands that NEVER fold facing a big open and re-raise one size ~always
    (BB-vs-open node `pf_<weak>_oop_x`).
Both are the "always-aggress, never-fold" pathology that a healthy blueprint never
shows (a weak hand must retain meaningful fold mass).
"""

OK, WARN, COLLAPSE = 'OK', 'WARN', 'COLLAPSE'

# Buckets treated as "weak" (should fold a lot) and "strong" (should rarely fold).
_WEAK = (0, 1, 2, 3)


def _norm(d):
    """A {action: mass} dict -> {action: prob}, or None if empty/None."""
    if not d:
        return None
    total = sum(v for v in d.values() if v > 0)
    if total <= 0:
        return None
    return {a: v / total for a, v in d.items() if v > 0}


def _fold_and_top_bet(dist, bet_prefixes):
    """(fold prob, largest single bet/raise-size share) for a normalized dist."""
    fold = dist.get('fold', 0.0)
    sizes = [p for a, p in dist.items() if a.startswith(bet_prefixes)]
    return fold, (max(sizes) if sizes else 0.0)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def strategy_shape_report(strat_fn, num_preflop_buckets=30):
    """Compute the preflop strategy-shape health report.

    strat_fn(key) -> {action: prob/mass} | None  (mass is renormalized here).
    Returns a dict with the metrics and a 'verdict' (OK / WARN / COLLAPSE), plus
    'reasons' (list of strings) when not OK. Missing keys are skipped, so this works
    on an early/partial run; 'n_open'/'n_bbx' report how many probe keys were found.
    """
    strong = num_preflop_buckets - 1

    # --- Open node: pf_<n>_ip_ (SB first-in, empty pattern) ---
    open_fold, open_top = {}, {}
    for n in range(num_preflop_buckets):
        d = _norm(strat_fn(f'pf_{n}_ip_'))
        if d is None:
            continue
        f, t = _fold_and_top_bet(d, ('bet_',))
        open_fold[n], open_top[n] = f, t

    weak_open_fold = _mean([open_fold.get(n) for n in _WEAK])
    weak_open_top = _mean([open_top.get(n) for n in _WEAK])
    pf0_fold = open_fold.get(0)
    strong_fold = open_fold.get(strong)
    gradient = (pf0_fold - strong_fold) if (pf0_fold is not None and strong_fold is not None) else None

    # --- BB-vs-5BB(xlarge) open node: pf_<n>_oop_x (the second collapse node) ---
    bbx_fold, bbx_top = {}, {}
    for n in range(num_preflop_buckets):
        d = _norm(strat_fn(f'pf_{n}_oop_x'))
        if d is None:
            continue
        f, t = _fold_and_top_bet(d, ('bet_', 'raise_'))
        bbx_fold[n], bbx_top[n] = f, t
    weak_bbx_fold = _mean([bbx_fold.get(n) for n in _WEAK])
    weak_bbx_top = _mean([bbx_top.get(n) for n in _WEAK])

    # --- Verdict ---
    reasons = []

    def collapsed(fold, top, label):
        if fold is not None and top is not None and fold < 0.05 and top > 0.75:
            reasons.append(f"{label}: weak hands fold {fold:.0%} but play one size {top:.0%} "
                           f"(collapse signature)")
            return True
        return False

    is_collapse = False
    is_collapse |= collapsed(weak_open_fold, weak_open_top, "open")
    is_collapse |= collapsed(weak_bbx_fold, weak_bbx_top, "BB-vs-5BB")
    if gradient is not None and gradient < 0.05 and (weak_open_fold or 1) < 0.10:
        reasons.append(f"no fold strength-gradient (pf_0 {pf0_fold:.0%} vs pf_{strong} "
                       f"{strong_fold:.0%})")
        is_collapse = True

    verdict = OK
    if is_collapse:
        verdict = COLLAPSE
    elif (weak_open_fold is not None and weak_open_top is not None
          and weak_open_fold < 0.20 and weak_open_top > 0.60):
        verdict = WARN
        reasons.append(f"weak-open fold {weak_open_fold:.0%} low / top-size "
                       f"{weak_open_top:.0%} high -- watch for collapse")

    return {
        'verdict': verdict,
        'reasons': reasons,
        'weak_open_fold': weak_open_fold,
        'weak_open_top_size': weak_open_top,
        'gradient_pf0_minus_strong': gradient,
        'pf0_open_fold': pf0_fold,
        'strong_open_fold': strong_fold,
        'weak_bbx_fold': weak_bbx_fold,
        'weak_bbx_top': weak_bbx_top,
        'n_open': len(open_fold),
        'n_bbx': len(bbx_fold),
        'open_fold_by_bucket': open_fold,
        'bbx_fold_by_bucket': bbx_fold,
    }


def _pct(x):
    return f"{x:.0%}" if x is not None else "n/a"


def format_shape_line(rep):
    """One-line checkpoint summary (mirrors the EV(served) line). ASCII only -- this
    prints to the training console, which on Windows is cp1252 and cannot encode
    emoji (a non-ASCII flag would raise UnicodeEncodeError and crash the checkpoint)."""
    flag = {OK: '', WARN: '!! ', COLLAPSE: '!!!! '}[rep['verdict']]
    return (f"  shape: {flag}{rep['verdict']} | weak-open fold {_pct(rep['weak_open_fold'])} "
            f"(top-size {_pct(rep['weak_open_top_size'])}) | grad "
            f"{_pct(rep['gradient_pf0_minus_strong'])} | BB-vs-5BB fold "
            f"{_pct(rep['weak_bbx_fold'])}"
            + (f"  <- {rep['reasons'][0]}" if rep['reasons'] else ""))
