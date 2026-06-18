# backend/bot/scripts/probe_rare_line_convergence.py
"""
READ-ONLY probe: are the RARE info sets in a blueprint still converging as training
continues, or has the rare tail plateaued the way aggregate BR has?

WHY this exists. Aggregate BR/exploitability is REACH-WEIGHTED toward the best-response
line, so it barely moves when a RARE info set is poorly converged. But a HUMAN wanders
into rare / off-tree lines constantly, and there the bot needs a sane strategy. So the
question "does running more iterations help live play even after BR plateaus?" is really
"are the rare info sets still converging?" -- which BR cannot answer. This probe answers
it directly from the per-info-set clocks + the average strategy itself.

It reports two things across a sequence of snapshots (oldest -> newest):

  (1) VISIT DISTRIBUTION -- percentiles of strategy_visit_count (the gamma clock; in the
      parallel path it counts ROUNDS the key was updated, not iterations, so read it
      RELATIVELY across snapshots, not as an absolute). Split by position (ip/oop) and
      street. Shows how under-visited the tail is and whether coverage is saturated.

  (2) STRATEGY DRIFT between CONSECUTIVE snapshots -- total-variation distance between the
      normalized average strategy of each shared key, averaged within (position x rarity
      tier). This is the DIRECT convergence signal: if the rare/OOP tier is still drifting
      a lot it has not settled (more iterations are still changing it); if the drift is
      small and shrinking pair-over-pair, it has converged.

READ: rare-OOP-tier drift LARGE but SHRINKING across pairs -> rare OOP lines are still
converging (more iters genuinely help human play). Drift already SMALL -> settled, more
iters buy little. Drift large and NOT shrinking -> the rare tail is cycling, not
converging (more iters won't help and may hurt -- see the seat-balance discussion).

Writes NOTHING. Safe against a DB a training run holds open (opens mode=ro).

Run from backend/bot/ (oldest -> newest):
    python scripts/probe_rare_line_convergence.py \
        analysis/blueprints/snapshots/snap_22500000.db \
        analysis/blueprints/snapshots/snap_30000000.db \
        analysis/blueprints/snapshots/snap_37500000.db \
        analysis/blueprints/snapshots/snap_45000000.db
"""
import argparse
import json
import os
import sqlite3


# --- key parsing (matches cfr/keys.py key format) ---------------------------- #
def _street(key):
    if '_river_' in key:
        return 'river'
    if '_turn_' in key:
        return 'turn'
    if '_flop_' in key:
        return 'flop'
    return 'preflop'


def _pos(key):
    if '_oop_' in key:
        return 'oop'
    if '_ip_' in key:
        return 'ip'
    return '?'


def _pctile(sorted_xs, q):
    if not sorted_xs:
        return 0
    return sorted_xs[min(len(sorted_xs) - 1, int(q * len(sorted_xs)))]


def _summ(xs):
    s = sorted(xs)
    return {
        'n': len(s),
        'p5': _pctile(s, 0.05), 'p10': _pctile(s, 0.10), 'p25': _pctile(s, 0.25),
        'p50': _pctile(s, 0.50), 'p75': _pctile(s, 0.75), 'p90': _pctile(s, 0.90),
        'max': s[-1] if s else 0,
    }


def _normalize(cum_strategy):
    """{action: mass} -> {action: prob}, or None if no mass."""
    total = sum(v for v in cum_strategy.values() if v > 0)
    if total <= 1e-12:
        return None
    return {a: v / total for a, v in cum_strategy.items() if v > 0}


def _tv(p, q):
    """Total-variation distance between two action->prob dicts (0..1)."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(a, 0.0) - q.get(a, 0.0)) for a in keys)


def _tier(svc):
    if svc < 10:
        return 'rare(<10)'
    if svc < 100:
        return 'mid(10-100)'
    return 'common(>=100)'


def load(path):
    """Load {key: (svc, normalized_strategy|None)} + total_iterations for one DB."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT value FROM training_metadata WHERE key='total_iterations'").fetchone()
        iters = json.loads(row[0]) if row else None
        rows = conn.execute(
            "SELECT key, strategy_visit_count, cumulative_strategy FROM info_sets").fetchall()
    finally:
        conn.close()
    data = {}
    for key, svc, cum in rows:
        data[key] = (svc, _normalize(json.loads(cum)))
    return iters, data


def per_snapshot(path, iters, data):
    all_svc = [svc for svc, _ in data.values()]
    oop_svc = [svc for k, (svc, _) in data.items() if _pos(k) == 'oop']
    ip_svc = [svc for k, (svc, _) in data.items() if _pos(k) == 'ip']
    by_street = {}
    for k, (svc, _) in data.items():
        by_street.setdefault(_street(k), []).append(svc)
    under = {t: sum(1 for x in all_svc if x < t) for t in (1, 5, 10, 25, 50)}
    under_oop = {t: sum(1 for x in oop_svc if x < t) for t in (1, 5, 10, 25, 50)}

    a, o, i = _summ(all_svc), _summ(oop_svc), _summ(ip_svc)
    print(f"\n=== {os.path.basename(path)} | iters={iters:,} | infosets={a['n']:,} ===")
    print("  strategy_visit_count (rounds touched) percentiles:")
    print(f"    ALL  n={a['n']:>7,}  p5={a['p5']:>5}  p10={a['p10']:>5}  p25={a['p25']:>5}"
          f"  p50={a['p50']:>6}  p90={a['p90']:>6}  max={a['max']:>6}")
    print(f"    OOP  n={o['n']:>7,}  p5={o['p5']:>5}  p10={o['p10']:>5}  p25={o['p25']:>5}"
          f"  p50={o['p50']:>6}  p90={o['p90']:>6}")
    print(f"    IP   n={i['n']:>7,}  p5={i['p5']:>5}  p10={i['p10']:>5}  p25={i['p25']:>5}"
          f"  p50={i['p50']:>6}  p90={i['p90']:>6}")
    print("  under-converged (svc < T):  ALL  "
          + "  ".join(f"<{t}:{under[t]:,}" for t in (1, 5, 10, 25, 50)))
    print("                              OOP  "
          + "  ".join(f"<{t}:{under_oop[t]:,}" for t in (1, 5, 10, 25, 50)))
    print("  by street (n / p10 / p50):  "
          + "  ".join(f"{st}:{_summ(v)['n']:,}/{_summ(v)['p10']}/{_summ(v)['p50']}"
                      for st, v in sorted(by_street.items())))
    return {'iters': iters, 'all': a, 'oop': o, 'under_oop': under_oop, 'under': under}


def pair_drift(iters_a, da, iters_b, db):
    """Mean total-variation strategy drift between two snapshots, by position x tier.
    Tier is by the LATER snapshot's svc (the line's current rarity)."""
    cells = {}  # (pos, tier) -> [tv,...]
    shared = 0
    for k, (svc_b, strat_b) in db.items():
        if strat_b is None or k not in da:
            continue
        _, strat_a = da[k]
        if strat_a is None:
            continue
        shared += 1
        cells.setdefault((_pos(k), _tier(svc_b)), []).append(_tv(strat_a, strat_b))
    print(f"\n--- strategy drift {iters_a:,} -> {iters_b:,}  (shared keys={shared:,}) ---")
    print(f"  {'pos':>4} | {'tier':>14} | {'n':>8} | {'mean TV drift':>13}")
    order = {'rare(<10)': 0, 'mid(10-100)': 1, 'common(>=100)': 2}
    for (pos, tier) in sorted(cells, key=lambda c: (c[0], order.get(c[1], 9))):
        tvs = cells[(pos, tier)]
        print(f"  {pos:>4} | {tier:>14} | {len(tvs):>8,} | {sum(tvs) / len(tvs):>13.4f}")
    return {(pos, tier): (sum(v) / len(v)) for (pos, tier), v in cells.items()}


def main():
    p = argparse.ArgumentParser(description="Rare-line convergence probe across snapshots.")
    p.add_argument('dbs', nargs='+', help="Snapshot DBs, oldest -> newest.")
    args = p.parse_args()

    loaded = [(path, *load(path)) for path in args.dbs]   # (path, iters, data)
    reps = [per_snapshot(path, iters, data) for (path, iters, data) in loaded]

    drifts = []
    for (pa, ia, da), (pb, ib, dbb) in zip(loaded, loaded[1:]):
        drifts.append((ia, ib, pair_drift(ia, da, ib, dbb)))

    # --- trend tables ---
    print("\n=== TREND 1: rare-line VISIT tail across snapshots ===")
    print(f"  {'iters':>12} | {'infosets':>9} | {'OOP p10':>8} | {'OOP p25':>8} | "
          f"{'OOP<10':>8} | {'ALL p10':>8} | {'ALL<10':>8}")
    for r in reps:
        print(f"  {r['iters']:>12,} | {r['all']['n']:>9,} | {r['oop']['p10']:>8} | "
              f"{r['oop']['p25']:>8} | {r['under_oop'][10]:>8,} | {r['all']['p10']:>8} | "
              f"{r['under'][10]:>8,}")

    print("\n=== TREND 2: rare-OOP strategy DRIFT pair-over-pair (the convergence signal) ===")
    print(f"  {'pair (iters)':>26} | {'OOP rare':>9} | {'OOP common':>10} | {'IP rare':>9}")
    for ia, ib, cells in drifts:
        oop_rare = cells.get(('oop', 'rare(<10)'))
        oop_comm = cells.get(('oop', 'common(>=100)'))
        ip_rare = cells.get(('ip', 'rare(<10)'))
        def f(x):
            return f"{x:.4f}" if x is not None else "  n/a"
        print(f"  {ia:>11,}->{ib:<12,} | {f(oop_rare):>9} | {f(oop_comm):>10} | {f(ip_rare):>9}")

    print("\n  READ:")
    print("  - TREND 1: OOP p10/p25 rising = rare OOP lines accruing visits; OOP<10 falling")
    print("    = fewer barely-touched OOP lines. If flat while infosets flat -> tail saturated.")
    print("  - TREND 2 is the decisive one: if OOP-rare drift is large but SHRINKING pair-over-")
    print("    pair, rare OOP lines are still converging -> more iterations help human play.")
    print("    If already small -> settled (stop). If large and NOT shrinking -> the rare tail")
    print("    is cycling, not converging -> more iterations won't help (and common OOP drifts).")


if __name__ == '__main__':
    main()
