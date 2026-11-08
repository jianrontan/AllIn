"""Strategy distance between blueprint DBs (statistical trainer agreement).

    python bench/compare_blueprints.py A.db B.db [C.db ...]

For every pair: keys in common, and the total-variation distance between the
normalised average strategies per key, averaged with weight = the keys' summed
average-strategy mass (a reach proxy), overall and per street. Used to check
that the Rust parallel trainer agrees with the Python one as closely as two
Python seeds agree with each other.
"""
import itertools
import json
import sqlite3
import sys

STREETS = ('preflop', 'flop', 'turn', 'river')


def load(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out = {}
    for key, strat in con.execute("SELECT key, cumulative_strategy FROM info_sets"):
        s = json.loads(strat)
        tot = sum(s.values())
        if tot > 1e-12:
            out[key] = (tot, {a: v / tot for a, v in s.items()})
    con.close()
    return out


def street(key):
    parts = key.split('_')
    return 'preflop' if len(parts) == 4 else parts[4]


def compare(a, b):
    common = a.keys() & b.keys()
    acc = {s: [0.0, 0.0] for s in STREETS + ('all',)}
    for k in common:
        (wa, pa), (wb, pb) = a[k], b[k]
        tv = 0.5 * sum(abs(pa.get(x, 0.0) - pb.get(x, 0.0)) for x in pa.keys() | pb.keys())
        w = wa + wb
        for s in (street(k), 'all'):
            acc[s][0] += w * tv
            acc[s][1] += w
    res = {s: (v[0] / v[1] if v[1] else float('nan')) for s, v in acc.items()}
    return len(common), res


def main(paths):
    dbs = {p: load(p) for p in paths}
    for p in paths:
        print(f"{len(dbs[p]):>8,} keys  {p}")
    print(f"\n{'pair':<12} {'common':>8}  " + "  ".join(f"{s:>8}" for s in STREETS + ('all',)))
    for (i, p), (j, q) in itertools.combinations(enumerate(paths), 2):
        n, r = compare(dbs[p], dbs[q])
        print(f"{i}-{j:<10} {n:>8,}  " + "  ".join(f"{r[s]:8.4f}" for s in STREETS + ('all',)))


if __name__ == '__main__':
    main(sys.argv[1:])
