"""TEST 2: low-SPR bet-sizing-collapse fingerprint.

Hypothesis: the bot overjams because at low SPR the pot-fraction menu collapses
(sized bets >= stack fold into all-in), so the effective menu degenerates toward
{small, all-in} and GTO sizes in the gap get rounded to a jam.

The info-set key carries no SPR, but the blueprint stores the ACTUAL legal menu
per node (legalActions). At low SPR that menu is short and all-in-heavy. So we
correlate, over postflop facing/opening nodes that COULD jam:

  * menu richness (how many distinct sized bet/raise options survived) -> proxy
    for SPR (rich = high SPR, sparse = low SPR);
  * avg P(all-in) the blueprint assigns.

If P(all-in) rises sharply as the menu thins, that's the collapse fingerprint =
a bet-sizing-abstraction cause of overjamming (independent of any trainer bias).
"""
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.storage.blueprint_db import BlueprintDB

DB = os.environ.get('ALLIN_BLUEPRINT_DB',
                    'analysis/blueprints/blueprint_par_20260529_233056.db')

_SIZED = {'s', 'm', 'l', 'o', 'x'}     # bet/raise size chars (excl. all-in 'a')


def main():
    db = BlueprintDB(DB, read_only=True)
    con = sqlite3.connect('file:' + DB + '?mode=ro', uri=True)
    keys = [r[0] for r in con.execute('select key from info_sets')]
    con.close()

    # bucket by # of distinct SIZED bet/raise options the node offered (the menu
    # richness). all-in must be in the menu (else jamming isn't even possible).
    by_menu = defaultdict(lambda: {'n': 0, 'sum_allin': 0.0, 'jam_gt50': 0})
    streets = ('_flop_', '_turn_', '_river_')
    for k in keys:
        if not any(s in k for s in streets):
            continue
        rec = db.get_record(k)
        if not rec:
            continue
        legal = rec.get('legalActions') or []
        if 'allin' not in legal:
            continue                                   # jam not offered -> skip
        n_sized = sum(1 for a in legal
                      if a.startswith(('bet_', 'raise_')))
        strat = rec.get('strategy') or {}
        pa = float(strat.get('allin', 0.0))
        b = by_menu[n_sized]
        b['n'] += 1
        b['sum_allin'] += pa
        if pa > 0.5:
            b['jam_gt50'] += 1
    db.close()

    lines = [f"DB: {DB}", "",
             "sized-bet options | nodes | avg P(allin) | jam>50%   "
             "  (fewer options = lower SPR = thinner menu)"]
    for nsz in sorted(by_menu):
        b = by_menu[nsz]
        if b['n'] == 0:
            continue
        lines.append(f"   {nsz:>2} sized opts  | {b['n']:>5} | "
                     f"   {b['sum_allin']/b['n']:.4f}   | "
                     f"{100*b['jam_gt50']/b['n']:5.1f}%")
    out = '\n'.join(lines)
    with open('scripts/_spr_out.txt', 'w', encoding='ascii', errors='replace') as f:
        f.write(out + '\n')
    print(out)


if __name__ == '__main__':
    main()
