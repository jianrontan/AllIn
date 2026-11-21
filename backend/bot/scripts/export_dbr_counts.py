#!/usr/bin/env python3
"""Export an anonymous player-held-out DBR count cache on a frozen blueprint.

The fixed DBR component is the empirical full-key action frequency. Confidence
n/(s+n) supplies the shrinkage; applying the serving blueprint posterior again
would count that shrinkage twice. Unseen/unsupported observations remain free.
No player identifiers or handles are written to either aggregate output.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

BOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOT))
from src.storage.blueprint_db import BlueprintDB
from src.exploitation.opponent_model import blueprint_model_stamp, check_model_stamp


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    with open(tmp, 'w') as f:
        json.dump(value, f, sort_keys=True, separators=(',', ':'), allow_nan=False)
        f.flush()
        import os
        os.fsync(f.fileno())
    tmp.replace(path)


def export_counts(base_path, model_dir, out_dir, holdout_fraction=.25, split_seed=20261121):
    if not 0 < holdout_fraction < 1:
        raise ValueError('holdout_fraction must be between zero and one')
    # SQLite uses the resolved database filename for its WAL sidecar.
    base_path = Path(base_path).resolve(strict=True)
    # Use a completed snapshot, not a mutable training database or a live WAL.
    if Path(str(base_path) + '-wal').exists() and Path(str(base_path) + '-wal').stat().st_size:
        raise ValueError('DBR base must be a quiescent snapshot without a nonempty WAL')
    base_hash = file_sha256(base_path)
    db = BlueprintDB(str(base_path), read_only=True)
    try:
        stamp = blueprint_model_stamp(db)
        players = []
        seen = set()
        for path in Path(model_dir).glob('model_*.json'):
            if path.name == 'model_population.json' or path.name.startswith('model_built_'):
                continue
            m = json.loads(path.read_text())
            check_model_stamp(m, stamp, 'anonymous count file')
            owner = m.get('playerId')
            if not owner or owner in seen:
                raise ValueError('missing or duplicate count owner')
            seen.add(owner)
            order = hashlib.sha256(f'{split_seed}|{owner}'.encode()).hexdigest()
            players.append((order, m))
        players.sort(key=lambda p: p[0])
        if len(players) < 2:
            raise ValueError('need at least two players for a held-out split')
        nh = max(1, min(len(players)-1, round(len(players)*holdout_fraction)))
        heldout, fit = players[:nh], players[nh:]
        def aggregate(rows):
            result = defaultdict(Counter)
            for _, m in rows:
                for k, counts in m['counts'].items():
                    for ch, n in counts.items():
                        if ch not in 'kcfsmlxo2a' or len(ch)!=1 or not isinstance(n,(int,float)) or not 0<=n<1e15:
                            raise ValueError('invalid aggregate action count')
                        result[k][ch] += n
            return {k:dict(c) for k,c in result.items()}
        train = aggregate(fit)
        held = aggregate(heldout)
        split_id = hashlib.sha256(json.dumps({'seed':split_seed,
            'fit':[order for order,_ in fit], 'heldout':[order for order,_ in heldout],
            'train_counts':train, 'heldout_counts':held}, sort_keys=True).encode()).hexdigest()
        # Bound to the frozen base's reachable key vocabulary; no mutable training prior.
        supported = {k:c for k,c in train.items() if db.get_average_strategy(k)}
        common = {'format':'allin-dbr-counts-v1', 'estimator':'empirical-full-v1',
                  'stamp':stamp, 'base_sha256':base_hash,
                  'base_iterations':db.get_metadata('total_iterations'), 'split_sha256':split_id,
                  'split_seed':split_seed, 'fit_players':len(fit), 'heldout_players':len(heldout)}
        out = Path(out_dir)
        if any((out/n).exists() for n in ('train.json','heldout/model_population.json','summary.json')):
            raise ValueError('refusing to overwrite an existing DBR split')
        atomic_json(out/'train.json',dict(common,rows=supported))
        atomic_json(out/'heldout/model_population.json',{'stamp':stamp,'counts':held})
        summary = dict(common,train_keys=len(supported),excluded_keys=len(train)-len(supported),
                       heldout_keys=len(held),fit_decisions=sum(sum(c.values()) for c in train.values()),
                       heldout_decisions=sum(sum(c.values()) for c in held.values()))
        atomic_json(out/'summary.json',summary)
        return summary
    finally:
        db.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--db',required=True)
    ap.add_argument('--model-dir',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--holdout-fraction',type=float,default=.25)
    ap.add_argument('--split-seed',type=int,default=20261121)
    a=ap.parse_args()
    summary=export_counts(a.db,a.model_dir,a.out_dir,a.holdout_fraction,a.split_seed)
    print(json.dumps(summary,sort_keys=True))

if __name__=='__main__':main()
