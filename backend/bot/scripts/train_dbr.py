#!/usr/bin/env python3
"""Explicit two-role DBR fork driver with atomic paired checkpoints and deadlines.

A manifest publishes immutable bot/opponent generations only after both writes and
hashes succeed. Resume verifies both hashes/configurations; partial generations
never replace the last resumable pair. The source blueprint is opened read-only.
"""
import argparse
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import sys
import time
import uuid
import fcntl
import re
import threading

BOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BOT))
from export_dbr_counts import atomic_json, file_sha256
from src.abstractions.abstraction_set import resolve_abstraction_dir, verify_tables
from src.storage.blueprint_db import BlueprintDB
from src.cfr.keys import db_key_format
from src.abstractions.sizing import db_menu_mode


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3 if sys.platform=='darwin' else 1024**2)


def backup(src,dst):
    with closing(sqlite3.connect(Path(src).resolve().as_uri()+'?mode=ro',uri=True)) as a:
        with closing(sqlite3.connect(dst)) as b:
            with b:
                a.backup(b)


def publish(trainer,root,total,run,previous=None,keep=2):
    """Publish after both role DBs are closed. Old published state survives failures."""
    root=Path(root)
    generation=f'g{total:012d}_{uuid.uuid4().hex[:12]}'
    folder=root/'checkpoints'/generation
    folder.mkdir(parents=True,exist_ok=False)
    bot,opp=folder/'bot.db',folder/'opponent.db'
    if previous:
        for name,path in [('bot',bot),('opponent',opp)]:backup(root/previous[name]['path'],path)
    # Native code clears dirty sets after BOTH DB writes. Never continue this
    # process after a publication failure: resume the previous manifest instead.
    trainer.checkpoint_dbr(str(bot),str(opp),total-1,generation,json.dumps(run,sort_keys=True))
    for path in (bot,opp):
        with closing(sqlite3.connect(path)) as db:
            with db:
                db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                db.execute('PRAGMA journal_mode=DELETE')
    state={'format':'allin-dbr-pair-v1','generation':generation,'iterations':total,'run':run,
           'published_at':datetime.now(timezone.utc).isoformat()}
    for name,path in [('bot',bot),('opponent',opp)]:
        with closing(sqlite3.connect(path)) as db:
            n,empty=db.execute("SELECT COUNT(*),SUM(cumulative_strategy='{}') FROM info_sets").fetchone()
        state[name]={'path':str(path.relative_to(root)),'sha256':file_sha256(path),
                     'info_sets':n,'zero_average_keys':empty or 0}
    atomic_json(folder/'pair.json',state)
    atomic_json(root/'latest.json',state)
    # Only complete owned generations are retired; keep current + a rollback pair.
    owned=sorted((root/'checkpoints').glob('g*'),key=lambda p:p.stat().st_mtime,reverse=True)
    protected={generation}
    if previous:protected.add(previous['generation'])
    for old in owned:
        if len(protected)>=max(2,keep):break
        if (old/'pair.json').exists():protected.add(old.name)
    for old in owned:
        if old.name not in protected and old.is_dir() and re.fullmatch(r'g[0-9]+_[0-9a-f]{12}',old.name):
            shutil.rmtree(old)
    return state


def load_pair(root):
    root=Path(root)
    state=json.loads((root/'latest.json').read_text())
    if state.get('format')!='allin-dbr-pair-v1':raise ValueError('unsupported DBR manifest')
    for name in ('bot','opponent'):
        path=(root/state[name]['path']).resolve()
        if not path.is_relative_to((root/'checkpoints').resolve()):raise ValueError('invalid checkpoint location')
        if file_sha256(path)!=state[name]['sha256']:raise ValueError(f'{name} checkpoint hash mismatch')
    return state


@contextmanager
def locked_run_state(root, resume):
    """Hold the writer lock while choosing, reading and advancing run state."""
    root = Path(root)
    if not resume:
        root.mkdir(parents=True, exist_ok=True)
    # Keep the lock file after closing: unlinking it could give another process a
    # different lock inode while a concurrent opener still holds the original.
    with open(root / '.run.lock', 'a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('another process owns this DBR run') from None
        if resume:
            state = load_pair(root)
        else:
            if any(path.name != '.run.lock' for path in root.iterdir()):
                raise ValueError('new DBR run directory must be empty')
            state = None
        yield state


PRESERVE_REGRET_RECIPE = 'preserve_regret_clock_v2'
LEGACY_RESET_RECIPE = 'reset_all_clocks_v1'


def recorded_fork_recipe(run):
    """Identify old runs without silently relabeling their learned state."""
    if type(run.get('version')) is int:
        if run['version'] == 1 and 'fork_recipe' not in run:
            return LEGACY_RESET_RECIPE
        if run['version'] == 2 and run.get('fork_recipe') == PRESERVE_REGRET_RECIPE:
            return PRESERVE_REGRET_RECIPE
    raise ValueError('unknown DBR run fork recipe/version; start a new run for a new recipe')


def canonical_run_config(value):
    """Normalize real-valued parameters without accepting coerced integer fields."""
    run = dict(value)
    for field in ('version', 'seed', 'workers', 'merge_every', 'checkpoint_every'):
        number = run.get(field)
        if type(number) is not int:
            raise ValueError(f'DBR run configuration {field} must be an integer')
        if (field == 'seed' and not 0 <= number < 2**64) or (field != 'seed' and number < 1):
            raise ValueError(f'invalid DBR run configuration {field}')
    for field in ('alpha', 'gamma', 'p_max', 'strength'):
        number = run.get(field)
        if type(number) not in (int, float):
            raise ValueError(f'DBR run configuration {field} must be a finite number')
        try:
            number = float(number)
        except OverflowError:
            raise ValueError(f'DBR run configuration {field} must be finite') from None
        if not math.isfinite(number):
            raise ValueError(f'DBR run configuration {field} must be finite')
        run[field] = number
    if run['alpha'] < 0 or run['gamma'] < 0 or not 0 <= run['p_max'] <= 1 or run['strength'] <= 0:
        raise ValueError('invalid DBR run configuration parameters')
    recorded_fork_recipe(run)
    return run


def main():
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--base',required=True,help='frozen base snapshot, never overwritten')
    ap.add_argument('--cache',required=True,help='anonymous fit-set empirical count cache')
    ap.add_argument('--run-dir',required=True)
    ap.add_argument('--resume',action='store_true',help='continue the recorded recipe; use a new directory for a corrected fork')
    ap.add_argument('--iterations',required=True,type=int,help='additional fork iterations')
    ap.add_argument('--p-max',type=float,default=.5)
    ap.add_argument('--strength',type=float,default=10.0)
    ap.add_argument('--seed',type=int,default=20261121)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--merge-every',type=int,default=2000)
    ap.add_argument('--checkpoint-every',type=int,default=2400000)
    ap.add_argument('--stop-at',help='timezone-aware ISO time; stops early to reserve checkpoint time')
    ap.add_argument('--max-seconds',type=float,help='additional monotonic wall-time bound, including setup')
    ap.add_argument('--checkpoint-reserve',type=float,default=180)
    ap.add_argument('--max-rss-gib',type=float,default=12)
    ap.add_argument('--river-cache-boards',type=int,default=140000)
    a=ap.parse_args()
    # SQLite resolves database symlinks before locating WAL sidecars. Use the
    # same canonical name for the frozen-base guard, hashing and native access.
    a.base=str(Path(a.base).resolve())
    if min(a.iterations,a.workers,a.merge_every,a.checkpoint_every)<1 or a.seed<0 or a.seed>=2**64:
        ap.error('positive iteration/worker/chunk counts and u64 seed required')
    if a.workers<2:
        # D's kept regret counters are per merge round; the serial path discounts per iteration (BUG-035).
        ap.error('DBR training requires --workers >= 2')
    if not 0<=a.p_max<=1 or not math.isfinite(a.strength) or a.strength<=0:
        ap.error('p-max in [0,1] and positive finite strength required')
    if not math.isfinite(a.checkpoint_reserve) or a.checkpoint_reserve<0 or not math.isfinite(a.max_rss_gib) or a.max_rss_gib<=0:
        ap.error('invalid checkpoint reserve / memory limit')
    start_clock=time.monotonic()
    budget=float('inf')
    wall_deadline=float('inf')
    if a.stop_at:
        stop=datetime.fromisoformat(a.stop_at)
        if stop.tzinfo is None:ap.error('--stop-at must include a timezone')
        wall_deadline=stop.timestamp()
        budget=min(budget,wall_deadline-time.time())
    if a.max_seconds is not None:
        if not math.isfinite(a.max_seconds) or a.max_seconds<=0:ap.error('max-seconds must be positive finite')
        budget=min(budget,a.max_seconds)
    deadline=start_clock+budget
    def remaining():
        # Absolute deadlines must also expire across a manual laptop sleep.
        return min(deadline-time.monotonic(),wall_deadline-time.time())
    if remaining()<=a.checkpoint_reserve:
        print('Deadline reserve already reached; no training started.');return
    # A backstop scoped to this process: even a stalled native round or checkpoint
    # cannot leave laptop-heavy threads running past the requested stop time.
    if math.isfinite(budget):
        def hard_stop():
            while remaining()>0:
                time.sleep(min(1.0,max(0.0,remaining())))
            print('Hard deadline reached; last published pair remains resumable.',file=sys.stderr,flush=True)
            os._exit(124)
        threading.Thread(target=hard_stop,daemon=True).start()
    wal=Path(str(a.base)+'-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('DBR base must be a quiescent snapshot without a nonempty WAL')
    cache=json.loads(Path(a.cache).read_text())
    if file_sha256(a.base)!=cache['base_sha256']:
        raise ValueError('base snapshot content differs from the DBR cache')
    root=Path(a.run_dir).resolve()
    with locked_run_state(root,a.resume) as state:
        base=BlueprintDB(a.base,read_only=True)
        try:
            kf=db_key_format(base);menu=db_menu_mode(base)
            alpha=base.get_metadata('alpha');gamma=base.get_metadata('gamma')
            adir=resolve_abstraction_dir(base,'DBR training');verify_tables(adir)
        finally:base.close()
        recipe=recorded_fork_recipe(state['run']) if state else PRESERVE_REGRET_RECIPE
        run={'version':2 if recipe==PRESERVE_REGRET_RECIPE else 1,'base_sha256':cache['base_sha256'],'cache_sha256':file_sha256(a.cache),
             'seed':a.seed,'workers':a.workers,'merge_every':a.merge_every,
             'checkpoint_every':a.checkpoint_every,'key_format':kf,'menu_mode':menu,
             'alpha':alpha,'gamma':gamma,'p_max':a.p_max,'strength':a.strength}
        if recipe==PRESERVE_REGRET_RECIPE:
            run['fork_recipe']=recipe
        run=canonical_run_config(run)
        if state and canonical_run_config(state['run'])!=run:
            raise ValueError('DBR run configuration mismatch; seed/schedule/cache must match')
        import allin_core
        trainer=allin_core.Trainer(menu,alpha,gamma,key_format=kf)
        trainer.load_abstraction(str(adir),a.river_cache_boards)
        if state:
            # Preserve the recorded JSON number types for native provenance
            # checks on old pairs (e.g. strength=10 versus canonical 10.0).
            # The validated continuation writes canonical metadata to its new pair.
            total=trainer.resume_dbr(str(root/state['bot']['path']),str(root/state['opponent']['path']),
                                     a.cache,a.p_max,a.strength,json.dumps(state['run'],sort_keys=True))
            if total!=state['iterations']:raise ValueError('manifest/role iteration mismatch')
        else:
            source=trainer.fork_dbr(a.base,a.cache,a.p_max,a.strength)
            total=0
            state=publish(trainer,root,total,run)
            print(f'Forked {source:,}-iteration base; retained regret counters and reset both role averages.',flush=True)
        if trainer.dbr_fork_recipe!=recipe:
            raise ValueError('fork recipe mismatch between manifest and native role tables')
        print(f'Fork recipe: {recipe}',flush=True)
        if recipe==LEGACY_RESET_RECIPE:
            print('Continuing a legacy reset-clock experiment; this is not a corrected warm-start fork.',flush=True)
        print(f'DBR p_max={a.p_max} strength={a.strength} | bot keys {trainer.num_info_sets:,} '
              f'free-opponent keys {trainer.opponent_info_sets:,} | RSS {rss_gib():.2f} GiB',flush=True)
        target=total+a.iterations
        last_log=0
        stop_reason=None
        def progress(d):
            nonlocal last_log,stop_reason
            now=time.monotonic()
            if remaining()<=a.checkpoint_reserve:stop_reason='deadline reserve'
            if rss_gib()>a.max_rss_gib:stop_reason='memory limit'
            if now-last_log>=30 or stop_reason:
                rate=d['iterations']/max(d['seconds'],1e-9)
                print(f"  fork iter {total+d['iterations']:,}/{target:,} | {rate:,.0f} it/s | "
                      f"keys {d['info_sets']:,}/{d['opponent_info_sets']:,} | RSS {rss_gib():.2f} GiB",flush=True)
                last_log=now
            return stop_reason is None
        while total<target and remaining()>a.checkpoint_reserve:
            n=min(a.checkpoint_every,target-total)
            stats=trainer.train(n,total,a.seed,workers=a.workers,merge_every=a.merge_every,
                                progress=progress,progress_every=10000)
            total+=stats['iterations']
            if stats['iterations']:
                state=publish(trainer,root,total,run,state)
                print(f'Published paired checkpoint at {total:,}; RSS {rss_gib():.2f} GiB.',flush=True)
            if stats['interrupted'] or stop_reason:break
        summary={'iterations':total,'target':target,'reason':stop_reason or ('complete' if total==target else 'deadline reserve'),
                 'seconds':time.monotonic()-start_clock,'peak_rss_gib':rss_gib(),
                 'bot':state['bot'],'opponent':state['opponent'],'run':run}
        atomic_json(root/'result.json',summary)
        print(json.dumps(summary,sort_keys=True),flush=True)

if __name__=='__main__':main()
