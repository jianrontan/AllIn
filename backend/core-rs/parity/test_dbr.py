"""Two-role DBR contracts using small simulator fixtures, never human data."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import pytest
core=pytest.importorskip('allin_core')
BOT=Path(__file__).parents[2]/'bot'
sys.path.insert(0,str(BOT));sys.path.insert(0,str(BOT/'scripts'))
from src.storage.blueprint_db import BlueprintDB
from src.abstractions.abstraction_set import LEGACY_ABSTRACTION_ID
from src.exploitation.opponent_model import blueprint_model_stamp
from train_dbr import publish,load_pair


def data():
    f=Path(__file__).parents[1]/'fixtures/mccfr_capped_pj_spr5a_g1.json'
    if not f.exists():pytest.skip('generate spr5a fixture first')
    return json.loads(f.read_text())


def run(t,rows,stream,start=0):
    return t.run_fixture([r['cards'] for r in rows],[r['fine'] for r in rows],
                        [r['post'] for r in rows],[r['raw'] for r in rows],
                        data()['fine_to_coarse'],stream,start_iteration=start)


def rows(path,table='info_sets'):
    with sqlite3.connect(path) as db:return db.execute(f'SELECT * FROM {table} ORDER BY key').fetchall()


def make_base_and_cache(tmp_path,key_format='spr5a'):
    d=data();t=core.Trainer('capped_pj',key_format=key_format)
    run(t,d['deals'][:20],d['uniforms']*20)
    base=tmp_path/'base.db';t.checkpoint(str(base),19)
    with sqlite3.connect(base) as db:
        db.execute('INSERT OR REPLACE INTO training_metadata VALUES (?,?)',('abstraction_id',json.dumps(LEGACY_ABSTRACTION_ID)))
        key=db.execute("SELECT key FROM info_sets WHERE key LIKE 'pf_%_ip_' LIMIT 1").fetchone()[0]
    db.close()
    with sqlite3.connect(base) as closed:
        closed.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        closed.execute('PRAGMA journal_mode=DELETE')
    closed.close()
    b=BlueprintDB(str(base),read_only=True)
    stamp=blueprint_model_stamp(b);b.close()
    cache={'format':'allin-dbr-counts-v1','estimator':'empirical-full-v1','stamp':stamp,
           'base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'split_sha256':'c'*64,
           'base_iterations':20,'rows':{key:{'c':10,'f':3,'s':2}}}
    path=tmp_path/'counts.json';path.write_text(json.dumps(cache))
    return base,path


@pytest.fixture
def base_and_cache(tmp_path):
    return make_base_and_cache(tmp_path)


def fork(base,cache,p=.5):
    t=core.Trainer('capped_pj',key_format='spr5a')
    assert t.fork_dbr(str(base),str(cache),p,10.0)==20
    return t


def test_fork_resets_averages_and_preserves_regret_clock(base_and_cache,tmp_path):
    base,cache=base_and_cache;t=fork(base,cache)
    state=publish(t,tmp_path/'run',0,{'seed':711})
    b=rows(tmp_path/'run'/state['bot']['path']);o=rows(tmp_path/'run'/state['opponent']['path'])
    assert b==o
    old={r[0]:r for r in rows(base)}
    assert all(r[2]==old[r[0]][2] for r in b)  # regrets retained
    assert all(json.loads(r[3])=={} and r[4:]==(old[r[0]][4],-1,0,-1) for r in b)
    assert any(r[4]>0 for r in b)
    assert t.dbr_fork_recipe=='preserve_regret_clock_v2'
    with pytest.raises(ValueError,match='paired checkpoint'):
        t.checkpoint(str(tmp_path/'wrong.db'),0)
    with pytest.raises(ValueError,match='resume_dbr'):
        core.Trainer('capped_pj',key_format='spr5a').resume(str(tmp_path/'run'/state['bot']['path']))


def test_paired_resume_matches_uninterrupted_and_rejects_wrong_pair(base_and_cache,tmp_path):
    base,cache=base_and_cache;d=data();ds=d['deals']*2;stream=d['uniforms']*100
    full=fork(base,cache);run(full,ds[:80],stream)
    fs=publish(full,tmp_path/'full',80,{'seed':711})
    a=fork(base,cache);used=run(a,ds[:40],stream)
    state=publish(a,tmp_path/'resume',40,{'seed':711})
    b=core.Trainer('capped_pj',key_format='spr5a')
    bp=str(tmp_path/'resume'/state['bot']['path']);op=str(tmp_path/'resume'/state['opponent']['path'])
    assert b.resume_dbr(bp,op,str(cache),.5,10.0,json.dumps({'seed':711}))==40
    run(b,ds[40:80],stream[used:],start=40)
    bs=publish(b,tmp_path/'resume',80,{'seed':711},state)
    for role in ('bot','opponent'):
        assert rows(tmp_path/'full'/fs[role]['path'])==rows(tmp_path/'resume'/bs[role]['path'])
    # Roles actually learn independently rather than aliasing one map.
    assert rows(tmp_path/'resume'/bs['bot']['path'])!=rows(tmp_path/'resume'/bs['opponent']['path'])
    fresh=core.Trainer('capped_pj',key_format='spr5a')
    with pytest.raises(ValueError,match='mismatch'):
        fresh.resume_dbr(bp,str(tmp_path/'resume'/bs['opponent']['path']),str(cache),.5,10.0,json.dumps({'seed':711}))
    with pytest.raises(ValueError,match='mismatch'):
        fresh.resume_dbr(bp,op,str(cache),.75,10.0,json.dumps({'seed':711}))
    assert load_pair(tmp_path/'resume')['iterations']==80


def test_failed_pair_publication_keeps_previous_manifest(base_and_cache,tmp_path):
    base,cache=base_and_cache;t=fork(base,cache)
    state=publish(t,tmp_path/'run',0,{'seed':711})
    class Broken:
        def checkpoint_dbr(self,*a):
            with sqlite3.connect(a[0]) as db:
                db.execute("UPDATE training_metadata SET value='999' WHERE key='total_iterations'")
            raise RuntimeError('simulated second-role failure')
    with pytest.raises(RuntimeError):publish(Broken(),tmp_path/'run',1,{'seed':711},state)
    assert load_pair(tmp_path/'run')==state


def test_observed_model_changes_learning_vs_zero_bias(base_and_cache,tmp_path):
    base,cache=base_and_cache;d=data();ds=d['deals']*2
    contents=json.loads(cache.read_text())
    contents['rows']={f'pf_{fine}_{seat}_':{'f':100} for deal in ds for fine in deal['fine'] for seat in ('ip','oop')}
    cache.write_text(json.dumps(contents))
    a=fork(base,cache,0.0);b=fork(base,cache,.75)
    stream=d['uniforms']*100
    run(a,ds[:100],stream);run(b,ds[:100],stream)
    sa=publish(a,tmp_path/'zero',100,{'seed':711})
    sb=publish(b,tmp_path/'biased',100,{'seed':711})
    assert rows(tmp_path/'zero'/sa['bot']['path'])!=rows(tmp_path/'biased'/sb['bot']['path'])


def test_export_is_disjoint_anonymous_and_stamp_bound(base_and_cache,tmp_path):
    from export_dbr_counts import export_counts
    base,cache=base_and_cache
    cached=json.loads(cache.read_text());key=next(iter(cached['rows']))
    folder=tmp_path/'synthetic-models';folder.mkdir()
    expected=0
    for i in range(8):
        expected+=10+i
        (folder/f'model_{i}.json').write_text(json.dumps({'stamp':cached['stamp'],
            'playerId':f'synthetic-owner-{i}','handle':'never-output-person','hands':100,
            'counts':{key:{'c':10+i}}}))
    out=tmp_path/'split'
    summary=export_counts(base,folder,out,.25,711)
    fit=json.loads((out/'train.json').read_text())
    held=json.loads((out/'heldout/model_population.json').read_text())
    assert summary['fit_players']==6 and summary['heldout_players']==2
    assert fit['rows'][key]['c']+held['counts'][key]['c']==expected
    output=''.join(p.read_text() for p in out.rglob('*.json'))
    assert 'synthetic-owner' not in output and 'never-output-person' not in output
    with pytest.raises(ValueError,match='overwrite'):export_counts(base,folder,out,.25,711)


def test_training_callback_can_request_a_clean_stop(base_and_cache,tmp_path):
    from src.abstractions.abstraction_set import resolve_abstraction_dir
    base,cache=base_and_cache;t=fork(base,cache)
    b=BlueprintDB(str(base),read_only=True);adir=resolve_abstraction_dir(b);b.close()
    t.load_abstraction(str(adir),1000)
    result=t.train(1000,0,711,workers=2,merge_every=8,progress=lambda d:False)
    assert result['interrupted'] and result['iterations']==16
    state=publish(t,tmp_path/'stopped',result['iterations'],{'seed':711})
    assert load_pair(tmp_path/'stopped')['iterations']==16
    assert state['bot']['info_sets']>0 and state['opponent']['info_sets']>0


def test_dbr_training_rejects_serial_worker(base_and_cache):
    # BUG-035: the serial path's per-iteration discount would rescale D's kept
    # per-merge-round regret counters away.
    from src.abstractions.abstraction_set import resolve_abstraction_dir
    base,cache=base_and_cache;t=fork(base,cache)
    b=BlueprintDB(str(base),read_only=True);adir=resolve_abstraction_dir(b);b.close()
    t.load_abstraction(str(adir),1000)
    with pytest.raises(Exception,match='workers >= 2'):t.train(16,0,711,workers=1)


def test_pair_hash_detects_database_corruption(base_and_cache,tmp_path):
    base,cache=base_and_cache;t=fork(base,cache)
    state=publish(t,tmp_path/'run',0,{'seed':711})
    with sqlite3.connect(tmp_path/'run'/state['opponent']['path']) as db:
        db.execute("UPDATE training_metadata SET value='123' WHERE key='total_iterations'")
    with pytest.raises(ValueError,match='opponent checkpoint hash mismatch'):
        load_pair(tmp_path/'run')


def test_driver_hard_deadline_stops_stalled_setup(base_and_cache,tmp_path):
    import subprocess
    base,cache=base_and_cache
    # Stand in for a stalled native setup, not a slow or hardware-dependent run.
    code="""import os,runpy,sys,time,types
class SlowTrainer:
    def __init__(self,*a,**kw):time.sleep(10)
sys.modules['allin_core']=types.SimpleNamespace(Trainer=SlowTrainer)
sys.path.insert(0,os.path.dirname(sys.argv[1]))
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    args=[sys.executable,'-c',code,str(BOT/'scripts/train_dbr.py'),
          '--base',str(base),'--cache',str(cache),'--run-dir',str(tmp_path/'timed'),
          '--iterations','100','--max-seconds','0.1','--checkpoint-reserve','0']
    result=subprocess.run(args,capture_output=True,text=True,timeout=5)
    assert result.returncode==124,result.stderr
    assert 'Hard deadline reached' in result.stderr
    assert not (tmp_path/'timed/latest.json').exists()


@pytest.mark.parametrize('legacy_missing_format',[False,True])
def test_v2_pair_round_trips_and_reads_older_implicit_format(tmp_path,legacy_missing_format):
    base,cache=make_base_and_cache(tmp_path,key_format='v2')
    trainer=core.Trainer('capped_pj',key_format='v2')
    assert trainer.fork_dbr(str(base),str(cache),.5,10)==20
    d=data();run(trainer,d['deals'][:20],d['uniforms']*20)
    state=publish(trainer,tmp_path/'pair',20,{'seed':711})
    paths=[tmp_path/'pair'/state[role]['path'] for role in ('bot','opponent')]
    for path in paths:
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT value FROM training_metadata WHERE key='key_format'").fetchone()==('"v2"',)
            if legacy_missing_format:db.execute("DELETE FROM training_metadata WHERE key='key_format'")
    fresh=core.Trainer('capped_pj',key_format='v2')
    assert fresh.resume_dbr(str(paths[0]),str(paths[1]),str(cache),.5,10,json.dumps({'seed':711}))==20
    assert fresh.num_info_sets==trainer.num_info_sets
    assert fresh.opponent_info_sets==trainer.opponent_info_sets


@pytest.mark.parametrize('alias_kind',[
    'same','new_dot','case_only','symlink_parent','directory_firmlink','symlink_file','hardlink','dangling_symlink','new_unicode',
])
def test_role_alias_rejection_is_before_writes_and_preserves_dirty_state(base_and_cache,tmp_path,alias_kind):
    import os
    base,cache=base_and_cache;trainer=fork(base,cache)
    d=data();run(trainer,d['deals'][:40],d['uniforms']*100)
    target=tmp_path/'role.db';other=target
    expected=None
    if alias_kind=='new_dot':
        other=str(tmp_path)+'/./role.db'
    elif alias_kind=='case_only':
        other=tmp_path/'ROLE.db'
    elif alias_kind=='symlink_parent':
        actual=tmp_path/'actual';actual.mkdir()
        link=tmp_path/'linked';link.symlink_to(actual,target_is_directory=True)
        target=actual/'role.db';other=link/'role.db'
    elif alias_kind=='directory_firmlink':
        if sys.platform!='darwin':pytest.skip('APFS firmlink regression is macOS-specific')
        alias_parent=Path('/System/Volumes/Data')/tmp_path.relative_to('/')
        if (not alias_parent.exists() or not os.path.samefile(tmp_path,alias_parent)
                or tmp_path.resolve()==alias_parent.resolve()):
            pytest.skip('temporary directory has no distinct firmlink spelling')
        other=alias_parent/'role.db'
    elif alias_kind in ('symlink_file','hardlink'):
        with sqlite3.connect(target) as db:db.execute('CREATE TABLE protected (value TEXT)')
        db.close();expected=target.read_bytes();other=tmp_path/'other.db'
        if alias_kind=='symlink_file':other.symlink_to(target)
        else:os.link(target,other)
    elif alias_kind=='dangling_symlink':
        other=tmp_path/'other.db';other.symlink_to(target)
    elif alias_kind=='new_unicode':
        target=tmp_path/'rôle.db';other=tmp_path/'other.db'
    with pytest.raises(ValueError,match='different database files|resolve DBR checkpoint target|ASCII'):
        trainer.checkpoint_dbr(str(target),str(other),39,'bad','{}')
    if expected is None:assert not target.exists()
    else:assert target.read_bytes()==expected
    assert not Path(str(target)+'-wal').exists()
    assert not Path(str(other)+'-wal').exists()
    # All pending role rows must still be available after the failed API call.
    state=publish(trainer,tmp_path/'valid',40,{})
    assert state['bot']['info_sets']==trainer.num_info_sets
    assert state['opponent']['info_sets']==trainer.opponent_info_sets
    assert rows(tmp_path/'valid'/state['bot']['path'])!=rows(tmp_path/'valid'/state['opponent']['path'])


def test_role_paths_keep_file_uri_looking_strings_as_literal_names(base_and_cache,tmp_path,monkeypatch):
    base,cache=base_and_cache;trainer=fork(base,cache)
    unicode_parent=tmp_path/'训练';unicode_parent.mkdir()
    monkeypatch.chdir(unicode_parent)
    bot='file:bot.db?mode=memory';opponent='file:opponent.db?mode=memory'
    trainer.checkpoint_dbr(bot,opponent,-1,'literal','{}')
    assert Path(bot).is_file() and Path(opponent).is_file()
    fresh=core.Trainer('capped_pj',key_format='spr5a')
    assert fresh.resume_dbr(bot,opponent,str(cache),.5,10,'{}')==0
