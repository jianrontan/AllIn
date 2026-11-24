"""DBR policy completion is shared by serving/eval, never written into training."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.storage.blueprint_db import BlueprintDB, FrozenBlueprint


def put(db, key, average):
    db.conn.execute('INSERT INTO info_sets (key,legal_actions,cumulative_regrets,cumulative_strategy,visit_count) VALUES (?,?,?,?,?)',
                    (key, '["fold","call"]', '{"call":7}', json.dumps(average), 23))
    db.conn.commit()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def pair(tmp_path, monkeypatch):
    monkeypatch.delenv('ALLIN_DBR_BASE_DB', raising=False)
    base, candidate = tmp_path/'D.db', tmp_path/'bot.db'
    stamp = {'key_format':'spr5a', 'menu_mode':'capped_pj', 'abstraction_id':'cards'}
    with closing(BlueprintDB(base)) as db:
        for k,v in dict(stamp, total_iterations=1600).items(): db.set_metadata(k,v)
        for key, average in [('empty', {'fold':1,'call':3}), ('zero', {'call':4}),
                             ('trained', {'fold':1}), ('base_empty', {}), ('missing', {'call':1})]:
            put(db, key, average)
    with closing(BlueprintDB(candidate)) as db:
        for k,v in dict(stamp, dbr_role='bot', dbr={'base_sha256':sha(base)}, fork_base_iterations=1600).items():
            db.set_metadata(k,v)
        for key, average in [('empty',{}), ('zero',{'fold':0,'call':0}),
                             ('trained',{'fold':3,'call':1}), ('new',{}), ('base_empty',{})]:
            put(db,key,average)
    return base,candidate


def test_effective_serving_and_evaluation_policy_and_raw_state(pair):
    base,candidate=pair
    before=(sha(base),sha(candidate))
    with closing(BlueprintDB(candidate,read_only=True,fallback_base_db=base)) as db:
        # API/explorer use records; evaluators/river leaf use average lookups.
        frozen=FrozenBlueprint(db)
        for key,want in [('empty',{'fold':.25,'call':.75}), ('zero',{'call':1}),
                         ('trained',{'fold':.75,'call':.25})]:
            record=db.get_record(key)
            assert record['strategy']==db.get_average_strategy(key)==frozen.get_average_strategy(key)==want
            assert record['visitCount']==23
            assert record['strategySource']==('dbr_average' if key=='trained' else 'frozen_base')
        # No D policy: ordinary consumers' untrained guards apply. A missing
        # candidate row is not completed, even when D has that row.
        for key in ('new','base_empty','missing'):
            assert db.get_record(key) is None
            assert db.get_average_strategy(key) is None
        assert json.loads(db.conn.execute("SELECT cumulative_strategy FROM info_sets WHERE key='empty'").fetchone()[0])=={}
        assert db.load_all_to_memory()['empty'].cumulative_regrets=={'call':7}
    assert (sha(base),sha(candidate))==before


def test_env_configuration_shared_across_threads(pair,monkeypatch):
    base,candidate=pair
    monkeypatch.setenv('ALLIN_DBR_BASE_DB',str(base))
    with closing(BlueprintDB(candidate,read_only=True)) as db:
        with ThreadPoolExecutor(max_workers=4) as pool:
            values=list(pool.map(db.get_average_strategy,['empty']*32))
        assert values==[{'fold':.25,'call':.75}]*32
        fallback=db._fallback_db
    assert not fallback._all_conns


def test_missing_base_fails_even_for_trained_candidate(pair):
    _,candidate=pair
    with closing(BlueprintDB(candidate,read_only=True)) as db:
        with pytest.raises(ValueError,match='ALLIN_DBR_BASE_DB'):db.get_average_strategy('trained')


def test_wrong_base_hash(pair,tmp_path):
    _,candidate=pair
    wrong=tmp_path/'wrong.db';wrong.write_bytes(b'wrong')
    with closing(BlueprintDB(candidate,read_only=True,fallback_base_db=wrong)) as db:
        with pytest.raises(ValueError,match='SHA256'):db.validate_policy()


@pytest.mark.parametrize('field,value,message',[
    ('key_format','v2','key_format'),('menu_mode','control','menu_mode'),
    ('abstraction_id','other','abstraction_id'),('fork_base_iterations',1500,'iteration'),
    ('dbr_role','opponent','bot-role')])
def test_candidate_binding_guards(pair,field,value,message):
    base,candidate=pair
    with closing(BlueprintDB(candidate)) as db:db.set_metadata(field,value)
    with closing(BlueprintDB(candidate,read_only=True,fallback_base_db=base)) as db:
        with pytest.raises(ValueError,match=message):db.validate_policy()


def test_wal_checked_at_physical_base_path(pair,tmp_path):
    base,candidate=pair
    link=tmp_path/'alias.db';link.symlink_to(base)
    Path(str(base)+'-wal').write_bytes(b'nonempty')
    with closing(BlueprintDB(candidate,read_only=True,fallback_base_db=link)) as db:
        with pytest.raises(ValueError,match='WAL'):db.validate_policy()


def test_writable_training_and_ordinary_blueprint_unchanged(pair,monkeypatch):
    base,candidate=pair
    monkeypatch.setenv('ALLIN_DBR_BASE_DB','does-not-exist')
    with closing(BlueprintDB(candidate)) as db:
        assert db.get_average_strategy('empty')=={'fold':.5,'call':.5}
        assert db.load_all_to_memory()['empty'].cumulative_strategy=={}
    with closing(BlueprintDB(base,read_only=True)) as db:
        assert db.get_average_strategy('base_empty')=={'fold':.5,'call':.5}
    with pytest.raises(ValueError,match='read-only'):BlueprintDB(candidate,fallback_base_db=base)


@pytest.mark.parametrize('attribution',[False,True])
def test_evaluation_workers_keep_explicit_base(pair,monkeypatch,attribution):
    """Exercise worker reopening without executing the deferred BR board walk."""
    from src.evaluation import best_response
    base,candidate=pair
    class Probe:
        def __init__(self,db,**kwargs):
            assert db.get_average_strategy('empty')=={'fold':.25,'call':.75}
        def board_contribution(self,*args):return (1.,2.)
    monkeypatch.setattr(best_response,'BestResponseEvaluator',Probe)
    with closing(BlueprintDB(candidate,read_only=True,fallback_base_db=base)) as db:
        path=db.policy_fallback_path
        if attribution:
            result=best_response._attr_board_chunk((str(candidate),'capped_pj',0.,[[]],None,False,path))
            assert result==(1.,2.,{})
        else:
            result=best_response._br_board_chunk((str(candidate),'capped_pj',[[]],0.,path))
            assert result==(1.,2.)
