"""Frozen human-model opponents preserve their prior's keys and menu."""
import json
import random
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

BOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BOT))
from scripts import h2h_eval
from scripts.h2h_eval import ModelPlayer
from src.storage.blueprint_db import BlueprintDB
from src.exploitation.opponent_model import HumanModel,blueprint_model_stamp


class Prior:
    def get_metadata(self,key,default=None):
        return {'key_format':'spr5a','menu_mode':'capped_pj'}.get(key,default)


class Cards:
    def get_bucket(self,hand,board):
        return 'pf_160' if board is None else 3


def test_frozen_model_keeps_history_keys_and_prior_menu():
    prior=Prior();seen=[]
    def model(key,legal):
        seen.append(key)
        return np.ones(len(legal))/len(legal)
    player=ModelPlayer(prior,Cards(),random.Random(7),model,menu_mode='capped_pj')
    player.street_patterns=['sc','kmc']
    _,info=player.act(0,['HA','DA'],['S2','C3','D4','H5'],2,
                     ['check','bet_small'],'',20,pot=20,to_call=0,stack=190)
    assert seen==[info['key']]
    assert '_turn_r' in seen[0] and '_tii_' in seen[0]
    assert player.src is prior
    assert player.game.preflop_cap_jam and player._capped.preflop_cap_jam
    assert player.game.voluntary_allin and not player._capped.voluntary_allin


def test_legacy_model_menu_stays_capped_by_default():
    player=ModelPlayer(Prior(),Cards(),random.Random(7),lambda k,a:np.ones(len(a)))
    assert player.menu_mode=='capped'
    assert not player.game.preflop_cap_jam


def test_cli_rejects_model_prior_for_blueprint_opponent(tmp_path):
    result=subprocess.run([sys.executable,str(BOT/'scripts/h2h_eval.py'),
                           '--db-a','unused','--db-b','unused',
                           '--model-base','unused','--out',str(tmp_path/'out.json')],
                          capture_output=True,text=True,timeout=10)
    assert result.returncode==2
    assert '--model-base requires --db-b model' in result.stderr
    assert not (tmp_path/'out.json').exists()



@pytest.mark.parametrize('extra',[[],['--model-dir','x'],['--model-base','x']])
def test_cli_requires_explicit_held_out_inputs_for_dbr_candidate(tmp_path,extra):
    candidate=tmp_path/'dbr_bot.db'
    db=BlueprintDB(str(candidate));db.set_metadata('dbr_role','bot');db.close()
    result=subprocess.run([sys.executable,str(BOT/'scripts/h2h_eval.py'),
                           '--db-a',str(candidate),'--db-b','model',*extra,
                           '--out',str(tmp_path/'out.json')],
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==2
    assert 'needs explicit --model-base' in result.stderr
    assert not (tmp_path/'out.json').exists()


@pytest.mark.parametrize('change',[None,'counts','prior_wal'])
@pytest.mark.parametrize('symlink_prior',[False,True])
def test_cli_binds_frozen_prior_and_discards_changed_inputs(tmp_path,monkeypatch,change,symlink_prior):
    candidate=tmp_path/'candidate.db';prior=tmp_path/'frozen.db'
    for path in (candidate,prior):
        db=BlueprintDB(str(path));db.set_metadata('key_format','spr5a')
        db.set_metadata('menu_mode','capped_pj');db.close()
    db=BlueprintDB(str(prior),read_only=True);stamp=blueprint_model_stamp(db);db.close()
    supplied_prior=prior
    if symlink_prior:
        supplied_prior=tmp_path/'prior-alias.db'
        supplied_prior.symlink_to(prior)
    counts=tmp_path/'model_population.json'
    counts.write_text(json.dumps({'stamp':stamp,'counts':{}}))
    output=tmp_path/'result.json';opened=[];writers=[]
    def open_db(*a,**kw):
        db=BlueprintDB(*a,**kw);opened.append(db);return db
    def human_model(folder,db):
        assert Path(db.db_path).resolve()==prior.resolve()
        return HumanModel(folder,db)
    def evaluate(**kwargs):
        if change=='counts':counts.write_text(counts.read_text()+' ')
        if change=='prior_wal':
            before=h2h_eval.content_sha256(prior)
            writer=sqlite3.connect(prior);writers.append(writer)
            writer.execute("INSERT INTO training_metadata VALUES ('mutation','true')");writer.commit()
            # A main-file hash alone cannot detect an active WAL writer.
            assert h2h_eval.content_sha256(prior)==before
            if symlink_prior:
                assert not Path(str(supplied_prior)+'-wal').exists()
                assert Path(str(prior)+'-wal').stat().st_size>0
    def match(a,b,**kwargs):
        assert Path(a.db_path)==candidate and Path(b.db_path).resolve()==prior.resolve()
        return SimpleNamespace(pa=SimpleNamespace(cards=Cards()),pb=SimpleNamespace(cards=Cards()),
                               rng=random.Random(7),evaluate=evaluate,deal_rng_scheme='legacy-shared')
    estimate={k:0.0 for k in ('aivat_mbb','aivat_stderr_mbb','raw_mbb','raw_stderr_mbb','var_reduction')}
    monkeypatch.setattr(h2h_eval,'BlueprintDB',open_db)
    monkeypatch.setattr(h2h_eval,'HeadToHeadMatch',match)
    monkeypatch.setattr(h2h_eval,'AIVATEstimator',lambda *a,**kw:SimpleNamespace(estimate=lambda rec:estimate))
    monkeypatch.setattr('src.exploitation.opponent_model.HumanModel',human_model)
    monkeypatch.setattr(sys,'argv',['h2h_eval.py','--db-a',str(candidate),'--db-b','model',
                        '--model-base',str(supplied_prior),'--model-dir',str(tmp_path),'--hands','1','--out',str(output)])
    try:
        if change:
            with pytest.raises(ValueError,match='result discarded'):h2h_eval.main()
            assert not output.exists()
        else:
            h2h_eval.main();result=json.loads(output.read_text())
            assert result['deal_rng_scheme']=='legacy-shared'
            assert result['population_model']['prior']==str(prior.resolve())
            assert result['population_model']['prior_sha256']==h2h_eval.content_sha256(prior)
    finally:
        for writer in writers:writer.close()
        for db in opened:db.close()
