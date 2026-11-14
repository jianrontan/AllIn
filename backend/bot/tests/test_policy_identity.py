"""Policy identity includes committed WAL data, assets, settings and native code."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.policy_identity import blueprint_hash, policy_identity


def test_blueprint_identity_is_stable_and_includes_committed_wal(tmp_path):
    path = tmp_path / 'blueprint.db'
    with sqlite3.connect(path) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE policy (value INTEGER)')
        db.execute('INSERT INTO policy VALUES (1)')
        db.commit()
        before = blueprint_hash(path)
        assert blueprint_hash(path) == before
        db.execute('UPDATE policy SET value=2')
        db.commit()
        assert blueprint_hash(path) != before


def test_identity_covers_policy_sources_settings_and_assets(tmp_path, monkeypatch):
    monkeypatch.setenv('ALLIN_MMAP_POSTFLOP', '0')
    monkeypatch.setenv('ALLIN_RIVER_BACKEND', 'python')
    monkeypatch.setenv('ALLIN_SUBMINIMUM_ANCHOR', '0')
    monkeypatch.setenv('ALLIN_TEST_SECRET', 'do-not-record-plaintext')
    path = tmp_path / 'blueprint.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE policy (value INTEGER)')
    assets = tmp_path / 'assets'
    assets.mkdir()
    table = assets / 'postflop_centroids_river.npz'
    table.write_bytes(b'centroids-v1')
    before = policy_identity(path, assets)
    for name in ('game/bot_strategy.py', 'cfr/translation.py', 'subgame/solve_control.py',
                 'subgame/rust_river_cfr.py', 'subgame/river_cfr.py'):
        assert 'backend/bot/src/' + name in before['sources']
    assert 'backend/core-rs/src/river/cfr.rs' in before['sources']
    assert 'backend/bot/scripts/river_menu_match.py' in before['sources']
    assert 'ALLIN_TEST_SECRET' not in before['settings']
    assert 'do-not-record-plaintext' not in str(before)
    monkeypatch.setenv('ALLIN_SUBMINIMUM_ANCHOR', '1')
    changed = policy_identity(path, assets)
    assert changed['environment_sha256'] != before['environment_sha256']
    table.write_bytes(b'centroids-v2')
    assert policy_identity(path, assets)['abstractions'] != changed['abstractions']


def test_loaded_native_binary_is_fingerprinted(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import pytest
    pytest.importorskip('allin_core')
    monkeypatch.setenv('ALLIN_RIVER_BACKEND', 'rust')
    binary = tmp_path / 'test_extension.so'
    binary.write_bytes(b'binary-v1')
    monkeypatch.setitem(sys.modules, 'allin_core.review_binary', SimpleNamespace(__file__=str(binary)))
    path = tmp_path / 'blueprint.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE policy (value INTEGER)')
    before = policy_identity(path, tmp_path)['native_binary']
    binary.write_bytes(b'binary-v2')
    after = policy_identity(path, tmp_path)['native_binary']
    assert after['allin_core.review_binary'] != before['allin_core.review_binary']


def test_flag_change_cannot_resume_generated_identity(tmp_path, monkeypatch):
    import importlib.util
    import json
    import pytest
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('identity_resume_test',scripts/'river_menu_match.py')
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setenv('ALLIN_RIVER_BACKEND','python')
    monkeypatch.setenv('ALLIN_MMAP_POSTFLOP','0')
    monkeypatch.setenv('ALLIN_SUBMINIMUM_ANCHOR','0')
    db_path=tmp_path/'blueprint.db'
    with sqlite3.connect(db_path) as db:
        db.execute('CREATE TABLE policy (value INTEGER)')
    identity=policy_identity(db_path,tmp_path)
    config=dict(db=str(db_path),a='F_EXTENDED',b='A',hands=10,iters=275,budget=24.,
                seed=101,depth_a=None,depth_b=None)
    path=tmp_path/'checkpoint.json'
    path.write_text(json.dumps(dict(config=config,source_sha256={},policy_identity=identity,
                                    completed=0,records=[])))
    runner.load_checkpoint(path,config,{},identity)
    monkeypatch.setenv('ALLIN_SUBMINIMUM_ANCHOR','1')
    with pytest.raises(ValueError,match='policy identity'):
        runner.load_checkpoint(path,config,{},policy_identity(db_path,tmp_path))


def test_mmap_cache_first_use_is_stable_and_contents_are_identified(tmp_path, monkeypatch):
    import numpy as np
    from src.abstractions.postflop_v2 import PostflopV2
    monkeypatch.setenv('ALLIN_RIVER_BACKEND','python')
    monkeypatch.setenv('ALLIN_MMAP_POSTFLOP','1')
    # Stamp validation has its own tests; exercise real cache extraction/loading here.
    monkeypatch.setattr(PostflopV2,'_verify_stamp',lambda *a: None)
    path=tmp_path/'blueprint.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE policy (value INTEGER)')
    assets=tmp_path/'assets'
    assets.mkdir()
    np.savez(assets/'postflop_table_flop.npz',ids=np.array([1,2]),buckets=np.array([0,1]))
    before=policy_identity(path,assets)
    assert policy_identity(path,assets)==before
    cache=assets/'postflop_table_flop.buckets.npy'
    np.save(cache,np.array([1,0]))
    assert policy_identity(path,assets)['abstractions']!=before['abstractions']
