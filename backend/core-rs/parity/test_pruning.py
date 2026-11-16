"""Pruning checkpoint contract; small fixture runs, no full training jobs."""
import json
import sqlite3
from pathlib import Path
import pytest
core = pytest.importorskip('allin_core')


def _rows(path, table):
    with sqlite3.connect(path) as db:
        return db.execute(f'SELECT * FROM {table} ORDER BY key').fetchall()


def test_pruning_resume_and_serving_snapshot(tmp_path):
    fixture = Path(__file__).parents[1] / 'fixtures/mccfr_capped_pj_spr5a_g1.json'
    if not fixture.exists():
        pytest.skip('generate mccfr_spr5a fixture first')
    data = json.loads(fixture.read_text())
    rows = (data['deals'] * 2)[:100]
    stream = data['uniforms'] * 50
    kw = dict(key_format='spr5a', prune=True, prune_c=1e-8, prune_min_visits=1)
    def run(t, rs, uniforms, start=0):
        return t.run_fixture([r['cards'] for r in rs], [r['fine'] for r in rs],
                             [r['post'] for r in rs], [r['raw'] for r in rs],
                             data['fine_to_coarse'], uniforms, start_iteration=start)
    full = core.Trainer('capped_pj', **kw)
    run(full, rows, stream)
    full_path = str(tmp_path/'full.db')
    full.checkpoint(full_path,99)
    a = core.Trainer('capped_pj', **kw)
    used = run(a,rows[:50],stream)
    path = str(tmp_path/'resume.db')
    a.checkpoint(path,49)
    with pytest.raises(ValueError,match='pruning mismatch'):
        core.Trainer('capped_pj',key_format='spr5a').resume(path)
    with pytest.raises(ValueError,match='pruning mismatch'):
        core.Trainer('capped_pj',**dict(kw,prune_c=.03)).resume(path)
    b = core.Trainer('capped_pj',**kw)
    assert b.resume(path) == 50
    run(b,rows[50:],stream[used:],start=50)
    b.checkpoint(path,99)
    assert _rows(path,'info_sets') == _rows(full_path,'info_sets')
    assert _rows(path,'pruning_state') == _rows(full_path,'pruning_state')
    # Production snapshots deliberately lack resumable training-only state.
    import sys
    sys.path.insert(0,str(Path(__file__).parents[2]/'bot/scripts'))
    from train_rust import _snapshot
    snap = _snapshot(path,100)
    with sqlite3.connect(snap) as db:
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='pruning_state'").fetchall()
    with pytest.raises(ValueError,match='serving snapshot'):
        core.Trainer('capped_pj',**kw).resume(snap)
