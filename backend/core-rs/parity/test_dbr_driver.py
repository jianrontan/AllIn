"""Driver restart, writer handoff and SQLite lifecycle contracts on tiny fixtures."""
from contextlib import closing
import fcntl
import json
from pathlib import Path
import sqlite3
import sys

import pytest

from test_dbr import base_and_cache, data, fork, run as run_fixture
import train_dbr as driver


def run_config(base, cache, strength=10.0):
    db = driver.BlueprintDB(str(base), read_only=True)
    try:
        alpha, gamma = db.get_metadata('alpha'), db.get_metadata('gamma')
    finally:
        db.close()
    return {
        'version': 2, 'fork_recipe': driver.PRESERVE_REGRET_RECIPE,
        'base_sha256': json.loads(cache.read_text())['base_sha256'],
        'cache_sha256': driver.file_sha256(cache),
        'seed': 711, 'workers': 2, 'merge_every': 8, 'checkpoint_every': 16,
        'key_format': 'spr5a', 'menu_mode': 'capped_pj',
        'alpha': alpha, 'gamma': gamma, 'p_max': .5, 'strength': strength,
    }


def arguments(base, cache, root, *, resume=True, explicit_strength=False):
    args = [
        'train_dbr.py', '--base', str(base), '--cache', str(cache),
        '--run-dir', str(root), '--iterations', '1', '--seed', '711',
        '--workers', '2', '--merge-every', '8', '--checkpoint-every', '16',
    ]
    if resume:
        args.append('--resume')
    if explicit_strength:
        args += ['--strength', '10']
    return args


@pytest.mark.parametrize('old_strength', [10, 10.0], ids=['old-int', 'old-float'])
@pytest.mark.parametrize('explicit_strength', [False, True], ids=['default', 'explicit'])
def test_resume_accepts_equivalent_strength_spellings(
        base_and_cache, tmp_path, monkeypatch, old_strength, explicit_strength):
    import allin_core
    base, cache = base_and_cache
    root = tmp_path / 'run'
    driver.publish(fork(base, cache), root, 0, run_config(base, cache, old_strength))
    monkeypatch.setattr(sys, 'argv', arguments(base, cache, root, explicit_strength=explicit_strength))
    driver.main()
    state = driver.load_pair(root)
    assert state['iterations'] == 1
    assert type(state['run']['strength']) is float
    assert state['run']['strength'] == 10.0
    # The canonical continuation itself remains resumable by the exact native
    # metadata check, rather than merely bypassing validation of the old pair.
    config = state['run']
    trainer = allin_core.Trainer('capped_pj', config['alpha'], config['gamma'], key_format='spr5a')
    assert trainer.resume_dbr(
        str(root / state['bot']['path']), str(root / state['opponent']['path']),
        str(cache), .5, 10.0, json.dumps(config, sort_keys=True)) == 1


@pytest.mark.parametrize('explicit_strength', [False, True], ids=['default', 'explicit'])
def test_new_run_records_canonical_strength(base_and_cache, tmp_path, monkeypatch, explicit_strength):
    base, cache = base_and_cache
    root = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', arguments(base, cache, root, resume=False,
                                              explicit_strength=explicit_strength))
    driver.main()
    assert type(driver.load_pair(root)['run']['strength']) is float


@pytest.mark.parametrize('field,value', [
    ('version', 1.0), ('seed', 711.0), ('seed', True), ('seed', -1), ('seed', 2**64),
    ('workers', 2.0), ('workers', True), ('merge_every', 8.0), ('checkpoint_every', 16.0),
])
def test_resume_rejects_invalid_integer_configuration(
        base_and_cache, tmp_path, monkeypatch, field, value):
    base, cache = base_and_cache
    root = tmp_path / 'run'
    config = run_config(base, cache)
    config[field] = value
    driver.publish(fork(base, cache), root, 0, config)
    before = (root / 'latest.json').read_bytes()
    monkeypatch.setattr(sys, 'argv', arguments(base, cache, root))
    with pytest.raises(ValueError, match=field):
        driver.main()
    assert (root / 'latest.json').read_bytes() == before


def test_resume_reads_latest_after_writer_finishes_during_lock_handoff(
        base_and_cache, tmp_path, monkeypatch):
    base, cache = base_and_cache
    root = tmp_path / 'run'
    trainer = fork(base, cache)
    fixture = data()
    stream = fixture['uniforms'] * 100
    used = run_fixture(trainer, fixture['deals'][:16], stream)
    config = run_config(base, cache)
    first = driver.publish(trainer, root, 16, config)
    original_flock = fcntl.flock
    newer = {}
    with open(root / '.run.lock', 'a+') as writer:
        original_flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def finish_writer_before_acquiring_resume_lock(fd, operation):
            if operation == (fcntl.LOCK_EX | fcntl.LOCK_NB) and not newer:
                # The old driver had already read generation 16 at this point.
                # Finish the active writer before the resumer acquires its lock.
                run_fixture(trainer, fixture['deals'][16:32], stream[used:], start=16)
                newer.update(driver.publish(trainer, root, 32, config, first))
                original_flock(writer, fcntl.LOCK_UN)
            return original_flock(fd, operation)

        monkeypatch.setattr(driver.fcntl, 'flock', finish_writer_before_acquiring_resume_lock)
        monkeypatch.setattr(sys, 'argv', arguments(base, cache, root))
        driver.main()
    state = driver.load_pair(root)
    assert state['iterations'] == 33
    assert (root / 'checkpoints' / newer['generation']).is_dir()


def test_locked_writer_is_rejected_before_reading_a_manifest(base_and_cache, tmp_path, monkeypatch):
    base, cache = base_and_cache
    root = tmp_path / 'run'
    driver.publish(fork(base, cache), root, 0, run_config(base, cache))
    with open(root / '.run.lock', 'a+') as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(driver, 'load_pair', lambda root: pytest.fail('read manifest before locking'))
        monkeypatch.setattr(sys, 'argv', arguments(base, cache, root))
        with pytest.raises(ValueError, match='another process owns'):
            driver.main()


def test_publication_and_backup_close_all_python_sqlite_connections(
        base_and_cache, tmp_path, monkeypatch):
    base, cache = base_and_cache
    trainer = fork(base, cache)
    config = run_config(base, cache)
    real_connect = sqlite3.connect
    connections = []

    class Connection(sqlite3.Connection):
        closed = False

        def close(self):
            self.closed = True
            super().close()

    def connect(*args, **kwargs):
        connection = real_connect(*args, factory=Connection, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(driver.sqlite3, 'connect', connect)
    root = tmp_path / 'run'
    state = driver.publish(trainer, root, 0, config)
    driver.publish(trainer, root, 0, config, state)
    assert len(connections) == 12
    assert all(connection.closed for connection in connections)


@pytest.mark.parametrize('component', ['driver', 'exporter'])
def test_symlinked_base_rejects_its_real_active_wal(tmp_path, monkeypatch, component):
    import export_dbr_counts as exporter
    base = tmp_path / 'base.db'
    db = driver.BlueprintDB(str(base))
    db.set_metadata('total_iterations', 20)
    db.close()
    alias = tmp_path / 'alias.db'
    alias.symlink_to(base)
    cache = tmp_path / 'cache.json'
    cache.write_text('{}')
    output = tmp_path / 'output'
    with closing(sqlite3.connect(base)) as writer:
        writer.execute("INSERT INTO training_metadata VALUES ('mutation','true')")
        writer.commit()
        assert Path(str(base) + '-wal').stat().st_size > 0
        assert not Path(str(alias) + '-wal').exists()
        # Neither component should hash/read a supposedly frozen base after
        # overlooking an active WAL under the symlink target's name.
        module = driver if component == 'driver' else exporter
        monkeypatch.setattr(module, 'file_sha256', lambda path: pytest.fail('hashed an active WAL base'))
        with pytest.raises(ValueError, match='quiescent snapshot'):
            if component == 'driver':
                monkeypatch.setattr(sys, 'argv', arguments(alias, cache, output, resume=False))
                driver.main()
            else:
                exporter.export_counts(alias, tmp_path / 'models', output)
    assert not output.exists()


def legacy_pair(root, state):
    """Emulate the former on-disk recipe, without changing the source base."""
    state=json.loads(json.dumps(state))
    state['run'].pop('fork_recipe',None);state['run']['version']=1
    for role in ('bot','opponent'):
        path=root/state[role]['path']
        db=sqlite3.connect(path)
        try:
            config=json.loads(db.execute("SELECT value FROM training_metadata WHERE key='dbr'").fetchone()[0])
            config['version']=1;config.pop('fork_recipe',None)
            db.execute("UPDATE training_metadata SET value=? WHERE key='dbr'",(json.dumps(config),))
            db.execute("UPDATE training_metadata SET value=? WHERE key='dbr_run'",(json.dumps(state['run']),))
            db.execute('UPDATE info_sets SET visit_count=0')
            db.commit()
        finally:db.close()
        state[role]['sha256']=driver.file_sha256(path)
    driver.atomic_json(root/'latest.json',state)
    driver.atomic_json(root/'checkpoints'/state['generation']/'pair.json',state)
    return state


def test_old_reset_clock_runs_resume_as_explicitly_legacy(
        base_and_cache,tmp_path,monkeypatch,capsys):
    base,cache=base_and_cache;root=tmp_path/'legacy'
    state=driver.publish(fork(base,cache),root,0,run_config(base,cache))
    legacy_pair(root,state)
    monkeypatch.setattr(sys,'argv',arguments(base,cache,root))
    driver.main()
    result=driver.load_pair(root)
    assert result['run']['version']==1 and 'fork_recipe' not in result['run']
    db=sqlite3.connect(root/result['bot']['path'])
    try:config=json.loads(db.execute("SELECT value FROM training_metadata WHERE key='dbr'").fetchone()[0])
    finally:db.close()
    assert config['version']==1 and 'fork_recipe' not in config
    assert 'legacy reset-clock experiment' in capsys.readouterr().out


def test_manifest_cannot_relabel_legacy_role_tables_as_corrected(
        base_and_cache,tmp_path,monkeypatch):
    base,cache=base_and_cache;root=tmp_path/'mismatch'
    state=driver.publish(fork(base,cache),root,0,run_config(base,cache))
    legacy=legacy_pair(root,state)
    # Claim the new recipe in both run records while retaining legacy native data.
    new_run=run_config(base,cache)
    legacy['run']=new_run
    for role in ('bot','opponent'):
        path=root/legacy[role]['path'];db=sqlite3.connect(path)
        try:
            db.execute("UPDATE training_metadata SET value=? WHERE key='dbr_run'",(json.dumps(new_run),));db.commit()
        finally:db.close()
        legacy[role]['sha256']=driver.file_sha256(path)
    driver.atomic_json(root/'latest.json',legacy)
    before=(root/'latest.json').read_bytes()
    monkeypatch.setattr(sys,'argv',arguments(base,cache,root))
    with pytest.raises(ValueError,match='fork recipe mismatch'):driver.main()
    assert (root/'latest.json').read_bytes()==before
