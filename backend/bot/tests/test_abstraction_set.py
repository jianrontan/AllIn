"""Versioned card abstractions (v3 plan step V3.0): fingerprint, stamps, resolution
and the fail-closed reader guards. See src/abstractions/abstraction_set.py."""
import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.abstractions import abstraction_set as A  # noqa: E402
from src.abstractions.postflop_features import centroid_hash, load_centroids  # noqa: E402
from src.storage.blueprint_db import BlueprintDB  # noqa: E402

REAL_ROOT = A.ROOT


class FakeDB:
    def __init__(self, meta):
        self.meta = meta

    def get_metadata(self, key, default=None):
        return self.meta.get(key, default)


def _copy_centroids(dst):
    os.makedirs(dst, exist_ok=True)
    for street in A.STREETS:
        name = f'postflop_centroids_{street}.npz'
        shutil.copy(os.path.join(REAL_ROOT, name), os.path.join(dst, name))


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """A temporary ROOT holding copies of the v2 centroids, plus a named 'v3'
    abstraction (same centroids, exhaustive flop bake in its manifest)."""
    root = str(tmp_path / 'abstractions')
    _copy_centroids(root)
    v3 = os.path.join(root, 'v3')
    _copy_centroids(v3)
    with open(os.path.join(v3, A.MANIFEST), 'w') as f:
        json.dump({'flop_runouts': 0}, f)
    monkeypatch.setattr(A, 'ROOT', root)
    return root, v3


def _db(path, meta):
    db = BlueprintDB(str(path))
    for k, v in meta.items():
        db.set_metadata(k, v)
    return db


# --------------------------------------------------------------------- fingerprint

def test_root_hashes_to_the_legacy_id():
    assert A.abstraction_id(REAL_ROOT) == A.LEGACY_ABSTRACTION_ID


def test_id_tracks_bake_settings_and_centroids(roots):
    root, v3 = roots
    assert A.abstraction_id(root) == A.LEGACY_ABSTRACTION_ID      # same files as v2
    assert A.abstraction_id(v3) != A.abstraction_id(root)        # flop_runouts 0 vs 200
    c, b = load_centroids('river', v3)
    np.savez(os.path.join(v3, 'postflop_centroids_river.npz'), centroids=c * 0.5, bins=b)
    before = A.abstraction_id(v3)
    np.savez(os.path.join(v3, 'postflop_centroids_river.npz'), centroids=c, bins=b)
    assert A.abstraction_id(v3) != before


@pytest.mark.parametrize('bad', ['../x', '/tmp', 'V3', 'a' * 33, 'v3/..', 5])
def test_bad_names_are_refused(bad):
    with pytest.raises(ValueError):
        A.abstraction_dir(bad)


def test_name_round_trip(roots, tmp_path):
    root, v3 = roots
    assert A.abstraction_dir(None) == root and A.abstraction_dir('') == root
    assert A.abstraction_name(root) is None
    assert A.abstraction_name(v3) == 'v3'
    with pytest.raises(ValueError):
        A.abstraction_name(str(tmp_path))                        # not under ROOT
    with pytest.raises(ValueError):
        A.abstraction_name(os.path.join(v3, 'deeper'))           # not a direct child


@pytest.mark.parametrize('manifest', [{'flop_runouts': -1}, {'flop_runouts': True},
                                      {'flop_runouts': '0'}, {'other': 1}, [0],
                                      {'flop_runouts': 1176}, {'flop_runouts': 5000}])
def test_bad_manifests_are_refused(roots, manifest):
    _, v3 = roots
    with open(os.path.join(v3, A.MANIFEST), 'w') as f:
        json.dump(manifest, f)
    with pytest.raises(ValueError):
        A.read_manifest(v3)


# ---------------------------------------------------------------------- resolution

def test_resolve(roots):
    root, v3 = roots
    assert A.resolve_abstraction_dir(None) == root
    assert A.resolve_abstraction_dir(object()) == root           # non-DB strategy source
    assert A.resolve_abstraction_dir(FakeDB({})) == root         # legacy, unstamped
    assert A.resolve_abstraction_dir(FakeDB(A.stamp_for(root))) == root
    assert A.resolve_abstraction_dir(FakeDB(A.stamp_for(v3))) == v3
    assert A.stamp_for(v3)['abstraction'] == 'v3' and 'abstraction' not in A.stamp_for(root)


@pytest.mark.parametrize('meta', [
    {'abstraction': 'v3', 'abstraction_id': '0' * 40},           # files changed
    {'abstraction': 'v4', 'abstraction_id': '0' * 40},           # missing directory
    {'abstraction': 'v3'},                                       # name without id
    {'abstraction_id': 'not-hex'},                               # malformed id
    {'abstraction': '../etc', 'abstraction_id': '0' * 40},       # traversal
])
def test_resolve_fails_closed(roots, meta):
    with pytest.raises(ValueError):
        A.resolve_abstraction_dir(FakeDB(meta))


def test_legacy_blueprints_refuse_a_changed_root(roots):
    root, _ = roots
    with open(os.path.join(root, A.MANIFEST), 'w') as f:
        json.dump({'flop_runouts': 0}, f)
    with pytest.raises(ValueError, match='changed'):
        A.resolve_abstraction_dir(FakeDB({}))


def test_serving_readers_refuse_a_named_abstraction(roots, tmp_path):
    _, v3 = roots
    assert A.require_root_abstraction(FakeDB({}), 'x') == A.ROOT
    with pytest.raises(ValueError, match='only supports the root'):
        A.require_root_abstraction(FakeDB(A.stamp_for(v3)), 'x')
    from src.game.bot_strategy import BlueprintStrategy
    db = _db(tmp_path / 'bp.db', A.stamp_for(v3))
    with pytest.raises(ValueError, match='only supports the root'):
        BlueprintStrategy(db)
    db.close()


def test_matches_bucket_each_side_with_its_own_abstraction(roots, tmp_path):
    root, v3 = roots
    from src.evaluation.match import HeadToHeadMatch
    from src.evaluation.cross_match import CrossMatch, Sizing
    a = _db(tmp_path / 'a.db', A.stamp_for(v3))
    b = _db(tmp_path / 'b.db', {})
    m = HeadToHeadMatch(a, b, seed=1)
    assert m.pa.cards.postflop.dir == v3 and m.pb.cards.postflop.dir == root
    same = HeadToHeadMatch(b, b, seed=1)
    assert same.pa.cards is same.pb.cards                        # one shared instance
    sizing = Sizing('t', (2.0, 2.5, 3.5), (0.66, 1.0, 1.5), (0.66, 1.0, 1.5),
                    (0.33, 0.66, 1.0))
    x = CrossMatch(a, sizing, b, sizing, seed=1)
    assert x.pa.cards.postflop.dir == v3 and x.pb.cards.postflop.dir == root
    a.close()
    b.close()


def test_lbr_and_best_response_resolve_the_blueprint_abstraction(roots, tmp_path):
    _, v3 = roots
    from src.evaluation.lbr import LBREvaluator
    from src.evaluation.best_response import BestResponseEvaluator
    db = _db(tmp_path / 'a.db', A.stamp_for(v3))
    assert LBREvaluator(db, seed=1).cards.postflop.dir == v3
    assert BestResponseEvaluator(db, seed=1).cards.postflop.dir == v3
    db.close()


def test_auto_resolution_skips_named_abstractions(roots, tmp_path, monkeypatch):
    _, v3 = roots
    from src import config
    monkeypatch.delenv('ALLIN_BLUEPRINT_DB', raising=False)
    d = tmp_path / 'bps'
    d.mkdir()
    _db(d / 'blueprint_plain.db', {'total_iterations': 10}).close()
    _db(d / 'blueprint_v3.db', {'total_iterations': 999, **A.stamp_for(v3)}).close()
    assert config.resolve_blueprint_path(d).name == 'blueprint_plain.db'


# ---------------------------------------------------------------------- tables

def _table(street, adir, **extra):
    c, b = load_centroids(street, adir)
    d = {'centroid_hash': centroid_hash(c, b), 'n_buckets': len(c), 'bins': b}
    d.update(extra)
    return d, c, b


def test_table_stamp_checks_flop_runouts(roots):
    root, _ = roots
    legacy = A.read_manifest(root)
    exhaustive = {'flop_runouts': 0}
    d, c, b = _table('flop', root)                              # pre-field table = 200
    assert A.check_table_stamp('flop', d, c, b, legacy) is True
    with pytest.raises(ValueError, match='flop_runouts'):
        A.check_table_stamp('flop', d, c, b, exhaustive)
    d0, _, _ = _table('flop', root, flop_runouts=0)
    assert A.check_table_stamp('flop', d0, c, b, exhaustive) is True
    with pytest.raises(ValueError, match='flop_runouts'):
        A.check_table_stamp('flop', d0, c, b, legacy)
    assert A.check_table_stamp('flop', {}, c, b, legacy) is False   # unstamped legacy
    dt, ct, bt = _table('turn', root)
    assert A.check_table_stamp('turn', dt, ct, bt, exhaustive) is True  # turn: no runouts


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(__file__), '..', 'scripts', f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bake_save_stamps_runouts_and_verify_tables_reads_them(roots):
    _, v3 = roots
    bake = _load_script('bake_postflop_table')
    bake._save('flop', [3, 1, 2], [0, 1, 2], v3, 0)
    with pytest.raises(ValueError, match='missing postflop_table_turn'):
        A.verify_tables(v3)                                      # named: turn required
    bake._save('turn', [3, 1, 2], [0, 1, 2], v3, 0)
    assert A.verify_tables(v3) == ['flop', 'turn']
    bake._save('flop', [3, 1, 2], [0, 1, 2], v3, 200)
    with pytest.raises(ValueError, match='flop_runouts'):
        A.verify_tables(v3)
    from src.abstractions.postflop_v2 import PostflopV2
    with pytest.raises(ValueError, match='flop_runouts'):
        PostflopV2(abstraction_dir=v3)._table('flop')


def test_named_abstractions_are_strict_about_tables(roots):
    root, v3 = roots
    from src.abstractions.postflop_v2 import PostflopV2
    # Missing flop/turn table: the root falls back lazily, a named abstraction refuses.
    assert PostflopV2(abstraction_dir=root)._table('flop') is None
    with pytest.raises(FileNotFoundError, match='missing'):
        PostflopV2(abstraction_dir=v3)._table('flop')
    assert PostflopV2(abstraction_dir=v3)._table('river') is None    # never baked
    # Unstamped table: the root warns, a named abstraction refuses.
    np.savez(os.path.join(v3, 'postflop_table_turn.npz'),
             ids=np.array([1], np.int64), buckets=np.array([0], np.uint8))
    with pytest.raises(ValueError, match='no centroid stamp'):
        PostflopV2(abstraction_dir=v3)._table('turn')
    shutil.copy(os.path.join(v3, 'postflop_table_turn.npz'),
                os.path.join(root, 'postflop_table_turn.npz'))
    with pytest.warns(UserWarning, match='no centroid stamp'):
        assert PostflopV2(abstraction_dir=root)._table('turn') is not None
    d, c, b = _table('turn', v3)
    del d['centroid_hash']
    with pytest.raises(ValueError, match='no centroid stamp'):
        A.check_table_stamp('turn', d, c, b, A.read_manifest(v3), allow_legacy=False)


def test_bake_runout_flag_only_binds_the_flop(roots, monkeypatch):
    root, _ = roots
    bake = _load_script('bake_postflop_table')
    monkeypatch.setattr(bake, '_canonical_boards', lambda n: [])     # nothing to bake
    monkeypatch.setattr(sys, 'argv', ['bake', '--street', 'turn', '--flop-runouts', '0',
                                      '--limit-boards', '1'])
    bake.main()                                                      # irrelevant: accepted
    monkeypatch.setattr(sys, 'argv', ['bake', '--street', 'flop', '--flop-runouts', '0',
                                      '--limit-boards', '1'])
    with pytest.raises(SystemExit):
        bake.main()                                                  # root manifest says 200


# ---------------------------------------------------------------------- resume

def test_check_resume(roots):
    root, v3 = roots
    A.check_resume(None, None, A.stamp_for(root), 't')           # legacy on unchanged root
    A.check_resume('v3', A.abstraction_id(v3), A.stamp_for(v3), 't')
    with pytest.raises(ValueError, match='resume abstraction mismatch'):
        A.check_resume(None, None, A.stamp_for(v3), 't')
    with pytest.raises(ValueError, match='resume abstraction mismatch'):
        A.check_resume('v3', '0' * 40, A.stamp_for(v3), 't')
    with pytest.raises(ValueError, match='resume abstraction mismatch'):
        A.check_resume('v3', A.abstraction_id(v3), A.stamp_for(root), 't')


def test_python_trainer_refuses_to_resume_a_named_abstraction(roots, tmp_path):
    _, v3 = roots
    from src.cfr.blueprint_trainer import BlueprintTrainer
    db = _db(tmp_path / 'bp.db', {'total_iterations': 1, **A.stamp_for(v3)})
    with pytest.raises(ValueError, match='resume abstraction mismatch'):
        BlueprintTrainer().resume_from_db(db)
    db.close()


def test_train_rust_stamps_and_refuses_cross_abstraction_resume(tmp_path, monkeypatch):
    """End to end through train_rust.main(): a new run is stamped, the stamp survives
    checkpoints, and resuming a blueprint stamped with another abstraction is refused
    before any training."""
    pytest.importorskip('allin_core')
    for street in ('flop', 'turn'):
        if not os.path.exists(os.path.join(REAL_ROOT, f'postflop_table_{street}.npz')):
            pytest.skip('baked root tables not present')
    tr = _load_script('train_rust')
    out = str(tmp_path / 'out')
    monkeypatch.setattr(sys, 'argv', ['train_rust', '--iterations', '2000', '--workers', '1',
                                      '--checkpoint-every', '1000', '--out-dir', out,
                                      '--seed', '1'])
    tr.main()
    (name,) = [f for f in os.listdir(out) if f.endswith('.db')]
    path = os.path.join(out, name)
    assert tr._read_abstraction_stamp(path) == (None, A.LEGACY_ABSTRACTION_ID)
    tr._write_abstraction_stamp(path, {'abstraction': 'v3', 'abstraction_id': '0' * 40})
    monkeypatch.setattr(sys, 'argv', ['train_rust', '--iterations', '1000', '--workers', '1',
                                      '--out-dir', out, '--resume', name])
    with pytest.raises(ValueError, match='resume abstraction mismatch'):
        tr.main()


def test_train_rust_stamp_round_trip(tmp_path):
    pytest.importorskip('allin_core')
    tr = _load_script('train_rust')
    path = str(tmp_path / 'bp.db')
    BlueprintDB(path).close()
    assert tr._read_abstraction_stamp(path) == (None, None)
    tr._write_abstraction_stamp(path, {'abstraction': 'v3', 'abstraction_id': 'a' * 40})
    assert tr._read_abstraction_stamp(path) == ('v3', 'a' * 40)


def test_kmeans_starts_are_distinct():
    """River features are spikes, so a random start can repeat a point; the copy
    then never wins a hand and its bucket stays empty (V3.3's first K=20 fit)."""
    import random
    m = _load_script('compute_postflop_buckets')
    spikes = np.eye(5)[[0, 0, 0, 0, 1, 2, 3, 4] * 25]          # 5 distinct, many repeats
    rng = random.Random(0)
    _, labels = m.kmeans_emd(spikes, 5, 10, rng)
    assert sorted(set(labels.tolist())) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        m.kmeans_emd(spikes, 6, 10, random.Random(0))
    # A start without repeats is kept as is and draws nothing from the RNG, so
    # fits that never hit a repeat (flop/turn) are unchanged.
    feats = np.random.default_rng(1).random((50, 4))
    rng, ref = random.Random(5), random.Random(5)
    assert m._distinct_starts(feats, [3, 9, 27], rng) == [3, 9, 27]
    assert rng.random() == ref.random()
