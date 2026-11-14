"""Conservative identity for resumable offline policy comparisons.

Legacy checkpoints cannot establish this identity and must not be resumed.
Blueprints must remain quiescent during a run; SQLite backup hashes committed
contents, including WAL pages, without changing the source database.
"""
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
from importlib.machinery import EXTENSION_SUFFIXES
import os
from pathlib import Path
import platform
import sqlite3
import sys
import tempfile

import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def blueprint_hash(path):
    with tempfile.TemporaryDirectory(prefix='allin-policy-') as directory:
        target = Path(directory) / 'snapshot.db'
        source = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
        destination = sqlite3.connect(target)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        return file_hash(target)


def policy_identity(db_path, abstraction_dir):
    bot = Path(__file__).resolve().parents[2]
    repo = bot.parents[1]
    files = list((bot / 'src').rglob('*.py'))
    files += list((bot / 'scripts').glob('*.py'))
    files += [bot / 'tests/run_maniac_live.py']  # imported match-record helper dependency
    native_root = repo / 'backend/core-rs'
    files += list((native_root / 'src').rglob('*.rs'))
    files += [native_root / name for name in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml')]
    sources = {str(p.relative_to(repo)): file_hash(p) for p in sorted(files)}
    asset_dir = Path(abstraction_dir)
    asset_files = list(asset_dir.glob('*.npz')) + list(asset_dir.glob('*.json'))
    if os.environ.get('ALLIN_MMAP_POSTFLOP') == '1':
        # Materialize the same mmap cache the policy will use before hashing it;
        # otherwise first-use extraction would change the identity on resume.
        from ..abstractions.postflop_v2 import PostflopV2
        tables = PostflopV2(abstraction_dir=str(asset_dir))
        for street in ('flop', 'turn'):
            if (asset_dir / f'postflop_table_{street}.npz').exists():
                tables._table(street)
        asset_files += list(asset_dir.glob('postflop_table_*.npy'))
    assets = {p.name: file_hash(p) for p in sorted(asset_files)}
    settings = {k: v for k, v in os.environ.items() if k.startswith('ALLIN_')}
    for key, value in {'ALLIN_RIVER_BACKEND': 'python', 'ALLIN_RIVER_ALGORITHM': 'cfrplus',
                       'ALLIN_SUBMINIMUM_ANCHOR': '0', 'ALLIN_RIVER_MENU': 'A'}.items():
        settings.setdefault(key, value)
    native = {}
    if settings['ALLIN_RIVER_BACKEND'] == 'rust':
        importlib.import_module('allin_core')
        for name, module in list(sys.modules.items()):
            path = getattr(module, '__file__', None)
            if name.startswith('allin_core') and path and any(str(path).endswith(s) for s in EXTENSION_SUFFIXES):
                native[name] = file_hash(path)
        if not native:
            raise RuntimeError('Cannot identify the loaded Rust extension binary')
    dependencies = {}
    for package in ('numpy', 'phevaluator', 'PyPokerEngine'):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    return {'version': 1, 'sources': sources, 'abstractions': assets,
            'dependencies': dependencies, 'implementation': platform.python_implementation(),
            'machine': platform.machine(),
            'blueprint_sha256': blueprint_hash(db_path),
            'settings': {k: settings[k] for k in ('ALLIN_RIVER_BACKEND', 'ALLIN_RIVER_ALGORITHM',
                                                  'ALLIN_SUBMINIMUM_ANCHOR', 'ALLIN_RIVER_MENU')},
            'environment_sha256': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in settings.items()},
            'native_binary': native, 'python': platform.python_version(), 'numpy': np.__version__}
