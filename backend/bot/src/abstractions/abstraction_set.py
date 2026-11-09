"""
Versioned card abstractions (v3 plan, step V3.0 -- docs/V3_BLUEPRINT_PLAN.md).

An ABSTRACTION is one directory of postflop bucket files: the per-street centroids
(`postflop_centroids_{flop,turn,river}.npz`, committed) and the baked lookup tables
(`postflop_table_{flop,turn}.npz`, git-ignored, rebuilt by
scripts/bake_postflop_table.py), plus an optional `abstraction.json` manifest of
the bake settings.

  * The ROOT directory (`analysis/abstractions/`) is the v2 abstraction the served
    blueprint was trained on. It has no manifest (legacy defaults).
  * A NAMED abstraction is a direct subdirectory, e.g. `analysis/abstractions/v3/`.

Every blueprint the Rust CLI trains is stamped with an `abstraction_id`: a hash of
the three centroid files and the manifest's bake settings. A named abstraction is
also stamped with its `abstraction` name. A blueprint with no stamp is a legacy
blueprint and means "the root directory, as it was when v2 was trained", so the
root must still hash to LEGACY_ABSTRACTION_ID.

Readers resolve a blueprint's directory with `resolve_abstraction_dir` and fail
closed on any mismatch: a blueprint must never be bucketed with files it was not
trained on. That would happen silently otherwise -- a re-baked table keeps the
centroid stamp, and a different river K just yields bucket ids the blueprint never
trained.
"""
import hashlib
import json
import os
import re

from .postflop_features import _ABSTRACTIONS_DIR, centroid_hash, load_centroids

ROOT = _ABSTRACTIONS_DIR
STREETS = ('flop', 'turn', 'river')
MANIFEST = 'abstraction.json'
# Bake settings a manifest may set, with the legacy values the root was built with.
# The v2 flop table was baked from 200 sampled turn+river runouts per board.
MANIFEST_DEFAULTS = {'flop_runouts': 200}
FLOP_RUNOUTS_ALL = 1176          # C(49, 2) turn+river runouts on a flop
# abstraction_id(ROOT) for the v2 abstraction (30/24/10 centroids, 200-runout flop
# bake). Unstamped blueprints were trained on exactly this, so they refuse to load
# if the root ever stops hashing to it.
LEGACY_ABSTRACTION_ID = '742b8dd454cbd7294f747a716db4aa36524ca1e7'

# Streets whose baked table a named abstraction must ship (the river is never baked).
REQUIRED_TABLES = ('flop', 'turn')

_NAME_RE = re.compile(r'^[a-z0-9_]{1,32}$')
_ID_RE = re.compile(r'^[0-9a-f]{40}$')


def abstraction_dir(name):
    """Directory of an abstraction name (None or '' = the root). Names are a
    restricted token so a stamp read from a blueprint can never point outside
    ROOT."""
    if name is None or name == '':
        return ROOT
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f"invalid abstraction name {name!r} (want [a-z0-9_]{{1,32}})")
    return os.path.join(ROOT, name)


def abstraction_name(adir):
    """Inverse of abstraction_dir: None for the root, else the subdirectory name.
    Refuses any other directory, so every stamped blueprint stays resolvable."""
    real, root = os.path.realpath(adir), os.path.realpath(ROOT)
    if real == root:
        return None
    base = os.path.basename(real)
    if os.path.dirname(real) == root and _NAME_RE.match(base):
        return base
    raise ValueError(
        f"abstraction directory {adir!r} must be {ROOT} or a direct subdirectory "
        f"named [a-z0-9_]{{1,32}}")


def read_manifest(adir):
    """The bake settings of an abstraction directory: MANIFEST_DEFAULTS overlaid
    with its abstraction.json, if any. Unknown keys and bad values are errors."""
    m = dict(MANIFEST_DEFAULTS)
    path = os.path.join(adir, MANIFEST)
    if not os.path.exists(path):
        return m
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")
    unknown = set(data) - set(MANIFEST_DEFAULTS)
    if unknown:
        raise ValueError(f"{path}: unknown manifest keys {sorted(unknown)}")
    fr = data.get('flop_runouts', m['flop_runouts'])
    if type(fr) is not int or not 0 <= fr < FLOP_RUNOUTS_ALL:
        # >= 1176 would bake the same exhaustive table as 0 under a different id.
        raise ValueError(f"{path}: flop_runouts must be an int in [0, {FLOP_RUNOUTS_ALL}) "
                         f"(0 = all {FLOP_RUNOUTS_ALL}), got {fr!r}")
    m['flop_runouts'] = fr
    return m


def abstraction_id(adir):
    """Fingerprint of an abstraction: the three centroid hashes plus the manifest's
    bake settings. Independent of whether the (git-ignored) tables are present; the
    tables are checked against the same settings when they load (check_table_stamp)."""
    m = read_manifest(adir)
    h = hashlib.sha1(b'allin-abstraction-v1')
    for street in STREETS:
        c, b = load_centroids(street, adir)
        h.update(f'|{street}:{centroid_hash(c, b)}'.encode())
    h.update(f"|flop_runouts={m['flop_runouts']}".encode())
    return h.hexdigest()


def stamp_for(adir):
    """Metadata to write into a blueprint trained on `adir`."""
    name = abstraction_name(adir)
    meta = {'abstraction_id': abstraction_id(adir)}
    if name is not None:
        meta['abstraction'] = name
    return meta


def _read_stamp(blueprint_db):
    name = blueprint_db.get_metadata('abstraction', None)
    sid = blueprint_db.get_metadata('abstraction_id', None)
    abstraction_dir(name)                                   # validates the name
    if sid is not None and not (isinstance(sid, str) and _ID_RE.match(sid)):
        raise ValueError(f"blueprint has a malformed abstraction_id {sid!r}")
    if name not in (None, '') and sid is None:
        raise ValueError(f"blueprint names abstraction {name!r} but has no abstraction_id")
    return (name or None), sid


def resolve_abstraction_dir(blueprint_db, who='reader'):
    """The verified abstraction directory a blueprint must be bucketed with.
    `blueprint_db` may be None or a non-DB strategy source: those get the root."""
    if blueprint_db is None or not hasattr(blueprint_db, 'get_metadata'):
        return ROOT
    name, sid = _read_stamp(blueprint_db)
    adir = abstraction_dir(name)
    if not os.path.isdir(adir):
        raise ValueError(f"{who}: the blueprint's abstraction {name!r} is missing ({adir})")
    want = sid if sid is not None else LEGACY_ABSTRACTION_ID   # legacy = the v2 root
    got = abstraction_id(adir)
    if got != want:
        raise ValueError(
            f"{who}: abstraction {name or 'root (v2)'!r} has changed since the blueprint "
            f"was trained (abstraction_id {got[:12]}... != stamped {want[:12]}...). "
            f"Its centroids or bake settings differ; restore them or retrain.")
    return adir


def is_root(adir):
    """True for the root (v2) abstraction directory."""
    return os.path.realpath(adir) == os.path.realpath(ROOT)


def require_root_abstraction(blueprint_db, who):
    """Fail closed for readers that can only bucket with the root (v2) abstraction
    -- the serving path until v3 serving (V3.6) lands."""
    adir = resolve_abstraction_dir(blueprint_db, who)
    if not is_root(adir):
        raise ValueError(
            f"{who} only supports the root (v2) abstraction, but this blueprint was "
            f"trained on {abstraction_name(adir)!r}")
    return adir


def check_table_stamp(street, d, centroids, bins, manifest, allow_legacy=True):
    """Verify a loaded table (an NpzFile or dict) was baked from these centroids and
    the manifest's bake settings. Returns True when verified and False for a legacy
    table with no stamp (the caller warns) -- allowed only when `allow_legacy`
    (the root). Raises on any mismatch, and on an unstamped table otherwise."""
    files = getattr(d, 'files', None)
    if files is None:
        files = list(d.keys())
    if 'centroid_hash' not in files:
        if allow_legacy:
            return False
        raise ValueError(
            f"postflop_table_{street}.npz has no centroid stamp; a named abstraction "
            f"needs stamped tables -- re-bake it with scripts/bake_postflop_table.py")
    # The bake checkpoints a partial table as it goes (complete=0) and stamps the
    # final save complete=1. Tables baked before the field existed are complete.
    if 'complete' in files and not int(d['complete']):
        raise ValueError(
            f"postflop_table_{street}.npz is a partial checkpoint from an interrupted "
            f"bake; re-run python scripts/bake_postflop_table.py --street {street}")
    want = centroid_hash(centroids, bins)
    got = str(d['centroid_hash'])
    if got != want:
        raise ValueError(
            f"postflop_table_{street}.npz was baked from DIFFERENT centroids "
            f"(stamp {got[:8]}... != current {want[:8]}...). The table is stale; "
            f"re-bake it: python scripts/bake_postflop_table.py --street {street}")
    nb, tb = int(d['n_buckets']), int(d['bins'])
    if nb != len(centroids) or tb != bins:
        raise ValueError(
            f"postflop_table_{street}.npz K/bins ({nb}/{tb}) != current centroids "
            f"({len(centroids)}/{bins}); re-bake --street {street}.")
    if street == 'flop':
        # Tables baked before the field existed used the legacy 200.
        runs = int(d['flop_runouts']) if 'flop_runouts' in files \
            else MANIFEST_DEFAULTS['flop_runouts']
        if runs != manifest['flop_runouts']:
            raise ValueError(
                f"postflop_table_flop.npz was baked with flop_runouts={runs}, but the "
                f"abstraction's manifest says {manifest['flop_runouts']}; re-bake it.")
    return True


def verify_tables(adir):
    """Check every baked table present in `adir` against its centroids and manifest
    (for callers that hand the directory to the Rust loader). A named abstraction
    must have stamped flop and turn tables. Returns the streets verified."""
    import numpy as np
    manifest = read_manifest(adir)
    root = is_root(adir)
    ok = []
    for street in STREETS:
        path = os.path.join(adir, f'postflop_table_{street}.npz')
        if not os.path.exists(path):
            if not root and street in REQUIRED_TABLES:
                raise ValueError(f"named abstraction {adir} is missing {os.path.basename(path)}")
            continue
        c, b = load_centroids(street, adir)
        with np.load(path) as d:
            if check_table_stamp(street, d, c, b, manifest, allow_legacy=root):
                ok.append(street)
    return ok


def check_resume(stored_name, stored_id, run_meta, who):
    """Refuse to resume a blueprint on a different abstraction than it was trained
    on. A legacy (unstamped) blueprint may only resume on the unchanged root."""
    run_name = run_meta.get('abstraction')
    run_id = run_meta['abstraction_id']
    want_id = stored_id if stored_id is not None else LEGACY_ABSTRACTION_ID
    if (stored_name or None) != (run_name or None) or want_id != run_id:
        raise ValueError(
            f"{who}: resume abstraction mismatch -- the blueprint was trained on "
            f"{stored_name or 'root (v2)'!r} ({want_id[:12]}...), this run uses "
            f"{run_name or 'root (v2)'!r} ({run_id[:12]}...). Resume with the same "
            f"abstraction, or start a fresh run.")
