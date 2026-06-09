# backend/bot/src/storage/blueprint_source.py
"""
Where the blueprint .db COMES FROM, decoupled from how it's read.

`BlueprintDB` correctly only knows how to open a local SQLite file. This thin
layer sits above `config.resolve_blueprint_path()` so the *source* of that file
can change without touching `BlueprintDB` or the API:

  * LocalFileSource  -- the default (and what v1 ships): the blueprint is a local
    file, baked into the Docker image. Wraps resolve_blueprint_path() /
    ALLIN_BLUEPRINT_DB.
  * S3ObjectSource   -- a STUB for later: downloads s3://bucket/key once to a
    cache dir on first use and returns the cached path (idempotent). NOT wired in
    v1; boto3 is imported lazily so the default path needs no AWS dependency.

`make_blueprint_source()` picks the implementation from ALLIN_BLUEPRINT_SOURCE
('local' default, 's3' later). The API calls
`make_blueprint_source().local_path()` where it used to call
resolve_blueprint_path(), so swapping in S3 is additive, not a rewrite.

Deliberately NOT here: an S3 *session* store (sessions are DynamoDB), a
background re-pull, or hot-reload -- a blueprint change is a restart.
"""
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from ..config import resolve_blueprint_path


class BlueprintSource(ABC):
    @abstractmethod
    def local_path(self):
        """Return a local filesystem Path that BlueprintDB can open."""


class LocalFileSource(BlueprintSource):
    """The blueprint is already a local file. Default source for v1."""

    def __init__(self, path=None):
        self._path = Path(path) if path else None

    def local_path(self):
        # An explicit path wins; otherwise defer to the resolver (which honors
        # ALLIN_BLUEPRINT_DB, else the highest-iteration DB in the blueprints dir).
        return self._path if self._path is not None else resolve_blueprint_path()


class S3ObjectSource(BlueprintSource):
    """Stub for a future S3-backed blueprint. Downloads the object ONCE to a
    cache dir and returns the cached path; subsequent calls reuse the cache (no
    re-download). Not wired into v1.

    The S3 client is injectable for testing (a botocore Stubber); in production
    it is created lazily so importing this module never requires boto3.
    """

    def __init__(self, s3_uri, cache_dir=None, client=None):
        self._uri = s3_uri
        self._bucket, self._key = self._parse_uri(s3_uri)
        self._cache_dir = Path(cache_dir or os.environ.get('ALLIN_BLUEPRINT_CACHE_DIR')
                               or tempfile.gettempdir())
        self._client = client            # injected (tests) or lazily created
        self._cached = None              # memoized local path once downloaded

    @staticmethod
    def _parse_uri(uri):
        if not uri.startswith('s3://'):
            raise ValueError(f"not an s3:// URI: {uri!r}")
        bucket, _, key = uri[len('s3://'):].partition('/')
        if not bucket or not key:
            raise ValueError(f"s3 URI must be s3://bucket/key, got {uri!r}")
        return bucket, key

    def _get_client(self):
        if self._client is None:
            import boto3                  # lazy: only when an S3 source is actually used
            self._client = boto3.client('s3')
        return self._client

    def local_path(self):
        # Already downloaded this process? reuse it (idempotent, no re-download).
        if self._cached is not None and self._cached.exists():
            return self._cached
        dest = self._cache_dir / Path(self._key).name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            # get_object (not download_file) so the call is directly stubbable with
            # botocore.stub.Stubber and writes are explicit.
            resp = self._get_client().get_object(Bucket=self._bucket, Key=self._key)
            with open(dest, 'wb') as f:
                f.write(resp['Body'].read())
        self._cached = dest
        return dest


def make_blueprint_source():
    """Build the blueprint source named by ALLIN_BLUEPRINT_SOURCE (default 'local').

    'local' -> LocalFileSource (resolver / ALLIN_BLUEPRINT_DB)
    's3'    -> S3ObjectSource(ALLIN_BLUEPRINT_S3_URI)   [not wired in v1]
    """
    src = os.environ.get('ALLIN_BLUEPRINT_SOURCE', 'local').strip().lower()
    if src in ('', 'local', 'file'):
        return LocalFileSource()
    if src == 's3':
        uri = os.environ.get('ALLIN_BLUEPRINT_S3_URI')
        if not uri:
            raise ValueError(
                "ALLIN_BLUEPRINT_SOURCE=s3 requires ALLIN_BLUEPRINT_S3_URI")
        return S3ObjectSource(uri)
    raise ValueError(
        f"Unknown ALLIN_BLUEPRINT_SOURCE={src!r} (use 'local' or 's3')")
