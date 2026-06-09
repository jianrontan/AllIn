# backend/bot/tests/test_blueprint_source.py
"""
Blueprint SOURCE seam (src/storage/blueprint_source.py): the layer that decides
where the blueprint .db comes from, above BlueprintDB (which only reads files).

Covers:
  * make_blueprint_source() defaults to LocalFileSource with no env set, and
    resolves to the same path the API would open.
  * S3ObjectSource downloads ONCE and caches -- a second local_path() does NOT
    re-download (proved by stubbing the S3 client to allow exactly one get_object).
  * factory validation (unknown source, s3 without a URI).

The S3 path is a stub (not wired in v1); LocalFileSource is the only path CI
exercises against the real blueprint.
"""
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.blueprint_source import (
    LocalFileSource, S3ObjectSource, make_blueprint_source)


class _OneShotS3:
    """Minimal stand-in for a boto3 S3 client: serves get_object exactly `allow`
    times, then raises -- so a second call (a re-download) is a hard failure. This
    is the botocore.stub.Stubber idea without the dependency: queue N responses,
    assert no extra calls."""

    def __init__(self, body, allow=1):
        self._body = body
        self._remaining = allow
        self.calls = 0

    def get_object(self, Bucket, Key):
        self.calls += 1
        if self._remaining <= 0:
            raise AssertionError(
                f"unexpected extra get_object (re-download) for s3://{Bucket}/{Key}")
        self._remaining -= 1
        return {'Body': io.BytesIO(self._body)}


def test_factory_defaults_to_local():
    os.environ.pop('ALLIN_BLUEPRINT_SOURCE', None)
    src = make_blueprint_source()
    assert isinstance(src, LocalFileSource), type(src)
    # local_path() resolves to a real, existing blueprint file.
    p = src.local_path()
    assert p.exists(), p
    print(f"PASS test_factory_defaults_to_local ({p.name})")


def test_local_explicit_path():
    src = make_blueprint_source()
    real = src.local_path()
    explicit = LocalFileSource(real)
    assert explicit.local_path() == real
    print("PASS test_local_explicit_path")


def test_s3_downloads_once_then_caches():
    body = b"fake-blueprint-bytes"
    client = _OneShotS3(body, allow=1)
    with tempfile.TemporaryDirectory() as d:
        src = S3ObjectSource("s3://my-bucket/snapshots/snap.db",
                             cache_dir=d, client=client)
        p1 = src.local_path()
        assert p1.exists() and p1.read_bytes() == body
        # Second call must NOT hit S3 again (the stub would raise on a 2nd call).
        p2 = src.local_path()
        assert p2 == p1
        assert client.calls == 1, client.calls
    print("PASS test_s3_downloads_once_then_caches (1 download, cached thereafter)")


def test_s3_reuses_existing_cache_file_without_download():
    body = b"already-on-disk"
    with tempfile.TemporaryDirectory() as d:
        # Pre-place the cache file: local_path() must NOT call get_object at all.
        with open(os.path.join(d, 'snap.db'), 'wb') as f:
            f.write(body)
        client = _OneShotS3(b"should-not-be-served", allow=0)
        src = S3ObjectSource("s3://b/k/snap.db", cache_dir=d, client=client)
        p = src.local_path()
        assert p.read_bytes() == body and client.calls == 0
    print("PASS test_s3_reuses_existing_cache_file_without_download")


def test_factory_rejects_bad_config():
    os.environ['ALLIN_BLUEPRINT_SOURCE'] = 'bogus'
    try:
        make_blueprint_source()
        raise AssertionError("expected ValueError for unknown source")
    except ValueError:
        pass
    os.environ['ALLIN_BLUEPRINT_SOURCE'] = 's3'
    os.environ.pop('ALLIN_BLUEPRINT_S3_URI', None)
    try:
        make_blueprint_source()
        raise AssertionError("expected ValueError for s3 without a URI")
    except ValueError:
        pass
    os.environ.pop('ALLIN_BLUEPRINT_SOURCE', None)
    # bad URI shapes
    for bad in ('not-s3', 's3://bucket-only'):
        try:
            S3ObjectSource(bad)
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError:
            pass
    print("PASS test_factory_rejects_bad_config")


TESTS = [
    test_factory_defaults_to_local,
    test_local_explicit_path,
    test_s3_downloads_once_then_caches,
    test_s3_reuses_existing_cache_file_without_download,
    test_factory_rejects_bad_config,
]

if __name__ == '__main__':
    passed = failed = 0
    for fn in TESTS:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\nResults: {passed} passed, {failed} failed out of {len(TESTS)}")
    sys.exit(1 if failed else 0)
