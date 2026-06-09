# backend/bot/src/game/session_store.py
"""
Where in-progress games are kept between requests.

`SessionStore` is the interface (the "plug shape"): save / get / delete a
game, identified by a session id. Games are stored as plain dicts
(GameSession.to_dict()), so any backing store that can hold JSON works.

  * InMemorySessionStore  — keeps games in this process's RAM. Used in
    development and the test suite. Games are lost on restart and are NOT shared
    across multiple backend processes (so it is unsafe under gunicorn with >1
    worker). This is the default (ALLIN_SESSION_STORE=memory).

  * DynamoDBSessionStore   — games live in a DynamoDB table, shared across every
    worker/box and surviving restarts. Native TTL expires stale games; a
    lease-based conditional-write lock serializes the load-modify-put of one
    session. Selected with ALLIN_SESSION_STORE=dynamodb (needs boto3 + AWS
    creds, or DynamoDB Local via ALLIN_DYNAMODB_ENDPOINT for testing).

`make_session_store()` picks the implementation from ALLIN_SESSION_STORE. A
future RedisSessionStore would implement the same methods and be a drop-in.
"""
import json
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager


class SessionLockTimeout(Exception):
    """Raised when a session lock can't be acquired within the timeout."""


class SessionStore(ABC):
    @abstractmethod
    def get(self, session_id):
        """Return the stored session dict, or None if absent/expired."""

    @abstractmethod
    def put(self, session_id, data):
        """Store (create or overwrite) a session dict."""

    @abstractmethod
    def delete(self, session_id):
        """Remove a session. No error if it does not exist."""

    @contextmanager
    def lock(self, session_id):
        """Serialize the load-modify-put of one session so concurrent requests
        for the same id can't clobber each other (lost update) or double-apply
        the bot. Default: no-op. A Redis/DynamoDB store would override this with
        a distributed lock; the InMemory store uses a per-session threading lock
        (the Flask dev server is multi-threaded)."""
        yield


class InMemorySessionStore(SessionStore):
    """Process-local store with a simple TTL sweep to bound memory use."""

    def __init__(self, ttl_seconds=3600):
        self._ttl = ttl_seconds
        self._data = {}        # session_id -> (expiry_epoch, session_dict)
        self._data_guard = threading.Lock()   # protects _data (cross-session)
        self._locks = {}       # session_id -> threading.RLock
        self._locks_guard = threading.Lock()

    def get(self, session_id):
        with self._data_guard:           # _data mutated cross-session; guard reads too
            entry = self._data.get(session_id)
            if entry is None:
                return None
            expiry, data = entry
            if time.time() > expiry:
                del self._data[session_id]
                return None
            return data

    def put(self, session_id, data):
        with self._data_guard:
            self._sweep_locked()
            self._data[session_id] = (time.time() + self._ttl, data)

    def delete(self, session_id):
        with self._data_guard:
            self._data.pop(session_id, None)
        with self._locks_guard:
            self._locks.pop(session_id, None)

    @contextmanager
    def lock(self, session_id):
        with self._locks_guard:
            lk = self._locks.get(session_id)
            if lk is None:
                lk = threading.RLock()
                self._locks[session_id] = lk
        with lk:
            yield

    def _sweep_locked(self):
        # Caller holds _data_guard. Snapshot keys before deleting (no mutate-during-
        # iterate). Pop per-session locks under their own guard.
        now = time.time()
        expired = [sid for sid, (exp, _) in list(self._data.items()) if now > exp]
        for sid in expired:
            self._data.pop(sid, None)
        if expired:
            with self._locks_guard:
                for sid in expired:
                    self._locks.pop(sid, None)


class DynamoDBSessionStore(SessionStore):
    """Shared, restart-surviving session store backed by a single DynamoDB table.

    Both the game state and the per-session lock live in one table keyed by a
    string `session_id`:

      * a SESSION item   key=<sid>           attrs: data (JSON str), expiry (epoch)
      * a LOCK item      key="__lock__<sid>" attrs: owner (token), expiry (epoch)

    `expiry` is the DynamoDB **TTL attribute** (enable TTL on `expiry` when you
    create the table), so finished/abandoned games and stale locks auto-delete.
    TTL deletion is lazy (minutes-late), so `get()` also treats an expired item
    as absent.

    The lock is a lease: acquire = conditional PutItem that only succeeds if no
    live lock exists (`attribute_not_exists OR expiry < now`); release = a
    conditional DeleteItem guarded by the owner token, so a holder whose lease
    already expired can't delete a lock another worker has since taken. A crashed
    holder's lock expires on its own via the lease, so the session never wedges.

    boto3 is imported lazily so the in-memory default needs no AWS dependency.
    """

    def __init__(self, table_name, region=None, endpoint_url=None,
                 ttl_seconds=3600, lock_lease_seconds=30,
                 lock_acquire_timeout=10.0, lock_poll_seconds=0.05):
        import boto3                       # lazy: only when this store is selected
        from botocore.exceptions import ClientError
        self._ClientError = ClientError
        self._ttl = ttl_seconds
        self._lease = lock_lease_seconds
        self._acquire_timeout = lock_acquire_timeout
        self._poll = lock_poll_seconds
        kwargs = {}
        if region:
            kwargs['region_name'] = region
        if endpoint_url:
            kwargs['endpoint_url'] = endpoint_url
        self._table = boto3.resource('dynamodb', **kwargs).Table(table_name)

    @staticmethod
    def _lock_key(session_id):
        return f"__lock__{session_id}"

    def get(self, session_id):
        resp = self._table.get_item(Key={'session_id': session_id})
        item = resp.get('Item')
        if item is None:
            return None
        if int(item.get('expiry', 0)) <= int(time.time()):
            return None                    # expired but not yet TTL-swept
        return json.loads(item['data'])

    def put(self, session_id, data):
        self._table.put_item(Item={
            'session_id': session_id,
            'data': json.dumps(data),
            'expiry': int(time.time()) + self._ttl,
        })

    def delete(self, session_id):
        self._table.delete_item(Key={'session_id': session_id})
        self._table.delete_item(Key={'session_id': self._lock_key(session_id)})

    @contextmanager
    def lock(self, session_id):
        lock_key = self._lock_key(session_id)
        token = uuid.uuid4().hex
        deadline = time.time() + self._acquire_timeout
        acquired = False
        while True:
            now = int(time.time())
            try:
                self._table.put_item(
                    Item={'session_id': lock_key, 'owner': token,
                          'expiry': now + self._lease},
                    ConditionExpression='attribute_not_exists(session_id) OR #e < :now',
                    ExpressionAttributeNames={'#e': 'expiry'},
                    ExpressionAttributeValues={':now': now},
                )
                acquired = True
                break
            except self._ClientError as e:
                code = e.response.get('Error', {}).get('Code')
                if code != 'ConditionalCheckFailedException':
                    raise                  # a real error, not just contention
                if time.time() >= deadline:
                    break
                time.sleep(self._poll)
        if not acquired:
            raise SessionLockTimeout(
                f"could not acquire lock for session {session_id} in "
                f"{self._acquire_timeout}s")
        try:
            yield
        finally:
            try:
                self._table.delete_item(
                    Key={'session_id': lock_key},
                    ConditionExpression='#o = :tok',
                    ExpressionAttributeNames={'#o': 'owner'},
                    ExpressionAttributeValues={':tok': token},
                )
            except self._ClientError as e:
                code = e.response.get('Error', {}).get('Code')
                if code != 'ConditionalCheckFailedException':
                    raise                  # lease already expired + retaken: ignore

    @staticmethod
    def create_table_if_missing(table_name, region=None, endpoint_url=None):
        """One-time provisioning helper (run once at deploy). Creates an
        on-demand table keyed by `session_id` and enables TTL on `expiry`."""
        import boto3
        from botocore.exceptions import ClientError
        kwargs = {}
        if region:
            kwargs['region_name'] = region
        if endpoint_url:
            kwargs['endpoint_url'] = endpoint_url
        client = boto3.client('dynamodb', **kwargs)
        try:
            client.create_table(
                TableName=table_name,
                AttributeDefinitions=[{'AttributeName': 'session_id',
                                       'AttributeType': 'S'}],
                KeySchema=[{'AttributeName': 'session_id', 'KeyType': 'HASH'}],
                BillingMode='PAY_PER_REQUEST',
            )
            client.get_waiter('table_exists').wait(TableName=table_name)
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') != 'ResourceInUseException':
                raise                      # already exists is fine
        client.update_time_to_live(
            TableName=table_name,
            TimeToLiveSpecification={'Enabled': True, 'AttributeName': 'expiry'},
        )


def make_session_store():
    """Build the session store named by ALLIN_SESSION_STORE (default 'memory').

    'memory'   -> InMemorySessionStore (dev + tests; not multi-worker safe)
    'dynamodb' -> DynamoDBSessionStore (prod; needs boto3 + AWS creds, or
                  DynamoDB Local via ALLIN_DYNAMODB_ENDPOINT)
    """
    # Sessions live 24h by default (ALLIN_SESSION_TTL_SECONDS); the InMemory store
    # sweeps on expiry, the DynamoDB store writes it to the TTL `expiry` attribute.
    ttl = int(os.environ.get('ALLIN_SESSION_TTL_SECONDS', '86400'))
    backend = os.environ.get('ALLIN_SESSION_STORE', 'memory').strip().lower()
    if backend in ('', 'memory', 'inmemory'):
        return InMemorySessionStore(ttl_seconds=ttl)
    if backend in ('dynamodb', 'dynamo'):
        return DynamoDBSessionStore(
            table_name=os.environ.get('ALLIN_DYNAMODB_TABLE', 'allin-sessions'),
            region=(os.environ.get('AWS_REGION')
                    or os.environ.get('AWS_DEFAULT_REGION')),
            endpoint_url=os.environ.get('ALLIN_DYNAMODB_ENDPOINT') or None,
            ttl_seconds=ttl,
        )
    raise ValueError(
        f"Unknown ALLIN_SESSION_STORE={backend!r} (use 'memory' or 'dynamodb')")
