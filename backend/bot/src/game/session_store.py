# backend/bot/src/game/session_store.py
"""
Where in-progress games are kept between requests.

`SessionStore` is the interface (the "plug shape"): save / get / delete a
game, identified by a session id. Games are stored as plain dicts
(GameSession.to_dict()), so any backing store that can hold JSON works.

  * InMemorySessionStore  — keeps games in this process's RAM. Used in
    development. Games are lost on restart and are NOT shared across multiple
    backend processes.

A future RedisSessionStore / DynamoDBSessionStore would implement the same
three methods and be a drop-in replacement — no other code changes.
"""
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager


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
        self._locks = {}       # session_id -> threading.RLock
        self._locks_guard = threading.Lock()

    def get(self, session_id):
        entry = self._data.get(session_id)
        if entry is None:
            return None
        expiry, data = entry
        if time.time() > expiry:
            del self._data[session_id]
            return None
        return data

    def put(self, session_id, data):
        self._sweep()
        self._data[session_id] = (time.time() + self._ttl, data)

    def delete(self, session_id):
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

    def _sweep(self):
        now = time.time()
        expired = [sid for sid, (exp, _) in self._data.items() if now > exp]
        for sid in expired:
            del self._data[sid]
            with self._locks_guard:
                self._locks.pop(sid, None)
