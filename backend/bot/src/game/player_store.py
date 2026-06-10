# backend/bot/src/game/player_store.py
"""
Where per-player leaderboard rows live (Phase: +EV leaderboard).

`PlayerStore` mirrors `SessionStore`'s interface pattern: an ABC with an
in-memory implementation for dev/tests and a DynamoDB implementation for prod,
chosen by `make_player_store()` from `ALLIN_STORE_BACKEND` (memory | dynamodb).

A player row is **account-ready** (anonymous now, Tier-2 Google sign-in later
binds to the SAME row, non-destructively):

    {
      playerId,            # PK (the client localStorage UUID)
      handle,              # display name (validated), or None
      hands, netBB,        # lifetime totals (human's perspective; bot = -netBB)
      firstSeen, lastSeen, # epoch seconds
      window_start,        # rolling hand-cap window start (epoch) or None
      hands_in_window,     # hands counted in the current window
      isRegistered,        # True once an account is linked
      # added by link_account(): email, authProvider, providerSub
    }

DynamoDB updates are atomic (UpdateItem ADD/SET). `top()` is a Scan+filter --
fine for launch volumes; a GSI on netBB/hands is the upgrade path.
"""
import os
import re
import threading
import time
from abc import ABC, abstractmethod

# Handle rules: 1-20 chars, ASCII letters/digits/underscore/hyphen, no spaces.
HANDLE_RE = re.compile(r'^[A-Za-z0-9_-]{1,20}$')

# Rolling hand-cap (Item 3): default 500 hands per 1h, env-overridable.
HAND_CAP = int(os.environ.get('ALLIN_HANDS_PER_WINDOW', '500'))
HAND_WINDOW_SECONDS = int(os.environ.get('ALLIN_HAND_WINDOW_SECONDS', '3600'))

# Small built-in blocklist so profanity rejection is deterministic without the
# optional `better-profanity` package; if that package IS installed we also run
# its (broader, curated) check. Substring match catches embedded attempts.
_PROFANITY_FALLBACK = {
    'fuck', 'shit', 'cunt', 'nigger', 'faggot', 'bitch', 'asshole',
    'rape', 'nazi', 'whore', 'slut', 'dick', 'pussy',
}


class InvalidHandle(Exception):
    """Raised when a handle fails validation (the API turns it into a 400)."""


class HandleTaken(Exception):
    """Raised when a (valid) username is already in use (the API -> 409)."""


class AccountConflict(Exception):
    """Raised when this browser's player is already bound to a DIFFERENT account
    (the API -> 403): sign out before signing into another Google account here."""


def _now():
    return int(time.time())


def _is_profane(handle):
    low = handle.lower()
    if any(bad in low for bad in _PROFANITY_FALLBACK):
        return True
    try:
        from better_profanity import profanity        # optional dependency
        return bool(profanity.contains_profanity(handle))
    except Exception:
        return False


def validate_handle(handle):
    """Return the handle if valid, else raise InvalidHandle."""
    if not isinstance(handle, str) or not HANDLE_RE.match(handle):
        raise InvalidHandle(
            "handle must be 1-20 characters: letters, digits, _ or - (no spaces)")
    if _is_profane(handle):
        raise InvalidHandle("that handle isn't allowed")
    return handle


def sanitize_display_name(name):
    """Best-effort display handle from a verified provider name/email (NOT
    user-typed, so we sanitize rather than reject): keep letters/digits/space/_/-,
    collapse whitespace, trim to 20. Returns None if nothing usable remains."""
    if not name:
        return None
    cleaned = re.sub(r'[^A-Za-z0-9 _-]', '', str(name))
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()[:20].strip()
    return cleaned or None


def _new_row(player_id):
    now = _now()
    # IMPORTANT: do NOT initialize window_start/hands_in_window here. The
    # DynamoDB conditional reset uses `attribute_not_exists(window_start) OR
    # window_start < :cutoff`. If we wrote `window_start: None` at create time
    # the attribute exists as NULL → first branch is False AND `NULL < n` is
    # also False under DynamoDB type-mismatch semantics, so the first-ever
    # window NEVER resets. Leaving the attributes absent until the first
    # record_hand_result lets `attribute_not_exists` correctly fire.
    return {
        'playerId': player_id, 'handle': None,
        'hands': 0, 'netBB': 0,  # int (not 0.0) -- DynamoDB rejects bare floats;
        'firstSeen': now, 'lastSeen': now,
        # window_start / hands_in_window deliberately absent (see comment above).
        'isRegistered': False,
    }


def _bb_per_100(row):
    h = row.get('hands') or 0
    return (row['netBB'] / h * 100.0) if h else 0.0


class PlayerStore(ABC):
    @abstractmethod
    def get(self, player_id):
        """Return the player's row dict, or None if absent."""

    @abstractmethod
    def create_if_absent(self, player_id):
        """Create a bare row for player_id if none exists. Return True iff created
        (so the caller can bump the global new-player count exactly once)."""

    @abstractmethod
    def upsert_handle(self, player_id, handle):
        """Validate + set the player's handle (creating the row if needed).
        Raises InvalidHandle on a bad handle. Returns the updated row."""

    @abstractmethod
    def record_hand_result(self, player_id, bb_delta):
        """Add one completed hand: bump hands + netBB and the rolling window
        (resetting it if expired). Atomic."""

    @abstractmethod
    def top(self, n=10, min_hands=50, accounts_only=False):
        """Players with >= min_hands, ranked by bb/100 desc, redacted for public
        display. accounts_only restricts to linked accounts (the ranked board)."""

    @abstractmethod
    def link_account(self, player_id, *, email, auth_provider, provider_sub):
        """Resolve the canonical account for `provider_sub` and return its row.
        If a canonical account already exists, ADOPT it (returning user / new
        device) WITHOUT merging this device's anonymous row. Otherwise bind THIS
        device's row to the account (first sign-in absorbs its anon history); raise
        AccountConflict if this row is already bound to a different account. Does NOT
        set a username -- the player picks a unique one next. The returned row's
        playerId is the canonical id the client should adopt."""

    # -- concrete helpers (store-agnostic, built on get()) --------------------
    def hand_cap_status(self, player_id):
        """(allowed, retry_after_seconds) for starting a new hand under the rolling
        cap. Window-expired or under-cap -> (True, 0); at/over cap -> (False, secs)."""
        row = self.get(player_id)
        if not row:
            return True, 0
        ws = row.get('window_start')
        if ws is None:
            return True, 0
        elapsed = _now() - int(ws)
        if elapsed >= HAND_WINDOW_SECONDS:
            return True, 0                       # window lapsed -> resets on next record
        if (row.get('hands_in_window') or 0) >= HAND_CAP:
            return False, HAND_WINDOW_SECONDS - elapsed
        return True, 0

    @staticmethod
    def public_row(row):
        """Redact a row for public display (drop playerId / email / provider sub)."""
        return {
            'handle': row.get('handle') or 'Anonymous',
            'hands': int(row.get('hands') or 0),
            'netBB': round(float(row.get('netBB') or 0.0), 2),
            'bbPer100': round(_bb_per_100(row), 2),
            'isRegistered': bool(row.get('isRegistered')),
        }


class InMemoryPlayerStore(PlayerStore):
    """Process-local player rows (dev + tests). Not shared across workers."""

    def __init__(self):
        self._rows = {}
        self._lock = threading.Lock()

    def get(self, player_id):
        with self._lock:
            r = self._rows.get(player_id)
            return dict(r) if r else None

    def create_if_absent(self, player_id):
        with self._lock:
            if player_id in self._rows:
                return False
            self._rows[player_id] = _new_row(player_id)
            return True

    def upsert_handle(self, player_id, handle):
        handle = validate_handle(handle)
        low = handle.lower()
        with self._lock:
            # Usernames are UNIQUE (case-insensitive) across live accounts.
            taken = any(pid != player_id and not row.get('merged_into')
                        and (row.get('handle') or '').lower() == low
                        for pid, row in self._rows.items())
            if taken:
                raise HandleTaken(handle)
            r = self._rows.setdefault(player_id, _new_row(player_id))
            r['handle'] = handle
            r['lastSeen'] = _now()
            return dict(r)

    def record_hand_result(self, player_id, bb_delta):
        with self._lock:
            r = self._rows.setdefault(player_id, _new_row(player_id))
            now = _now()
            ws = r.get('window_start')
            if ws is None or now - int(ws) >= HAND_WINDOW_SECONDS:
                r['window_start'] = now
                r['hands_in_window'] = 0
            r['hands_in_window'] = (r.get('hands_in_window') or 0) + 1
            r['hands'] = (r.get('hands') or 0) + 1
            r['netBB'] = (r.get('netBB') or 0.0) + float(bb_delta)
            r['lastSeen'] = now

    def top(self, n=10, min_hands=50, accounts_only=False):
        with self._lock:
            rows = [dict(r) for r in self._rows.values()
                    if (r.get('hands') or 0) >= min_hands
                    and not r.get('merged_into')
                    and (not accounts_only or r.get('isRegistered'))]
        rows.sort(key=_bb_per_100, reverse=True)
        return [self.public_row(r) for r in rows[:n]]

    def link_account(self, player_id, *, email, auth_provider, provider_sub):
        with self._lock:
            # ONE canonical account per provider_sub. If it already exists, this is
            # a returning user (possibly on a new device): ADOPT it as-is -- do NOT
            # merge/sum this device's anonymous row. The returned row's playerId is
            # the canonical id the client adopts.
            canonical_id = next(
                (pid for pid, row in self._rows.items()
                 if row.get('providerSub') == provider_sub
                 and not row.get('merged_into')),
                None)
            if canonical_id is not None:
                c = self._rows[canonical_id]
                c['email'] = email
                c['lastSeen'] = _now()
                return dict(c)
            # First sign-in for this account: bind THIS device's row, so it absorbs
            # the device's anonymous history. Refuse if the row is already bound to a
            # different account (don't clobber it).
            existing = self._rows.get(player_id)
            if existing and existing.get('providerSub') \
                    and existing.get('providerSub') != provider_sub:
                raise AccountConflict()
            r = self._rows.setdefault(player_id, _new_row(player_id))
            r['email'] = email
            r['authProvider'] = auth_provider
            r['providerSub'] = provider_sub
            r['isRegistered'] = True
            r['lastSeen'] = _now()
            return dict(r)


class DynamoDBPlayerStore(PlayerStore):
    """DynamoDB-backed player rows for prod. Atomic ADD/SET updates; Scan-based
    top() (GSI on bb/100 is the documented upgrade path). boto3 is lazy-imported
    so the in-memory default needs no AWS dependency.

    NOTE: exercised in CI via moto / DynamoDB Local (see test_player_store.py),
    not in the no-AWS dev environment.
    """

    def __init__(self, table_name, region=None, endpoint_url=None):
        import boto3
        from botocore.exceptions import ClientError
        from boto3.dynamodb.conditions import Attr
        from botocore.config import Config
        self._ClientError = ClientError
        self._Attr = Attr
        kwargs = {'config': Config(retries={'mode': 'adaptive', 'max_attempts': 5})}
        if region:
            kwargs['region_name'] = region
        if endpoint_url:
            kwargs['endpoint_url'] = endpoint_url
        self._table = boto3.resource('dynamodb', **kwargs).Table(table_name)

    @staticmethod
    def _clean(item):
        """DynamoDB returns Decimals; coerce numerics back to int/float."""
        if item is None:
            return None
        from decimal import Decimal
        out = {}
        for k, v in item.items():
            if isinstance(v, Decimal):
                out[k] = int(v) if v % 1 == 0 else float(v)
            else:
                out[k] = v
        return out

    def get(self, player_id):
        return self._clean(self._table.get_item(Key={'playerId': player_id}).get('Item'))

    def create_if_absent(self, player_id):
        try:
            self._table.put_item(
                Item=_new_row(player_id),
                ConditionExpression='attribute_not_exists(playerId)')
            return True
        except self._ClientError as e:
            if e.response.get('Error', {}).get('Code') == 'ConditionalCheckFailedException':
                return False
            raise

    def _scan_all(self, flt):
        items, resp = [], self._table.scan(FilterExpression=flt)
        items.extend(resp.get('Items', []))
        while 'LastEvaluatedKey' in resp:
            resp = self._table.scan(FilterExpression=flt,
                                    ExclusiveStartKey=resp['LastEvaluatedKey'])
            items.extend(resp.get('Items', []))
        return items

    def upsert_handle(self, player_id, handle):
        handle = validate_handle(handle)
        low = handle.lower()
        # Uniqueness via scan (launch volume; a handle-reservation item or a GSI on
        # a lowercased handle is the race-proof upgrade). Reject if another live row
        # already holds the name.
        for it in self._scan_all(self._Attr('merged_into').not_exists()):
            if it.get('playerId') != player_id and (it.get('handle') or '').lower() == low:
                raise HandleTaken(handle)
        self.create_if_absent(player_id)
        self._table.update_item(
            Key={'playerId': player_id},
            UpdateExpression='SET handle = :h, lastSeen = :now',
            ExpressionAttributeValues={':h': handle, ':now': _now()})
        return self.get(player_id)

    def record_hand_result(self, player_id, bb_delta):
        from decimal import Decimal
        now = _now()
        self.create_if_absent(player_id)
        # Reset the rolling window first IF it's absent or expired (conditional, so
        # it only fires when needed); then unconditionally add to the counters.
        # NOTE: reset+increment is two updates, so at the window boundary under
        # concurrent same-player hands the count can be off by a few. The cap is a
        # SOFT anti-DoS throttle (a human plays hands sequentially), so this is
        # accepted; a single atomic update / reservation is the upgrade if needed.
        try:
            self._table.update_item(
                Key={'playerId': player_id},
                UpdateExpression='SET window_start = :now, hands_in_window = :zero',
                ConditionExpression=(
                    'attribute_not_exists(window_start) OR window_start < :cutoff'),
                ExpressionAttributeValues={
                    ':now': now, ':zero': 0,
                    ':cutoff': now - HAND_WINDOW_SECONDS})
        except self._ClientError as e:
            if e.response.get('Error', {}).get('Code') != 'ConditionalCheckFailedException':
                raise                            # window still valid -> no reset
        self._table.update_item(
            Key={'playerId': player_id},
            UpdateExpression=('ADD hands :one, netBB :delta, hands_in_window :one '
                              'SET lastSeen = :now'),
            ExpressionAttributeValues={
                ':one': 1, ':delta': Decimal(str(bb_delta)), ':now': now})

    def top(self, n=10, min_hands=50, accounts_only=False):
        # Scan + filter (launch volumes only; add a GSI on a bb/100 attribute later).
        flt = self._Attr('hands').gte(min_hands) & self._Attr('merged_into').not_exists()
        if accounts_only:
            flt = flt & self._Attr('isRegistered').eq(True)
        items, resp = [], self._table.scan(FilterExpression=flt)
        items.extend(resp.get('Items', []))
        while 'LastEvaluatedKey' in resp:
            resp = self._table.scan(FilterExpression=flt,
                                    ExclusiveStartKey=resp['LastEvaluatedKey'])
            items.extend(resp.get('Items', []))
        rows = [self._clean(it) for it in items]
        rows.sort(key=_bb_per_100, reverse=True)
        return [self.public_row(r) for r in rows[:n]]

    def link_account(self, player_id, *, email, auth_provider, provider_sub):
        # ONE canonical account per provider_sub. If it exists (returning user, maybe
        # a new device), ADOPT it as-is -- no merge/sum of this device's anon row.
        canon = [self._clean(it) for it in self._scan_all(
            self._Attr('providerSub').eq(provider_sub)
            & self._Attr('merged_into').not_exists())]
        if canon:
            c = canon[0]                          # one canonical per sub by construction
            self._table.update_item(
                Key={'playerId': c['playerId']},
                UpdateExpression='SET email = :e, lastSeen = :now',
                ExpressionAttributeValues={':e': email, ':now': _now()})
            return self.get(c['playerId'])
        # First sign-in: bind THIS device's row (absorbs its anon stats). Refuse if
        # it's already bound to a different account.
        cur = self.get(player_id)
        if cur and cur.get('providerSub') and cur.get('providerSub') != provider_sub:
            raise AccountConflict()
        self.create_if_absent(player_id)
        self._table.update_item(
            Key={'playerId': player_id},
            UpdateExpression=('SET email = :e, authProvider = :a, providerSub = :s, '
                              'isRegistered = :t, lastSeen = :now'),
            ExpressionAttributeValues={':e': email, ':a': auth_provider,
                                       ':s': provider_sub, ':t': True, ':now': _now()})
        return self.get(player_id)

    @staticmethod
    def create_table_if_missing(table_name, region=None, endpoint_url=None):
        """One-time provisioning helper (run once at deploy). Also enables
        Point-In-Time Recovery so a wiped table is restorable (the leaderboard
        is the most precious launch-window data; without PITR, gone is gone)."""
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
                AttributeDefinitions=[{'AttributeName': 'playerId', 'AttributeType': 'S'}],
                KeySchema=[{'AttributeName': 'playerId', 'KeyType': 'HASH'}],
                BillingMode='PAY_PER_REQUEST')
            client.get_waiter('table_exists').wait(TableName=table_name)
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') != 'ResourceInUseException':
                raise
        # PITR: idempotent — already-enabled returns ContinuousBackupsUnavailable
        # or succeeds silently. moto / DynamoDB Local doesn't support PITR; tolerate.
        try:
            client.update_continuous_backups(
                TableName=table_name,
                PointInTimeRecoverySpecification={'PointInTimeRecoveryEnabled': True})
        except ClientError as e:
            code = e.response.get('Error', {}).get('Code', '')
            if code not in ('ContinuousBackupsUnavailableException',
                            'UnknownOperationException', 'ValidationException'):
                raise


def make_player_store():
    """Build the player store named by ALLIN_STORE_BACKEND (default 'memory')."""
    backend = os.environ.get('ALLIN_STORE_BACKEND', 'memory').strip().lower()
    if backend in ('', 'memory', 'inmemory'):
        return InMemoryPlayerStore()
    if backend in ('dynamodb', 'dynamo'):
        return DynamoDBPlayerStore(
            table_name=os.environ.get('ALLIN_PLAYERS_TABLE', 'allin-players'),
            region=(os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION')),
            endpoint_url=os.environ.get('ALLIN_DYNAMODB_ENDPOINT') or None)
    raise ValueError(f"Unknown ALLIN_STORE_BACKEND={backend!r} (use 'memory' or 'dynamodb')")
