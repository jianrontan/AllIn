# backend/bot/src/game/global_stats_store.py
"""
The single global "+EV counter" row: the bot's lifetime record vs the whole field.

Mirrors the PlayerStore/SessionStore pattern: an ABC with InMemory (dev/tests)
and DynamoDB (prod) implementations, chosen by `make_global_stats_store()` from
ALLIN_STORE_BACKEND. One logical singleton row:

    {totalHands, totalNetBB, totalPlayers}

totalNetBB is the HUMAN field's net; the bot's net is its negation (what the
"+EV counter" headlines). Updates are atomic (DynamoDB ADD on the singleton).
"""
import json
import logging
import os
import threading
import time

_LOG = logging.getLogger(__name__)
_SINGLETON_ID = 'global'
# Shared per-version aggregate snapshot lives in its OWN item (separate from the counter row) so the
# leaderboard reads one coherent blob across ALL gunicorn workers instead of each worker running its
# own recap scan on an independent clock (the old per-worker cache flickered between workers).
_SNAPSHOT_ID = 'version_snapshot'
# Cap the per-player map stored in the snapshot so the DynamoDB item can't exceed the 400KB limit as
# the player base grows. The leaderboard only shows the top _LEADERBOARD_MAX (200) anyway; 500/version
# leaves headroom for merge-resolution. At launch volumes this never truncates. NB the cap is taken by
# the SAME key the board ranks on (Net BB) -- capping by hands would let a high-net/low-hands player be
# evicted before ranking and silently vanish from the board past the cap.
_SNAPSHOT_PLAYER_CAP = 500


def _cap_snapshot_by_player(data, cap=_SNAPSHOT_PLAYER_CAP):
    """Bound the snapshot: keep only the top `cap` players per version by Net BB (the board's ranking
    key; hands as tiebreak). Returns (capped_data, truncated_bool). 'totals' are untouched (exact)."""
    by_player = (data or {}).get('byPlayer') or {}
    capped, truncated = {}, False
    for ver, pmap in by_player.items():
        items = sorted(pmap.items(),
                       key=lambda kv: ((kv[1] or {}).get('humanNetBB', 0.0),
                                       (kv[1] or {}).get('hands', 0)),
                       reverse=True)
        if len(items) > cap:
            items, truncated = items[:cap], True
        capped[ver] = dict(items)
    out = dict(data or {})
    out['byPlayer'] = capped
    return out, truncated


class GlobalStatsStore:
    def get(self):
        """Return {totalHands, totalNetBB, totalPlayers, byVersion}, where byVersion is
        {version: {hands, netBB}} -- LIVE per-bot-version running counters (no hand-table scan)."""
        raise NotImplementedError

    def record_hand_result(self, bb_delta, is_new_player=False, version=None):
        """Add one completed hand to the global totals (and a new player if flagged). When `version`
        (the bot-version label, e.g. 'v1'/'v2') is given, also bump that version's running counter --
        this is what makes the +EV card's v1/v2 numbers live without scanning the hand table."""
        raise NotImplementedError

    def record_new_player(self):
        """Count a newly-seen player without a hand result (called at first
        /api/game/new). Kept separate so the hand-end hook never double-counts."""
        raise NotImplementedError

    def record_merged_player(self):
        """Decrement totalPlayers when one player row is MERGED into another (sign-in on a
        new device): the merged anon was counted at /game/new but is no longer a distinct
        player, so undo that one count (otherwise totalPlayers over-counts vs the board)."""
        raise NotImplementedError

    def get_version_snapshot(self):
        """Return the SHARED per-version aggregate snapshot, or None if never computed:
            {'data': {'totals': {...}, 'byPlayer': {...}}, 'computedAt': <int epoch>}
        Shared across workers so every worker serves the same leaderboard numbers."""
        raise NotImplementedError

    def put_version_snapshot(self, data, computed_at):
        """Persist the shared per-version aggregate snapshot (the recap-scan result). Called by
        whichever worker won try_acquire_version_refresh(). `data` = {'totals':..., 'byPlayer':...}."""
        raise NotImplementedError

    def try_acquire_version_refresh(self, lease_seconds=90):
        """Best-effort cross-worker lock: return True iff THIS caller may run the slow recap scan
        now (and should then call put_version_snapshot). Only one worker per lease window wins;
        the rest serve the existing snapshot. The lease auto-expires so a crashed holder can't
        wedge refreshes forever."""
        raise NotImplementedError


class InMemoryGlobalStatsStore(GlobalStatsStore):
    def __init__(self):
        self._d = {'totalHands': 0, 'totalNetBB': 0.0, 'totalPlayers': 0}
        self._byv = {}                     # {version: {'hands': int, 'netBB': float}}
        self._snapshot = None              # {'data':..., 'computedAt': int}
        self._refresh_lease = 0.0          # epoch until which a refresh is "held"
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            out = dict(self._d)
            out['byVersion'] = {v: dict(d) for v, d in self._byv.items()}
            return out

    def record_hand_result(self, bb_delta, is_new_player=False, version=None):
        with self._lock:
            self._d['totalHands'] += 1
            self._d['totalNetBB'] += float(bb_delta)
            if is_new_player:
                self._d['totalPlayers'] += 1
            if version:
                bv = self._byv.setdefault(version, {'hands': 0, 'netBB': 0.0})
                bv['hands'] += 1
                bv['netBB'] += float(bb_delta)

    def record_new_player(self):
        with self._lock:
            self._d['totalPlayers'] += 1

    def record_merged_player(self):
        with self._lock:
            self._d['totalPlayers'] = max(0, self._d['totalPlayers'] - 1)

    def get_version_snapshot(self):
        with self._lock:
            if self._snapshot is None:
                return None
            return {'data': self._snapshot['data'], 'computedAt': self._snapshot['computedAt']}

    def put_version_snapshot(self, data, computed_at):
        with self._lock:
            self._snapshot = {'data': data, 'computedAt': int(computed_at)}

    def try_acquire_version_refresh(self, lease_seconds=90):
        with self._lock:
            now = time.time()
            if now >= self._refresh_lease:
                self._refresh_lease = now + lease_seconds
                return True
            return False


class DynamoDBGlobalStatsStore(GlobalStatsStore):
    """Atomic counters on one singleton item. boto3 is lazy-imported. Exercised in
    CI via moto / DynamoDB Local, not in the no-AWS dev environment."""

    def __init__(self, table_name, region=None, endpoint_url=None):
        import boto3
        from botocore.config import Config
        from botocore.exceptions import ClientError
        self._ClientError = ClientError
        kwargs = {'config': Config(retries={'mode': 'adaptive', 'max_attempts': 5})}
        if region:
            kwargs['region_name'] = region
        if endpoint_url:
            kwargs['endpoint_url'] = endpoint_url
        self._table = boto3.resource('dynamodb', **kwargs).Table(table_name)

    def get(self):
        from decimal import Decimal
        item = self._table.get_item(Key={'statId': _SINGLETON_ID}).get('Item') or {}
        def num(v):
            return (int(v) if isinstance(v, Decimal) and v % 1 == 0
                    else float(v) if isinstance(v, Decimal) else v)
        # Per-version running counters live in flat `vh_<version>` (hands) / `vn_<version>` (net)
        # attributes -- a flat prefix (not a nested map) so the per-hand `ADD` is trivial + can't
        # collide with other attrs. Reassemble them into byVersion here.
        by_version = {}
        for k, v in item.items():
            if k.startswith('vh_'):
                by_version.setdefault(k[3:], {})['hands'] = num(v)
            elif k.startswith('vn_'):
                by_version.setdefault(k[3:], {})['netBB'] = num(v)
        by_version = {ver: {'hands': d.get('hands', 0), 'netBB': d.get('netBB', 0.0)}
                      for ver, d in by_version.items()}
        return {
            'totalHands': num(item.get('totalHands', 0)),
            'totalNetBB': num(item.get('totalNetBB', 0.0)),
            'totalPlayers': num(item.get('totalPlayers', 0)),
            'byVersion': by_version,
        }

    def record_hand_result(self, bb_delta, is_new_player=False, version=None):
        from decimal import Decimal
        delta = Decimal(str(bb_delta))
        expr = 'ADD totalHands :one, totalNetBB :delta'
        vals = {':one': 1, ':delta': delta}
        names = {}
        if is_new_player:
            expr += ', totalPlayers :one'
        if version:
            # Bump this version's running counters in the SAME atomic update. Names are aliased
            # (#vh/#vn) because the attribute name embeds the version string.
            expr += ', #vh :one, #vn :delta'
            names = {'#vh': f'vh_{version}', '#vn': f'vn_{version}'}
        kwargs = {'Key': {'statId': _SINGLETON_ID}, 'UpdateExpression': expr,
                  'ExpressionAttributeValues': vals}
        if names:
            kwargs['ExpressionAttributeNames'] = names
        self._table.update_item(**kwargs)

    def record_new_player(self):
        self._table.update_item(
            Key={'statId': _SINGLETON_ID},
            UpdateExpression='ADD totalPlayers :one',
            ExpressionAttributeValues={':one': 1})

    def record_merged_player(self):
        # ADD -1 (atomic). Paired with record_new_player so it won't drive the count negative.
        self._table.update_item(
            Key={'statId': _SINGLETON_ID},
            UpdateExpression='ADD totalPlayers :neg',
            ExpressionAttributeValues={':neg': -1})

    def get_version_snapshot(self):
        item = self._table.get_item(Key={'statId': _SNAPSHOT_ID}).get('Item') or {}
        snap = item.get('snapshot')
        if not snap:
            return None
        try:
            data = json.loads(snap)
        except (ValueError, TypeError):
            return None
        ca = item.get('computedAt') or 0
        return {'data': data, 'computedAt': int(ca)}

    def put_version_snapshot(self, data, computed_at):
        capped, truncated = _cap_snapshot_by_player(data)
        if truncated:
            _LOG.warning("version snapshot byPlayer truncated to top %d/version", _SNAPSHOT_PLAYER_CAP)
        # SET (not put_item) so it never clobbers refreshLease on the same item. `snapshot` is a
        # DynamoDB RESERVED WORD (and moto doesn't enforce that -- it only fails on real DynamoDB /
        # DynamoDB Local), so alias both names via ExpressionAttributeNames.
        self._table.update_item(
            Key={'statId': _SNAPSHOT_ID},
            UpdateExpression='SET #snap = :s, #ca = :c',
            ExpressionAttributeNames={'#snap': 'snapshot', '#ca': 'computedAt'},
            ExpressionAttributeValues={':s': json.dumps(capped), ':c': int(computed_at)})

    def try_acquire_version_refresh(self, lease_seconds=90):
        now = int(time.time())
        try:
            self._table.update_item(
                Key={'statId': _SNAPSHOT_ID},
                UpdateExpression='SET refreshLease = :lease',
                ConditionExpression='attribute_not_exists(refreshLease) OR refreshLease < :now',
                ExpressionAttributeValues={':lease': now + int(lease_seconds), ':now': now})
            return True
        except self._ClientError as e:
            if e.response.get('Error', {}).get('Code') == 'ConditionalCheckFailedException':
                return False
            raise

    @staticmethod
    def create_table_if_missing(table_name, region=None, endpoint_url=None):
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
                AttributeDefinitions=[{'AttributeName': 'statId', 'AttributeType': 'S'}],
                KeySchema=[{'AttributeName': 'statId', 'KeyType': 'HASH'}],
                BillingMode='PAY_PER_REQUEST')
            client.get_waiter('table_exists').wait(TableName=table_name)
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') != 'ResourceInUseException':
                raise
        # PITR for disaster recovery; idempotent. Tolerate moto / DDB Local where
        # the operation isn't supported.
        try:
            client.update_continuous_backups(
                TableName=table_name,
                PointInTimeRecoverySpecification={'PointInTimeRecoveryEnabled': True})
        except ClientError as e:
            code = e.response.get('Error', {}).get('Code', '')
            if code not in ('ContinuousBackupsUnavailableException',
                            'UnknownOperationException', 'ValidationException'):
                raise


def make_global_stats_store():
    """Build the global stats store named by ALLIN_STORE_BACKEND (default 'memory')."""
    backend = os.environ.get('ALLIN_STORE_BACKEND', 'memory').strip().lower()
    if backend in ('', 'memory', 'inmemory'):
        return InMemoryGlobalStatsStore()
    if backend in ('dynamodb', 'dynamo'):
        return DynamoDBGlobalStatsStore(
            table_name=os.environ.get('ALLIN_GLOBAL_TABLE', 'allin-global'),
            region=(os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION')),
            endpoint_url=os.environ.get('ALLIN_DYNAMODB_ENDPOINT') or None)
    raise ValueError(f"Unknown ALLIN_STORE_BACKEND={backend!r} (use 'memory' or 'dynamodb')")
