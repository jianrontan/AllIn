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
import os
import threading
import time

_SINGLETON_ID = 'global'


class GlobalStatsStore:
    def get(self):
        """Return {totalHands, totalNetBB, totalPlayers}."""
        raise NotImplementedError

    def record_hand_result(self, bb_delta, is_new_player=False):
        """Add one completed hand to the global totals (and a new player if flagged)."""
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


class InMemoryGlobalStatsStore(GlobalStatsStore):
    def __init__(self):
        self._d = {'totalHands': 0, 'totalNetBB': 0.0, 'totalPlayers': 0}
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            return dict(self._d)

    def record_hand_result(self, bb_delta, is_new_player=False):
        with self._lock:
            self._d['totalHands'] += 1
            self._d['totalNetBB'] += float(bb_delta)
            if is_new_player:
                self._d['totalPlayers'] += 1

    def record_new_player(self):
        with self._lock:
            self._d['totalPlayers'] += 1

    def record_merged_player(self):
        with self._lock:
            self._d['totalPlayers'] = max(0, self._d['totalPlayers'] - 1)


class DynamoDBGlobalStatsStore(GlobalStatsStore):
    """Atomic counters on one singleton item. boto3 is lazy-imported. Exercised in
    CI via moto / DynamoDB Local, not in the no-AWS dev environment."""

    def __init__(self, table_name, region=None, endpoint_url=None):
        import boto3
        from botocore.config import Config
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
        return {
            'totalHands': num(item.get('totalHands', 0)),
            'totalNetBB': num(item.get('totalNetBB', 0.0)),
            'totalPlayers': num(item.get('totalPlayers', 0)),
        }

    def record_hand_result(self, bb_delta, is_new_player=False):
        from decimal import Decimal
        expr = 'ADD totalHands :one, totalNetBB :delta'
        vals = {':one': 1, ':delta': Decimal(str(bb_delta))}
        if is_new_player:
            expr += ', totalPlayers :one'
        self._table.update_item(Key={'statId': _SINGLETON_ID},
                                UpdateExpression=expr, ExpressionAttributeValues=vals)

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
