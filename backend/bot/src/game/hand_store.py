# backend/bot/src/game/hand_store.py
"""
Per-hand capture: one persisted "recap" row per completed hand.

WHY this exists. v1 doesn't surface hand history in the UI, but every hand
played from launch onward is potential coach / training / analytics material
that's irretrievable if not captured at the moment. This store is the cheap
write-only side of that: ~1 DynamoDB PutItem per completed hand, no consumer
yet. The read API (`/api/hands`) and the coach/RAG layer that use this data
are post-launch.

Same factory-of-stores pattern as PlayerStore / GlobalStatsStore /
SessionStore — pick `InMemory*` (dev) or `DynamoDB*` (prod) via
`ALLIN_STORE_BACKEND`. Table name overridable via `ALLIN_HANDS_TABLE`
(default `allin-hands`).

Schema:
    PK = playerId (String) — query "this user's hands"
    SK = handKey  (String) — `<13-digit epoch ms>#<sessionId>#<handNumber>`,
                              so ScanIndexForward=False gives most-recent-first.

The recap dict itself is built by `recap_from_session(session)` from
`GameSession.data` at the `hand_over` transition. Card formats are display
(`Ah`), not engine (`HA`). See `recap_from_session` for the field list.
"""
import os
import threading
import time
from abc import ABC, abstractmethod

from .cards import to_display_list


_STREETS = ('preflop', 'flop', 'turn', 'river')


def _now_ms():
    """Epoch milliseconds. Higher resolution than seconds so two hands dealt
    in the same second still sort deterministically by the handKey suffix."""
    return int(time.time() * 1000)


def _hand_key(ts_ms, session_id, hand_number):
    # 13 digits fits milliseconds through year 2286; zero-padded so lexicographic
    # order matches numeric order (a DynamoDB sort key sorts as a string).
    return f"{ts_ms:013d}#{session_id}#{hand_number}"


def recap_from_session(session, *, blueprint_name=None, ts_ms=None):
    """Build the persisted recap dict from a hand_over GameSession.

    Returns the dict the store writes. Captures everything we'd want for a
    later hand-history UI or coach, but no engine-internal state (range
    trackers, bet pattern, info-set keys) — those can be reconstructed if
    needed by replaying the actionLog. Card formats are display.

    Raises ValueError if the session isn't at `hand_over`.
    """
    d = session.data
    if d.get('status') != 'hand_over':
        raise ValueError("recap_from_session: session is not at hand_over")

    human_seat = d.get('human_seat')
    human_hole = d['p0_cards'] if human_seat == 0 else d['p1_cards']
    bot_hole = d['p1_cards'] if human_seat == 0 else d['p0_cards']

    # Only the cards that were actually revealed during the hand (a pre-river
    # fold leaves the rest of the board face-down; an all-in runout reveals all
    # five). `revealed_board` is the count, set by `_settle()`.
    revealed = int(d.get('revealed_board') or 0)
    community = list(d.get('community') or [])[:revealed]

    # action_log is the chronological per-action record across ALL streets
    # (history resets every street so isn't usable here). Each entry is
    # {player, street, action, chips}.
    action_log = [dict(a) for a in (d.get('action_log') or [])]

    result = dict(d.get('result') or {})
    human_delta = float(result.get('humanDelta') or 0.0)
    human_net_after = float(d.get('human_net') or 0.0)
    human_net_before = round(human_net_after - human_delta, 2)

    ts = _now_ms() if ts_ms is None else int(ts_ms)
    session_id = d.get('session_id')
    hand_number = int(d.get('hand_number') or 0)

    return {
        'playerId': d.get('player_id'),
        'sessionId': session_id,
        'handNumber': hand_number,
        'handKey': _hand_key(ts, session_id, hand_number),
        'ts': ts,                                            # ms since epoch
        'humanSeat': human_seat,                             # 0=SB/IP, 1=BB/OOP
        'humanHole': to_display_list(human_hole),
        'botHole': to_display_list(bot_hole),
        'community': to_display_list(community),
        'actionLog': action_log,
        'startingPot': round(float(d.get('starting_pot') or 0.0), 2),
        'result': result,                                    # {humanDelta, winner, reason, finalPot}
        'humanNetBefore': human_net_before,
        'humanNetAfter': round(human_net_after, 2),
        'menuMode': getattr(session, 'menu_mode', None),
        'blueprint': blueprint_name,
    }


class HandStore(ABC):
    @abstractmethod
    def put(self, recap):
        """Persist a recap dict (built by `recap_from_session`). Idempotent on
        (playerId, handKey)."""

    @abstractmethod
    def list_for_player(self, player_id, *, n=20):
        """Return up to `n` most-recent recap dicts for the player, newest first."""

    @abstractmethod
    def get(self, player_id, hand_key):
        """Fetch one recap by its compound key, or None."""


class InMemoryHandStore(HandStore):
    """Process-local recap rows (dev + tests). NOT shared across workers."""

    def __init__(self):
        self._rows = {}                 # (playerId, handKey) -> recap
        self._by_player = {}            # playerId -> list of handKeys (insertion order)
        self._lock = threading.Lock()

    def put(self, recap):
        pid = recap.get('playerId')
        hk = recap.get('handKey')
        if not pid or not hk:
            raise ValueError("recap is missing playerId/handKey")
        with self._lock:
            key = (pid, hk)
            if key not in self._rows:
                self._by_player.setdefault(pid, []).append(hk)
            self._rows[key] = dict(recap)

    def list_for_player(self, player_id, *, n=20):
        with self._lock:
            keys = list(self._by_player.get(player_id) or [])
            # handKey starts with a zero-padded epoch -> lexicographic sort = chronological.
            keys.sort(reverse=True)
            keys = keys[: max(0, int(n))]
            return [dict(self._rows[(player_id, k)]) for k in keys]

    def get(self, player_id, hand_key):
        with self._lock:
            r = self._rows.get((player_id, hand_key))
            return dict(r) if r else None


class DynamoDBHandStore(HandStore):
    """DynamoDB-backed recap rows for prod. One PutItem per completed hand;
    list_for_player is a single Query with ScanIndexForward=False. boto3 is
    lazy-imported so the in-memory default needs no AWS dependency.

    Exercised in CI via moto / DynamoDB Local (see test_hand_store.py), not
    in the no-AWS dev environment.
    """

    def __init__(self, table_name, region=None, endpoint_url=None):
        import boto3
        from botocore.exceptions import ClientError
        from botocore.config import Config
        self._ClientError = ClientError
        kwargs = {'config': Config(retries={'mode': 'adaptive', 'max_attempts': 5})}
        if region:
            kwargs['region_name'] = region
        if endpoint_url:
            kwargs['endpoint_url'] = endpoint_url
        self._table = boto3.resource('dynamodb', **kwargs).Table(table_name)

    @staticmethod
    def _to_ddb(item):
        """DynamoDB rejects bare floats; coerce to Decimal. Lists/dicts recurse."""
        from decimal import Decimal
        if isinstance(item, float):
            return Decimal(str(item))
        if isinstance(item, dict):
            return {k: DynamoDBHandStore._to_ddb(v) for k, v in item.items()}
        if isinstance(item, list):
            return [DynamoDBHandStore._to_ddb(v) for v in item]
        return item

    @staticmethod
    def _from_ddb(item):
        """Decimals -> int/float for the public dict shape."""
        from decimal import Decimal
        if item is None:
            return None
        if isinstance(item, Decimal):
            return int(item) if item % 1 == 0 else float(item)
        if isinstance(item, dict):
            return {k: DynamoDBHandStore._from_ddb(v) for k, v in item.items()}
        if isinstance(item, list):
            return [DynamoDBHandStore._from_ddb(v) for v in item]
        return item

    def put(self, recap):
        pid = recap.get('playerId')
        hk = recap.get('handKey')
        if not pid or not hk:
            raise ValueError("recap is missing playerId/handKey")
        self._table.put_item(Item=self._to_ddb(recap))

    def list_for_player(self, player_id, *, n=20):
        resp = self._table.query(
            KeyConditionExpression='playerId = :p',
            ExpressionAttributeValues={':p': player_id},
            ScanIndexForward=False,                 # newest first
            Limit=max(1, int(n)))
        return [self._from_ddb(it) for it in resp.get('Items', [])]

    def get(self, player_id, hand_key):
        item = self._table.get_item(
            Key={'playerId': player_id, 'handKey': hand_key}).get('Item')
        return self._from_ddb(item)

    @staticmethod
    def create_table_if_missing(table_name, region=None, endpoint_url=None):
        """One-time provisioning helper. Run once at deploy."""
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
                AttributeDefinitions=[
                    {'AttributeName': 'playerId', 'AttributeType': 'S'},
                    {'AttributeName': 'handKey', 'AttributeType': 'S'},
                ],
                KeySchema=[
                    {'AttributeName': 'playerId', 'KeyType': 'HASH'},
                    {'AttributeName': 'handKey', 'KeyType': 'RANGE'},
                ],
                BillingMode='PAY_PER_REQUEST')
            client.get_waiter('table_exists').wait(TableName=table_name)
            # PITR for disaster recovery. Hand recaps are the most precious
            # launch-window data — a wipe is unrecoverable without this.
            try:
                client.update_continuous_backups(
                    TableName=table_name,
                    PointInTimeRecoverySpecification={'PointInTimeRecoveryEnabled': True})
            except ClientError as e2:
                code2 = e2.response.get('Error', {}).get('Code', '')
                if code2 not in ('ContinuousBackupsUnavailableException',
                                 'UnknownOperationException', 'ValidationException'):
                    raise
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') != 'ResourceInUseException':
                raise


def make_hand_store():
    """Build the hand store named by ALLIN_STORE_BACKEND (default 'memory')."""
    backend = os.environ.get('ALLIN_STORE_BACKEND', 'memory').strip().lower()
    if backend in ('', 'memory', 'inmemory'):
        return InMemoryHandStore()
    if backend in ('dynamodb', 'dynamo'):
        return DynamoDBHandStore(
            table_name=os.environ.get('ALLIN_HANDS_TABLE', 'allin-hands'),
            region=(os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION')),
            endpoint_url=os.environ.get('ALLIN_DYNAMODB_ENDPOINT') or None)
    raise ValueError(f"Unknown ALLIN_STORE_BACKEND={backend!r} (use 'memory' or 'dynamodb')")
