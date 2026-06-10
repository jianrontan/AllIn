#!/usr/bin/env python3
# scripts/provision_dynamodb.py
"""
One-time DynamoDB table provisioning for the AllIn deploy. Run once with admin
AWS credentials (`aws configure` first); idempotent on re-runs (existing tables
are left alone, PITR is re-enabled, TTL is re-enabled).

Each store class owns its own create_table_if_missing() with the right schema +
PITR + TTL where applicable, so this script is a thin orchestrator. Add a new
store here when one ships; everything else stays in the store class.

Usage:
    python scripts/provision_dynamodb.py                  # default region from $AWS_REGION
    python scripts/provision_dynamodb.py --region ap-southeast-1
    python scripts/provision_dynamodb.py --endpoint http://localhost:8000   # DDB Local
"""
import argparse
import os
import sys

# Make the bot package importable without a pip install -e.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, 'backend', 'bot', 'src'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        '--region',
        default=(os.environ.get('AWS_REGION')
                 or os.environ.get('AWS_DEFAULT_REGION')
                 or 'ap-southeast-1'),
        help='AWS region (default: $AWS_REGION or ap-southeast-1)')
    ap.add_argument(
        '--endpoint', default=None,
        help='DynamoDB endpoint override (e.g. http://localhost:8000 for DDB Local)')
    ap.add_argument(
        '--sessions-table',  default=os.environ.get('ALLIN_DYNAMODB_TABLE', 'allin-sessions'))
    ap.add_argument(
        '--players-table',   default=os.environ.get('ALLIN_PLAYERS_TABLE',  'allin-players'))
    ap.add_argument(
        '--global-table',    default=os.environ.get('ALLIN_GLOBAL_TABLE',   'allin-global'))
    ap.add_argument(
        '--hands-table',     default=os.environ.get('ALLIN_HANDS_TABLE',    'allin-hands'))
    args = ap.parse_args()

    # Lazy imports — fail fast with a clean message if boto3 is missing.
    try:
        from game.session_store      import DynamoDBSessionStore
        from game.player_store       import DynamoDBPlayerStore
        from game.global_stats_store import DynamoDBGlobalStatsStore
        from game.hand_store         import DynamoDBHandStore
    except ImportError as e:
        sys.exit(f"boto3 (or backend/bot/src on sys.path) missing: {e}")

    kw = {'region': args.region, 'endpoint_url': args.endpoint}
    print(f"region={args.region}  endpoint={args.endpoint or '(default AWS)'}")
    for name, cls in (
        (args.sessions_table,  DynamoDBSessionStore),
        (args.players_table,   DynamoDBPlayerStore),
        (args.global_table,    DynamoDBGlobalStatsStore),
        (args.hands_table,     DynamoDBHandStore),
    ):
        print(f"  - {name}: provisioning … ", end='', flush=True)
        cls.create_table_if_missing(name, **kw)
        print("ok")
    print("done")


if __name__ == '__main__':
    main()
