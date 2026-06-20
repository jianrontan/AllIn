#!/usr/bin/env python3
"""Clone prod DynamoDB tables into a LOCAL DynamoDB (DynamoDB Local / LocalStack) so dev can run
against a PERSISTENT, prod-faithful copy of ALL the data -- e.g. to build/test a stats UI -- WITHOUT
touching prod and WITHOUT writing dev hands back into the prod tables.

Why this and not the in-memory seed: the in-memory store is wiped on restart and only holds hands.
A local DynamoDB persists across restarts (with -dbPath), holds EVERY table (hands/players/global/
sessions), and is the SAME code path prod uses (ALLIN_STORE_BACKEND=dynamodb + ALLIN_DYNAMODB_ENDPOINT).

Setup:
  1. Run DynamoDB Local with on-disk persistence (mount a volume so it survives restarts):
       docker run -d -p 8000:8000 -v dynamodb-local-data:/home/dynamodblocal/data \
         amazon/dynamodb-local -jar DynamoDBLocal.jar -sharedDb -dbPath /home/dynamodblocal/data
  2. Clone prod -> local (needs prod READ creds; only ever WRITES to --endpoint, never prod):
       python scripts/clone_dynamo_to_local.py --endpoint http://localhost:8000
  3. Point the API at the local copy:
       ALLIN_STORE_BACKEND=dynamodb ALLIN_SESSION_STORE=dynamodb \
       ALLIN_DYNAMODB_ENDPOINT=http://localhost:8000 AWS_ACCESS_KEY_ID=local \
       AWS_SECRET_ACCESS_KEY=local python strategy_api.py

Each table is described on the source, re-created on the target with the SAME key schema (+ GSIs),
then scanned and batch-written. Re-running upserts (existing target tables are reused). The
client-level scan/put copies the raw DynamoDB item format, so no type marshalling is needed.
"""
import argparse
import os
import time

import boto3


def _scan_all(client, table):
    items, resp = [], client.scan(TableName=table)
    items += resp.get('Items', [])
    while 'LastEvaluatedKey' in resp:
        resp = client.scan(TableName=table, ExclusiveStartKey=resp['LastEvaluatedKey'])
        items += resp.get('Items', [])
        print(f"    ...scanned {len(items)} items", flush=True)
    return items


def clone_table(src, dst, name):
    desc = src.describe_table(TableName=name)['Table']
    kwargs = dict(TableName=name,
                  AttributeDefinitions=desc['AttributeDefinitions'],
                  KeySchema=desc['KeySchema'],
                  BillingMode='PAY_PER_REQUEST')
    gsis = desc.get('GlobalSecondaryIndexes')
    if gsis:
        kwargs['GlobalSecondaryIndexes'] = [
            {'IndexName': g['IndexName'], 'KeySchema': g['KeySchema'], 'Projection': g['Projection']}
            for g in gsis]
    try:
        dst.create_table(**kwargs)
        dst.get_waiter('table_exists').wait(TableName=name)
        print(f"  created local table {name}")
    except dst.exceptions.ResourceInUseException:
        print(f"  local table {name} already exists -- upserting")

    items = _scan_all(src, name)
    for i in range(0, len(items), 25):                       # batch_write_item caps at 25 / call
        req = {name: [{'PutRequest': {'Item': it}} for it in items[i:i + 25]]}
        resp = dst.batch_write_item(RequestItems=req)
        unp = resp.get('UnprocessedItems') or {}
        attempts = 0
        while unp and attempts < 5:                           # rare on local DDB; retry to be safe
            time.sleep(0.2)
            resp = dst.batch_write_item(RequestItems=unp)
            unp = resp.get('UnprocessedItems') or {}
            attempts += 1
    print(f"  cloned {len(items)} items into {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoint',
                    default=os.environ.get('ALLIN_DYNAMODB_ENDPOINT', 'http://localhost:8000'),
                    help='LOCAL DynamoDB endpoint (the WRITE target)')
    ap.add_argument('--region', default=os.environ.get('AWS_REGION', 'ap-southeast-1'))
    ap.add_argument('--tables', nargs='*', default=[
        os.environ.get('ALLIN_HANDS_TABLE', 'allin-hands'),
        os.environ.get('ALLIN_PLAYERS_TABLE', 'allin-players'),
        os.environ.get('ALLIN_GLOBAL_TABLE', 'allin-global'),
        os.environ.get('ALLIN_DYNAMODB_TABLE', 'allin-sessions'),
    ])
    args = ap.parse_args()

    # This is NOT an EC2 box, so never let boto3 hang ~2 min on the instance-metadata service when
    # creds aren't in env/profile -- fail fast on missing creds instead.
    os.environ.setdefault('AWS_EC2_METADATA_DISABLED', 'true')
    src = boto3.client('dynamodb', region_name=args.region)                    # prod (default creds)
    dst = boto3.client('dynamodb', region_name=args.region, endpoint_url=args.endpoint)  # local

    # Pre-flight: instant, clear errors instead of a silent hang.
    try:
        who = boto3.client('sts', region_name=args.region).get_caller_identity()
        print(f"prod creds OK (account {who.get('Account')})")
    except Exception as e:
        raise SystemExit(f"\nAWS creds not usable in THIS shell: {e}\n"
                         f"  -> run `aws configure` (writes ~/.aws, seen by all shells) OR export "
                         f"AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION here.")
    try:
        dst.list_tables()
    except Exception as e:
        raise SystemExit(f"\nLocal DynamoDB at {args.endpoint} not reachable: {e}\n"
                         f"  -> is DynamoDB Local running?  docker ps   /   curl {args.endpoint}")
    print(f"cloning {args.tables}\n  from prod ({args.region}) -> local ({args.endpoint})")
    for name in args.tables:
        print(f"table {name}:")
        try:
            clone_table(src, dst, name)
        except Exception as e:                               # one bad table shouldn't abort the rest
            print(f"  SKIP {name}: {type(e).__name__}: {e}")
    print(f"\ndone. run: ALLIN_STORE_BACKEND=dynamodb ALLIN_SESSION_STORE=dynamodb "
          f"ALLIN_DYNAMODB_ENDPOINT={args.endpoint} python strategy_api.py")


if __name__ == '__main__':
    main()
