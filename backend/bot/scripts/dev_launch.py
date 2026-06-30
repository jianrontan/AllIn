#!/usr/bin/env python3
"""Dev launcher: start the BACKEND (against the LOCAL DynamoDB, exploitation ON) + the FRONTEND, and
FORCE every session to a chosen profiled player (default 'ron') so you play AS them and the bot
EXPLOITS their fitted model + history. One command to test the exploiter end-to-end.

Prereqs (see CLAUDE.md "Local exploitation / stats testing"): Docker Desktop running + DynamoDB Local up
(`docker start allin-ddb`). The local DB must hold the cloned prod data -- from a prior clone, or pull it
now with --refresh.

--refresh pulls the LATEST prod data into the local DynamoDB before launching (so you test ron's CURRENT
history). It READS prod, so you need prod READ creds in THIS shell FIRST -- ONE of:

    aws configure                                # IAM access keys -> persists in ~/.aws (one-time, until keys rotate), OR
    aws sso login                                # if your account uses AWS SSO (re-login every few hours)
    aws dynamodb list-tables --region ap-southeast-1   # verify creds work BEFORE --refresh

Read-only creds suffice (the clone only READS prod, only WRITES the local --endpoint). WITHOUT --refresh
NO creds are needed -- it just uses whatever's already in the local DB.

    docker start allin-ddb                       # boot DynamoDB Local (data persists across sessions)
    python scripts/dev_launch.py --refresh       # PULL latest prod data, then play as ron (NEEDS creds)
    python scripts/dev_launch.py                 # play as ron, exploit on, PROD-equivalent (auto/0.2)
    python scripts/dev_launch.py --handle Lay    # play as a different profiled player
    python scripts/dev_launch.py --arm off       # CONTROL arm (pure blueprint) to A/B against
    docker stop allin-ddb                        # stop DynamoDB Local. Ctrl-C stops both servers.

Watch the DEBUG OVERLAY mid-hand for the bot's read of you (confidence + top hands + whether the exploit
tilt / river solve fired). Turn solving is REMOVED (dead end) -- this is blueprint + river solver + the
pre-river exploit tilt. River solves are ~2-3s on a dev box.
"""
import argparse
import os
import subprocess
import sys

_BOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # backend/bot
_BACKEND = os.path.dirname(_BOT)                                         # backend
_API = os.path.join(_BACKEND, 'api')                                     # backend/api
_REPO = os.path.dirname(_BACKEND)                                        # repo root
_FRONTEND = os.path.join(_REPO, 'frontend')


def resolve_player(handle, endpoint, region):
    """Find a player's UUID by handle in the LOCAL DynamoDB players table (low-level scan)."""
    import boto3
    c = boto3.client('dynamodb', region_name=region, endpoint_url=endpoint,
                     aws_access_key_id='local', aws_secret_access_key='local')
    table = os.environ.get('ALLIN_PLAYERS_TABLE', 'allin-players')
    resp = c.scan(TableName=table)
    items = resp.get('Items', [])
    while 'LastEvaluatedKey' in resp:
        resp = c.scan(TableName=table, ExclusiveStartKey=resp['LastEvaluatedKey'])
        items += resp.get('Items', [])
    for it in items:
        if it.get('handle', {}).get('S', '').lower() == handle.lower():
            return it['playerId']['S']
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--handle', default='ron', help="profiled player to play AS / exploit")
    ap.add_argument('--arm', default='on', choices=['on', 'off'],
                    help="'on' = exploit (treatment); 'off' = pure-blueprint control (A/B)")
    ap.add_argument('--endpoint', default='http://localhost:8000')
    ap.add_argument('--region', default=os.environ.get('AWS_REGION', 'ap-southeast-1'))
    ap.add_argument('--blueprint',
                    default=os.path.join(_BOT, 'analysis', 'blueprints', 'snapshots', 'snap_52500000.db'))
    ap.add_argument('--anchor', default='auto', choices=['auto', 'belief', 'blueprint', 'confidence'],
                    help="river gadget anchor. DEFAULT 'auto' = PROD-EQUIVALENT (exploits only when "
                         "proven <=blueprint, else clamps). 'belief' = max exploit, NO safety floor.")
    ap.add_argument('--gate', default='0.2',
                    help="exploit/all-in confidence gate. DEFAULT 0.2 = PROD. Lower (e.g. 0.1) only to "
                         "FORCE the exploit to engage on weaker reads for mechanism testing.")
    ap.add_argument('--bot-version', dest='bot_version', default='v2.0.0',
                    help="ALLIN_BOT_VERSION tag for hands played this session (the bot-version filter on "
                         "the stats; default v2.0.0 = the 30/24 bot, MATCHING prod's label so dev hands "
                         "bucket the same way). Bump when the bot meaningfully changes.")
    ap.add_argument('--refresh', action='store_true',
                    help="first PULL the latest prod data into the local DynamoDB "
                         "(runs clone_dynamo_to_local.py) so you test against ron's CURRENT history. Needs "
                         "prod READ creds in THIS shell (aws configure); only WRITES to the local --endpoint.")
    args = ap.parse_args()

    os.environ.setdefault('AWS_EC2_METADATA_DISABLED', 'true')
    if args.refresh:
        print("refreshing local DynamoDB from prod (clone_dynamo_to_local.py; needs prod creds) ...")
        clone = os.path.join(_BOT, 'scripts', 'clone_dynamo_to_local.py')
        # Run with the AMBIENT env (your real prod creds) -- NOT the 'local' dummy creds the backend uses;
        # the clone READS prod and only WRITES to --endpoint. cwd=_BOT so its sys.path resolves.
        if subprocess.run([sys.executable, clone, '--endpoint', args.endpoint,
                           '--region', args.region], cwd=_BOT).returncode != 0:
            print("  clone FAILED (prod creds missing in this shell? run `aws configure`). aborting.")
            sys.exit(1)
    try:
        pid = resolve_player(args.handle, args.endpoint, args.region)
    except Exception as e:
        print(f"could not reach the local DynamoDB at {args.endpoint}: {e}\n"
              f"  -> is DynamoDB Local up + cloned?  docker ps  /  python scripts/clone_dynamo_to_local.py")
        sys.exit(1)
    if not pid:
        print(f"no player with handle '{args.handle}' in the local DynamoDB. Cloned the data? "
              f"(scripts/clone_dynamo_to_local.py)")
        sys.exit(1)
    env = dict(os.environ)
    env.update({
        'ALLIN_STORE_BACKEND': 'dynamodb', 'ALLIN_SESSION_STORE': 'dynamodb',
        'ALLIN_DYNAMODB_ENDPOINT': args.endpoint,
        'AWS_ACCESS_KEY_ID': 'local', 'AWS_SECRET_ACCESS_KEY': 'local',
        'AWS_DEFAULT_REGION': args.region,
        'ALLIN_BLUEPRINT_DB': args.blueprint,
        'ALLIN_OPPONENT_MODEL_DIR': os.path.join(_BOT, 'analysis', 'opponent_models'),
        'ALLIN_EXPLOIT': '1', 'ALLIN_AB_ARM': args.arm,
        'ALLIN_EXPLOIT_RECENT_N': '100', 'ALLIN_EXPLOIT_RECENT_REFRESH': '50',
        'ALLIN_GADGET_ANCHOR': args.anchor, 'ALLIN_MMAP_POSTFLOP': '1',
        'ALLIN_GUARD_CONFIDENCE': args.gate,       # prod 0.2; lower only to force-engage for testing
        'ALLIN_DEV_FORCE_PLAYER': pid,             # backend: every session is this player -> exploit them
        'VITE_DEV_PLAYER_ID': pid,                 # frontend: vite exposes this -> auto-sign-in as them
        'ALLIN_BOT_VERSION': args.bot_version,     # tag dev hands like prod (default v2 = the 30/24 bot)
        'ALLIN_DEBUG_OVERLAY': '1',                # watch the bot's read of you mid-hand
    })
    # Feature banner so it's never a guessing game what's actually ENABLED for this run.
    print(f"playing AS '{args.handle}' ({pid[:8]}...) -- frontend auto-signs-in (VITE_DEV_PLAYER_ID)")
    print("  FEATURES: " + " | ".join([
        f"arm={args.arm}({'EXPLOIT' if args.arm == 'on' else 'CONTROL'})",
        "exploit=ON", f"anchor={args.anchor}",
        "river-solve=ON (~2-3s on this dev box; 24s budget is the prod 0.25-vCPU cap; turn-solve SHELVED)",
        f"gate={args.gate}", "live-last-N=100/50"]))
    safe = (args.anchor == 'auto' and args.gate == '0.2')
    print("  -> river clamps <=blueprint (auto), engages only on confident reads (gate 0.2)" if safe else
          f"  -> AGGRESSIVE (anchor={args.anchor}, gate={args.gate}): more exploit, weaker safety floor")
    print("starting backend (http://localhost:5000) + frontend (http://localhost:5173) ... Ctrl-C to stop")
    backend = subprocess.Popen([sys.executable, 'strategy_api.py'], cwd=_API, env=env)
    frontend = subprocess.Popen('npm run dev', cwd=_FRONTEND, shell=True, env=env)
    try:
        backend.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in (frontend, backend):
            try:
                p.terminate()
            except Exception:
                pass


if __name__ == '__main__':
    main()
