#!/usr/bin/env python3
"""Dev launcher: start the BACKEND (against the LOCAL DynamoDB, exploitation ON) + the FRONTEND, and
FORCE every session to a chosen profiled player (default 'ron') so you play AS them and the bot
EXPLOITS their fitted model + history. One command to test the exploiter end-to-end.

Prereqs (see CLAUDE.md "Local exploitation / stats testing"): DynamoDB Local running + cloned
(docker ps shows allin-ddb; scripts/clone_dynamo_to_local.py has run). Ctrl-C stops both servers.

  python scripts/dev_launch.py                 # play as ron, exploit on (live last-N 100/refresh 50)
  python scripts/dev_launch.py --handle Lay    # play as a different profiled player
  python scripts/dev_launch.py --arm off       # CONTROL arm (pure blueprint) to A/B against
"""
import argparse
import os
import subprocess
import sys

_BOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # backend/bot
_BACKEND = os.path.dirname(_BOT)                                         # backend
_API = os.path.join(_BACKEND, 'api')                                    # backend/api
_REPO = os.path.dirname(_BACKEND)                                       # repo root
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
    ap.add_argument('--no-turn-solve', dest='turn_solve', action='store_false',
                    help="DISABLE the exploitative turn solve (ON by default; ~12s/solve)")
    ap.add_argument('--anchor', default='auto', choices=['auto', 'belief', 'blueprint', 'confidence'],
                    help="river gadget anchor. DEFAULT 'auto' = PROD-EQUIVALENT (exploits only when "
                         "proven <=blueprint, else clamps). 'belief' = max exploit, NO safety floor.")
    ap.add_argument('--gate', default='0.2',
                    help="exploit/all-in confidence gate. DEFAULT 0.2 = PROD. Lower (e.g. 0.1) only to "
                         "FORCE the exploit to engage on weaker reads for mechanism testing.")
    args = ap.parse_args()

    os.environ.setdefault('AWS_EC2_METADATA_DISABLED', 'true')
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
        'ALLIN_DEBUG_OVERLAY': '1',                # watch the bot's read of you mid-hand
        'ALLIN_TURN_SOLVE': '1' if args.turn_solve else '0',
    })
    # Feature banner so it's never a guessing game what's actually ENABLED for this run.
    print(f"playing AS '{args.handle}' ({pid[:8]}...) -- frontend auto-signs-in (VITE_DEV_PLAYER_ID)")
    print("  FEATURES: " + " | ".join([
        f"arm={args.arm}({'EXPLOIT' if args.arm == 'on' else 'CONTROL'})",
        "exploit=ON", f"anchor={args.anchor}",
        f"turn-solve={'ON' if args.turn_solve else 'off'}",
        f"river-solve=ON", f"gate={args.gate}",
        "live-last-N=100/50"]))
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
