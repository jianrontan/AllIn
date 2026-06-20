#!/usr/bin/env python3
"""Export allin-hands recaps to local JSONL for offline opponent modeling (Phase 6 / E0).

Read-only scan of the prod `allin-hands` table (one recap per completed hand, written by
`game.hand_store.recap_from_session`). Joins handles from `allin-players` for readability and
optional handle-based filtering. Needs AWS credentials, like `scripts/player_stats.py`.

Run from backend/bot:
    python scripts/export_hands.py                       # all players -> analysis/opponent_models/hands_export.jsonl
    python scripts/export_hands.py --min-hands 50        # drop small-sample players
    python scripts/export_hands.py --players XYyyyy Lay Choonweng ron Kahtong   # by handle (or playerId)

Output: one JSON object per line (the recap as stored, Decimals coerced to int/float), with an
added `_handle` field. A per-player summary is printed to stdout. This is the input to the E0
cleaning/filtering pass and the per-(bucket,ctx) frequency fitter (see docs/EXPLOITATION_PLAN.md).
"""
import argparse
import json
import os
from collections import defaultdict
from decimal import Decimal


def _clean(o):
    """DynamoDB Decimals -> int/float recursively, so the row is JSON-serializable."""
    if isinstance(o, Decimal):
        return int(o) if o % 1 == 0 else float(o)
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def _scan_all(table):
    """Paginated full scan -> list of items (the table is small, ~thousands of rows)."""
    rows, resp = [], table.scan()
    rows += resp['Items']
    while 'LastEvaluatedKey' in resp:
        resp = table.scan(ExclusiveStartKey=resp['LastEvaluatedKey'])
        rows += resp['Items']
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', default=os.environ.get('AWS_REGION', 'ap-southeast-1'))
    ap.add_argument('--hands-table', default=os.environ.get('ALLIN_HANDS_TABLE', 'allin-hands'))
    ap.add_argument('--players-table',
                    default=os.environ.get('ALLIN_PLAYERS_TABLE', 'allin-players'))
    ap.add_argument('--out', default='analysis/opponent_models/hands_export.jsonl')
    ap.add_argument('--min-hands', type=int, default=0,
                    help='only export players with >= this many captured hands')
    ap.add_argument('--players', nargs='*',
                    help='restrict to these players (handle OR playerId); default = all')
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='drop these players (handle OR playerId) from the export')
    args = ap.parse_args()

    import boto3
    db = boto3.resource('dynamodb', region_name=args.region)

    # 1. handle <-> playerId from the players table (for readability + handle filtering).
    prows = _scan_all(db.Table(args.players_table))
    handle_of = {p['playerId']: (p.get('handle') or 'anon')
                 for p in prows if 'playerId' in p and not p.get('merged_into')
                 and not str(p['playerId']).startswith('handle#')}   # skip handle reservations
    # Remap a merged-away (second-device) playerId to its canonical account, so a returning
    # player's hands aren't split across two models (data-review fix). Anon-only multi-device
    # can't be linked and is left as-is (acceptable).
    merged_of = {p['playerId']: p['merged_into'] for p in prows
                 if 'playerId' in p and p.get('merged_into')}
    # Resolve --players (a mix of handles and ids) to a playerId allow-set.
    allow = None
    if args.players:
        want = set(args.players)
        id_by_handle = {h: pid for pid, h in handle_of.items()}
        allow = {x if x in handle_of else id_by_handle.get(x) for x in want}
        allow.discard(None)
        missing = want - {handle_of.get(a, a) for a in allow} - allow
        if missing:
            print(f"WARNING: could not resolve {sorted(missing)} to a playerId "
                  f"(handle not in {args.players_table}); skipped.")

    # Resolve --exclude (handle/id) to a deny-set of playerIds.
    deny = set()
    if args.exclude:
        id_by_handle = {h: pid for pid, h in handle_of.items()}
        deny = {x if x in handle_of else id_by_handle.get(x) for x in args.exclude}
        deny.discard(None)

    # 2. scan the hands table, group by playerId.
    hands = _scan_all(db.Table(args.hands_table))
    by_player = defaultdict(list)
    for h in hands:
        pid = h.get('playerId')
        if pid is None:
            continue
        pid = merged_of.get(pid, pid)              # second-device hands -> canonical account
        if pid in deny or (allow is not None and pid not in allow):
            continue
        by_player[pid].append(h)

    # 3. apply --min-hands, sort each player's hands chronologically (handKey is epoch-prefixed),
    #    and write JSONL + a per-player summary.
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    written = 0
    summary = []
    with open(args.out, 'w', encoding='utf-8') as f:
        for pid, rows in by_player.items():
            if len(rows) < args.min_hands:
                continue
            rows.sort(key=lambda r: str(r.get('handKey', '')))      # chronological
            for r in rows:
                rec = _clean(r)
                rec['_handle'] = handle_of.get(pid, 'anon')
                f.write(json.dumps(rec, separators=(',', ':')) + '\n')
                written += 1
            summary.append((handle_of.get(pid, 'anon'), pid, len(rows)))

    summary.sort(key=lambda x: x[2], reverse=True)
    print(f"scanned {len(hands)} hand recaps across {len(by_player)} players; "
          f"wrote {written} hands for {len(summary)} players to {args.out}\n")
    print(f"  {'handle':<18}{'hands':>7}  playerId")
    print("  " + "-" * 50)
    for hdl, pid, n in summary:
        print(f"  {hdl:<18}{n:>7}  {pid}")


if __name__ == '__main__':
    main()
