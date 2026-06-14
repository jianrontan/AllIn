#!/usr/bin/env python3
"""Scan allin-players and print the player-stat distribution (not just the
leaderboard cut). Read-only. Run from anywhere:  python scripts/player_stats.py"""
import argparse
import os
from decimal import Decimal


def num(v):
    return float(v) if isinstance(v, Decimal) else (v or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', default=os.environ.get('AWS_REGION', 'ap-southeast-1'))
    ap.add_argument('--table', default=os.environ.get('ALLIN_PLAYERS_TABLE', 'allin-players'))
    ap.add_argument('--csv', help='also write every player row to this CSV path')
    ap.add_argument('--rank', action='store_true',
                    help='print every player ranked by BB/hand (player perspective)')
    ap.add_argument('--min-hands', type=int, default=0,
                    help='with --rank, hide players below this many hands '
                         '(filters out small-sample noise; 50 is the credible floor)')
    args = ap.parse_args()

    import boto3
    table = boto3.resource('dynamodb', region_name=args.region).Table(args.table)

    rows, resp = [], table.scan()
    rows += resp['Items']
    while 'LastEvaluatedKey' in resp:
        resp = table.scan(ExclusiveStartKey=resp['LastEvaluatedKey'])
        rows += resp['Items']

    # Drop handle-reservation items (PK 'handle#...') and merged-away rows -- not players.
    players = [r for r in rows
               if not str(r.get('playerId', '')).startswith('handle#')
               and not r.get('merged_into')]

    total = len(players)
    registered = sum(1 for p in players if p.get('isRegistered'))
    played = [p for p in players if num(p.get('hands')) > 0]
    print(f"rows scanned:        {len(rows)}")
    print(f"real players:        {total}  ({registered} registered, {total - registered} anon)")
    print(f"played >=1 hand:     {len(played)}")

    buckets = [(0, 1), (1, 10), (10, 50), (50, 100), (100, 500), (500, 10 ** 9)]
    print("\nhands played:")
    for lo, hi in buckets:
        n = sum(1 for p in players if lo <= num(p.get('hands')) < hi)
        label = f"{lo}-{hi - 1}" if hi < 10 ** 9 else f"{lo}+"
        print(f"  {label:>10}: {n}")

    nets = sorted(num(p.get('netBB')) for p in played)
    if nets:
        beat = sum(1 for x in nets if x > 0)
        mid = nets[len(nets) // 2]
        print(f"\nnetBB (player side): min {nets[0]:.0f}  median {mid:.0f}  max {nets[-1]:.0f}")
        print(f"players beating the bot (netBB>0): {beat}/{len(nets)}")

    print("\nmost active:")
    for p in sorted(played, key=lambda p: num(p.get('hands')), reverse=True)[:5]:
        print(f"  {(p.get('handle') or 'anon'):<16} "
              f"{int(num(p.get('hands'))):>5} hands  {num(p.get('netBB')):>8.0f} BB")

    if args.rank:
        # BB/hand from the PLAYER's perspective: positive = beating the bot.
        # Small samples are mostly variance, so --min-hands filters the noise
        # (50+ is the statistically credible floor; below that, ignore the rank).
        ranked = [p for p in played if num(p.get('hands')) >= args.min_hands]
        ranked.sort(key=lambda p: num(p.get('netBB')) / num(p.get('hands')),
                    reverse=True)
        title = (f"\nranked by BB/hand (>= {args.min_hands} hands):"
                 if args.min_hands else "\nranked by BB/hand (all players):")
        print(title)
        print(f"  {'#':>3} {'player':<18}{'hands':>6}{'netBB':>9}{'BB/hand':>9}  acct")
        print("  " + "-" * 50)
        for i, p in enumerate(ranked, 1):
            h, n = num(p.get('hands')), num(p.get('netBB'))
            name = p.get('handle') or 'anon'
            acct = 'yes' if p.get('isRegistered') else ''
            print(f"  {i:>3} {name:<18}{int(h):>6}{n:>9.0f}{n / h:>9.2f}  {acct}")

    if args.csv:
        import csv
        cols = ['playerId', 'handle', 'hands', 'netBB', 'isRegistered',
                'firstSeen', 'lastSeen']
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            for p in players:
                w.writerow({c: num(p.get(c)) if c in ('hands', 'netBB') else p.get(c)
                            for c in cols})
        print(f"\nwrote {len(players)} rows to {args.csv}")


if __name__ == '__main__':
    main()
