#!/usr/bin/env python
"""
One-time backfill for the LIVE per-version counters on the global stats row.

WHY. `GlobalStats` gained live per-bot-version running counters (`vh_<version>`/`vn_<version>`),
bumped on every hand, so the +EV card's v1/v2 numbers are live with no hand-table scan. New rows
start those counters at 0, so the historical prefix (hands played before the counters shipped) must
be seeded from the per-hand recap table once, at the v2 cutover. After this runs, the live ADD path
keeps them current.

WHAT IT DOES (deliberately conservative -- see scripts review / BUG_LOG):
  - Computes per-version {hands, humanNetBB} from the recaps (`HandStore.version_aggregates`).
  - SEEDS ONLY the per-version attrs `vh_<v>`/`vn_<v>` on the global row. It does NOT touch
    `totalHands`/`totalNetBB` -- those are already live-correct in prod (the counter is incremented
    per hand), and overwriting them from the recap sum could move the public headline by the
    counter-vs-recap drift (the two writes are independent try/excepts, so they can differ by a few).
  - Uses a SET (idempotent: re-running with the same data overwrites, never accumulates) with aliased
    dynamic attribute names + Decimal for the net (DynamoDB rejects bare floats).
  - Dry-run by DEFAULT; prints the plan. Pass --apply to write.

LABEL CONSOLIDATION. Historical prod recaps may be tagged with a build SHA / 'v1.0.0' (the old
ALLIN_GIT_SHA fallback, since removed) rather than a coarse 'v1'. Since all pre-v2 prod hands were the
v1 bot, pass `--consolidate-to v1` to fold EVERY label found in the scan into 'v1' (so the card's v1
filter shows them). Verify the labels first with a plain dry-run. NB run `--consolidate-to` ONLY
BEFORE the v2 cutover -- after it, v2 recaps exist and would be folded into v1 too. CAVEAT: this fixes
only the GLOBAL CARD counters; it does NOT re-tag recap rows, so the leaderboard's per-version dropdown
(still recap-scan-backed) keeps the historical SHA/'v1.0.0' labels and its v1 slice undercounts the
pre-counter hands. The card is exact; the leaderboard v1 historical slice is approximate unless you
also re-tag the recaps (separate migration). The 'all' views are unaffected.

SAFETY (race). A SET races a concurrent live ADD on the same attr (lost update). Run it in a brief
QUIESCE window (scale the backend to zero / drain) so no hand completes mid-backfill -- then it's
exact. At launch volume that's a few seconds. Order: deploy the counter-writing code -> backfill
(quiesced, still on the v1 label) -> flip ALLIN_BOT_VERSION=v2 + the v2 blueprint.

ROLLBACK. Capture the global row before applying. If the counter-writing code was deployed first
(the recommended order), live hands have already ADDed nonzero vh_/vn_, so a blind REMOVE is NOT
clean -- it would also drop the legitimately live-counted hands. Prefer re-running with the captured
pre-values, or PITR (`allin-global` has it enabled) as the backstop. REMOVE is clean ONLY if the
backfill ran before any live hand was ever counted.

Run from backend/bot/ with the same store env as the API (ALLIN_GLOBAL_TABLE, ALLIN_DYNAMODB_ENDPOINT,
AWS creds/region). Example (local DynamoDB dry-run):
    ALLIN_DYNAMODB_ENDPOINT=http://localhost:8000 AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local \
    AWS_DEFAULT_REGION=ap-southeast-1 python scripts/backfill_global_version_counters.py
Add --apply to write, --consolidate-to v1 to fold all labels into v1.
"""
import argparse
import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.hand_store import make_hand_store

_SINGLETON_ID = 'global'


def _aggregate(consolidate_to=None):
    """Per-version {hands, humanNetBB} from the recaps, optionally folding all labels into one."""
    totals = make_hand_store().version_aggregates()['totals']
    if consolidate_to:
        merged = {'hands': 0, 'humanNetBB': 0.0}
        for d in totals.values():
            merged['hands'] += d['hands']
            merged['humanNetBB'] += d['humanNetBB']
        return {consolidate_to: merged}
    return totals


def _global_table():
    import boto3
    from botocore.config import Config
    kwargs = {'config': Config(retries={'mode': 'adaptive', 'max_attempts': 5})}
    region = os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION')
    if region:
        kwargs['region_name'] = region
    ep = os.environ.get('ALLIN_DYNAMODB_ENDPOINT')
    if ep:
        kwargs['endpoint_url'] = ep
    name = os.environ.get('ALLIN_GLOBAL_TABLE', 'allin-global')
    return boto3.resource('dynamodb', **kwargs).Table(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true',
                    help='actually write (default is a dry-run that only prints the plan)')
    ap.add_argument('--consolidate-to', metavar='LABEL', default=None,
                    help="fold ALL scanned version labels into this one (e.g. v1 for the cutover)")
    args = ap.parse_args()

    if os.environ.get('ALLIN_STORE_BACKEND', '').strip().lower() not in ('dynamodb', 'dynamo'):
        # version_aggregates on the in-memory store is process-local + empty; this tool targets DDB.
        os.environ['ALLIN_STORE_BACKEND'] = 'dynamodb'

    agg = _aggregate(args.consolidate_to)
    print("Per-version totals to SEED onto the global row (vh_<v>/vn_<v>), from the recap table:")
    for v, d in sorted(agg.items()):
        print(f"  {v:>14}: hands={d['hands']:>8}  humanNetBB={round(d['humanNetBB'], 2)}")
    print(f"  (totalHands/totalNetBB are LEFT UNTOUCHED -- live counter is authoritative)")

    if not args.apply:
        print("\nDRY-RUN. Re-run with --apply (in a quiesced window) to write. "
              "Verify the labels above are what you expect (use --consolidate-to v1 to fold "
              "SHA/'v1.0.0' historical labels into v1).")
        return

    expr = 'SET'
    names, vals, parts = {}, {}, []
    for i, (v, d) in enumerate(sorted(agg.items())):
        parts.append(f' #h{i} = :h{i}, #n{i} = :n{i}')
        names[f'#h{i}'] = f'vh_{v}'
        names[f'#n{i}'] = f'vn_{v}'
        vals[f':h{i}'] = int(d['hands'])
        vals[f':n{i}'] = Decimal(str(round(d['humanNetBB'], 2)))
    if not parts:
        print("nothing to backfill (no recaps).")
        return
    _global_table().update_item(
        Key={'statId': _SINGLETON_ID},
        UpdateExpression=expr + ','.join(parts),
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=vals)
    print("\nAPPLIED. Verify with GET /api/stats (byVersion) and a read-only re-run of this dry-run.")


if __name__ == '__main__':
    main()
