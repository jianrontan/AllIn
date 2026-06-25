#!/usr/bin/env python
"""
One-time migration: re-tag the HISTORICAL recap rows with an explicit `botVersion = v1.0.0`.

WHY. The +EV card and the leaderboard split hands by `hand_store.recap_version(recap)`, which returns the
explicit `botVersion` tag verbatim if set, else derives a label from the blueprint name. A prod-clone scan
(2026-06-24) shows historical v1 hands have `botVersion` UNSET -- they derive `v1` from `blueprint_final`,
so they already bucket correctly after the semver-map change. This tool makes that label EXPLICIT and
DURABLE: it stamps the single canonical `v1.0.0` on every pre-cutover hand (independent of the blueprint
name / map). It is belt-and-suspenders unless a dry-run reveals SHA-tagged hands (the old ALLIN_GIT_SHA
fallback) -- those would return the SHA verbatim and NOT bucket to v1, and this tool re-tags them too,
preserving the original SHA in `buildSha`. See docs/private/V2_MIGRATION_PLAN.md.

WHAT IT DOES (deliberately conservative -- mirrors backfill_global_version_counters.py):
  - Scans `allin-hands`, projecting only the key + version-relevant attrs.
  - For each row that is NOT already v1.0.0 and is NOT a v2 hand, update_item:
      SET botVersion = 'v1.0.0'  [, buildSha = <old SHA>  if the old tag looks like a SHA and buildSha absent]
  - SKIPS v2 hands (botVersion startswith 'v2', or a v2 blueprint stem) so an accidental post-cutover
    re-run can't clobber v2 tags. SKIPS rows already at the target (idempotent: safe to resume/re-run).
  - Dry-run by DEFAULT: prints the per-old-label counts so you eyeball it. Pass --apply to write.

SCOPE. This fixes the RECAP ROWS (the leaderboard's per-version dropdown, which is recap-scan-backed). The
+EV CARD's live counters are a SEPARATE surface -- seed those with backfill_global_version_counters.py.
Run THIS first (so the recap aggregate the backfill reads is already labelled v1.0.0 and you can drop its
--consolidate-to), then the backfill. Both are one-time, both before the cutover.

SAFETY. Each write is a SET on an EXISTING row's `botVersion`; a concurrent live hand is a PutItem on a
NEW row (different handKey), so there's no lost-update race on the rows this touches -- it is safe to run
live. Still prefer running it pre-cutover (the table is then unambiguously all-v1). PITR on `allin-hands`
is the rollback backstop (confirm it's enabled first); the change is also trivially reversible per-row.

Run from backend/bot/ with the same store env as the API (ALLIN_HANDS_TABLE, ALLIN_DYNAMODB_ENDPOINT,
AWS creds/region). Example (local DynamoDB dry-run):
    ALLIN_DYNAMODB_ENDPOINT=http://localhost:8000 AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local \
    AWS_DEFAULT_REGION=ap-southeast-1 python scripts/retag_recaps_v1.py
Add --apply to write.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_V2_BLUEPRINT_STEMS = ('blueprint_final_v2', 'snap_52500000')
# A build SHA from `${GITHUB_SHA} | cut -c1-7` is exactly 7 hex chars. Used only to decide whether the
# old botVersion is worth preserving in buildSha (semver labels like v1.0.0 start with 'v' -> not a SHA).
_SHA_RE = re.compile(r'^[0-9a-f]{7,40}$')


def _is_v2(bv, blueprint):
    """A v2 hand we must NOT relabel to v1: explicit v2 tag, or a v2 blueprint stem."""
    if bv and str(bv).startswith('v2'):
        return True
    bp = str(blueprint or '')
    return any(stem in bp for stem in _V2_BLUEPRINT_STEMS)


def _hands_table():
    import boto3
    from botocore.config import Config
    kwargs = {'config': Config(retries={'mode': 'adaptive', 'max_attempts': 8})}
    region = os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION')
    if region:
        kwargs['region_name'] = region
    ep = os.environ.get('ALLIN_DYNAMODB_ENDPOINT')
    if ep:
        kwargs['endpoint_url'] = ep
    name = os.environ.get('ALLIN_HANDS_TABLE', 'allin-hands')
    return boto3.resource('dynamodb', **kwargs).Table(name)


def _scan_rows(table):
    """Yield each recap's (playerId, handKey, botVersion, blueprint, buildSha), paginated."""
    # Alias every name: 'result' would be reserved (not projected here, but stay consistent); botVersion/
    # blueprint/buildSha aren't reserved but aliasing is cheap insurance against a silent scan failure.
    kwargs = {
        'ProjectionExpression': 'playerId, handKey, #bv, #bp, #bs',
        'ExpressionAttributeNames': {'#bv': 'botVersion', '#bp': 'blueprint', '#bs': 'buildSha'},
    }
    resp = table.scan(**kwargs)
    while True:
        for it in resp.get('Items', []):
            yield (it.get('playerId'), it.get('handKey'),
                   it.get('botVersion'), it.get('blueprint'), it.get('buildSha'))
        if 'LastEvaluatedKey' not in resp:
            return
        resp = table.scan(ExclusiveStartKey=resp['LastEvaluatedKey'], **kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true',
                    help='actually write (default is a dry-run that only prints the plan)')
    ap.add_argument('--target', default='v1.0.0',
                    help='the label to stamp on historical hands (default v1.0.0)')
    args = ap.parse_args()
    target = args.target

    if os.environ.get('ALLIN_STORE_BACKEND', '').strip().lower() not in ('dynamodb', 'dynamo'):
        os.environ['ALLIN_STORE_BACKEND'] = 'dynamodb'

    table = _hands_table()

    # First pass: classify (no writes). Count by the CURRENT label so the dry-run is auditable.
    to_retag = []          # (pid, hk, old_bv, preserve_sha)
    skipped_v2 = skipped_target = missing_key = 0
    by_old_label = {}
    for pid, hk, bv, bp, build_sha in _scan_rows(table):
        if not pid or not hk:
            missing_key += 1
            continue
        old = bv if bv else f'(blueprint:{bp})' if bp else '(none)'
        by_old_label[old] = by_old_label.get(old, 0) + 1
        if bv == target:
            skipped_target += 1
            continue
        if _is_v2(bv, bp):
            skipped_v2 += 1
            continue
        preserve_sha = bool(bv) and not str(bv).startswith('v') \
            and bool(_SHA_RE.match(str(bv))) and not build_sha
        to_retag.append((pid, hk, bv, preserve_sha))

    print(f"Recaps to re-tag -> botVersion = {target!r}: {len(to_retag)}")
    print(f"  already {target}: {skipped_target}   v2 (protected): {skipped_v2}   "
          f"missing key (skipped): {missing_key}")
    print("Current labels seen in the scan:")
    for lbl, n in sorted(by_old_label.items(), key=lambda kv: -kv[1]):
        print(f"  {str(lbl):>24}: {n}")

    if not args.apply:
        print("\nDRY-RUN. Verify the labels above are all v1 (no v2 hidden among them), then re-run "
              "with --apply. v2-tagged and already-target rows are left untouched.")
        return

    if not to_retag:
        print("\nNothing to re-tag.")
        return

    written = 0
    for pid, hk, old_bv, preserve_sha in to_retag:
        expr = 'SET #bv = :v'
        names = {'#bv': 'botVersion'}
        vals = {':v': target}
        if preserve_sha:
            expr += ', #bs = :sha'
            names['#bs'] = 'buildSha'
            vals[':sha'] = str(old_bv)
        table.update_item(Key={'playerId': pid, 'handKey': hk},
                          UpdateExpression=expr,
                          ExpressionAttributeNames=names,
                          ExpressionAttributeValues=vals)
        written += 1
        if written % 500 == 0:
            print(f"  ... {written}/{len(to_retag)}")
    print(f"\nAPPLIED. Re-tagged {written} recap(s) to {target!r}. "
          f"Verify with GET /api/leaderboard?version={target} and a read-only re-run of this dry-run.")


if __name__ == '__main__':
    main()
