// frontend/src/components/EvCounter.jsx
// The public "+EV counter": the bot's lifetime record vs the whole field.
// Polls /api/stats on ONE shared loop across all mounted instances (see subscribeStats):
// adaptive 8s until the per-version breakdown is ready, then 60s. The bot's net is the
// negation of the human field's net (totalNetBB is the humans' aggregate P/L).
import React, { useEffect, useState } from 'react';
import { getStats } from '../api';
import VersionFilter from './VersionFilter';

const fmtSigned = (bb) => `${bb > 0 ? '+' : ''}${bb.toLocaleString(undefined, {
    maximumFractionDigits: 0,
})}`;
const fmtRate = (v) => `${v > 0 ? '+' : ''}${v.toFixed(2)}`;   // BB/hand, 2 dp

// ONE shared /api/stats poll for ALL mounted EvCounter instances (e.g. AiGame renders a desktop
// + a mobile copy). Instances subscribe; the loop runs only while >=1 is mounted, a late
// subscriber gets the last cached value immediately, and the interval is adaptive: 8s until the
// per-version breakdown (byVersion, filled by a server-side background scan) is ready, then 60s.
let _stats = null;
const _subs = new Set();
let _started = false;

function _tick() {
    getStats()
        .then((s) => { _stats = s; _subs.forEach((fn) => fn(s)); })
        .catch(() => {})
        .finally(() => {
            // Adaptive: poll fast (8s) until the per-version breakdown lands, then settle to 60s.
            const ready = _stats?.byVersion && Object.keys(_stats.byVersion).length > 0;
            setTimeout(_tick, ready ? 60000 : 8000);
        });
}
function subscribeStats(fn) {
    _subs.add(fn);
    if (_stats) fn(_stats);                          // a late subscriber gets the cached value at once
    // Start exactly ONE loop for the app's lifetime. We deliberately do NOT stop it on the last
    // unsubscribe and do NOT restart it on re-subscribe: tying the loop to mount/unmount let a
    // re-render churn (e.g. the 400ms "thinking" animation re-mounting this card) restart the loop
    // every tick and spam /api/stats. A single 8s→60s loop is negligible and immune to that.
    if (!_started) { _started = true; _tick(); }
    return () => { _subs.delete(fn); };
}

function EvCounter({ compact = false, version: versionProp, onVersionChange }) {
    const [stats, setStats] = useState(_stats);
    // version is CONTROLLED when the parent passes version + onVersionChange (so one dropdown can
    // drive both this card AND the leaderboard); otherwise it's local (standalone use, e.g. the
    // compact in-game counter, which doesn't render the dropdown anyway).
    const [localVersion, setLocalVersion] = useState('all');
    const version = versionProp !== undefined ? versionProp : localVersion;
    const setVersion = onVersionChange || setLocalVersion;

    // Subscribe to the shared poll (module-level, above); unsubscribe on unmount.
    useEffect(() => subscribeStats(setStats), []);

    // Render the card template even while the API is down: the numeric slots
    // show dashes (the same as the loading state) rather than the whole card
    // vanishing -- a disappeared element reads as "broken page" where a dashed
    // one reads as "loading". The 60s poll keeps retrying, so a transient
    // outage self-heals in place.
    const loading = !stats;
    const byVersion = stats?.byVersion || {};
    const versions = Object.keys(byVersion).sort();
    // Self-heal: if the selected version vanished from the data (retired / churn), reset to 'all' so
    // the controlled <select> never holds a value with no matching <option> and the page isn't stuck
    // querying a phantom version. (Effect, not during-render -- setVersion may be the parent's setter.)
    useEffect(() => {
        if (version !== 'all' && versions.length > 0 && !versions.includes(version)) setVersion('all');
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [version, versions.join('|')]);
    // Displayed totals depend on the selected bot version. 'all' = the global counter; a specific
    // version = its recap aggregate (per-version player count isn't tracked, so it's hidden then).
    let hands, botNet, players;
    if (version === 'all') {
        // 'all' = sum of every version from the SAME source the v1/v2 slices use (the recap
        // aggregate), so the dropdown reconciles: all == v1 + v2 + ... Until the background scan
        // fills byVersion, fall back to the global counter so the card isn't blank.
        if (versions.length > 0) {
            const s = versions.reduce((a, v) => {
                a.hands += Number(byVersion[v]?.hands) || 0;
                a.net += Number(byVersion[v]?.humanNetBB) || 0;
                return a;
            }, { hands: 0, net: 0 });
            hands = s.hands;
            botNet = -s.net;                            // bot net = -(human field net)
        } else {
            hands = Number(stats?.totalHands) || 0;
            botNet = -(Number(stats?.totalNetBB) || 0);
        }
        players = Number(stats?.totalPlayers) || 0;
    } else {
        const d = byVersion[version] || { hands: 0, humanNetBB: 0 };
        hands = Number(d.hands) || 0;
        botNet = -(Number(d.humanNetBB) || 0);
        players = null;
    }
    const bbPerHand = hands ? botNet / hands : 0;
    const winning = botNet >= 0;
    const color = loading ? 'text-neutral-500' : (winning ? 'text-emerald-400' : 'text-rose-400');
    const DASH = '—';

    if (compact) {
        return (
            <span className="text-xs tabular-nums text-neutral-400">
                Bot <span className={`font-semibold ${color}`}>
                    {loading ? DASH : `${fmtSigned(botNet)} BB`}
                </span>
                <span className="text-neutral-600">
                    {' '}({loading ? DASH : `${fmtRate(bbPerHand)} BB/hand`}) ·{' '}
                    {loading ? DASH : `${hands.toLocaleString()} hands`}
                </span>
            </span>
        );
    }

    return (
        <div className="relative rounded-2xl border border-neutral-800 bg-neutral-900/60 px-6 py-4 text-center">
            <div className="mb-1">
                <span className="text-xs uppercase tracking-[0.2em] text-neutral-500">Bot vs humans</span>
            </div>
            <div className={`text-3xl font-bold tabular-nums ${color}`}>
                {loading ? DASH : `${fmtSigned(botNet)} BB`}
            </div>
            <div className={`mt-0.5 text-base font-semibold tabular-nums ${color}`}>
                {loading ? DASH : `${fmtRate(bbPerHand)} BB/hand`}
            </div>
            <div className="mt-1 text-sm text-neutral-500 tabular-nums">
                {loading
                    ? <>over {DASH} hands · {DASH} players</>
                    : <>over {hands.toLocaleString()} hands{players != null
                        ? <> · {players.toLocaleString()} players</> : null}</>}
            </div>
            {/* Version dropdown: on mobile it sits in normal flow BELOW the stats (centered), so it never
                crowds the centered header on the narrow card; on desktop (sm+) the wider card pins it to
                the top-right corner (absolute, so its DOM order here doesn't affect the desktop position). */}
            <div className="mt-3 flex justify-center sm:mt-0 sm:block sm:absolute sm:top-3 sm:right-3">
                <VersionFilter value={version} onChange={setVersion} versions={versions} />
            </div>
        </div>
    );
}

export default EvCounter;
