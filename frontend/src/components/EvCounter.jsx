// frontend/src/components/EvCounter.jsx
// The public "+EV counter": the bot's lifetime record vs the whole field.
// Polls /api/stats every 60s (shared across instances, see subscribeStats). The
// bot's net is the negation of the human field's net (totalNetBB = humans' P/L).
import React, { useEffect, useState } from 'react';
import { getStats } from '../api';

const fmtSigned = (bb) => `${bb > 0 ? '+' : ''}${bb.toLocaleString(undefined, {
    maximumFractionDigits: 0,
})}`;
const fmtRate = (v) => `${v > 0 ? '+' : ''}${v.toFixed(2)}`;   // BB/hand, 2 dp

// Shared poll: ONE /api/stats fetch loop no matter how many EvCounter instances
// are mounted (e.g. AiGame renders a desktop copy + a mobile copy). Instances
// subscribe; the 60s interval runs only while at least one is mounted, and a late
// subscriber immediately receives the last cached value. Errors are swallowed --
// the cards keep showing dashes and the next tick retries. The counter is a
// slow-moving lifetime stat, so 60s (with the backend's 5s per-worker cache) is
// plenty; a player's OWN stats refresh separately (getMe on each hand_over).
let _stats = null;
const _subs = new Set();
let _timer = null;

function _tick() {
    getStats().then((s) => { _stats = s; _subs.forEach((fn) => fn(s)); }).catch(() => {});
}

function subscribeStats(fn) {
    _subs.add(fn);
    if (_stats) fn(_stats);
    if (_timer === null) {
        _tick();
        _timer = setInterval(_tick, 60000);
    }
    return () => {
        _subs.delete(fn);
        if (_subs.size === 0 && _timer !== null) {
            clearInterval(_timer);
            _timer = null;
        }
    };
}

function EvCounter({ compact = false }) {
    const [stats, setStats] = useState(_stats);
    useEffect(() => subscribeStats(setStats), []);

    // Render the card template even while the API is down: the numeric slots
    // show dashes (the same as the loading state) rather than the whole card
    // vanishing -- a disappeared element reads as "broken page" where a dashed
    // one reads as "loading". The 60s poll keeps retrying, so a transient
    // outage self-heals in place.
    const loading = !stats;
    const hands = Number(stats?.totalHands) || 0;
    const players = Number(stats?.totalPlayers) || 0;
    const botNet = -(Number(stats?.totalNetBB) || 0);   // bot net = -(human field net)
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
        <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 px-6 py-4 text-center">
            <div className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-1">
                Bot vs humans
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
                    : <>over {hands.toLocaleString()} hands · {players.toLocaleString()} players</>}
            </div>
        </div>
    );
}

export default EvCounter;
