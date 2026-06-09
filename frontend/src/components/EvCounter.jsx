// frontend/src/components/EvCounter.jsx
// The public "+EV counter": the bot's lifetime record vs the whole field.
// Polls /api/stats every ~30s. The bot's net is the negation of the human field's
// net (totalNetBB is the humans' aggregate P/L).
import React, { useEffect, useState } from 'react';
import { getStats } from '../api';

const fmtSigned = (bb) => `${bb > 0 ? '+' : ''}${bb.toLocaleString(undefined, {
    maximumFractionDigits: 0,
})}`;
const fmtRate = (v) => `${v > 0 ? '+' : ''}${v.toFixed(2)}`;   // BB/hand, 2 dp

function EvCounter({ compact = false }) {
    const [stats, setStats] = useState(null);
    const [err, setErr] = useState(false);

    useEffect(() => {
        let alive = true;
        const tick = () => getStats().then((s) => alive && setStats(s))
            .catch(() => alive && setErr(true));
        tick();
        const id = setInterval(tick, 30000);
        return () => { alive = false; clearInterval(id); };
    }, []);

    if (err && !stats) return null;            // stay invisible if the API is down

    // Render the card template immediately; until stats arrive, show a dash in
    // each numeric slot instead of a "Loading…" line, so the layout doesn't shift
    // when returning to the page (the values just fill in a moment later).
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
