// frontend/src/components/Leaderboard.jsx
// The landing-page leaderboard. A toggle switches between "Signed-in" (ranked
// accounts, the default) and "All players" (includes anonymous); both require
// minHands. Paginated (pageSize per page). The API returns { players, total }.
import React, { useEffect, useState } from 'react';
import { getLeaderboard } from '../api';

function Leaderboard({ title, minHands = 50, pageSize = 20, version = 'all' }) {
    const [accountsOnly, setAccountsOnly] = useState(true);   // default: signed-in only
    const [page, setPage] = useState(0);                       // 0-based
    const [data, setData] = useState(null);                    // { players, total, yourRank } | null
    const [err, setErr] = useState(null);

    // Both cuts use the real minimum (pagination verified, so the min-0 testing override is gone).
    const effMinHands = minHands;

    // version is a CONTROLLED prop -- one dropdown on the bot-stats card drives both. Reset to page 1
    // when it changes, DURING RENDER (React's derived-state pattern) so the fetch effect runs ONCE at
    // page 0, not the stale page first then page 0 (which double-fetched).
    const [prevVersion, setPrevVersion] = useState(version);
    if (version !== prevVersion) {
        setPrevVersion(version);
        setPage(0);
    }

    useEffect(() => {
        let alive = true;
        setErr(null);
        getLeaderboard({ n: pageSize, minHands: effMinHands, accountsOnly, offset: page * pageSize, version })
            .then((d) => alive && setData({
                players: d.players || [], total: d.total || 0, yourRank: d.yourRank ?? null,
            }))
            .catch((e) => alive && setErr(e.message));
        return () => { alive = false; };
    }, [accountsOnly, version, page, effMinHands, pageSize]);

    const total = data?.total || 0;
    const pageCount = Math.max(1, Math.ceil(total / pageSize));
    const rows = data?.players || [];
    // The caller's rank (page-independent) + the page that holds them, for the "Find me" jump.
    const myRank = data?.yourRank ?? null;
    const myPage = myRank != null ? Math.floor((myRank - 1) / pageSize) : null;
    const desc = `${accountsOnly ? 'Signed-in accounts' : 'All players (incl. anonymous)'}`
        + ` with ${effMinHands}+ hands, ranked by BB/hand.`;

    // Switching the filter resets to the first page (the result set changes).
    const switchFilter = (val) => { if (val !== accountsOnly) { setAccountsOnly(val); setPage(0); } };

    return (
        <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5 w-full">
            <div className="flex items-center justify-between mb-1 gap-3 flex-wrap">
                <h3 className="text-sm uppercase tracking-wider text-amber-300/80">{title}</h3>
                <div className="inline-flex rounded-lg border border-neutral-800 overflow-hidden text-xs">
                    {[['Signed-in', true], ['All players', false]].map(([label, val]) => (
                        <button key={label} type="button" onClick={() => switchFilter(val)}
                            className={'px-3 py-1 transition-colors '
                                + (accountsOnly === val
                                    ? 'bg-amber-500 text-neutral-950 font-semibold'
                                    : 'text-neutral-400 hover:text-neutral-200')}>
                            {label}
                        </button>
                    ))}
                </div>
            </div>
            <p className="text-[11px] text-neutral-600 mb-3">{desc}</p>
            {err && <p className="text-sm text-rose-400">{err}</p>}
            {!err && data && rows.length === 0 && (
                <p className="text-sm text-neutral-600">
                    {page > 0
                        ? 'No players on this page.'
                        : `Only signed-in players make the leaderboard — sign in and play ${minHands}+ hands to claim a spot.`}
                </p>
            )}
            {!err && rows.length > 0 && (
                <>
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="text-neutral-500 text-xs uppercase tracking-wider">
                                <th className="text-left font-medium pb-2">#</th>
                                <th className="text-left font-medium pb-2">Player</th>
                                <th className="text-right font-medium pb-2">BB/hand</th>
                                <th className="text-right font-medium pb-2">Hands</th>
                                <th className="text-right font-medium pb-2">Net BB</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map((r, i) => {
                                // Defensive defaults: a malformed row must not crash the table.
                                const hands = Number(r.hands) || 0;
                                const net = Number(r.netBB) || 0;
                                // BB/hand, matching the bot's EvCounter (net ÷ hands). Ranking is
                                // unchanged: the backend still sorts by bbPer100 (this ×100).
                                const bbPerHand = hands ? net / hands : 0;
                                const rank = page * pageSize + i + 1;   // continue across pages
                                const isYou = !!r.isYou;                 // the caller's own row (server-marked)
                                return (
                                    <tr key={`${rank}-${r.handle}`}
                                        className={'border-t border-neutral-800/70 '
                                            + (isYou ? 'bg-amber-500/10' : '')}>
                                        <td className="py-1.5 text-neutral-500 tabular-nums">{rank}</td>
                                        <td className="py-1.5">
                                            <span className={isYou
                                                ? 'text-amber-300 font-semibold '
                                                  + '[text-shadow:0_0_8px_rgba(245,158,11,0.75)]'
                                                : 'text-neutral-200'}>
                                                {r.handle || 'Anonymous'}
                                            </span>
                                            {r.isRegistered && (
                                                <span title="signed-in account"
                                                    className="ml-1 text-emerald-500/80">✓</span>
                                            )}
                                        </td>
                                        <td className={'py-1.5 text-right tabular-nums font-semibold '
                                            + (bbPerHand >= 0 ? 'text-emerald-400' : 'text-rose-400')}>
                                            {bbPerHand > 0 ? '+' : ''}{bbPerHand.toFixed(2)}
                                        </td>
                                        <td className="py-1.5 text-right tabular-nums text-neutral-400">
                                            {hands.toLocaleString()}
                                        </td>
                                        <td className="py-1.5 text-right tabular-nums text-neutral-400">
                                            {net > 0 ? '+' : ''}{net.toLocaleString()}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                    {pageCount > 1 && (
                        <div className="flex items-center justify-between mt-3 text-xs text-neutral-500">
                            <button type="button" disabled={page <= 0}
                                onClick={() => setPage((p) => Math.max(0, p - 1))}
                                className="px-3 py-1 rounded border border-neutral-800 disabled:opacity-40
                                           hover:text-neutral-200 disabled:hover:text-neutral-500">
                                ← Prev
                            </button>
                            <div className="flex items-center gap-3">
                                <span className="tabular-nums">Page {page + 1} of {pageCount}</span>
                                {myPage != null && myPage !== page && (
                                    <button type="button" onClick={() => setPage(myPage)}
                                        className="px-2 py-1 rounded border border-amber-500/40
                                                   text-amber-300/90 hover:bg-amber-500/10 transition-colors">
                                        Find me (#{myRank})
                                    </button>
                                )}
                            </div>
                            <button type="button" disabled={page >= pageCount - 1}
                                onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
                                className="px-3 py-1 rounded border border-neutral-800 disabled:opacity-40
                                           hover:text-neutral-200 disabled:hover:text-neutral-500">
                                Next →
                            </button>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}

export default Leaderboard;
