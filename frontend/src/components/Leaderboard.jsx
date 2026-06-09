// frontend/src/components/Leaderboard.jsx
// One leaderboard cut. Two are shown on the landing page:
//   - Ranked (accountsOnly, minHands=50): signed-in accounts only.
//   - Most active (anonymous allowed): a lower min-hands "who's playing" cut.
import React, { useEffect, useState } from 'react';
import { getLeaderboard } from '../api';

function Leaderboard({ title, accountsOnly = false, minHands = 50, n = 10, note }) {
    const [rows, setRows] = useState(null);
    const [err, setErr] = useState(null);

    useEffect(() => {
        let alive = true;
        getLeaderboard({ n, minHands, accountsOnly })
            .then((d) => alive && setRows(d.players || []))
            .catch((e) => alive && setErr(e.message));
        return () => { alive = false; };
    }, [accountsOnly, minHands, n]);

    return (
        <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5 w-full">
            <h3 className="text-sm uppercase tracking-wider text-amber-300/80 mb-1">{title}</h3>
            {note && <p className="text-[11px] text-neutral-600 mb-3">{note}</p>}
            {err && <p className="text-sm text-rose-400">{err}</p>}
            {!err && rows && rows.length === 0 && (
                <p className="text-sm text-neutral-600">
                    No one has cleared the minimum hand count yet, be the first.
                </p>
            )}
            {!err && rows && rows.length > 0 && (
                <table className="w-full text-sm">
                    <thead>
                        <tr className="text-neutral-500 text-xs uppercase tracking-wider">
                            <th className="text-left font-medium pb-2">#</th>
                            <th className="text-left font-medium pb-2">Player</th>
                            <th className="text-right font-medium pb-2">BB/100</th>
                            <th className="text-right font-medium pb-2">Hands</th>
                            <th className="text-right font-medium pb-2">Net BB</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map((r, i) => {
                            // Defensive defaults: a malformed row must not crash the table.
                            const bb100 = Number(r.bbPer100) || 0;
                            const hands = Number(r.hands) || 0;
                            const net = Number(r.netBB) || 0;
                            return (
                                <tr key={i} className="border-t border-neutral-800/70">
                                    <td className="py-1.5 text-neutral-500 tabular-nums">{i + 1}</td>
                                    <td className="py-1.5 text-neutral-200">
                                        {r.handle || 'Anonymous'}
                                        {r.isRegistered && (
                                            <span title="signed-in account"
                                                className="ml-1 text-emerald-500/80">✓</span>
                                        )}
                                    </td>
                                    <td className={'py-1.5 text-right tabular-nums font-semibold '
                                        + (bb100 >= 0 ? 'text-emerald-400' : 'text-rose-400')}>
                                        {bb100 > 0 ? '+' : ''}{bb100}
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
            )}
        </div>
    );
}

export default Leaderboard;
