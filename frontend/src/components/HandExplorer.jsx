// frontend/src/components/HandExplorer.jsx
// "What should I do with this hand?" — enter real cards + a betting line,
// the backend abstracts it to a bucket key and returns the blueprint strategy.
import React, { useState } from 'react';
import { getStrategyFromHand } from '../api';
import StrategyResult from './StrategyResult';

const STREETS = ['preflop', 'flop', 'turn', 'river'];
const BOARD_COUNT = { preflop: 0, flop: 3, turn: 4, river: 5 };

const ACTION_BUTTONS = [
    { label: 'Check', action: 'check' },
    { label: 'Call', action: 'call' },
    { label: 'Fold', action: 'fold' },
    { label: 'Bet S', action: 'bet', size: 'small' },
    { label: 'Bet M', action: 'bet', size: 'medium' },
    { label: 'Bet L', action: 'bet', size: 'large' },
    { label: 'Raise S', action: 'raise', size: 'small' },
    { label: 'Raise M', action: 'raise', size: 'medium' },
    { label: 'Raise L', action: 'raise', size: 'large' },
    { label: 'All-in', action: 'allin' },
];

const normalizeCard = (raw) => {
    const c = (raw || '').replace(/[^a-zA-Z0-9]/g, '').slice(0, 2);
    if (c.length === 0) return '';
    if (c.length === 1) return c[0].toUpperCase();
    return c[0].toUpperCase() + c[1].toLowerCase();
};

const LABEL = 'block mb-2 text-xs uppercase tracking-wider text-neutral-500';
const CARD_INPUT = 'w-12 h-14 text-center text-lg font-semibold rounded-lg ' +
    'bg-black/60 border border-neutral-700 text-neutral-100 ' +
    'focus:border-amber-500 focus:outline-none';
const seg = (active) =>
    'px-3.5 py-1.5 rounded-lg text-sm transition-colors ' +
    (active
        ? 'bg-amber-500 text-neutral-950 font-semibold'
        : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700');
const chip = 'px-3 py-1.5 rounded-lg text-sm bg-neutral-800 text-neutral-200 ' +
    'hover:bg-neutral-700 transition-colors';

function HandExplorer() {
    const [hole, setHole] = useState(['', '']);
    const [street, setStreet] = useState('preflop');
    const [community, setCommunity] = useState(['', '', '', '', '']);
    const [position, setPosition] = useState('ip');
    const [actions, setActions] = useState([]);

    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const boardN = BOARD_COUNT[street];

    const setHoleCard = (i, v) =>
        setHole((p) => p.map((c, idx) => (idx === i ? normalizeCard(v) : c)));
    const setCommunityCard = (i, v) =>
        setCommunity((p) => p.map((c, idx) => (idx === i ? normalizeCard(v) : c)));

    const lookup = async () => {
        setError(null);
        setLoading(true);
        setResult(null);
        try {
            const data = await getStrategyFromHand({
                holeCards: hole,
                communityCards: community.slice(0, boardN),
                position,
                actions,
            });
            setResult(data);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <p className="text-neutral-400 text-sm mb-6">
                Enter your hand and the betting so far on the current street. The bot
                only ever sees the <i>bucket</i> your hand falls into.
            </p>

            {/* Hole cards */}
            <div className="mb-6">
                <label className={LABEL}>Your hole cards</label>
                <div className="flex gap-2">
                    {[0, 1].map((i) => (
                        <input key={i} className={CARD_INPUT} maxLength={2}
                            placeholder={i === 0 ? 'As' : 'Kh'}
                            value={hole[i]}
                            onChange={(e) => setHoleCard(i, e.target.value)} />
                    ))}
                </div>
            </div>

            {/* Street */}
            <div className="mb-6">
                <label className={LABEL}>Street</label>
                <div className="flex flex-wrap gap-2">
                    {STREETS.map((s) => (
                        <button key={s} className={seg(street === s)}
                            onClick={() => { setStreet(s); setActions([]); }}>
                            {s}
                        </button>
                    ))}
                </div>
            </div>

            {/* Community cards */}
            {boardN > 0 && (
                <div className="mb-6">
                    <label className={LABEL}>Community cards ({street})</label>
                    <div className="flex gap-2">
                        {Array.from({ length: boardN }).map((_, i) => (
                            <input key={i} className={CARD_INPUT} maxLength={2}
                                placeholder={['Qd', 'Jc', 'Tc', '9h', '8s'][i]}
                                value={community[i]}
                                onChange={(e) => setCommunityCard(i, e.target.value)} />
                        ))}
                    </div>
                </div>
            )}

            {/* Position */}
            <div className="mb-6">
                <label className={LABEL}>Your position</label>
                <div className="flex flex-wrap gap-2">
                    <button className={seg(position === 'ip')}
                        onClick={() => setPosition('ip')}>In position (button/SB)</button>
                    <button className={seg(position === 'oop')}
                        onClick={() => setPosition('oop')}>Out of position (BB)</button>
                </div>
            </div>

            {/* Betting line builder */}
            <div className="mb-6">
                <label className={LABEL}>Betting line this street</label>
                <div className="flex flex-wrap gap-2 mb-2">
                    {ACTION_BUTTONS.map((b) => (
                        <button key={b.label} className={chip}
                            onClick={() => setActions((p) => [...p, b])}>
                            {b.label}
                        </button>
                    ))}
                </div>
                <div className="min-h-[2.5rem] p-2 rounded-lg bg-black/50
                                flex flex-wrap gap-2 items-center">
                    {actions.length === 0 && (
                        <span className="text-neutral-600 text-sm px-1">
                            No actions yet (start of street)
                        </span>
                    )}
                    {actions.map((a, i) => (
                        <span key={i} className="px-2.5 py-1 rounded-md bg-neutral-700
                                                  text-xs text-neutral-200">
                            {a.action}{a.size ? ` ${a.size}` : ''}
                        </span>
                    ))}
                </div>
                {actions.length > 0 && (
                    <div className="flex gap-2 mt-2">
                        <button className={chip}
                            onClick={() => setActions((p) => p.slice(0, -1))}>Undo</button>
                        <button className={chip}
                            onClick={() => setActions([])}>Clear</button>
                    </div>
                )}
            </div>

            <button onClick={lookup} disabled={loading}
                className="px-7 py-3 rounded-xl font-semibold bg-amber-500
                           text-neutral-950 hover:bg-amber-400 disabled:opacity-50
                           transition-colors">
                {loading ? 'Looking up…' : 'Look up strategy'}
            </button>

            <StrategyResult result={result} loading={loading} error={error} />
        </div>
    );
}

export default HandExplorer;
