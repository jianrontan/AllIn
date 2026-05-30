// frontend/src/components/HandExplorer.jsx
// "What should I do with this hand?" — enter real cards + a betting line,
// the backend abstracts it to a bucket key and returns the blueprint strategy.
import React, { useState } from 'react';
import { getStrategyFromHand } from '../api';
import StrategyResult from './StrategyResult';

const STREETS = ['preflop', 'flop', 'turn', 'river'];
const BOARD_COUNT = { preflop: 0, flop: 3, turn: 4, river: 5 };

// Sizeless actions. Fold is intentionally absent: a fold ends the hand, so it
// never appears in a betting line you'd query a strategy for.
const DISCRETE = [
    { label: 'Check', action: 'check' },
    { label: 'Call', action: 'call' },
    { label: 'All-in', action: 'allin' },
];

// Preflop sizes are an absolute BB ladder (not pot fractions), so preflop sizing
// is expressed as a raise-TO total in BB: the three trained opens, plus any
// custom amount. Postflop uses free pot-fraction sizing.
const PREFLOP_PRESETS = [2, 2.5, 3.5, 5];            // raise-to totals in BB (5 = xlarge open)

// Postflop quick sizes as pot fractions; any custom % is allowed too. With
// pseudo-harmonic translation the backend maps an off-grid size onto the trained
// grid (⅓ / ⅔ / pot) and blends the responses, so any size returns a strategy.
const POSTFLOP_PRESETS = [0.33, 0.5, 0.66, 0.75, 1.0, 1.25, 1.5];

// A bet/raise is a "raise" once someone has already put money in this street.
const verbFor = (line) =>
    line.some((a) => ['bet', 'raise', 'allin'].includes(a.action)) ? 'raise' : 'bet';

const chipLabel = (a) => {
    if (a.action === 'check') return 'Check';
    if (a.action === 'call') return 'Call';
    if (a.action === 'allin') return 'All-in';
    if (a.bb != null) return `${a.action === 'raise' ? 'Raise to' : 'Open to'} ${a.bb} BB`;
    const verb = a.action === 'raise' ? 'Raise' : 'Bet';
    if (a.fraction != null) return `${verb} ${Math.round(a.fraction * 100)}%`;
    return `${verb} ${a.size}`;
};

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
    const [sizeInput, setSizeInput] = useState('');   // custom size: BB (preflop) or % pot

    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const boardN = BOARD_COUNT[street];

    const setHoleCard = (i, v) =>
        setHole((p) => p.map((c, idx) => (idx === i ? normalizeCard(v) : c)));
    const setCommunityCard = (i, v) =>
        setCommunity((p) => p.map((c, idx) => (idx === i ? normalizeCard(v) : c)));

    const addDiscrete = (action) => setActions((p) => [...p, { action }]);
    const addBb = (bb) => setActions((p) => [...p, { action: verbFor(p), bb }]);
    const addFraction = (fraction) =>
        setActions((p) => [...p, { action: verbFor(p), fraction }]);
    const addCustom = () => {
        const v = parseFloat(sizeInput);
        if (!isFinite(v) || v <= 0) return;
        if (street === 'preflop') addBb(v); else addFraction(v / 100);
        setSizeInput('');
    };

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
                    {DISCRETE.map((b) => (
                        <button key={b.action} className={chip}
                            onClick={() => addDiscrete(b.action)}>
                            {b.label}
                        </button>
                    ))}
                </div>

                {street === 'preflop' ? (
                    <div className="space-y-2 mb-2">
                        <div className="flex flex-wrap gap-2 items-center">
                            <span className="text-xs text-neutral-500 mr-1">Raise to</span>
                            {PREFLOP_PRESETS.map((bb) => (
                                <button key={bb} className={chip}
                                    onClick={() => addBb(bb)}>{bb} BB</button>
                            ))}
                        </div>
                        <div className="flex flex-wrap gap-2 items-center">
                            <span className="text-xs text-neutral-500 mr-1">Custom</span>
                            <input className={`${CARD_INPUT} w-16 h-9 text-base`}
                                placeholder="2.7" value={sizeInput}
                                onChange={(e) => setSizeInput(e.target.value.replace(/[^0-9.]/g, ''))}
                                onKeyDown={(e) => e.key === 'Enter' && addCustom()} />
                            <span className="text-sm text-neutral-400">BB</span>
                            <button className={chip} onClick={addCustom}>Add</button>
                        </div>
                    </div>
                ) : (
                    <div className="space-y-2 mb-2">
                        <div className="flex flex-wrap gap-2 items-center">
                            <span className="text-xs text-neutral-500 mr-1">Bet / raise</span>
                            {POSTFLOP_PRESETS.map((f) => (
                                <button key={f} className={chip}
                                    onClick={() => addFraction(f)}>
                                    {Math.round(f * 100)}%
                                </button>
                            ))}
                        </div>
                        <div className="flex flex-wrap gap-2 items-center">
                            <span className="text-xs text-neutral-500 mr-1">Custom</span>
                            <input className={`${CARD_INPUT} w-16 h-9 text-base`}
                                placeholder="55" value={sizeInput}
                                onChange={(e) => setSizeInput(e.target.value.replace(/[^0-9.]/g, ''))}
                                onKeyDown={(e) => e.key === 'Enter' && addCustom()} />
                            <span className="text-sm text-neutral-400">% pot</span>
                            <button className={chip} onClick={addCustom}>Add</button>
                        </div>
                    </div>
                )}

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
                            {chipLabel(a)}
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
