// frontend/src/components/KeyExplorer.jsx
// Browse the blueprint by building an info-set key directly from the
// abstraction vocabulary (or by pasting a raw key).
import React, { useEffect, useState } from 'react';
import { getAbstractions, getStrategyByKey } from '../api';
import StrategyResult from './StrategyResult';

const LABEL = 'block mb-2 text-xs uppercase tracking-wider text-neutral-500';
const SELECT = 'px-3 py-2 rounded-lg bg-black/60 border border-neutral-700 ' +
    'text-neutral-100 text-sm focus:border-amber-500 focus:outline-none';
const chip = 'px-3 py-1.5 rounded-lg text-sm bg-neutral-800 text-neutral-200 ' +
    'hover:bg-neutral-700 transition-colors';

// Parse a raw info-set key back into dropdown fields, so hand-editing the key
// keeps the dropdowns in sync (otherwise the next dropdown change silently
// rebuilds the key from stale state and clobbers the edit). Returns null for an
// unparseable/free-form string (dropdowns are then left as-is).
const parseKey = (text) => {
    const m = text.trim().match(/^(pf_\d+)_(.*)$/);
    if (!m) return null;
    const bucket = m[1];
    const rest = m[2].split('_');
    if (rest[0] === 'ip' || rest[0] === 'oop') {
        // preflop: {bucket}_{position}_{pattern}
        return { street: 'preflop', bucket, position: rest[0],
            pattern: rest.slice(1).join('_') };
    }
    if (rest.length >= 4 && (rest[1] === 'ip' || rest[1] === 'oop')
        && ['flop', 'turn', 'river'].includes(rest[2])) {
        // postflop: {bucket}_{strength}_{position}_{street}_{pattern}
        const strength = Number(rest[0]);
        if (Number.isInteger(strength)) {
            return { street: rest[2], bucket, strength, position: rest[1],
                pattern: rest.slice(3).join('_') };
        }
    }
    return null;
};

function KeyExplorer() {
    const [abstractions, setAbstractions] = useState(null);

    const [street, setStreet] = useState('preflop');
    const [bucket, setBucket] = useState('pf_9');
    const [strength, setStrength] = useState(4);
    const [position, setPosition] = useState('ip');
    const [pattern, setPattern] = useState('');
    const [keyText, setKeyText] = useState('pf_9_ip_');

    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        getAbstractions().then(setAbstractions).catch((e) => setError(e.message));
    }, []);

    const composeKey = (s, b, st, pos, pat) =>
        s === 'preflop'
            ? `${b}_${pos}_${pat}`
            : `${b}_${st}_${pos}_${s}_${pat}`;

    // Postflop bucket count is per-street (12 flop / 12 turn / 10 river), so a
    // bucket valid on the flop may be out of range on the river — clamp it.
    const bucketsForStreet = (s) =>
        (s === 'preflop' ? [] : (abstractions?.postflopBuckets?.[s] || []));

    // Any dropdown change rewrites the key text field.
    const sync = (next) => {
        const s = next.street ?? street;
        const b = next.bucket ?? bucket;
        let st = next.strength ?? strength;
        const pos = next.position ?? position;
        const pat = next.pattern ?? pattern;
        const buckets = bucketsForStreet(s);
        if (buckets.length && st > buckets.length - 1) st = buckets.length - 1;
        if (next.street !== undefined) setStreet(next.street);
        if (next.bucket !== undefined) setBucket(next.bucket);
        if (next.strength !== undefined || st !== strength) setStrength(st);
        if (next.position !== undefined) setPosition(next.position);
        if (next.pattern !== undefined) setPattern(next.pattern);
        setKeyText(composeKey(s, b, st, pos, pat));
    };

    // Editable key field: update the text, and if it parses, sync the dropdowns
    // to match (without recomposing the text, so the user's edit is preserved).
    const onKeyTextEdit = (text) => {
        setKeyText(text);
        const p = parseKey(text);
        if (!p) return;
        setStreet(p.street);
        setBucket(p.bucket);
        setPosition(p.position);
        setPattern(p.pattern);
        if (p.street !== 'preflop') {
            const buckets = bucketsForStreet(p.street);
            let st = p.strength ?? 0;
            if (buckets.length && st > buckets.length - 1) st = buckets.length - 1;
            setStrength(st);
        }
    };

    const lookup = async () => {
        const key = keyText.trim();
        if (!key) { setError('Key is empty'); return; }
        setError(null);
        setLoading(true);
        setResult(null);
        try {
            setResult(await getStrategyByKey(key));
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    if (!abstractions) {
        return (
            <div className="text-neutral-500">
                {error || 'Loading abstractions…'}
            </div>
        );
    }

    // Exclude fold: a fold ends the hand, so 'f' never appears in a queryable
    // info-set pattern (heads-up: no one acts after a fold).
    const patternChars = Object.entries(abstractions.patternChars)
        .filter(([ch]) => ch !== 'f');

    return (
        <div>
            <p className="text-neutral-400 text-sm mb-6">
                Build a key from the abstraction buckets, or paste one directly.
            </p>

            <div className="flex flex-wrap gap-5 mb-6">
                <div>
                    <label className={LABEL}>Street</label>
                    <select className={SELECT} value={street}
                        onChange={(e) => sync({ street: e.target.value })}>
                        {abstractions.streets.map((s) => <option key={s}>{s}</option>)}
                    </select>
                </div>

                <div>
                    <label className={LABEL}>Starting-hand bucket</label>
                    <select className={SELECT} value={bucket}
                        onChange={(e) => sync({ bucket: e.target.value })}>
                        {abstractions.preflopBuckets.map((b) => <option key={b}>{b}</option>)}
                    </select>
                </div>

                {street !== 'preflop' && (
                    <div>
                        <label className={LABEL}>Postflop bucket</label>
                        <select className={SELECT} value={strength}
                            onChange={(e) => sync({ strength: Number(e.target.value) })}>
                            {bucketsForStreet(street).map((b) => (
                                <option key={b} value={b}>{b}</option>
                            ))}
                        </select>
                    </div>
                )}

                <div>
                    <label className={LABEL}>Position</label>
                    <select className={SELECT} value={position}
                        onChange={(e) => sync({ position: e.target.value })}>
                        {abstractions.positions.map((p) => (
                            <option key={p.value} value={p.value}>{p.label}</option>
                        ))}
                    </select>
                </div>
            </div>

            {/* Betting-pattern builder */}
            <div className="mb-6">
                <label className={LABEL}>Betting pattern (this street)</label>
                <div className="flex flex-wrap gap-2 mb-2">
                    {patternChars.map(([ch, name]) => (
                        <button key={ch} className={chip}
                            onClick={() => sync({ pattern: pattern + ch })}>
                            {ch} · {name}
                        </button>
                    ))}
                    <button className={chip}
                        onClick={() => sync({ pattern: pattern.slice(0, -1) })}>
                        ⌫ back
                    </button>
                    <button className={chip}
                        onClick={() => sync({ pattern: '' })}>clear</button>
                </div>
                <code className="text-sm text-neutral-500">pattern: &quot;{pattern}&quot;</code>
            </div>

            {/* Raw key (editable) */}
            <div className="mb-6">
                <label className={LABEL}>Info-set key (editable)</label>
                <input className={`${SELECT} w-full font-mono`}
                    value={keyText}
                    onChange={(e) => onKeyTextEdit(e.target.value)} />
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

export default KeyExplorer;
