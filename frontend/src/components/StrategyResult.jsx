// frontend/src/components/StrategyResult.jsx
// Shared result panel for both the Hand Explorer and the Key Explorer.
import React from 'react';

const ACTION_LABELS = {
    fold: 'Fold', call: 'Call', check: 'Check', allin: 'All-in',
    bet_small: 'Bet — small (≈ ⅓ pot)',
    bet_medium: 'Bet — medium (≈ ⅔ pot)',
    bet_large: 'Bet — large (≈ pot)',
    bet_xlarge: 'Open — xlarge (5 BB)',
    bet_overbet: 'Bet — overbet (1.5× pot)',
    raise_small: 'Raise — small', raise_medium: 'Raise — medium',
    raise_large: 'Raise — large',
    raise_overbet: 'Raise — overbet (1.5× pot)',
};

const actionLabel = (a) => ACTION_LABELS[a] || a;

// Bracket size chars (action translation) -> human label.
const CHAR_LABEL = { s: '⅓ pot', m: '⅔ pot', l: 'pot', o: '1.5× pot', x: '5 BB open', a: 'all-in' };

const barColor = (a) => {
    if (a === 'fold') return 'bg-rose-500';
    if (a === 'call' || a === 'check') return 'bg-sky-500';
    if (a === 'allin') return 'bg-violet-500';
    return 'bg-emerald-500';
};

const PANEL = 'mt-5 rounded-2xl border border-neutral-800 bg-neutral-900/70 p-5';

function StrategyResult({ result, loading, error }) {
    if (error) {
        return (
            <div className={`${PANEL} border-rose-800/60`}>
                <span className="text-rose-400 font-semibold">Error: </span>
                <span className="text-neutral-300">{error}</span>
            </div>
        );
    }
    if (loading) {
        return <div className={`${PANEL} text-neutral-400`}>Looking up strategy…</div>;
    }
    if (!result) {
        return (
            <div className={`${PANEL} text-neutral-500`}>
                Enter a situation above and look it up to see the blueprint&rsquo;s strategy.
            </div>
        );
    }

    const { key, found, strategy, visitCount, cardBucket, strengthBucket,
        translated, brackets } = result;

    return (
        <div className={PANEL}>
            <div className="mb-4">
                <div className="text-xs uppercase tracking-wider text-neutral-500 mb-1">
                    Info-set key
                </div>
                <code className="inline-block bg-black/60 text-amber-200 px-2.5 py-1
                                  rounded-md text-sm break-all">
                    {key}
                </code>
            </div>

            {(cardBucket || strengthBucket != null) && (
                <div className="mb-4 flex flex-wrap gap-x-6 gap-y-1 text-sm text-neutral-400">
                    {cardBucket && (
                        <span>Starting-hand bucket:{' '}
                            <b className="text-neutral-200">{cardBucket}</b></span>
                    )}
                    {strengthBucket != null && (
                        <span>Postflop strength bucket:{' '}
                            <b className="text-neutral-200">{strengthBucket}</b></span>
                    )}
                </div>
            )}

            {!found && (
                <div className="rounded-lg border border-amber-700/60 bg-amber-950/40
                                px-4 py-3 text-sm text-amber-200">
                    <b>Not in the blueprint.</b> This exact situation was never reached
                    during training, so there is no learned strategy for it. Try a more
                    common line.
                </div>
            )}

            {translated && brackets && (
                <div className="mb-4 rounded-lg border border-sky-800/50 bg-sky-950/30
                                px-4 py-3 text-sm text-sky-200">
                    <b>Off-grid bet → action translation.</b> Your size sits between two
                    trained sizes, so the strategy below is the pseudo-harmonic blend of:
                    <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-xs text-sky-300/90">
                        {brackets.map((b) => (
                            <span key={b.char}>
                                {(b.weight * 100).toFixed(0)}% on <b>{CHAR_LABEL[b.char] || b.char}</b>
                                {b.found ? '' : ' (untrained)'}
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {found && strategy && (
                <>
                    <h4 className="text-sm uppercase tracking-wider text-neutral-500 mb-3">
                        Blueprint strategy
                    </h4>
                    <div className="space-y-3">
                        {Object.entries(strategy)
                            .sort((a, b) => b[1] - a[1])
                            .map(([action, prob]) => (
                                <div key={action}>
                                    <div className="flex justify-between text-sm mb-1">
                                        <span className="text-neutral-300">
                                            {actionLabel(action)}
                                        </span>
                                        <span className="font-semibold tabular-nums">
                                            {(prob * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                    <div className="h-3 rounded-full bg-neutral-800 overflow-hidden">
                                        <div
                                            className={`h-full rounded-full ${barColor(action)}`}
                                            style={{ width: `${Math.max(prob * 100, 0.5)}%` }}
                                        />
                                    </div>
                                </div>
                            ))}
                    </div>
                    <div className="mt-4 text-xs text-neutral-500">
                        {visitCount != null
                            ? `Trained on ${visitCount.toLocaleString()} visits to this info set.`
                            : 'Blended across the bracketing sizes (per-size visit counts omitted).'}
                    </div>
                </>
            )}
        </div>
    );
}

export default StrategyResult;
