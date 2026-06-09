// frontend/src/components/StrategyResult.jsx
// Shared result panel for both the Hand Explorer and the Key Explorer.
import React from 'react';

const ACTION_LABELS = {
    fold: 'Fold', call: 'Call', check: 'Check', allin: 'All-in',
    bet_small: 'Bet - small (≈ ⅓ pot)',
    bet_medium: 'Bet - medium (≈ ⅔ pot)',
    bet_large: 'Bet - large (≈ pot)',
    bet_xlarge: 'Open - xlarge (5 BB)',
    bet_overbet: 'Bet - overbet (1.5× pot)',
    bet_overbet2: 'Bet - overbet2 (2× pot)',
    raise_small: 'Raise - small', raise_medium: 'Raise - medium',
    raise_large: 'Raise - large',
    raise_overbet: 'Raise - overbet (1.5× pot)',
    raise_overbet2: 'Raise - overbet2 (2× pot)',
};

const actionLabel = (a) => ACTION_LABELS[a] || a;

// Bracket size chars (action translation) -> human label.
const CHAR_LABEL = {
    s: '⅓ pot', m: '⅔ pot', l: 'pot', o: '1.5× pot', '2': '2× pot',
    x: '5 BB open', a: 'all-in',
};

// Colour for both engine action names (blueprint) and the river solver's
// friendly labels ("Bet 10 BB", "Raise to 30 BB", "Check", "All-in", ...).
const barColor = (a) => {
    if (a === 'fold' || a === 'Fold') return 'bg-rose-500';
    if (['call', 'check', 'Call', 'Check'].includes(a)) return 'bg-sky-500';
    if (a === 'allin' || a === 'All-in') return 'bg-violet-500';
    return 'bg-emerald-500';
};

const PANEL = 'mt-5 rounded-2xl border border-neutral-800 bg-neutral-900/70 p-5';

// One {action: prob} distribution as a sorted stack of labelled bars.
function StrategyBars({ strategy }) {
    // Defensive: a malformed/empty strategy must not crash the panel.
    const entries = (strategy && typeof strategy === 'object')
        ? Object.entries(strategy) : [];
    if (entries.length === 0) {
        return <div className="text-sm text-neutral-600">No strategy data.</div>;
    }
    return (
        <div className="space-y-3">
            {entries
                .sort((a, b) => b[1] - a[1])
                .map(([action, prob]) => (
                    <div key={action}>
                        <div className="flex justify-between text-sm mb-1">
                            <span className="text-neutral-300">{actionLabel(action)}</span>
                            <span className="font-semibold tabular-nums">
                                {(prob * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div className="h-3 rounded-full bg-neutral-800 overflow-hidden">
                            <div className={`h-full rounded-full ${barColor(action)}`}
                                style={{ width: `${Math.max(prob * 100, 0.5)}%` }} />
                        </div>
                    </div>
                ))}
        </div>
    );
}

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
        translated, brackets, showSolver, solver, solverError } = result;

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

            {/* River subgame solver (when the toggle was on for a river spot). Shown
                FIRST - it's the exact-card answer; the blueprint follows for context. */}
            {showSolver && (
                <div className="mb-5 rounded-xl border border-fuchsia-800/50 bg-fuchsia-950/20 p-4">
                    <h4 className="text-sm uppercase tracking-wider text-fuchsia-300 mb-1">
                        River solver strategy
                    </h4>
                    {solverError ? (
                        <div className="text-sm text-rose-300">
                            Solve unavailable: {solverError}
                        </div>
                    ) : solver ? (
                        <>
                            <p className="text-[11px] text-neutral-500 mb-3">
                                Real-time CFR+ solve of the exact board &amp; line - ungated
                                (no SPR/EV gate). Ranges built by replaying the entered hand
                                through the blueprint · pot {solver.potEntryBb} BB · stacks{' '}
                                {solver.effectiveStackBb} BB
                                {solver.confidence != null ? ` · range confidence ${Math.round(solver.confidence * 100)}%` : ''} ·{' '}
                                {solver.iters} iters
                                {solver.gap != null ? ` · gap ${solver.gap}` : ''}
                                {solver.converged ? ' · converged' : ' · budget-capped'}.
                            </p>
                            <StrategyBars strategy={solver.strategy} />
                        </>
                    ) : null}
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

            {translated && Array.isArray(brackets) && (
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
                        {showSolver ? 'Blueprint strategy (for comparison)' : 'Blueprint strategy'}
                    </h4>
                    <StrategyBars strategy={strategy} />
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
