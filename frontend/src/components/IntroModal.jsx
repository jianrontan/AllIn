// frontend/src/components/IntroModal.jsx
// First-visit "what is this" popup. Shown on first visit (gated by the
// allin.introDismissed localStorage flag, managed by the parent) and reopenable
// from the header "?" button. Copy is fixed per the deployment handoff.
import React, { useState } from 'react';

const BULLETS = [
    <>Heads-up no-limit hold&rsquo;em vs one AI.</>,
    <>Every hand starts both stacks at <b>100&nbsp;BB</b>, no rebuys, no top-ups,
        but your <b>cross-hand P/L</b> is tracked.</>,
    <>Custom bet sizes welcome, type any amount in BB.</>,
    <>The bot is <b>trained, not scripted</b>, it learned through ~25&nbsp;M hands of
        self-play and re-solves the river live.</>,
    <>Your record contributes to the <b>+EV counter</b> on the homepage. Sign in
        to appear on the ranked leaderboard.</>,
];

// onClose is called with whether "don't show again" was checked.
function IntroModal({ open, onClose }) {
    const [dontShow, setDontShow] = useState(false);
    if (!open) return null;
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4"
            onClick={() => onClose(dontShow)}>
            <div className="w-full max-w-md rounded-2xl border border-amber-600/40
                            bg-neutral-900 p-6 shadow-2xl"
                onClick={(e) => e.stopPropagation()}>
                <h2 className="text-xl font-bold text-amber-300 mb-4">Welcome to AllIn</h2>
                <ul className="space-y-2.5 text-sm text-neutral-300">
                    {BULLETS.map((b, i) => (
                        <li key={i} className="flex gap-2">
                            <span className="text-amber-500 shrink-0">•</span>
                            <span>{b}</span>
                        </li>
                    ))}
                </ul>
                <label className="mt-5 flex items-center gap-2 text-xs text-neutral-400 cursor-pointer">
                    <input type="checkbox" checked={dontShow}
                        onChange={(e) => setDontShow(e.target.checked)}
                        className="h-4 w-4 accent-amber-500" />
                    Don&rsquo;t show this again
                </label>
                <button onClick={() => onClose(dontShow)}
                    className="mt-4 w-full px-5 py-3 rounded-xl font-semibold bg-amber-500
                               text-neutral-950 hover:bg-amber-400 transition-colors">
                    Continue
                </button>
            </div>
        </div>
    );
}

export default IntroModal;
