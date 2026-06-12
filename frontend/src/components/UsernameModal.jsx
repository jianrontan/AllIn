// frontend/src/components/UsernameModal.jsx
// Required (on sign-in) unique-username picker for signed-in players. Pre-filled
// with a suggestion from the Google profile name; the player edits to a unique,
// valid handle (1-20 chars, letters/digits/_/-, no profanity). They can defer
// ("later") but won't appear on the ranked leaderboard until they set one.
import React, { useState } from 'react';
import { upsertHandle } from '../api';

function UsernameModal({ open, suggested, onSaved, onSkip }) {
    const [value, setValue] = useState(suggested || '');
    const [busy, setBusy] = useState(false);
    const [err, setErr] = useState(null);
    // Re-seed the field when a new suggestion arrives (e.g. after sign-in resolves).
    React.useEffect(() => { if (open) setValue(suggested || ''); }, [open, suggested]);
    // Escape = the explicit "Skip for now" button (the modal already has a skip
    // path; keyboard users shouldn't be trapped without one).
    React.useEffect(() => {
        if (!open) return;
        const onKey = (e) => { if (e.key === 'Escape') onSkip?.(); };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [open, onSkip]);
    if (!open) return null;

    const save = async () => {
        const v = value.trim();
        if (!v) return;
        setBusy(true);
        setErr(null);
        try {
            onSaved?.(await upsertHandle(v));
        } catch (e) {
            setErr(e.message);            // 400 invalid / 409 taken, server's wording
        } finally {
            setBusy(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4"
            onClick={() => onSkip?.()}>
            <div role="dialog" aria-modal="true" aria-label="Choose a username"
                onClick={(e) => e.stopPropagation()}
                className="w-full max-w-sm rounded-2xl border border-amber-600/40
                            bg-neutral-900 p-6 shadow-2xl">
                <h2 className="text-lg font-bold text-amber-300 mb-1">Choose a username</h2>
                <p className="text-xs text-neutral-500 mb-4">
                    Your name on the ranked leaderboard. 1-20 characters: letters,
                    digits, _ or - (no spaces). Must be unique.
                </p>
                <input autoFocus value={value}
                    onChange={(e) => setValue(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && save()}
                    placeholder="username"
                    className="w-full px-3 py-2 rounded-lg bg-neutral-800 text-neutral-100
                               border border-neutral-700 focus:border-amber-500 outline-none" />
                {err && <p className="mt-2 text-xs text-rose-400">{err}</p>}
                <button onClick={save} disabled={busy || !value.trim()}
                    className="mt-4 w-full px-5 py-3 rounded-xl font-semibold bg-amber-500
                               text-neutral-950 hover:bg-amber-400 disabled:opacity-50
                               transition-colors">
                    {busy ? 'Saving…' : 'Save username'}
                </button>
                <button onClick={() => onSkip?.()}
                    className="mt-2 w-full text-xs text-neutral-500 hover:text-neutral-300">
                    Skip for now (you won&rsquo;t appear on the leaderboard)
                </button>
            </div>
        </div>
    );
}

export default UsernameModal;
