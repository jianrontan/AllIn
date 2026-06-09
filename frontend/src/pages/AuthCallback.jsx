// frontend/src/pages/AuthCallback.jsx
// Cognito Hosted-UI redirect target (/auth/callback). The ID token arrives in
// the URL fragment (implicit flow). We bind it to this browser's EXISTING
// anonymous playerId (non-destructive upgrade) via POST /api/auth/google, then
// show the player's carried-over record - never a 0-state.
import React, { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { authGoogle, setAccount, adoptPlayerId } from '../api';
import UsernameModal from '../components/UsernameModal';

function AuthCallback() {
    const [state, setState] = useState('working');   // working | done | error
    const [row, setRow] = useState(null);
    const [err, setErr] = useState(null);
    const [needUsername, setNeedUsername] = useState(false);
    const ran = useRef(false);

    useEffect(() => {
        if (ran.current) return;                     // StrictMode double-invoke guard
        ran.current = true;
        // alive flag: if the user navigates away (Home, AiGame) before the
        // authGoogle call resolves, suppress all UI state setters. The
        // adoptPlayerId / setAccount side-effects (localStorage writes) DO
        // still fire — those should land regardless so the rest of the app
        // sees the now-signed-in identity.
        let alive = true;
        const frag = new URLSearchParams(window.location.hash.replace(/^#/, ''));
        const idToken = frag.get('id_token');
        const oauthErr = frag.get('error_description') || frag.get('error');
        if (oauthErr) {
            // Do NOT render the URL-supplied error_description: the fragment is
            // attacker-controllable, so echoing it would let anyone display
            // arbitrary text on our trusted origin (phishing). Log it, show fixed copy.
            console.warn('OAuth callback error:', oauthErr);
            setErr('Google sign-in was cancelled or failed. Please try again.');
            setState('error');
            return;
        }
        if (!idToken) { setErr('No ID token in the callback URL.'); setState('error'); return; }
        // Scrub the token from the address bar.
        window.history.replaceState(null, '', window.location.pathname);
        authGoogle(idToken)
            .then((r) => {
                adoptPlayerId(r.playerId);   // localStorage write — always lands
                setAccount(r);               // localStorage write — always lands
                if (!alive) return;
                setRow(r);
                setNeedUsername(!r.usernameSet);
                setState('done');
            })
            .catch((e) => { if (alive) { setErr(e.message); setState('error'); } });
        return () => { alive = false; };
    }, []);

    return (
        <div className="min-h-screen flex flex-col items-center justify-center px-6
                        bg-[radial-gradient(ellipse_at_center,#0c2a1f_0%,#0a0a0a_72%)]">
            {state === 'working' && <p className="text-neutral-300">Signing you in…</p>}
            {state === 'error' && (
                <>
                    <p className="text-rose-400 mb-3">Sign-in failed: {err}</p>
                    <Link to="/" className="text-amber-400 hover:text-amber-300">← Home</Link>
                </>
            )}
            {state === 'done' && (
                <>
                    <h1 className="text-2xl font-bold text-amber-300 mb-2">
                        {row?.handle ? `Welcome back, ${row.handle}` : 'Signed in'}
                    </h1>
                    <p className="text-neutral-400 mb-6 tabular-nums">
                        You&rsquo;ve played {(row?.hands || 0).toLocaleString()} hands
                        {row?.handle
                            ? ' · you’re on the ranked leaderboard.'
                            : ' · pick a username to join the leaderboard.'}
                    </p>
                    <div className="flex gap-3">
                        <Link to="/ai-game"
                            className="px-6 py-3 rounded-xl font-semibold bg-amber-500
                                       text-neutral-950 hover:bg-amber-400">Play</Link>
                        <Link to="/"
                            className="px-6 py-3 rounded-xl font-semibold border border-amber-500/40
                                       text-amber-200 hover:bg-amber-500/10">Home</Link>
                    </div>
                </>
            )}

            <UsernameModal open={state === 'done' && needUsername}
                suggested={row?.suggestedHandle}
                onSaved={(r) => { setAccount(r); setRow(r); setNeedUsername(false); }}
                onSkip={() => setNeedUsername(false)} />
        </div>
    );
}

export default AuthCallback;
