// frontend/src/components/LoginPrompt.jsx
// Optional "join the leaderboard" login popup. You play anonymously by default;
// signing in (Google, via Cognito Hosted UI) is the only way onto the ranked
// leaderboard. Self-hides when Cognito isn't configured (dev), so it never shows
// a dead button. Always skippable.
import React from 'react';
import { cognitoConfigured } from '../config';
import GoogleSignInButton from './GoogleSignInButton';

function LoginPrompt({ open, onClose }) {
    if (!open || !cognitoConfigured()) return null;
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4"
            onClick={() => onClose?.()}>
            <div className="w-full max-w-sm rounded-2xl border border-amber-600/40
                            bg-neutral-900 p-6 shadow-2xl text-center"
                onClick={(e) => e.stopPropagation()}>
                <h2 className="text-lg font-bold text-amber-300 mb-2">Join the leaderboard?</h2>
                <p className="text-sm text-neutral-400 mb-5">
                    Sign in with Google to claim a ranked spot. Otherwise you play
                    anonymously, and your results still feed the +EV counter.
                </p>
                <div className="flex flex-col items-center gap-3">
                    <GoogleSignInButton registered={false} />
                    <button onClick={() => onClose?.()}
                        className="text-sm text-neutral-500 hover:text-neutral-300">
                        Maybe later
                    </button>
                </div>
            </div>
        </div>
    );
}

export default LoginPrompt;
