// frontend/src/components/GoogleSignInButton.jsx
// Redirects to the Cognito Hosted UI (Google IdP). If already signed in, shows
// the account state. When the build wasn't given the Cognito public config (dev
// without auth) the button still renders, but DISABLED with a hint, so its place
// in the UI is visible (it only works once VITE_COGNITO_* are set at build time).
import React from 'react';
import { cognitoConfigured, hostedUiUrl } from '../config';

function GoogleSignInButton({ registered, handle }) {
    if (registered) {
        return (
            <span className="text-sm text-neutral-400">
                Signed in{handle ? <> as <b className="text-neutral-200">{handle}</b></> : ''}
                <span className="ml-1 text-emerald-500/80" title="ranked-eligible">✓</span>
            </span>
        );
    }
    if (!cognitoConfigured()) {
        return (
            <button disabled
                title="Google sign-in isn't configured yet (set VITE_COGNITO_DOMAIN and VITE_COGNITO_APP_CLIENT_ID at build time)"
                className="px-4 py-2 rounded-lg text-sm font-medium border border-neutral-800
                           text-neutral-500 opacity-60 cursor-not-allowed">
                Sign in with Google
            </button>
        );
    }
    return (
        <button onClick={() => { window.location.href = hostedUiUrl(); }}
            className="px-4 py-2 rounded-lg text-sm font-medium border border-neutral-700
                       text-neutral-200 hover:bg-neutral-800 transition-colors">
            Sign in with Google
        </button>
    );
}

export default GoogleSignInButton;
