// frontend/src/config.js
// Public, build-time configuration. These are NOT secrets: a Cognito User Pool
// domain + App Client ID are public by design (like a Stripe publishable key).
// The Google OAuth *client secret* never reaches the frontend - it lives only in
// the Cognito IdP config in AWS.

export const COGNITO = {
    domain: import.meta.env.VITE_COGNITO_DOMAIN || '',           // e.g. auth.jianrontan.com
    clientId: import.meta.env.VITE_COGNITO_APP_CLIENT_ID || '',
    redirectUri:
        import.meta.env.VITE_COGNITO_REDIRECT_URI
        || (typeof window !== 'undefined' ? window.location.origin + '/auth/callback' : ''),
};

// True only when the build was given the Cognito public values; otherwise the
// "Sign in with Google" UI hides itself (dev can run without auth).
export const cognitoConfigured = () => !!(COGNITO.domain && COGNITO.clientId);

// Strip a trailing '/' from the Cognito domain so we don't build URLs like
// `https://auth.example.com//oauth2/authorize` (Cognito rejects with a redirect
// mismatch on the doubled slash).
function _cognitoBase() {
    let d = COGNITO.domain;
    if (!d.startsWith('http')) d = `https://${d}`;
    return d.replace(/\/+$/, '');
}

// CSRF state for the OAuth flow. Without this, an attacker who completes a
// Cognito sign-in halfway and captures the redirect can trick a victim into
// visiting /auth/callback#id_token=... and bind the victim's playerId to the
// attacker's Google account (login CSRF / session fixation). We generate a
// fresh cryptographically-random state per sign-in, persist it in
// sessionStorage (per-tab, per-origin), include it in the authorize URL, and
// REQUIRE it to match on the callback. Cognito echoes it back verbatim.
const STATE_KEY = 'allin_oauth_state';

function _randomState() {
    if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
        const a = new Uint8Array(16);
        crypto.getRandomValues(a);
        return Array.from(a, (b) => b.toString(16).padStart(2, '0')).join('');
    }
    // Vanishingly-unlikely fallback for ancient browsers; v4-ish.
    return 'x'.repeat(32).replace(/x/g, () => Math.floor(Math.random() * 16).toString(16));
}

export function takeOAuthState() {
    // One-shot read + delete. Called from AuthCallback; if absent, the callback
    // didn't originate from a hostedUiUrl() click in this same tab → reject.
    try {
        const v = sessionStorage.getItem(STATE_KEY);
        sessionStorage.removeItem(STATE_KEY);
        return v;
    } catch { return null; }
}

// Cognito Hosted UI authorize URL. Implicit flow (response_type=token) returns
// the ID token in the URL fragment to /auth/callback - no client secret needed
// for a public SPA. identity_provider=Google jumps straight to the Google dance.
export function hostedUiUrl() {
    const state = _randomState();
    try { sessionStorage.setItem(STATE_KEY, state); } catch { /* private mode */ }
    const params = new URLSearchParams({
        client_id: COGNITO.clientId,
        response_type: 'token',
        scope: 'openid email profile',
        redirect_uri: COGNITO.redirectUri,
        identity_provider: 'Google',
        state,
    });
    return `${_cognitoBase()}/oauth2/authorize?${params.toString()}`;
}

// Cognito Hosted UI sign-out URL. Clears the Cognito session cookie so a
// subsequent "Sign in with Google" click prompts Google again instead of
// silently re-binding whichever Google session is in the browser. Used by the
// frontend sign-out flow alongside clearing local storage.
export function hostedUiSignOutUrl() {
    const logoutUri = (typeof window !== 'undefined')
        ? window.location.origin + '/' : COGNITO.redirectUri;
    const params = new URLSearchParams({
        client_id: COGNITO.clientId,
        logout_uri: logoutUri,
    });
    return `${_cognitoBase()}/logout?${params.toString()}`;
}
