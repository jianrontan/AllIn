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

// Cognito Hosted UI authorize URL. Implicit flow (response_type=token) returns
// the ID token in the URL fragment to /auth/callback - no client secret needed
// for a public SPA. identity_provider=Google jumps straight to the Google dance.
export function hostedUiUrl() {
    const base = COGNITO.domain.startsWith('http')
        ? COGNITO.domain
        : `https://${COGNITO.domain}`;
    const params = new URLSearchParams({
        client_id: COGNITO.clientId,
        response_type: 'token',
        scope: 'openid email profile',
        redirect_uri: COGNITO.redirectUri,
        identity_provider: 'Google',
    });
    return `${base}/oauth2/authorize?${params.toString()}`;
}
