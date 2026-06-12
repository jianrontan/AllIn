// frontend/src/api.js
// Single point of contact with the backend. Swapping transport (e.g. to
// WebSockets later) or the base URL touches only this file.

// The localhost fallback is DEV-ONLY: a production bundle built without
// VITE_API_BASE must not silently point every visitor at localhost (each
// would "work" on a developer machine and break for everyone else). The
// production build fails loudly instead -- see the guard in vite.config.js.
const API_BASE = import.meta.env.VITE_API_BASE
    || (import.meta.env.DEV ? 'http://localhost:5000' : '');

// The player id this browser owns. Sent with every game request so the backend
// can verify session ownership (a leaked/guessed session id alone is not enough
// to read or act on a game). Set once the player id is known (see setPlayerId).
let playerId = null;
export const setPlayerId = (id) => { playerId = id || null; };

const PLAYER_ID_KEY = 'allin_player_id';
const uuidv4 = () =>
    (crypto.randomUUID
        ? crypto.randomUUID()
        : 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
            const r = (Math.random() * 16) | 0;
            return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
        }));

// The browser's stable anonymous identity. Created + persisted on first call;
// also primes the module-level playerId used by game requests.
export function getPlayerId() {
    let id = localStorage.getItem(PLAYER_ID_KEY);
    if (!id) {
        id = uuidv4();
        localStorage.setItem(PLAYER_ID_KEY, id);
    }
    playerId = id;
    return id;
}

// Adopt a canonical playerId returned by sign-in (one account per Google sub):
// a returning user on a new device switches this browser to the account's id.
export function adoptPlayerId(id) {
    if (!id) return;
    localStorage.setItem(PLAYER_ID_KEY, id);
    playerId = id;
}

// Active game session id, persisted so reloads CONTINUE the same session
// (preserving hand_number + human_net + the dealt hand) instead of starting a
// fresh one. Sessions live 24h on the backend; if the stored id 404s on
// /api/game/state, the caller clears it and starts a new game cleanly.
const SESSION_ID_KEY = 'allin_session_id';
export const getStoredSessionId = () => localStorage.getItem(SESSION_ID_KEY);
export const setStoredSessionId = (id) => {
    if (id) localStorage.setItem(SESSION_ID_KEY, id);
};
export const clearStoredSessionId = () => localStorage.removeItem(SESSION_ID_KEY);

// Lightweight cached account state (handle + whether signed in), persisted so the
// header can render it without a self-lookup endpoint. Updated after sign-in
// (/auth/callback) and after a handle change.
const ACCOUNT_KEY = 'allin_account';
export const getAccount = () => {
    try { return JSON.parse(localStorage.getItem(ACCOUNT_KEY) || 'null'); }
    catch { return null; }
};
export const setAccount = (row) => {
    if (!row) return;
    const prev = getAccount() || {};
    localStorage.setItem(ACCOUNT_KEY, JSON.stringify({
        handle: row.handle ?? prev.handle ?? null,
        isRegistered: row.isRegistered ?? prev.isRegistered ?? false,
    }));
};

// Sign out: clear ALL local identity (playerId, account cache, active session).
// Callers typically follow with `window.location.href = hostedUiSignOutUrl()`
// so the Cognito session cookie is also dropped (otherwise a subsequent
// "Sign in with Google" silently re-binds whatever Google session is in the
// browser). Mints a fresh anonymous playerId on next page load via getPlayerId.
export function signOutLocal() {
    try {
        localStorage.removeItem(PLAYER_ID_KEY);
        localStorage.removeItem(ACCOUNT_KEY);
        localStorage.removeItem(SESSION_ID_KEY);
    } catch { /* private mode / quota issues — best effort */ }
    playerId = null;
}

async function request(path, options) {
    let res;
    try {
        res = await fetch(API_BASE + path, options);
    } catch {
        // Tag network-level failures so UI copy can stay friendly (the raw
        // message embeds API_BASE, which reads as broken/leaky on prod).
        const e = new Error('Could not reach the server. Check your connection and try again.');
        e.isNetworkError = true;
        throw e;
    }
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
        const e = new Error(body.error || `Request failed (HTTP ${res.status})`);
        e.status = res.status;
        throw e;
    }
    return body;
}

const jsonPost = (path, payload) => request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {}),
});

// POST a game request, attaching the owning playerId for the ownership check.
const gamePost = (path, payload) => jsonPost(path, { ...(payload || {}), playerId });

// --- Strategy explorer -------------------------------------------------------
export const getAbstractions = () => request('/api/abstractions');

export const getStrategyByKey = (key) =>
    request('/api/strategy?key=' + encodeURIComponent(key));

export const getStrategyFromHand = (payload) =>
    jsonPost('/api/strategy/from-hand', payload);

// River subgame solver, on demand: the UNGATED solved strategy for a concrete
// river spot (real cards + board + line + river-entry pot). Range-vs-range with
// uniform river-entry ranges. Only valid on the river (5 community cards).
export const riverSolve = (payload) =>
    jsonPost('/api/strategy/river-solve', payload);

// --- Play against the bot (used in Phase 3) ---------------------------------
// Start a game. Set the player id first (setPlayerId) from any saved value; the
// backend reuses it or mints one and echoes it back, which the caller stores.
export const newGame = () => gamePost('/api/game/new', {});
export const getGameState = (id) =>
    request('/api/game/state?id=' + encodeURIComponent(id)
        + (playerId ? '&playerId=' + encodeURIComponent(playerId) : ''));
// `extra` carries unrestricted-sizing fields, e.g. { amountBb } for a
// bet_custom / raise_custom action.
export const sendGameAction = (id, action, extra = {}) =>
    gamePost('/api/game/action', { id, action, ...extra });
// Runs the bot's pending turn(s). Split from sendGameAction so the UI can reveal
// the freshly-dealt board + a "thinking" indicator before the (slow) river solve.
export const sendBotAction = (id) => gamePost('/api/game/bot-action', { id });
export const nextHand = (id) => gamePost('/api/game/next-hand', { id });

// --- +EV leaderboard + accounts ---------------------------------------------
export const getStats = () => request('/api/stats');
// The caller's OWN row (lifetime hands + net + bb/100). Returns a 0-state row
// if the player is unknown — never 404 — so the UI renders cleanly on first
// load before any hand.
export const getMe = () =>
    request('/api/me?playerId=' + encodeURIComponent(getPlayerId()));

// /api/healthz — exposes `debugOverlay` so the UI can hide the Debug button
// entirely when the backend has the overlay redacted. Cached after the first
// successful call.
export const getHealth = () => request('/api/healthz');
export const getLeaderboard = ({ n = 10, minHands = 50, accountsOnly = false } = {}) =>
    request(`/api/leaderboard?n=${n}&min_hands=${minHands}`
        + (accountsOnly ? '&accounts_only=true' : ''));
// Set the caller's unique username (signed-in players, on sign-in or rename).
// Throws on 400 (invalid) / 409 (taken) with the server's message.
export const upsertHandle = (handle) =>
    jsonPost('/api/player', { playerId: getPlayerId(), handle });
// Bind a Cognito-issued Google ID token; returns the canonical account row
// ({playerId, usernameSet, suggestedHandle, ...}). The caller adopts playerId.
export const authGoogle = (idToken) =>
    jsonPost('/api/auth/google', { idToken, playerId: getPlayerId() });
