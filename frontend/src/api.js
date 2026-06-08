// frontend/src/api.js
// Single point of contact with the backend. Swapping transport (e.g. to
// WebSockets later) or the base URL touches only this file.

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000';

// The player id this browser owns. Sent with every game request so the backend
// can verify session ownership (a leaked/guessed session id alone is not enough
// to read or act on a game). Set once the player id is known (see setPlayerId).
let playerId = null;
export const setPlayerId = (id) => { playerId = id || null; };

async function request(path, options) {
    let res;
    try {
        res = await fetch(API_BASE + path, options);
    } catch {
        throw new Error(`Cannot reach the API server at ${API_BASE}. Is it running?`);
    }
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
        throw new Error(body.error || `Request failed (HTTP ${res.status})`);
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
