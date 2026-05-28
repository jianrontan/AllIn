// frontend/src/api.js
// Single point of contact with the backend. Swapping transport (e.g. to
// WebSockets later) or the base URL touches only this file.

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000';

async function request(path, options) {
    let res;
    try {
        res = await fetch(API_BASE + path, options);
    } catch {
        throw new Error('Cannot reach the API server. Is it running on :5000?');
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

// --- Strategy explorer -------------------------------------------------------
export const getAbstractions = () => request('/api/abstractions');

export const getStrategyByKey = (key) =>
    request('/api/strategy?key=' + encodeURIComponent(key));

export const getStrategyFromHand = (payload) =>
    jsonPost('/api/strategy/from-hand', payload);

// --- Play against the bot (used in Phase 3) ---------------------------------
export const newGame = (playerId) => jsonPost('/api/game/new', { playerId });
export const getGameState = (id) =>
    request('/api/game/state?id=' + encodeURIComponent(id));
// `extra` carries unrestricted-sizing fields, e.g. { amountBb } for a
// bet_custom / raise_custom action.
export const sendGameAction = (id, action, extra = {}) =>
    jsonPost('/api/game/action', { id, action, ...extra });
// Runs the bot's pending turn(s). Split from sendGameAction so the UI can reveal
// the freshly-dealt board + a "thinking" indicator before the (slow) river solve.
export const sendBotAction = (id) => jsonPost('/api/game/bot-action', { id });
export const nextHand = (id) => jsonPost('/api/game/next-hand', { id });
