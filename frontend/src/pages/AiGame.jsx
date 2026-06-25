// frontend/src/pages/AiGame.jsx
// Play full heads-up hands against the blueprint bot under the abstracted
// rules it was trained on. The backend is authoritative; this page renders
// the redacted state it returns and posts the player's actions.
//
// All amounts are shown in big blinds (the backend works in chips; 1 BB = 2).
import React, { useEffect, useRef, useState } from 'react';
import { Link, useNavigate, useBlocker } from 'react-router-dom';
import { newGame, sendGameAction, sendBotAction, nextHand, getGameState,
    getPlayerId, adoptPlayerId, getAccount, getMe, getHealth,
    getStoredSessionId, setStoredSessionId, clearStoredSessionId } from '../api';
import { fmtBB, fmtBBSigned } from '../format';
import PlayingCard from '../components/PlayingCard';
import EvCounter from '../components/EvCounter';
import IntroModal from '../components/IntroModal';
import LoginPrompt from '../components/LoginPrompt';
import { AnnouncementsButton } from '../components/Announcements';
import GoogleSignInButton from '../components/GoogleSignInButton';

// Standardized snake_case localStorage key (matches the other allin_* keys
// owned by api.js — sessionId, account, playerId). Old `allin.introDismissed`
// users will see the popup once after this change; acceptable trade-off.
const INTRO_KEY = 'allin_intro_dismissed';

// Abstract action -> button verb. The BB amount is shown separately, below
// the verb, so the player always sees exactly how much is being committed.
const ACTION_VERB = {
    fold: 'Fold', check: 'Check', call: 'Call', allin: 'All-in',
    bet_small: 'Bet small', bet_medium: 'Bet medium', bet_large: 'Bet large',
    bet_xlarge: 'Bet xlarge', bet_overbet: 'Overbet',
    raise_small: 'Raise small', raise_medium: 'Raise medium', raise_large: 'Raise large',
    raise_overbet: 'Raise overbet',
    // bet_xlarge is preflop-open-only; actionVerb maps a preflop bet_* to raise_*,
    // so this entry is what the xlarge open actually renders as ("Raise xlarge").
    raise_xlarge: 'Raise xlarge',
};

// Canonical display order. The backend's stack-constraint logic inserts
// 'allin' wherever it lands in its list; this pins a stable order with
// all-in always last, bets/raises ascending by size (overbet after large,
// xlarge - the 4th preflop open - after large too).
const ACTION_ORDER = {
    fold: 0, check: 1, call: 2,
    bet_small: 3, bet_medium: 4, bet_large: 5, bet_xlarge: 6, bet_overbet: 7,
    raise_small: 8, raise_medium: 9, raise_large: 10, raise_overbet: 11,
    allin: 12,
};

const sortedActions = (actions) =>
    [...actions].sort(
        (a, b) => (ACTION_ORDER[a.action] ?? 99) - (ACTION_ORDER[b.action] ?? 99));

// Preflop the blinds are already posted, so the first aggressive action is a
// raise over the big blind - the backend models it as `bet_*`, but we show it
// as "Raise". Postflop, `bet_*` is a genuine bet.
const actionVerb = (action, street) => {
    if (street === 'preflop' && action.startsWith('bet_')) {
        return ACTION_VERB['raise_' + action.slice(4)] || action;
    }
    return ACTION_VERB[action] || action;
};

// Action-log verb: drops the small/medium/large size word (the exact size is
// already shown as the BB amount), so the log reads "Raise 3 BB", "Bet 4.6 BB".
const logVerb = (action, street) => {
    if (action.startsWith('raise_')) return 'Raise';
    if (action.startsWith('bet_')) return street === 'preflop' ? 'Raise' : 'Bet';
    return ACTION_VERB[action] || action;     // fold / check / call / all-in
};

const actionClasses = (a) => {
    if (a === 'fold') return 'bg-rose-700 hover:bg-rose-600';
    if (a === 'call' || a === 'check') return 'bg-sky-700 hover:bg-sky-600';
    if (a === 'allin') return 'bg-violet-700 hover:bg-violet-600';
    return 'bg-emerald-700 hover:bg-emerald-600';
};

const FELT = {
    background: 'radial-gradient(ellipse at center, #11815a 0%, #064534 78%)',
};

// Empty board placeholder, the same fluid size as a PlayingCard (via the shared
// `--card-w` variable) so the board reserves space for all five cards from the
// start - the table never resizes as the flop/turn/river land, and empty slots
// track the cards as the board scales with the window. When `onReveal` is set (a
// fold left cards undealt) the slot itself becomes the "reveal the run-out" button.
function EmptySlot({ onReveal }) {
    const sizeStyle = { width: 'var(--card-w, 3.5rem)', aspectRatio: '7 / 10' };
    if (onReveal) {
        return (
            <button onClick={onReveal} style={sizeStyle}
                title="Reveal the cards that didn't come out"
                className="rounded-lg border-2 border-dashed border-amber-500/50
                           text-amber-300/80 hover:text-amber-200 hover:border-amber-400/80
                           hover:bg-amber-500/10 flex items-center justify-center
                           transition-colors">
                <span className="text-[10px] leading-tight font-semibold">Reveal</span>
            </button>
        );
    }
    return <div style={sizeStyle}
        className="rounded-lg border-2 border-dashed border-white/10" />;
}

function Seat({ name, stackChips, active, holding }) {
    return (
        <div className="flex items-center justify-center gap-2 text-sm">
            <span className={'inline-block w-2 h-2 rounded-full ' +
                (active ? 'bg-amber-400 shadow-[0_0_8px] shadow-amber-400' : 'bg-transparent')} />
            <span className="text-neutral-200 font-medium">{name}</span>
            <span className="text-neutral-400">· {fmtBB(stackChips)} BB stack</span>
            {holding && (
                <span className="px-2 py-0.5 rounded-full bg-black/30 text-xs
                                 text-neutral-200">
                    {holding}
                </span>
            )}
        </div>
    );
}

// Colour the "last action" pill by action family, matching the action buttons:
// fold rose, check/call sky, all-in violet, bet/raise (incl. custom) emerald.
const lastActionTone = (a) => {
    if (a === 'fold') return 'bg-rose-500/20 text-rose-200';
    if (a === 'check' || a === 'call') return 'bg-sky-500/20 text-sky-200';
    if (a === 'allin') return 'bg-violet-500/20 text-violet-200';
    if (a.startsWith('bet_') || a.startsWith('raise_')) return 'bg-emerald-500/20 text-emerald-200';
    return 'bg-white/10 text-neutral-200';
};

// "What just happened": a small pill next to a seat showing that player's most
// recent action this hand (verb + size), so the flow is readable at a glance
// without scanning the action log. It re-animates on every change via a keyed
// remount (see the lastActionIn keyframe in index.css).
function LastActionPill({ entry }) {
    if (!entry) return null;
    const amt = entry.chips > 0 ? ` ${fmtBB(entry.chips)} BB` : '';
    return (
        <span key={`${entry.action}|${entry.chips}|${entry.street}`}
            className={'px-2 py-0.5 rounded-full text-xs font-medium ' +
                'animate-[lastActionIn_220ms_ease-out] ' + lastActionTone(entry.action)}>
            {logVerb(entry.action, entry.street)}{amt}
        </span>
    );
}

// The chips a player currently has wagered in front of them this street.
function BetChip({ chips }) {
    if (!chips || chips <= 0) return null;
    return (
        <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-3.5 rounded-full bg-amber-400
                             ring-2 ring-amber-600 ring-offset-1 ring-offset-emerald-900" />
            <span className="text-sm font-semibold text-amber-100 tabular-nums">
                {fmtBB(chips)} BB
            </span>
        </div>
    );
}

// The bot's hand-level Bayesian belief about the human's hole cards (Phase 3
// range tracker). `confidence` is how well the human's actions have matched the
// blueprint model the bot updates against - it drops if you play unexpectedly.
// Showing it is safe: it's a guess about the player's OWN cards, and never
// reveals the bot's cards.
function BotRead({ read, final }) {
    if (!read || !read.topHands || read.topHands.length === 0) return null;
    const confPct = Math.round(read.confidence * 100);
    return (
        <div>
            <h4 className="text-xs uppercase tracking-wider text-neutral-500 mb-2">
                Bot&rsquo;s read of your hand
                {final && <span className="ml-1.5 text-neutral-600 normal-case">· final</span>}
            </h4>
            <div className="text-sm text-neutral-400 mb-2">
                Read confidence:{' '}
                <span className="text-neutral-200 font-medium tabular-nums">{confPct}%</span>
                <span className="text-neutral-600">
                    {' '}· top hands it {final ? 'put' : 'puts'} you on
                </span>
            </div>
            <div className="flex flex-wrap gap-1.5">
                {read.topHands.map((h, i) => (
                    <span key={i} className="px-2 py-1 rounded bg-black/30 text-xs
                                             tabular-nums text-neutral-300">
                        {h.label}{' '}
                        <span className="text-neutral-500">{(h.prob * 100).toFixed(1)}%</span>
                    </span>
                ))}
            </div>
        </div>
    );
}

// One {action: prob} distribution as a stack of labelled bars (debug overlay).
function DistBars({ rows }) {
    if (!rows || rows.length === 0) return <div className="text-neutral-700">-</div>;
    return (
        <div className="space-y-0.5">
            {rows.map(([a, p]) => (
                <div key={a} className="flex items-center gap-2">
                    <span className="w-28 shrink-0 truncate text-neutral-300" title={a}>{a}</span>
                    <span className="flex-1 h-1.5 rounded bg-neutral-800 overflow-hidden">
                        <span className="block h-full bg-emerald-500"
                            style={{ width: `${Math.round((p || 0) * 100)}%` }} />
                    </span>
                    <span className="w-9 text-right tabular-nums text-neutral-500">
                        {Math.round((p || 0) * 100)}%
                    </span>
                </div>
            ))}
        </div>
    );
}

// Debug overlay: the bot's per-decision trace this hand. For each bot action it
// shows the blueprint info-set key the bot was queried with, the strategy stored
// there, the action chosen, and - on the river - the subgame solver's solved
// strategy + EV gate. SPOILER: the info-set key encodes the bot's card bucket, so
// this lives behind a toggle and is meant for inspection, not mid-hand peeking.
function BotDebug({ debug }) {
    if (!debug || debug.length === 0) {
        return <div className="text-xs text-neutral-600">No bot decisions yet this hand.</div>;
    }
    const dist = (d) => Object.entries(d || {})
        .filter(([, p]) => p > 0.0001)
        .sort((a, b) => b[1] - a[1]);

    const badge = (s) => {
        if (s.mode === 'river_solver' || s.mode === 'turn_solver') {
            const p = s.mode === 'turn_solver' ? 'turn ' : '';
            return s.deviated
                ? [p + 'solver · deviated', 'bg-fuchsia-900/60 text-fuchsia-200']
                : [p + 'solver · kept BP', 'bg-sky-900/60 text-sky-200'];
        }
        if (s.mode === 'exploit_tilt') return ['exploit tilt', 'bg-emerald-900/60 text-emerald-200'];
        if (s.mode === 'allin_guard') return ['all-in guard', 'bg-amber-900/60 text-amber-200'];
        if (s.mode === 'deep_raise_guard') return ['deep-raise guard', 'bg-amber-900/60 text-amber-200'];
        if (s.mode === 'first_act_value') return ['first-act value', 'bg-amber-900/60 text-amber-200'];
        if (s.mode === 'premium_no_fold') return ['premium no-fold', 'bg-amber-900/60 text-amber-200'];
        if (s.mode === 'fallback') return ['solver fallback', 'bg-rose-900/60 text-rose-200'];
        return ['blueprint', 'bg-neutral-800 text-neutral-400'];
    };

    return (
        <div className="space-y-3">
            {/* Most recent decision first (newest on top). */}
            {debug.slice().reverse().map((r, i) => {
                const s = r.solver;
                const [label, cls] = s ? badge(s) : [null, null];
                return (
                    <div key={i}
                        className="rounded-lg border border-neutral-800 bg-black/30 p-2.5 text-xs">
                        <div className="flex items-center justify-between mb-1.5">
                            <span className="uppercase tracking-wider text-neutral-500">{r.street}</span>
                            {s && (
                                <span className={'px-1.5 py-0.5 rounded text-[10px] font-semibold ' + cls}>
                                    {label}
                                </span>
                            )}
                        </div>
                        <div className="font-mono text-[11px] text-emerald-300 break-all mb-1.5">
                            {r.infoSetKey}
                        </div>
                        <div className="text-neutral-500 mb-1">
                            chose <span className="text-neutral-200 font-semibold break-all">{r.chosen}</span>
                        </div>
                        {s && s.exploitOn !== undefined && (
                            <div className="text-[10px] mb-1">
                                <span className="text-neutral-500">exploit </span>
                                <span className={s.exploitOn ? 'text-emerald-400' : 'text-neutral-500'}>
                                    {s.exploitOn ? 'ON' : 'off'}
                                </span>
                                {s.exploitOn && s.readConfidence != null && (
                                    <span className={s.readConfidence < s.guardConfidence
                                        ? 'text-amber-400' : 'text-emerald-300'}>
                                        {' · read ' + s.readConfidence}
                                        {s.readConfidence < s.guardConfidence
                                            ? ' < ' + s.guardConfidence + ' → blueprint (retreated)'
                                            : ' ≥ ' + s.guardConfidence}
                                    </span>
                                )}
                            </div>
                        )}
                        {s && s.guardEq !== undefined && (
                            <div className="text-[10px] mb-1 text-neutral-400">
                                all-in guard: eq{' '}
                                <span className={s.guardNeed != null && s.guardEq >= s.guardNeed
                                    ? 'text-emerald-300' : 'text-amber-400'}>{s.guardEq}</span>
                                {s.guardNeed != null && <span> vs need {s.guardNeed}</span>}
                                {' '}· EV(call) {s.guardEvCall}
                            </div>
                        )}
                        {r.deepJamRoute && (
                            <div className="text-[10px] mb-1 text-amber-300">
                                deep stack: blueprint wanted ALL-IN {r.deepJamRoute.allinPct}% (not legal here)
                                {' '}→ re-translated to sized bets ({r.deepJamRoute.scheme})
                            </div>
                        )}
                        {r.rawBlueprint && (
                            <>
                                <div className="text-neutral-600 mb-0.5">blueprint intent (raw · incl. all-in)</div>
                                <DistBars rows={dist(r.rawBlueprint)} />
                            </>
                        )}
                        <div className="text-neutral-600 mb-0.5">
                            {r.deepJamRoute ? 'served (re-translated)' : 'blueprint strategy'}
                        </div>
                        <DistBars rows={dist(r.strategy)} />
                        {s && s.tiltedStrategy && (
                            <>
                                <div className="text-emerald-400/80 mt-2 mb-0.5">exploit-tilted (served)</div>
                                <DistBars rows={dist(s.tiltedStrategy)} />
                            </>
                        )}
                        {s && (s.mode === 'river_solver' || s.mode === 'turn_solver') && (
                            <>
                                <div className="text-neutral-600 mt-2 mb-0.5">
                                    solved ({s.mode === 'turn_solver' ? 'turn' : 'river'} subgame)
                                </div>
                                <DistBars rows={dist(s.solvedStrategy)} />
                                <div className="mt-1.5 text-[11px] text-neutral-500 tabular-nums leading-relaxed">
                                    EV solved {s.evSolved} vs BP {s.evBaseline}
                                    {' '}(Δ {s.evDelta}, margin {s.evMargin})<br />
                                    {s.iters} iters · gap {s.gap}{s.converged ? ' · converged' : ''}
                                </div>
                            </>
                        )}
                    </div>
                );
            })}
        </div>
    );
}

// Confirm leaving a live hand — styled like the app's other overlays
// (IntroModal / LoginPrompt) instead of the browser's window.confirm.
function LeaveHandModal({ open, busy, onConfirm, onCancel }) {
    if (!open) return null;
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4"
            onClick={busy ? undefined : onCancel}>
            <div className="w-full max-w-sm rounded-2xl border border-amber-600/40
                            bg-neutral-900 p-6 shadow-2xl text-center"
                onClick={(e) => e.stopPropagation()}>
                <h2 className="text-lg font-bold text-amber-300 mb-2">Leave this hand?</h2>
                <p className="text-sm text-neutral-400 mb-5">
                    Your hand will be folded and the hand counts as a loss. You can
                    start a fresh hand any time.
                </p>
                <div className="flex flex-col gap-2.5">
                    <button onClick={onConfirm} disabled={busy}
                        className="w-full px-5 py-3 rounded-xl font-semibold bg-rose-600
                                   text-white hover:bg-rose-500 disabled:opacity-50
                                   transition-colors">
                        {busy ? 'Folding…' : 'Fold & leave'}
                    </button>
                    <button onClick={onCancel} disabled={busy}
                        className="text-sm text-neutral-500 hover:text-neutral-300
                                   disabled:opacity-50">
                        Stay in the hand
                    </button>
                </div>
            </div>
        </div>
    );
}

function AiGame() {
    const [view, setView] = useState(null);
    const [showDebug, setShowDebug] = useState(false);
    // When a hand ends by a fold before the river, the player can reveal the
    // community cards that never came out. Reset each new hand (see effect below).
    const [showRunout, setShowRunout] = useState(false);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState(null);
    const [customAmt, setCustomAmt] = useState('');
    // True while the bot's turn (incl. a slow river solve) is in flight, so the UI
    // can show a "thinking" indicator after the new card has been revealed.
    const [thinking, setThinking] = useState(false);
    const [dots, setDots] = useState('');
    // Guards against React StrictMode invoking the mount effect twice in dev,
    // which would otherwise deal (and orphan) a second game session.
    const sessionStarted = useRef(false);
    // Synchronous in-flight guard. `busy` re-renders asynchronously, so a fast
    // double-click can fire two requests before the buttons disable; this blocks
    // the second immediately (and avoids the server's spurious 409 banner).
    const inFlight = useRef(false);
    // Identity + first-visit popup + optional login nudge. Lazy initializer so
    // the signed-in chip renders on the first frame (no "Sign in" flash).
    const [account, setAccountState] = useState(() => getAccount());
    const [showIntro, setShowIntro] = useState(false);
    const [showLogin, setShowLogin] = useState(false);
    // Leave-this-hand confirm overlay (driven by the navigation blocker below).
    // `leaving` is true while the fold request is in flight so the buttons disable.
    const [leaving, setLeaving] = useState(false);
    // Lifetime stats for THIS player (hands + net across all their sessions).
    // Refreshes on mount and again after every completed hand. Distinct from
    // view.handNumber / view.humanNet (which are PER-SESSION).
    const [lifetime, setLifetime] = useState(null);
    // Debug overlay availability — `null` until /api/healthz answers; `true`
    // shows the toggle, `false` hides the whole button. Prevents an inert
    // "Debug" button when the backend has ALLIN_DEBUG_OVERLAY=0.
    const [debugAvailable, setDebugAvailable] = useState(null);
    const loginAsked = useRef(false);
    const navigate = useNavigate();
    // Null-safe (view is null until the first deal). Shared by the unload guard and
    // the leave confirm.
    const liveHand = view?.status === 'in_hand';

    // Intercept in-app navigation away from a live hand (browser Back, the Home
    // link, anything) so it routes through the leave-this-hand confirm instead of
    // silently abandoning the hand. The modal opens off `blocker.state` (below);
    // confirm proceeds, cancel resets. Requires the data router (see App.jsx).
    const blocker = useBlocker(
        ({ currentLocation, nextLocation }) =>
            liveHand && currentLocation.pathname !== nextLocation.pathname);

    // While a hand is live, warn on a real page unload (tab close / reload). The
    // browser only allows its own generic prompt here; the actual fold is the
    // server's inactivity sweeper (a beforeunload handler can't reliably complete
    // a request). In-app navigation away is handled by the Home link's confirm.
    // Depend on the boolean, not `view`, so it only re-binds on the live transition.
    useEffect(() => {
        if (!liveHand) return;
        const handler = (e) => { e.preventDefault(); e.returnValue = ''; };
        window.addEventListener('beforeunload', handler);
        return () => window.removeEventListener('beforeunload', handler);
    }, [liveHand]);

    useEffect(() => {
        getPlayerId();                       // ensure the anonymous id exists
        setAccountState(getAccount());
        if (localStorage.getItem(INTRO_KEY) !== 'true') setShowIntro(true);
    }, []);

    // Closing the first-visit popup: persist the "don't show again" flag, then
    // (once) nudge an un-signed-in player to log in for the leaderboard. The
    // LoginPrompt self-hides when Cognito isn't configured, so dev sees nothing.
    const closeIntro = (dontShow) => {
        setShowIntro(false);
        if (dontShow) localStorage.setItem(INTRO_KEY, 'true');
        if (!loginAsked.current && !getAccount()?.isRegistered) {
            loginAsked.current = true;
            setShowLogin(true);
        }
    };

    const startSession = async () => {
        setBusy(true);
        setError(null);
        try {
            // Send any saved player id so the backend reuses it; it echoes back
            // the authoritative id, which we persist and keep for the ownership
            // check on every later request. `getPlayerId()` is the canonical
            // setter — it both reads from localStorage AND primes api.js's
            // module-level `playerId` used by every game request.
            getPlayerId();

            // Try to CONTINUE the previous session: a reload mid-session should
            // preserve hand_number + human_net + the dealt hand. The backend keeps
            // sessions for 24h; a stale/missing id 404s, and we silently fall
            // through to a fresh game.
            const storedId = getStoredSessionId();
            if (storedId) {
                try {
                    const v = await getGameState(storedId);
                    if (v.playerId) adoptPlayerId(v.playerId);
                    setView(v);
                    return;                          // restored cleanly
                } catch {
                    clearStoredSessionId();          // expired / unknown -> fresh game
                }
            }
            const v = await newGame();
            if (v.playerId) adoptPlayerId(v.playerId);
            if (v.sessionId) setStoredSessionId(v.sessionId);
            setView(v);
        } catch (e) {
            setError(e.message);
        } finally {
            setBusy(false);
        }
    };

    useEffect(() => {
        if (sessionStarted.current) return;
        sessionStarted.current = true;
        startSession();
    }, []);

    // Collapse the fold run-out reveal whenever a new hand is dealt.
    const handNumber = view ? view.handNumber : null;
    useEffect(() => { setShowRunout(false); }, [handNumber]);

    // Persist the active session id so a reload (or a Home -> AiGame round-trip)
    // resumes the SAME session instead of dealing a new one. The backend stores
    // sessions for 24h; an expired/missing id 404s on /state and startSession
    // falls through to newGame cleanly (clearing the stored id).
    useEffect(() => {
        if (view?.sessionId) setStoredSessionId(view.sessionId);
    }, [view?.sessionId]);

    // Lifetime stats: load on mount, refresh whenever a hand ends. getMe() returns
    // a 0-state row for unknown players, so this never 404s.
    const fetchLifetime = async () => {
        try { setLifetime(await getMe()); } catch { /* silent: dev / stats outage */ }
    };
    useEffect(() => { fetchLifetime(); }, []);

    // Check ONCE on mount whether the backend exposes the debug overlay
    // (ALLIN_DEBUG_OVERLAY). If it doesn't, the Debug button is hidden — no
    // dead toggle, no "No bot decisions yet" panel that never populates.
    useEffect(() => {
        getHealth()
            .then((h) => setDebugAvailable(!!h.debugOverlay))
            .catch(() => setDebugAvailable(false));
    }, []);
    const handEnded = view?.status === 'hand_over';
    useEffect(() => { if (handEnded) fetchLifetime(); }, [handEnded, handNumber]);

    // Animated ellipsis for the "Bot is thinking" indicator: '' -> . -> .. -> ...
    useEffect(() => {
        if (!thinking) { setDots(''); return; }
        const seq = ['', '.', '..', '...'];
        let i = 0;
        const id = setInterval(() => { i = (i + 1) % seq.length; setDots(seq[i]); }, 400);
        return () => clearInterval(id);
    }, [thinking]);

    // Returns true on success, false if it bailed or errored (so callers can
    // decide whether to clear input).
    const run = async (fn) => {
        if (inFlight.current) return false;   // ignore overlapping submits
        inFlight.current = true;
        setBusy(true);
        setError(null);
        let ok = false;
        try {
            let v = await fn();
            setView(v);                       // reveal the new card immediately
            // The bot's turn runs separately so the freshly-dealt card shows first;
            // loop it (with a "thinking" indicator) until it's the human's turn or
            // the hand ends. The backend also pauses the bot whenever ITS action
            // deals a new board card (stop_on_new_card), so each loop pass renders
            // that card before the bot's next (possibly slow river-solve) decision
            // - i.e. you see the river, then the bot thinks, never the reverse.
            // Cap generously so a legitimate hand (multi-street, board-pause stops,
            // an uncapped re-raise war) never strands at the limit; it only bounds a
            // runaway from a backend bug. If it ever trips, the Reconnect button
            // re-fetches authoritative state.
            let guard = 0;
            while (v && v.status === 'in_hand' && v.toAct === 'bot' && guard < 24) {
                setThinking(true);
                v = await sendBotAction(v.sessionId);
                setView(v);
                guard += 1;
            }
            ok = true;
        } catch (e) {
            setError(e.message);
            // The action may have been rejected because the hand already ended (e.g. the
            // inactivity sweeper folded an abandoned idle hand). Re-sync authoritative state so
            // the UI shows hand_over + "Next hand" instead of stranding on a 409, and drop the
            // banner when the hand simply ended.
            if (view?.sessionId) {
                try {
                    const fresh = await getGameState(view.sessionId);
                    setView(fresh);
                    if (fresh.status === 'hand_over') setError(null);
                } catch { /* re-sync failed -- keep the original error */ }
            }
        } finally {
            setThinking(false);
            setBusy(false);
            inFlight.current = false;
        }
        return ok;
    };

    const doAction = (action) => run(() => sendGameAction(view.sessionId, action));
    const dealNext = () => run(() => nextHand(view.sessionId));

    // Recover from a transient network error mid-hand: re-fetch authoritative
    // state instead of stranding the hand (a reload would start a new session).
    const resync = async () => {
        if (!view) return;
        setError(null);
        try {
            setView(await getGameState(view.sessionId));
        } catch (e) {
            setError(e.message);
        }
    };

    // Unrestricted custom bet/raise: it's a raise when there's a bet to call,
    // otherwise a bet. amountBb is the raise-to TOTAL in big blinds.
    const facingBet = !!view && (view.legalActions || []).some((la) => la.action === 'call');
    const customAmtNum = parseFloat(customAmt);
    const customValid = !!(view && view.customBounds) && !isNaN(customAmtNum)
        && customAmtNum >= view.customBounds.minBb
        && customAmtNum <= view.customBounds.maxBb;
    // Snap a typed/spun amount back into the legal range when the field loses
    // focus, so the spinner arrows (step) can't strand it on an illegal value.
    const clampCustom = () => {
        if (customAmt === '' || !(view && view.customBounds)) return;
        const n = parseFloat(customAmt);
        if (isNaN(n)) { setCustomAmt(''); return; }
        const { minBb, maxBb } = view.customBounds;
        setCustomAmt(String(Math.min(maxBb, Math.max(minBb, n))));
    };
    const doCustom = async () => {
        if (!customValid) return;
        const action = facingBet ? 'raise_custom' : 'bet_custom';
        const amt = customAmtNum;
        // Clear the input only after the bet is accepted, so a rejected bet keeps
        // what was typed for an easy retry.
        const ok = await run(() => sendGameAction(view.sessionId, action, { amountBb: amt }));
        if (ok) setCustomAmt('');
    };

    if (!view) {
        return (
            <div className="min-h-screen flex flex-col items-center justify-center
                            bg-[radial-gradient(ellipse_at_center,#0c2a1f_0%,#0a0a0a_72%)]">
                <h1 className="text-2xl font-bold mb-2">Play with AI</h1>
                {error
                    ? (
                        <>
                            <p className="text-rose-400">{error}</p>
                            {/* Without a retry the only escape from a failed first
                                deal (backend cold/degraded/transient) is leaving --
                                the kind of dead-end a first-time visitor bounces off. */}
                            <button onClick={startSession} disabled={busy}
                                className="mt-4 px-5 py-2 rounded-lg bg-neutral-800
                                           text-neutral-200 hover:bg-neutral-700
                                           disabled:opacity-50 text-sm font-medium">
                                Try again
                            </button>
                        </>
                    )
                    : <p className="text-neutral-400">Dealing…</p>}
                <Link to="/" className="mt-4 text-sm text-amber-400 hover:text-amber-300">
                    ← Home
                </Link>
            </div>
        );
    }

    const handOver = view.status === 'hand_over';
    const yourTurn = view.toAct === 'you';

    // Confirmed leave (modal "Fold & leave"): fold when it's the human's turn, then
    // let the blocked navigation proceed to wherever the user was headed. If it's
    // the bot's turn we can't fold — the inactivity sweeper resolves it server-side.
    const confirmLeave = async () => {
        if (leaving) return;                    // guard a fast double-click
        setLeaving(true);
        try {
            if (view.toAct === 'you') await sendGameAction(view.sessionId, 'fold');
        } catch {
            // Ignore — the inactivity sweeper folds it server-side as a backstop.
        }
        if (blocker.state === 'blocked') blocker.proceed();
        else navigate('/');                     // fallback if the blocker isn't active
    };

    // Cancelled leave: stay in the hand and release the blocked navigation.
    const cancelLeave = () => {
        if (leaving) return;
        if (blocker.state === 'blocked') blocker.reset();
    };

    // Fold run-out: community cards remain undealt when the hand ended before the
    // river. They can be revealed (dimmed) into the empty board slots on request.
    const community = view.community || [];
    const canRevealRunout = handOver && view.fullBoard
        && view.fullBoard.length > community.length;
    const boardCards = (handOver && showRunout && view.fullBoard)
        ? view.fullBoard : community;
    // True once the fold run-out is revealed: the made-hand labels then show what
    // each player WOULD have had on the full board ("You would have:" etc.).
    const runoutShown = canRevealRunout && showRunout;

    // Each seat's most recent action this hand (the action log is per-hand), for
    // the on-table "last action" pills. Last entry per seat wins.
    const lastBySeat = {};
    for (const e of view.actionLog) lastBySeat[e.seat] = e;

    // Your lifetime P/L line ("You ±X BB (±Y BB/hand) · Z hands"), shown next to
    // the bot's record. Rendered in both the desktop top bar and the mobile stat
    // strip, so it's a function (fresh elements per call). netBB is already in BB.
    const youStat = () => {
        if (!lifetime) return null;
        const bb = Number(lifetime.netBB) || 0;       // coerce: a partial /api/me shape must not crash toLocaleString
        const hands = Number(lifetime.hands) || 0;
        const rate = hands ? bb / hands : 0;
        const cls = bb > 0 ? 'text-emerald-400' : bb < 0 ? 'text-rose-400' : 'text-neutral-400';
        const f = (v, dp) =>
            `${v > 0 ? '+' : ''}${v.toLocaleString(undefined, { maximumFractionDigits: dp })}`;
        return (
            <span className="text-xs tabular-nums text-neutral-400">
                You <span className={`font-semibold ${cls}`}>{f(bb, 0)} BB</span>
                <span className="text-neutral-600">
                    {' '}({f(rate, 2)} BB/hand) · {hands.toLocaleString()} hands
                </span>
            </span>
        );
    };

    return (
        <div className="min-h-screen bg-[radial-gradient(ellipse_at_center,#0c2a1f_0%,#0a0a0a_62%)]">
            {/* Full-bleed: the top bar and side columns reach the screen edges
                (Home top-left, Net/Debug top-right, actions far right) while the
                board stays centred in the flexible middle column. Tight padding on
                phones, roomier margins from the `sm:` breakpoint up. */}
            <div className="w-full px-3 py-4 sm:px-8 sm:py-7">
                {/* Top bar:
                    LEFT  — Home link, title, then Bot / You stat lines stacked.
                    RIGHT — `?` (intro), Debug (if available), Sign in with Google.
                */}
                <div className="flex items-start justify-between gap-3 mb-4 sm:mb-6">
                    <div className="flex flex-col gap-1 min-w-0">
                        <Link to="/" className="text-sm text-amber-400 hover:text-amber-300">
                            ← Home
                        </Link>
                        <h1 className="text-xl sm:text-2xl font-bold">Play with AI</h1>
                        {/* Bot record + your lifetime P/L, stacked under the title.
                            Hidden on phones (they don't fit beside the right-side
                            cluster); shown there in a full-width strip below the bar. */}
                        <span className="hidden sm:inline mt-1"><EvCounter compact /></span>
                        {lifetime && <span className="hidden sm:inline">{youStat()}</span>}
                    </div>
                    <div className="flex items-center gap-3 shrink-0">
                        <button onClick={() => setShowIntro(true)} title="What is this?"
                            className="text-xs w-6 h-6 rounded-full border border-neutral-700
                                       text-neutral-400 hover:text-neutral-200">?</button>
                        {debugAvailable && (
                            <button onClick={() => setShowDebug((v) => !v)}
                                className={'text-xs px-2 py-1 rounded-lg border transition-colors ' +
                                    (showDebug
                                        ? 'border-fuchsia-600 text-fuchsia-300 bg-fuchsia-950/40'
                                        : 'border-neutral-700 text-neutral-400 hover:text-neutral-200')}>
                                {showDebug ? 'Debug ✓' : 'Debug'}
                            </button>
                        )}
                        <AnnouncementsButton />
                        <GoogleSignInButton registered={account?.isRegistered}
                            handle={account?.handle} />
                    </div>
                </div>

                {/* Mobile-only: the bot's record + your lifetime P/L as a full-width
                    strip below the top bar (the top bar itself is too cramped on a
                    phone to fit them beside the sign-in cluster). */}
                <div className="sm:hidden flex flex-col gap-0.5 mb-4">
                    <EvCounter compact />
                    {youStat()}
                </div>

                <IntroModal open={showIntro} onClose={closeIntro} />
                <LoginPrompt open={showLogin} onClose={() => setShowLogin(false)} />
                <LeaveHandModal open={blocker.state === 'blocked'} busy={leaving}
                    onConfirm={confirmLeave} onCancel={cancelLeave} />

                {/* Body: left = read/debug/log, centre = table, right = actions.
                    The centre column flexes so the table sits in the middle of the
                    screen; equal side columns keep it visually centred. On mobile it
                    stacks: table, then actions, then the read/debug panel. */}
                <div className="grid grid-cols-1 lg:grid-cols-[18rem_minmax(0,1fr)_18rem]
                                gap-5 lg:gap-6 items-start">

                {/* LEFT: debug overlay (read + per-decision trace) + action log.
                    Everything spoiler-shaped (the bot's read of your range, the
                    per-decision info-set keys, the solver's internals) lives behind
                    the Debug toggle so a casual visitor sees only the felt and the
                    action log. The toggle itself only exists when the backend has
                    ALLIN_DEBUG_OVERLAY=1; in prod with the overlay off there is no
                    button and nothing to render here. */}
                <aside className="order-3 lg:order-1 flex flex-col gap-6">
                    {showDebug && (
                        <>
                            <BotRead read={view.botRead} final={handOver} />
                            <div>
                                <h4 className="text-xs uppercase tracking-wider text-neutral-500 mb-1">
                                    Bot debug
                                </h4>
                                <p className="text-[11px] text-neutral-600 mb-2">
                                    Info-set keys &amp; solver internals - reveals the bot&rsquo;s
                                    hand bucket (spoiler).
                                </p>
                                <BotDebug debug={view.botDebug} />
                            </div>
                        </>
                    )}
                    {view.actionLog.length > 0 && (
                        <div>
                            <h4 className="text-xs uppercase tracking-wider text-neutral-500 mb-2">
                                Action log
                            </h4>
                            <div className="text-sm text-neutral-400 space-y-0.5">
                                {view.actionLog.map((e, i) => (
                                    <div key={i}>
                                        <span className={e.seat === 'you'
                                            ? 'text-emerald-400' : 'text-amber-400'}>
                                            {e.seat}
                                        </span>
                                        {' '}· {e.street} · {logVerb(e.action, e.street)}
                                        {e.chips > 0 ? ` ${fmtBB(e.chips)} BB` : ''}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </aside>

                {/* CENTRE: felt table (fluid) + caption + result + error.
                    `containerType: inline-size` makes this column a query container so
                    the board sizes off the SPACE AVAILABLE here (cqw), not the raw
                    viewport - a wider screen widens this column and grows the board. */}
                <main className="order-1 lg:order-2 w-full flex flex-col items-center gap-3"
                    style={{ containerType: 'inline-size' }}>

                {/* Felt table. `--card-w` drives every card's size. It's the SMALLER
                    of a width budget (cqw - the centre column) and a height budget
                    (vh - three card rows must fit a short window), clamped to a sane
                    min/max so it grows with the screen but never gets comical or
                    overflows vertically. */}
                <div className="w-fit rounded-[1.75rem] sm:rounded-[2.25rem] ring-4 ring-amber-600/60
                                shadow-2xl shadow-black/60 px-3 py-3 sm:px-6 sm:py-4"
                    style={{ ...FELT, '--card-w': 'clamp(2.75rem, min(7.5cqw, 8.5vh), 5rem)' }}>
                    {/* Bot — the made hand ("???" until showdown) sits on the seat
                        line; the prominent pill below the cards carries the dynamic
                        last-action / "thinking" label. */}
                    <Seat name="Bot" stackChips={view.botStack}
                        active={view.toAct === 'bot'}
                        holding={runoutShown
                            ? `would have ${view.botFullHand}`
                            : (view.botHand || '???')} />
                    <div className="flex justify-center gap-2 mt-1.5">
                        {(view.botCards || [null, null]).map((c, i) => (
                            <PlayingCard key={i} card={c} hidden={!view.botCards} />
                        ))}
                    </div>
                    {/* Fixed height so the table never jumps between "no action yet",
                        an action pill, and the thinking indicator. */}
                    <div className="h-6 flex items-center justify-center mt-1.5">
                        {thinking
                            ? <span className="px-3 py-0.5 rounded-full bg-black/30 text-xs
                                               tracking-wide text-amber-100/90">
                                Bot is thinking<span className="font-semibold">{dots}</span>
                              </span>
                            : <LastActionPill entry={lastBySeat.bot} />}
                    </div>
                    <div className="h-6 flex items-center justify-center mt-1.5">
                        <BetChip chips={view.botBet} />
                    </div>

                    {/* Pot + board. Five fixed-size slots are always rendered, so the
                        table never resizes as the flop/turn/river land; empty slots
                        are dashed placeholders. On a fold the run-out fills the empty
                        slots dimmed (cards that never actually came out). */}
                    <div className="flex flex-col items-center gap-2 py-2
                                    border-y border-white/10">
                        <span className="px-4 py-1 rounded-full bg-black/40 text-sm
                                         text-amber-200 font-medium">
                            Pot {fmtBB(view.totalPot)} BB · {view.street}
                        </span>
                        <div className="flex justify-center gap-2 items-center">
                            {[0, 1, 2, 3, 4].map((i) => {
                                const card = boardCards[i];
                                if (!card) return (
                                    <EmptySlot key={i}
                                        onReveal={canRevealRunout && !showRunout
                                            ? () => setShowRunout(true) : undefined} />
                                );
                                // A revealed run-out card (beyond what was actually
                                // dealt) is dimmed so it reads as "would-have-come".
                                const isRunout = i >= community.length;
                                return (
                                    <div key={i}
                                        className={isRunout ? 'opacity-40 grayscale' : ''}>
                                        <PlayingCard card={card} />
                                    </div>
                                );
                            })}
                        </div>
                        {/* Hand / seat caption, right below the board */}
                        <p className="text-xs text-neutral-300/70 tracking-wide text-center">
                            Hand #{view.handNumber} · you are{' '}
                            {view.yourSeat === 'button' ? 'on the button (SB)' : 'in the big blind'}
                        </p>
                    </div>

                    {/* You — mirror of the bot: action pill nearest the board (above
                        your cards), your made hand on the seat line. */}
                    <div className="h-6 flex items-center justify-center mt-1.5">
                        <BetChip chips={view.yourBet} />
                    </div>
                    <div className="h-6 flex items-center justify-center mt-1.5">
                        <LastActionPill entry={lastBySeat.you} />
                    </div>
                    <div className="flex justify-center gap-2 mt-1.5">
                        {(view.yourCards || []).map((c, i) => <PlayingCard key={i} card={c} />)}
                    </div>
                    {/* mt-1.5 mirrors the bot's seat↔cards gap (the bot gets it from its
                        cards' mt-1.5; here the seat is last, so it needs its own). */}
                    <div className="mt-1.5">
                        <Seat name="You" stackChips={view.yourStack} active={yourTurn}
                            holding={runoutShown
                                ? `would have ${view.yourFullHand}`
                                : view.yourHand} />
                    </div>
                </div>

                {error && (
                    <div className="mt-4 flex items-center justify-center gap-3 text-sm">
                        <span className="text-rose-400">{error}</span>
                        <button onClick={resync} disabled={busy}
                            className="px-3 py-1 rounded-lg bg-neutral-800 text-neutral-200
                                       hover:bg-neutral-700 disabled:opacity-50">
                            Reconnect
                        </button>
                    </div>
                )}

                {/* Result */}
                {handOver && view.result && (
                    <div className={'mt-3 rounded-xl px-4 py-2 text-center ' +
                        (view.result.winner === 'you'
                            ? 'bg-emerald-900/60 border border-emerald-700'
                            : view.result.winner === 'bot'
                                ? 'bg-rose-900/50 border border-rose-800'
                                : 'bg-neutral-800 border border-neutral-700')}>
                        <span className="font-semibold">
                            {view.result.winner === 'you' ? 'You win'
                                : view.result.winner === 'bot' ? 'Bot wins'
                                    : 'Split pot'}
                        </span>
                        <span className="text-neutral-300">
                            {' '}({view.result.reason}) ·{' '}
                            {fmtBBSigned(view.result.humanDelta)} BB
                        </span>
                    </div>
                )}

                </main>

                {/* RIGHT: actions, ordered by bet size, custom bet at the end */}
                <aside className="order-2 lg:order-3 flex flex-col gap-2 sm:gap-2.5">
                    <h4 className="text-xs uppercase tracking-wider text-neutral-500">
                        {yourTurn ? 'Your action' : handOver ? 'Hand over' : 'Bot to act'}
                    </h4>

                    {handOver && (
                        <button onClick={dealNext} disabled={busy}
                            className="w-full px-5 py-3 rounded-xl font-semibold bg-amber-500
                                       text-neutral-950 hover:bg-amber-400
                                       disabled:opacity-50 transition-colors">
                            Next hand
                        </button>
                    )}

                    {!handOver && !yourTurn && (
                        <div className="text-neutral-500 text-sm py-2">
                            Bot is acting…
                            {/* With no request in flight, "bot to act" is a STUCK
                                state -- the drive loop normally only stops on your
                                turn or hand over. It happens if the loop's guard
                                cap trips or a response was lost. Continue DRIVES
                                the pending bot turn(s) (a plain state re-fetch
                                would return the same stuck state). */}
                            {!busy && !thinking && (
                                <button
                                    onClick={() => run(() => sendBotAction(view.sessionId))}
                                    className="ml-2 px-2 py-0.5 rounded bg-neutral-800
                                               text-neutral-300 hover:bg-neutral-700
                                               text-xs align-middle">
                                    Continue
                                </button>
                            )}
                        </div>
                    )}

                    {/* Action grid: 3 columns on mobile/tablet (the aside is full
                        width there, so 2 wasted horizontal space), back to 2 at lg
                        where the aside narrows to 18rem. Shorter/tighter buttons on
                        mobile so the full set fits with less scrolling. */}
                    {!handOver && yourTurn && (
                        <div className="grid grid-cols-3 lg:grid-cols-2 gap-1.5 sm:gap-2">
                            {sortedActions(view.legalActions).map((la) => (
                                <button key={la.action} onClick={() => doAction(la.action)}
                                    disabled={busy}
                                    className={'rounded-lg px-1.5 sm:px-2 min-h-[2.5rem] sm:min-h-[3.25rem] ' +
                                        'flex flex-col items-center justify-center text-center leading-tight ' +
                                        'text-white disabled:opacity-50 transition-colors ' +
                                        actionClasses(la.action)}>
                                    {/* Fixed min-height + vertical centering keeps every button
                                        the same size and centres its label whether or not it has
                                        a bet-amount line (Bet/Raise vs Fold/Check). */}
                                    <span className="text-xs sm:text-sm font-semibold">
                                        {actionVerb(la.action, view.street)}
                                    </span>
                                    {la.chips > 0 && (
                                        <span className="text-[10px] sm:text-[11px] opacity-80 tabular-nums">
                                            {fmtBB(la.chips)} BB
                                        </span>
                                    )}
                                </button>
                            ))}
                        </div>
                    )}

                    {/* Custom (unrestricted) bet/raise - at the end of the action list.
                        Tighter padding/heights on mobile to match the compact buttons. */}
                    {!handOver && yourTurn && view.customBounds && (
                        <div className="mt-1 rounded-xl border border-neutral-800 p-2.5 sm:p-3">
                            <div className="text-xs text-neutral-400 mb-1.5">
                                {(facingBet || view.street === 'preflop')
                                    ? 'Raise to a custom size' : 'Bet a custom size'}
                            </div>
                            <div className="flex items-center gap-2">
                                <input
                                    type="number" inputMode="decimal"
                                    min={view.customBounds.minBb}
                                    max={view.customBounds.maxBb}
                                    step="0.5"
                                    value={customAmt}
                                    onChange={(e) => setCustomAmt(e.target.value)}
                                    onBlur={clampCustom}
                                    onKeyDown={(e) => { if (e.key === 'Enter') doCustom(); }}
                                    placeholder={`${view.customBounds.minBb}-${view.customBounds.maxBb}`}
                                    className="w-full px-3 py-1.5 sm:py-2 rounded-lg bg-neutral-800 text-white text-sm
                                               border border-neutral-700 focus:border-amber-500
                                               outline-none tabular-nums" />
                                <span className="text-sm text-neutral-400">BB</span>
                            </div>
                            <button onClick={doCustom} disabled={busy || !customValid}
                                className="mt-1.5 sm:mt-2 w-full px-4 py-1.5 sm:py-2 rounded-lg text-white text-sm font-semibold
                                           bg-sky-700 hover:bg-sky-600 disabled:opacity-40
                                           transition-colors">
                                Confirm bet
                            </button>
                        </div>
                    )}
                </aside>

                </div>{/* end body grid */}
            </div>
        </div>
    );
}

export default AiGame;
