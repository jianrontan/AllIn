// frontend/src/pages/AiGame.jsx
// Play full heads-up hands against the blueprint bot under the abstracted
// rules it was trained on. The backend is authoritative; this page renders
// the redacted state it returns and posts the player's actions.
//
// All amounts are shown in big blinds (the backend works in chips; 1 BB = 2).
import React, { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { newGame, sendGameAction, sendBotAction, nextHand, getGameState } from '../api';
import { fmtBB, fmtBBSigned } from '../format';
import PlayingCard from '../components/PlayingCard';

const PLAYER_ID_KEY = 'allin_player_id';

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
// xlarge — the 4th preflop open — after large too).
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
// raise over the big blind — the backend models it as `bet_*`, but we show it
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
// start — the table never resizes as the flop/turn/river land, and empty slots
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

function Seat({ name, stackChips, active }) {
    return (
        <div className="flex items-center justify-center gap-2 text-sm">
            <span className={'inline-block w-2 h-2 rounded-full ' +
                (active ? 'bg-amber-400 shadow-[0_0_8px] shadow-amber-400' : 'bg-transparent')} />
            <span className="text-neutral-200 font-medium">{name}</span>
            <span className="text-neutral-400">· {fmtBB(stackChips)} BB stack</span>
        </div>
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
// blueprint model the bot updates against — it drops if you play unexpectedly.
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
    if (!rows || rows.length === 0) return <div className="text-neutral-700">—</div>;
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
// there, the action chosen, and — on the river — the subgame solver's solved
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
        if (s.mode === 'river_solver')
            return s.deviated
                ? ['solver · deviated', 'bg-fuchsia-900/60 text-fuchsia-200']
                : ['solver · kept BP', 'bg-sky-900/60 text-sky-200'];
        if (s.mode === 'allin_guard') return ['all-in guard', 'bg-amber-900/60 text-amber-200'];
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
                        <div className="text-neutral-600 mb-0.5">blueprint strategy</div>
                        <DistBars rows={dist(r.strategy)} />
                        {s && s.mode === 'river_solver' && (
                            <>
                                <div className="text-neutral-600 mt-2 mb-0.5">solved (river subgame)</div>
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

    const startSession = async () => {
        setBusy(true);
        setError(null);
        try {
            const playerId = localStorage.getItem(PLAYER_ID_KEY) || undefined;
            const v = await newGame(playerId);
            if (v.playerId) localStorage.setItem(PLAYER_ID_KEY, v.playerId);
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
            // — i.e. you see the river, then the bot thinks, never the reverse.
            let guard = 0;
            while (v && v.status === 'in_hand' && v.toAct === 'bot' && guard < 8) {
                setThinking(true);
                v = await sendBotAction(v.sessionId);
                setView(v);
                guard += 1;
            }
            ok = true;
        } catch (e) {
            setError(e.message);
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
    const facingBet = !!view && view.legalActions.some((la) => la.action === 'call');
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
                <h1 className="text-2xl font-bold mb-2">Play With AI</h1>
                {error
                    ? <p className="text-rose-400">{error}</p>
                    : <p className="text-neutral-400">Dealing…</p>}
                <Link to="/" className="mt-4 text-sm text-amber-400 hover:text-amber-300">
                    ← Home
                </Link>
            </div>
        );
    }

    const handOver = view.status === 'hand_over';
    const yourTurn = view.toAct === 'you';
    const net = view.humanNet;
    // Fold run-out: community cards remain undealt when the hand ended before the
    // river. They can be revealed (dimmed) into the empty board slots on request.
    const canRevealRunout = handOver && view.fullBoard
        && view.fullBoard.length > view.community.length;
    const boardCards = (handOver && showRunout && view.fullBoard)
        ? view.fullBoard : view.community;
    // True once the fold run-out is revealed: the made-hand labels then show what
    // each player WOULD have had on the full board ("You would have:" etc.).
    const runoutShown = canRevealRunout && showRunout;

    return (
        <div className="min-h-screen bg-[radial-gradient(ellipse_at_center,#0c2a1f_0%,#0a0a0a_62%)]">
            {/* Full-bleed: the top bar and side columns reach the screen edges
                (Home top-left, Net/Debug top-right, actions far right) while the
                board stays centred in the flexible middle column. */}
            <div className="w-full px-8 py-7 sm:px-10">
                {/* Top bar: Home + title top-left, debug toggle + net top-right */}
                <div className="flex items-start justify-between gap-4 mb-6">
                    <div>
                        <Link to="/" className="text-sm text-amber-400 hover:text-amber-300">
                            ← Home
                        </Link>
                        <h1 className="mt-1 text-2xl font-bold">Play With AI</h1>
                    </div>
                    <div className="flex items-center gap-3 shrink-0">
                        <button onClick={() => setShowDebug((v) => !v)}
                            className={'text-xs px-2 py-1 rounded-lg border transition-colors ' +
                                (showDebug
                                    ? 'border-fuchsia-600 text-fuchsia-300 bg-fuchsia-950/40'
                                    : 'border-neutral-700 text-neutral-400 hover:text-neutral-200')}>
                            {showDebug ? 'Debug ✓' : 'Debug'}
                        </button>
                        <span className={'text-sm font-semibold tabular-nums ' +
                            (net > 0 ? 'text-emerald-400'
                                : net < 0 ? 'text-rose-400' : 'text-neutral-400')}>
                            Net&nbsp;&nbsp;{fmtBBSigned(net)} BB
                        </span>
                    </div>
                </div>

                {/* Body: left = read/debug/log, centre = table, right = actions.
                    The centre column flexes so the table sits in the middle of the
                    screen; equal side columns keep it visually centred. On mobile it
                    stacks: table, then actions, then the read/debug panel. */}
                <div className="grid grid-cols-1 lg:grid-cols-[18rem_minmax(0,1fr)_18rem]
                                gap-6 items-start">

                {/* LEFT: bot read + debug overlay + action log */}
                <aside className="order-3 lg:order-1 flex flex-col gap-6">
                    {showDebug && (
                        <div>
                            <h4 className="text-xs uppercase tracking-wider text-neutral-500 mb-1">
                                Bot debug
                            </h4>
                            <p className="text-[11px] text-neutral-600 mb-2">
                                Info-set keys &amp; solver internals — reveals the bot&rsquo;s
                                hand bucket (spoiler).
                            </p>
                            <BotDebug debug={view.botDebug} />
                        </div>
                    )}
                    <BotRead read={view.botRead} final={handOver} />
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
                    viewport — a wider screen widens this column and grows the board. */}
                <main className="order-1 lg:order-2 w-full flex flex-col items-center gap-3"
                    style={{ containerType: 'inline-size' }}>

                {/* Felt table. `--card-w` drives every card's size. It's the SMALLER
                    of a width budget (cqw — the centre column) and a height budget
                    (vh — three card rows must fit a short window), clamped to a sane
                    min/max so it grows with the screen but never gets comical or
                    overflows vertically. */}
                <div className="w-fit rounded-[2.25rem] ring-4 ring-amber-600/60
                                shadow-2xl shadow-black/60 px-6 py-4"
                    style={{ ...FELT, '--card-w': 'clamp(2.5rem, min(7.5cqw, 8.5vh), 5rem)' }}>
                    {/* Bot */}
                    <Seat name="Bot" stackChips={view.botStack}
                        active={view.toAct === 'bot'} />
                    <div className="flex justify-center gap-2 mt-1.5">
                        {(view.botCards || [null, null]).map((c, i) => (
                            <PlayingCard key={i} card={c} hidden={!view.botCards} />
                        ))}
                    </div>
                    {/* Always shown so the table height doesn't jump; "???" until
                        the bot's cards are revealed at showdown. */}
                    <div className="flex justify-center mt-1.5">
                        <span className="px-3 py-0.5 rounded-full bg-black/30 text-xs
                                         tracking-wide text-amber-100/90 tabular-nums">
                            {thinking
                                ? <>Bot is thinking<span className="font-semibold">{dots}</span></>
                                : runoutShown
                                    ? <>Bot would have: <span className="font-semibold">{view.botFullHand}</span></>
                                    : <>Bot has: <span className="font-semibold">{view.botHand || '???'}</span></>}
                        </span>
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
                                const isRunout = i >= view.community.length;
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

                    {/* You */}
                    <div className="h-6 flex items-center justify-center mt-1.5">
                        <BetChip chips={view.yourBet} />
                    </div>
                    <div className="flex justify-center gap-2 mt-1.5">
                        {view.yourCards.map((c, i) => <PlayingCard key={i} card={c} />)}
                    </div>
                    <div className="flex justify-center my-1.5">
                        <span className="px-3 py-0.5 rounded-full bg-black/30 text-xs
                                         tracking-wide text-emerald-100/90">
                            {runoutShown ? 'You would have: ' : 'You have: '}
                            <span className="font-semibold">
                                {runoutShown ? view.yourFullHand : view.yourHand}
                            </span>
                        </span>
                    </div>
                    <Seat name="You" stackChips={view.yourStack} active={yourTurn} />
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
                <aside className="order-2 lg:order-3 flex flex-col gap-2.5">
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
                        <div className="text-neutral-500 text-sm py-2">Bot is acting…</div>
                    )}

                    {/* Compact action grid, ascending by bet size (two per row). */}
                    {!handOver && yourTurn && (
                        <div className="grid grid-cols-2 gap-2">
                            {sortedActions(view.legalActions).map((la) => (
                                <button key={la.action} onClick={() => doAction(la.action)}
                                    disabled={busy}
                                    className={'rounded-lg px-3 py-2 text-center leading-tight ' +
                                        'text-white disabled:opacity-50 transition-colors ' +
                                        actionClasses(la.action)}>
                                    <span className="block text-sm font-semibold">
                                        {actionVerb(la.action, view.street)}
                                    </span>
                                    {la.chips > 0 && (
                                        <span className="block text-[11px] opacity-80 tabular-nums">
                                            {fmtBB(la.chips)} BB
                                        </span>
                                    )}
                                </button>
                            ))}
                        </div>
                    )}

                    {/* Custom (unrestricted) bet/raise — at the end of the action list */}
                    {!handOver && yourTurn && view.customBounds && (
                        <div className="mt-1 rounded-xl border border-neutral-800 p-3">
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
                                    placeholder={`${view.customBounds.minBb}–${view.customBounds.maxBb}`}
                                    className="w-full px-3 py-2 rounded-lg bg-neutral-800 text-white text-sm
                                               border border-neutral-700 focus:border-amber-500
                                               outline-none tabular-nums" />
                                <span className="text-sm text-neutral-400">BB</span>
                            </div>
                            <button onClick={doCustom} disabled={busy || !customValid}
                                className="mt-2 w-full px-4 py-2 rounded-lg text-white text-sm font-semibold
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
