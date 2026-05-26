// frontend/src/pages/AiGame.jsx
// Play full heads-up hands against the blueprint bot under the abstracted
// rules it was trained on. The backend is authoritative; this page renders
// the redacted state it returns and posts the player's actions.
//
// All amounts are shown in big blinds (the backend works in chips; 1 BB = 2).
import React, { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { newGame, sendGameAction, nextHand } from '../api';
import { fmtBB, fmtBBSigned } from '../format';
import PlayingCard from '../components/PlayingCard';

const PLAYER_ID_KEY = 'allin_player_id';

// Abstract action -> button verb. The BB amount is shown separately, below
// the verb, so the player always sees exactly how much is being committed.
const ACTION_VERB = {
    fold: 'Fold', check: 'Check', call: 'Call', allin: 'All-in',
    bet_small: 'Bet small', bet_medium: 'Bet medium', bet_large: 'Bet large',
    raise_small: 'Raise small', raise_medium: 'Raise medium', raise_large: 'Raise large',
};

// Canonical display order. The backend's stack-constraint logic inserts
// 'allin' wherever it lands in its list; this pins a stable order with
// all-in always last, bets/raises ascending by size.
const ACTION_ORDER = {
    fold: 0, check: 1, call: 2,
    bet_small: 3, bet_medium: 4, bet_large: 5,
    raise_small: 6, raise_medium: 7, raise_large: 8,
    allin: 9,
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

const actionClasses = (a) => {
    if (a === 'fold') return 'bg-rose-700 hover:bg-rose-600';
    if (a === 'call' || a === 'check') return 'bg-sky-700 hover:bg-sky-600';
    if (a === 'allin') return 'bg-violet-700 hover:bg-violet-600';
    return 'bg-emerald-700 hover:bg-emerald-600';
};

const FELT = {
    background: 'radial-gradient(ellipse at center, #11815a 0%, #064534 78%)',
};

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
function BotRead({ read }) {
    if (!read || !read.topHands || read.topHands.length === 0) return null;
    const confPct = Math.round(read.confidence * 100);
    return (
        <div>
            <h4 className="text-xs uppercase tracking-wider text-neutral-500 mb-2">
                Bot&rsquo;s read of your hand
            </h4>
            <div className="text-sm text-neutral-400 mb-2">
                Read confidence:{' '}
                <span className="text-neutral-200 font-medium tabular-nums">{confPct}%</span>
                <span className="text-neutral-600">
                    {' '}· top hands it puts you on
                </span>
            </div>
            <div className="flex flex-wrap gap-1.5">
                {read.topHands.map((h, i) => (
                    <span key={i} className="px-2 py-1 rounded bg-black/30 text-xs
                                             tabular-nums text-neutral-300">
                        {h.cards.join(' ')}{' '}
                        <span className="text-neutral-500">{(h.prob * 100).toFixed(1)}%</span>
                    </span>
                ))}
            </div>
        </div>
    );
}

function AiGame() {
    const [view, setView] = useState(null);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState(null);
    const [customAmt, setCustomAmt] = useState('');
    // Guards against React StrictMode invoking the mount effect twice in dev,
    // which would otherwise deal (and orphan) a second game session.
    const sessionStarted = useRef(false);

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

    const run = async (fn) => {
        setBusy(true);
        setError(null);
        try {
            setView(await fn());
        } catch (e) {
            setError(e.message);
        } finally {
            setBusy(false);
        }
    };

    const doAction = (action) => run(() => sendGameAction(view.sessionId, action));
    const dealNext = () => run(() => nextHand(view.sessionId));

    // Unrestricted custom bet/raise: it's a raise when there's a bet to call,
    // otherwise a bet. amountBb is the raise-to TOTAL in big blinds.
    const facingBet = !!view && view.legalActions.some((la) => la.action === 'call');
    const customAmtNum = parseFloat(customAmt);
    const customValid = !!(view && view.customBounds) && !isNaN(customAmtNum)
        && customAmtNum >= view.customBounds.minBb
        && customAmtNum <= view.customBounds.maxBb;
    const doCustom = () => {
        if (!customValid) return;
        const action = facingBet ? 'raise_custom' : 'bet_custom';
        setCustomAmt('');
        run(() => sendGameAction(view.sessionId, action, { amountBb: customAmtNum }));
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

    return (
        <div className="min-h-screen flex justify-center
                        bg-[radial-gradient(ellipse_at_top,#0c2a1f_0%,#0a0a0a_62%)]">
            <div className="w-full max-w-5xl px-6 py-5">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <Link to="/" className="text-sm text-amber-400 hover:text-amber-300">
                        ← Home
                    </Link>
                    <span className={'text-sm font-semibold tabular-nums ' +
                        (net > 0 ? 'text-emerald-400'
                            : net < 0 ? 'text-rose-400' : 'text-neutral-400')}>
                        Net&nbsp;&nbsp;{fmtBBSigned(net)} BB
                    </span>
                </div>

                <h1 className="mt-2 text-2xl font-bold">Play With AI</h1>
                <p className="text-neutral-500 text-sm mb-4">
                    Hand #{view.handNumber} · you are{' '}
                    {view.yourSeat === 'button' ? 'on the button (SB)' : 'in the big blind'}
                </p>

                {/* Two columns on desktop: table + actions on the left, the
                    bot's read and the action log in a sidebar so they don't push
                    the table off-screen. Single column (stacked) on mobile. */}
                <div className="flex flex-col lg:flex-row lg:items-start gap-6">
                <div className="w-full lg:flex-1 lg:max-w-xl">

                {/* Felt table */}
                <div className="rounded-[2.25rem] ring-4 ring-amber-600/60
                                shadow-2xl shadow-black/60 px-6 py-4" style={FELT}>
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
                                         tracking-wide text-amber-100/90">
                            Bot has: <span className="font-semibold">{view.botHand || '???'}</span>
                        </span>
                    </div>
                    <div className="h-6 flex items-center justify-center mt-1.5">
                        <BetChip chips={view.botBet} />
                    </div>

                    {/* Pot + board */}
                    <div className="flex flex-col items-center gap-2 py-2
                                    border-y border-white/10">
                        <span className="px-4 py-1 rounded-full bg-black/40 text-sm
                                         text-amber-200 font-medium">
                            Pot {fmtBB(view.totalPot)} BB · {view.street}
                        </span>
                        <div className="flex justify-center gap-2 min-h-[4.25rem] items-center">
                            {view.community.length === 0
                                ? <span className="text-emerald-200/50 text-sm">
                                    no board yet
                                  </span>
                                : view.community.map((c, i) => (
                                    <PlayingCard key={i} card={c} />
                                ))}
                        </div>
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
                            You have: <span className="font-semibold">{view.yourHand}</span>
                        </span>
                    </div>
                    <Seat name="You" stackChips={view.yourStack} active={yourTurn} />
                </div>

                {error && <p className="text-rose-400 text-sm mt-4">{error}</p>}

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

                {/* Actions */}
                <div className="mt-3 flex flex-wrap gap-2.5">
                    {handOver && (
                        <button onClick={dealNext} disabled={busy}
                            className="px-7 py-2.5 rounded-xl font-semibold bg-amber-500
                                       text-neutral-950 hover:bg-amber-400
                                       disabled:opacity-50 transition-colors">
                            Next hand
                        </button>
                    )}
                    {!handOver && yourTurn && sortedActions(view.legalActions).map((la) => (
                        <button key={la.action} onClick={() => doAction(la.action)}
                            disabled={busy}
                            className={'px-5 py-2 rounded-xl text-white text-center ' +
                                'min-w-[5.5rem] disabled:opacity-50 transition-colors ' +
                                actionClasses(la.action)}>
                            <span className="block font-semibold text-sm leading-tight">
                                {actionVerb(la.action, view.street)}
                            </span>
                            {la.chips > 0 && (
                                <span className="block text-xs opacity-80 tabular-nums">
                                    {fmtBB(la.chips)} BB
                                </span>
                            )}
                        </button>
                    ))}
                    {!handOver && !yourTurn && (
                        <span className="text-neutral-500 text-sm py-3">
                            Bot is acting…
                        </span>
                    )}
                </div>

                {/* Custom (unrestricted) bet/raise box */}
                {!handOver && yourTurn && view.customBounds && (
                    <div className="mt-3 flex items-center gap-2">
                        <span className="text-sm text-neutral-400">
                            {(facingBet || view.street === 'preflop') ? 'Raise to' : 'Bet'}
                        </span>
                        <input
                            type="number" inputMode="decimal"
                            min={view.customBounds.minBb}
                            max={view.customBounds.maxBb}
                            step="0.5"
                            value={customAmt}
                            onChange={(e) => setCustomAmt(e.target.value)}
                            onKeyDown={(e) => { if (e.key === 'Enter') doCustom(); }}
                            placeholder={`${view.customBounds.minBb}–${view.customBounds.maxBb}`}
                            className="w-24 px-3 py-2 rounded-xl bg-neutral-800 text-white text-sm
                                       border border-neutral-700 focus:border-amber-500
                                       outline-none tabular-nums" />
                        <span className="text-sm text-neutral-400">BB</span>
                        <button onClick={doCustom} disabled={busy || !customValid}
                            className="px-5 py-2 rounded-xl text-white text-sm font-semibold
                                       bg-sky-700 hover:bg-sky-600 disabled:opacity-40
                                       transition-colors">
                            Confirm
                        </button>
                    </div>
                )}

                </div>{/* end main column */}

                {/* Sidebar: bot read + action log */}
                <aside className="w-full lg:w-80 lg:flex-shrink-0 flex flex-col gap-7">

                {/* Bot's read of your hand (range tracker) */}
                {!handOver && <BotRead read={view.botRead} />}

                {/* Action log */}
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
                                    {' '}· {e.street} · {actionVerb(e.action, e.street)}
                                    {e.chips > 0 ? ` ${fmtBB(e.chips)} BB` : ''}
                                </div>
                            ))}
                        </div>
                    </div>
                )}
                </aside>{/* end sidebar */}
                </div>{/* end two-column row */}
            </div>
        </div>
    );
}

export default AiGame;
