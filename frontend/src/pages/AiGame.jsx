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

const actionVerb = (action) => ACTION_VERB[action] || action;

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

function AiGame() {
    const [view, setView] = useState(null);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState(null);
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
            <div className="w-full max-w-xl px-6 py-8">
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
                <p className="text-neutral-500 text-sm mb-5">
                    Hand #{view.handNumber} · you are{' '}
                    {view.yourSeat === 'button' ? 'on the button (SB)' : 'in the big blind'}
                </p>

                {/* Felt table */}
                <div className="rounded-[2.25rem] ring-4 ring-amber-600/60
                                shadow-2xl shadow-black/60 px-6 py-7" style={FELT}>
                    {/* Bot */}
                    <Seat name="Bot" stackChips={view.botStack}
                        active={view.toAct === 'bot'} />
                    <div className="flex justify-center gap-2 mt-2">
                        {(view.botCards || [null, null]).map((c, i) => (
                            <PlayingCard key={i} card={c} hidden={!view.botCards} />
                        ))}
                    </div>
                    <div className="h-7 flex items-center justify-center mt-2">
                        <BetChip chips={view.botBet} />
                    </div>

                    {/* Pot + board */}
                    <div className="flex flex-col items-center gap-3 py-3
                                    border-y border-white/10">
                        <span className="px-4 py-1 rounded-full bg-black/40 text-sm
                                         text-amber-200 font-medium">
                            Pot {fmtBB(view.pot)} BB · {view.street}
                        </span>
                        <div className="flex justify-center gap-2 min-h-[5rem] items-center">
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
                    <div className="h-7 flex items-center justify-center mb-2">
                        <BetChip chips={view.yourBet} />
                    </div>
                    <div className="flex justify-center gap-2 mb-2">
                        {view.yourCards.map((c, i) => <PlayingCard key={i} card={c} />)}
                    </div>
                    <Seat name="You" stackChips={view.yourStack} active={yourTurn} />
                </div>

                {error && <p className="text-rose-400 text-sm mt-4">{error}</p>}

                {/* Result */}
                {handOver && view.result && (
                    <div className={'mt-4 rounded-xl px-4 py-3 text-center ' +
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
                <div className="mt-5 flex flex-wrap gap-2.5">
                    {handOver && (
                        <button onClick={dealNext} disabled={busy}
                            className="px-7 py-3 rounded-xl font-semibold bg-amber-500
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
                                {actionVerb(la.action)}
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

                {/* Action log */}
                {view.actionLog.length > 0 && (
                    <div className="mt-7">
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
                                    {' '}· {e.street} · {actionVerb(e.action)}
                                    {e.chips > 0 ? ` ${fmtBB(e.chips)} BB` : ''}
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default AiGame;
