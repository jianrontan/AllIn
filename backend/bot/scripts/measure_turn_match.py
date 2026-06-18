# backend/bot/scripts/measure_turn_match.py
"""
Stage-2 N0' real-game gate -- does the CONSISTENCY-FIXED turn stack win chips vs the
blueprint, and does it beat the RIVER-ONLY stack (the +1801 mbb baseline)?

Plays one BOT strategy (turn solver / river solver / blueprint) against a BLUEPRINT
opponent through the real GameSession, alternating seats each hand, and reports the
bot's chip win rate (mbb/hand), raw and AIVAT-corrected.

CRITICAL (the whole point of the revival): the hand is driven through `advance_bot_turns`,
exactly like serving -- so the bot gets its live trackers, turn/river-entry fields, AND the
continual-re-solving chaining (1a own-range / 1b CFV / 1c exact-leaf gate). The OLD version
of this script drove the loop with a bare `apply_action(a)`, which bypassed advance_bot_turns
and therefore measured the turn solver WITHOUT the consistency fixes -- a silent false signal.
If you change the play loop, keep it on advance_bot_turns.

Compare runs:
  * blueprint vs blueprint   -> sanity, should be ~0
  * river  vs blueprint      -> the CURRENTLY-deployed stack's edge (the +1801 baseline)
  * turn   vs blueprint      -> the consistency-fixed turn-solver stack's edge
The (turn - river) gap is the turn solver's marginal value over today's deployment. The
Stage-2 GATE: turn must beat river (not merely beat the blueprint) -- if it does not WITH the
consistency fixes live, the M-pre thin-band explanation wins and the turn revival stops.

--aivat reports the c1 (preflop-equity) + c2 (river-runout) + c3 (all-in-EV) corrected mbb/hand
alongside raw. c2 uses the bot's OWN on-model river-entry belief about the opponent (river_entry_opp)
as B's range -- valid here because the opponent IS the blueprint that belief models (the maniac path
in compare_gadget_policies skips c2 because a maniac is off-model). Power budget: ~5-10k hands (the 250-hand
+/-1363 mbb run was hopeless). Slow (live solves); run in the background, ONE FREE CORE while
training -- do not parallelize into the training cores.

Run from backend/bot/:
    python scripts/measure_turn_match.py --bot turn  --hands 6000 --aivat --n-buckets 20 --leaf-rivers 4 --turn-iters 80
    python scripts/measure_turn_match.py --bot river --hands 6000 --aivat
    python scripts/measure_turn_match.py --bot blueprint --hands 6000 --aivat
"""
import argparse
import math
import os
import random
import sys
import time

import numpy as np

_BOT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BOT_ROOT)
sys.path.insert(0, os.path.join(_BOT_ROOT, 'tests'))   # for run_maniac_live.human_action

from run_maniac_live import human_action               # aggressive (maxbet/jam) opponent styles
from src.config import resolve_blueprint_path
from src.storage.blueprint_db import BlueprintDB, FrozenBlueprint
from src.abstractions.sizing import db_menu_mode, BIG_BLIND
from src.cfr.poker_game import STARTING_STACK
from src.game.game_session import GameSession, advance_bot_turns
from src.game.bot_strategy import BlueprintStrategy
from src.game.range_tracker import RangeTracker
from src.subgame.river_subgame_solver import RiverSubgameSolver
from src.subgame.turn_subgame_solver import TurnSubgameSolver


def _build_bot(kind, rawdb, fb, args):
    if kind == 'blueprint':
        return BlueprintStrategy(fb)
    # PIN the policy to the DEPLOYED serving config (backend/api/strategy_api.py:114) so the
    # gate verdict transfers to production: the RiverSubgameSolver class default is the
    # UNSAFE-v1 direct solve (safe_gadget=False, gadget_anchor='belief', purify=0) -- safety
    # is opt-IN and serving turns it on. Measuring the default would grade a stack we don't
    # ship. iters/budget are CLI-tunable but DEFAULT to the served 200/5.0. Seed the solver's
    # own RNG (else the bot's sampled actions are non-reproducible across runs).
    served = dict(check_every=40, safe_gadget=True, gadget_anchor='auto',
                  purify_threshold=0.01, max_iters=args.river_iters,
                  time_budget=args.river_budget, rng=np.random.default_rng(args.seed))
    if kind == 'river':
        s = RiverSubgameSolver(rawdb, **served)
    elif kind == 'turn':
        s = TurnSubgameSolver(rawdb, n_buckets=args.n_buckets, leaf_rivers=args.leaf_rivers,
                              turn_max_iters=args.turn_iters, turn_time_budget=args.turn_budget,
                              max_spr_turn=args.turn_max_spr,
                              multivalued_leaf=getattr(args, 'multivalued_leaf', False), **served)
    else:
        raise ValueError(kind)
    s.db = fb                       # cache blueprint reads across hands (huge speedup)
    return s


def _play_and_record(session, bot, opponent, bot_seat, max_steps, want_aivat):
    """Drive one hand through advance_bot_turns (so the continual-re-solving chaining is
    live) with the OPPONENT seat playing the blueprint. If want_aivat, also build an AIVAT
    record (aivat.AIVATEstimator format) by wrapping apply_action to log each action's
    PRE-street + the folder seat -- the same proven shape as compare_gadget_policies.
    `result` (the cross-hand chip delta) is filled by the caller. Returns the record or None."""
    d = session.data
    log = {'streets': [], 'folder': None}
    orig = session.apply_action

    def wrapped(action, solved_hero_probs=None):
        if action == 'fold':
            log['folder'] = session.current_player()
        log['streets'].append(d['street'])
        return orig(action, solved_hero_probs=solved_hero_probs)

    if want_aivat:
        session.apply_action = wrapped
    try:
        steps = 0
        while d['status'] == 'in_hand' and steps < max_steps:
            advance_bot_turns(session, bot)
            if d['status'] != 'in_hand':
                break
            if session.is_human_turn():
                key = session.info_set_key(d['human_seat'])
                legal = session.legal_actions()
                session.apply_action(opponent.decide(key, legal, {}))
            steps += 1
    finally:
        if want_aivat:
            session.apply_action = orig

    if not want_aivat:
        return None

    res = d.get('result') or {}
    # TOTAL committed per seat = pre-final-street invested + this street's contribution
    # (the final street's bets live in `history`, not yet folded into p*_invested), via the
    # engine's own accounting (includes blinds, matches the final pot). raw p*_invested
    # understates it and misses every all-in -- this is what AIVAT's c3 needs.
    g = session.game
    st = min(d['street'], 3)
    p0_tot = d['p0_invested'] + g.get_player_contribution_this_round(
        d['history'], st, d['starting_pot'], 0, d['p0_invested'], d['p1_invested'])
    p1_tot = d['p1_invested'] + g.get_player_contribution_this_round(
        d['history'], st, d['starting_pot'], 1, d['p0_invested'], d['p1_invested'])
    showdown = res.get('reason') == 'showdown'
    both_allin = (abs(p0_tot - STARTING_STACK) < 1e-6 and abs(p1_tot - STARTING_STACK) < 1e-6)
    allin_street = log['streets'][-1] if (showdown and both_allin and log['streets']) else None
    a_cards = d['p0_cards'] if bot_seat == 0 else d['p1_cards']
    b_cards = d['p1_cards'] if bot_seat == 0 else d['p0_cards']
    # c2 (river-runout CV): if the hand reached river betting, the bot snapshotted its OWN
    # on-model belief about the opponent at river entry (river_entry_opp). Hand it to AIVAT
    # as B's turn-board range -- more accurate than replaying LBR's BotRange, and valid here
    # because the opponent IS the blueprint the tracker models. (No 'events' -> c2 from this.)
    river_range = None
    board = list(d['community'])
    if d.get('river_entry_opp') is not None and len(board) >= 5:
        opp = RangeTracker.from_dict(d['river_entry_opp'], session.cards)
        river_range = {'range': opp, 'turn_board': board[:4], 'river_card': board[4],
                       'pot': float(d['starting_pot'])}      # pot carried into the river
    return {
        'seat_of_A': bot_seat,
        'hand_a': list(a_cards),
        'hand_b': list(b_cards),
        'board': board,
        'events': [],                                  # c2 comes from river_range, not events
        'river_range': river_range,
        'allin_street': allin_street,
        'folded': log['folder'],
        'invested': [p0_tot, p1_tot],
        # 'result' filled in by the caller (the cross-hand human_net delta).
    }


def _turn_stats_report(bot, indent="  "):
    """#3 fire-rate / latency / per-deviation-EV for a TurnSubgameSolver (no-op otherwise).
    The vacuous-gate finding was invisible because the harness never tallied these."""
    s = getattr(bot, 'stats', None)
    if not s or not s.get('turn_calls'):
        return
    print(f"{indent}turn fire: solves(past SPR gate)={s['turn_calls']} "
          f"deviated={s.get('turn_deviated', 0)} kept-blueprint={s.get('turn_kept', 0)} "
          f"nonconverged/timeout={s.get('turn_timeout', 0)}")
    secs = getattr(bot, 'turn_solve_seconds', [])
    if secs:
        a = np.array(secs)
        print(f"{indent}turn solve wall-s: p50={np.percentile(a, 50):.1f} "
              f"p90={np.percentile(a, 90):.1f} max={a.max():.1f} "
              f"(budget={getattr(bot, 'turn_time_budget', 0):.0f}s)")
    evd = getattr(bot, 'turn_deviation_evs', [])
    if evd:
        print(f"{indent}turn deviation gate evDelta (chips): mean={np.mean(evd):+.1f} "
              f"min={np.min(evd):+.1f} max={np.max(evd):+.1f} n={len(evd)}")


def _opponent_action(style, opponent, sess):
    """The opponent's action. 'maxbet'/'jam'/etc -> aggressive pot-builder (run_maniac_live)
    so turns arrive at LOW SPR (small, fast-solving trees -> the turn solver actually fires);
    'blueprint' -> a passive blueprint villain (rarely builds the pot, so the turn ~never
    fires -- that was the vacuous-gate trap)."""
    if style != 'blueprint':
        return human_action(sess, style)
    return opponent.decide(sess.info_set_key(sess.data['human_seat']), sess.legal_actions(), {})


def _hand_chips(seed_h, human_seat, bot, opponent, opp_style, strat_fn, menu, max_steps,
                want_aivat=False):
    """#4 CRN: play ONE hand on a FRESH session seeded for pairing; return the bot's chips.
    PAIRING-LEAK FIX: deck/opponent/blueprint-sampling share the global `random` while the
    solver samples from `_rng`; once the turn bot fires and consumes `_rng` (not `random`),
    the opponent's later `random` draws DESYNC between arms -> spurious divergence (the 15-vs-3
    bug). Fix: reseed BOTH `random` and the bot's `_rng` to a DETERMINISTIC per-decision seed
    before every bot/opponent decision, so the two arms are BIT-IDENTICAL until a REAL turn
    deviation -- regardless of how much `_rng` a solve consumed internally. Non-deviation hands
    then diff exactly 0; only genuine turn deviations contribute. human_seat alternates."""
    random.seed(seed_h)
    sess = GameSession.new('crn', 'opp', strategy_fn=strat_fn, menu_mode=menu,
                           max_raises_per_street=float('inf'))
    if human_seat != 0:                       # re-deal the SAME deck with the seat swapped
        random.seed(seed_h)
        sess._deal_hand(hand_number=1, human_seat=human_seat)

    dcount = [0]
    orig_decide = bot.decide

    def seeded_decide(key, legal, public):    # per-decision deterministic RNG (arm-independent)
        s = seed_h * 1_000_003 + dcount[0]
        dcount[0] += 1
        random.seed(s)
        if getattr(bot, '_rng', None) is not None:
            bot._rng = np.random.default_rng(s)
        return orig_decide(key, legal, public)

    bot.decide = seeded_decide
    _log = {'streets': [], 'folder': None}
    orig_apply = sess.apply_action
    if want_aivat:                            # log streets/folder for the AIVAT record
        def _wrapped_apply(action, solved_hero_probs=None):
            if action == 'fold':
                _log['folder'] = sess.current_player()
            _log['streets'].append(sess.data['street'])
            return orig_apply(action, solved_hero_probs=solved_hero_probs)
        sess.apply_action = _wrapped_apply
    ocount = [0]
    try:
        steps = 0
        while sess.data['status'] == 'in_hand' and steps < max_steps:
            advance_bot_turns(sess, bot)
            if sess.data['status'] != 'in_hand':
                break
            if sess.is_human_turn():
                random.seed(seed_h * 7_000_003 + ocount[0])   # opponent RNG: own namespace
                ocount[0] += 1
                sess.apply_action(_opponent_action(opp_style, opponent, sess))
            steps += 1
    finally:
        bot.decide = orig_decide
        sess.apply_action = orig_apply
    bot_chips = -float(sess.data['human_net'])     # fresh session: bot's chips this hand
    if want_aivat:
        rec = _aivat_record(sess, 1 - human_seat, _log)
        rec['result'] = bot_chips
        return bot_chips, rec
    return bot_chips


def _aivat_record(session, bot_seat, log):
    """Assemble an AIVATEstimator record from a FINISHED session + the per-action `log`
    ({'streets':[...], 'folder': seat|None}). Same shape as _play_and_record's record, so
    the paired arms feed the estimator identically -- and non-diverged hands produce
    IDENTICAL records, so they cancel in the (turn - river) AIVAT diff (pairing preserved).
    'result' is filled by the caller (the hand's bot-chip delta)."""
    d = session.data
    res = d.get('result') or {}
    g = session.game
    st = min(d['street'], 3)
    p0_tot = d['p0_invested'] + g.get_player_contribution_this_round(
        d['history'], st, d['starting_pot'], 0, d['p0_invested'], d['p1_invested'])
    p1_tot = d['p1_invested'] + g.get_player_contribution_this_round(
        d['history'], st, d['starting_pot'], 1, d['p0_invested'], d['p1_invested'])
    showdown = res.get('reason') == 'showdown'
    both_allin = (abs(p0_tot - STARTING_STACK) < 1e-6 and abs(p1_tot - STARTING_STACK) < 1e-6)
    allin_street = log['streets'][-1] if (showdown and both_allin and log['streets']) else None
    a_cards = d['p0_cards'] if bot_seat == 0 else d['p1_cards']
    b_cards = d['p1_cards'] if bot_seat == 0 else d['p0_cards']
    # c2 (river-runout CV): mirror _play_and_record so the PAIRED arms get c2 too. Without
    # this the paired AIVAT was c3-only, and turn deviations (which reach showdown far more
    # often than all-in) had NO variance removed -- the dominant non-all-in luck source went
    # unaddressed. Non-diverged hands build the IDENTICAL river_range in both arms -> cancels.
    river_range = None
    board = list(d['community'])
    if d.get('river_entry_opp') is not None and len(board) >= 5:
        opp = RangeTracker.from_dict(d['river_entry_opp'], session.cards)
        river_range = {'range': opp, 'turn_board': board[:4], 'river_card': board[4],
                       'pot': float(d['starting_pot'])}
    return {'seat_of_A': bot_seat, 'hand_a': list(a_cards), 'hand_b': list(b_cards),
            'board': board, 'events': [], 'river_range': river_range,
            'allin_street': allin_street, 'folded': log['folder'],
            'invested': [p0_tot, p1_tot]}


def run_paired(path, hands, seed, args):
    """#4 CRN-paired river-vs-turn: play each hand on the SAME deck with BOTH stacks and diff
    the per-hand chips. ~all non-fire hands cancel exactly -> the (turn-river) stderr is
    driven only by the handful of hands the turn solver acts on, a several-fold variance cut
    vs two independent arms. This is the decisive instrument once the solver actually fires."""
    rawdb = BlueprintDB(path, read_only=True)
    fb = FrozenBlueprint(rawdb)
    menu = db_menu_mode(rawdb)
    # ONE bot object for BOTH arms, toggling max_spr_turn to disable/enable the turn solver.
    # Two SEPARATE objects can't be paired cleanly: they consume `_rng` differently before
    # the (Monte-Carlo, sampled) deep-raise guard -> the guard fires for one arm but not the
    # other -> spurious divergence (chased to ground 2026-06-15). With one object the guard,
    # blueprint, and river paths run IDENTICALLY; only the turn-solver branch differs, so a
    # diff means a genuine turn deviation. Determinism so non-deviation hands are bit-equal:
    #  * purify=1.0 -> blueprint path is ARGMAX (immune to the global-`random` state);
    #  * time budgets +inf -> every solve runs FIXED iters (a time-budgeted solve stops at a
    #    contention-dependent iter count -> non-deterministic). SPR cap still bounds tree size.
    # NB this measures the ARGMAX (modal) policy, not the served 0.01-mix -- the right call for
    # isolating whether the turn SOLVE finds better strategy.
    bot = _build_bot('turn', rawdb, fb, args)
    bot.purify_threshold = 1.0
    bot.time_budget = float('inf')
    bot.turn_time_budget = float('inf')
    turn_cap = args.turn_max_spr
    opponent = BlueprintStrategy(fb)
    strat_fn = BlueprintStrategy(fb).range_model_fn()
    opp_style = args.opponent
    import src.subgame.river_subgame_solver as _rss
    _base_river_spr = _rss.SOLVER_MAX_SPR
    measure = getattr(args, 'measure', 'turn')
    # arm A = baseline, arm B = +the-solver-under-test. 'turn': A=river-only, B=river+turn.
    # 'river': A=blueprint (NO solving), B=river solver. Toggle SOLVER_MAX_SPR (module const,
    # read at the river fire gate) for the river arm; max_spr_turn (instance) for the turn arm.
    la, lb = ('blueprint', 'river') if measure == 'river' else ('river', 'turn')
    def _set_arm(with_solver):
        if measure == 'river':
            bot.max_spr_turn = -1.0                                # turn OFF in both arms
            _rss.SOLVER_MAX_SPR = _base_river_spr if with_solver else -1.0
        else:
            _rss.SOLVER_MAX_SPR = _base_river_spr                  # river ON in both arms
            bot.max_spr_turn = turn_cap if with_solver else -1.0
    print(f"PAIRED (CRN) {lb}-vs-{la} | {os.path.basename(path)} | menu={menu} | hands={hands} "
          f"seed={seed} opp={opp_style} | turn(n={args.n_buckets},rivers={args.leaf_rivers},"
          f"it={args.turn_iters},spr<={turn_cap:g})")
    print("  ONE-bot ARGMAX, fixed-iters (deterministic); per-decision RNG reseed -> "
          "non-deviation hands cancel EXACTLY\n", flush=True)
    diffs = []
    r_tot = t_tot = 0.0
    ndiv = 0
    _div_hands = []                          # per-deviation ledger: (hand, seat, realized mbb)
    estimator = None
    turn_recs, river_recs = [], []
    if args.aivat:                           # AIVAT the paired diff: c3 collapses the all-in
        from src.evaluation.aivat import AIVATEstimator   # variance that dominates vs maxbet.
        estimator = AIVATEstimator(BlueprintDB(path, read_only=True), seed=seed)
    t0 = time.time()
    for h in range(hands):
        seed_h = seed * 1_000_003 + h
        human_seat = h % 2
        _set_arm(False)                  # baseline arm (la): river-only, or pure blueprint
        rc = _hand_chips(seed_h, human_seat, bot, opponent, opp_style, strat_fn, menu,
                         args.max_steps, want_aivat=args.aivat)
        _set_arm(True)                   # +solver arm (lb): +turn solver, or +river solver
        tc = _hand_chips(seed_h, human_seat, bot, opponent, opp_style, strat_fn, menu,
                         args.max_steps, want_aivat=args.aivat)
        if args.aivat:
            rc, _rrec = rc; tc, _trec = tc
            river_recs.append(_rrec); turn_recs.append(_trec)
        d = tc - rc
        diffs.append(d)
        r_tot += rc
        t_tot += tc
        if abs(d) > 1e-9:
            ndiv += 1
            _div_hands.append((h, human_seat, d / BIG_BLIND * 1000.0))   # per-deviation ledger
        if (h + 1) % max(1, hands // 10) == 0:
            nn = len(diffs)
            md = sum(diffs) / nn / BIG_BLIND * 1000.0
            print(f"  hand {h + 1:5d}: ({lb}-{la}) {md:+8.1f} mbb/hand  diverged={ndiv}/{nn}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    rawdb.close()
    _rss.SOLVER_MAX_SPR = _base_river_spr     # restore the module constant
    n = len(diffs)
    mean_d = sum(diffs) / n if n else 0.0
    var = sum((x - mean_d) ** 2 for x in diffs) / n if n else 0.0
    md_mbb = mean_d / BIG_BLIND * 1000.0
    se_mbb = (math.sqrt(var / n) / BIG_BLIND * 1000.0) if n else 0.0
    print(f"\n  {la} arm {r_tot / n / BIG_BLIND * 1000.0:+.1f}  {lb} arm "
          f"{t_tot / n / BIG_BLIND * 1000.0:+.1f} mbb/hand (raw, paired draws)")
    print(f"  PAIRED ({lb} - {la}) = {md_mbb:+.1f} +/- {se_mbb:.1f} mbb/hand over {n} hands "
          f"(diverged on {ndiv} = {100.0 * ndiv / max(1, n):.1f}%; {time.time() - t0:.0f}s)")
    if estimator is not None and turn_recs:
        # PAIRED control variate: subtract the per-hand control-variate DIFF (Δc2 river-runout
        # + Δc3 all-in-EV; Δc1=0 since both arms share the deck) from the raw DIFF with UNIT
        # coefficient -- the classic AIVAT estimator. Each c_k has conditional mean 0, so this
        # is UNBIASED (E[adiff]=E[D]) regardless of correlation, and it matches the (verified)
        # unpaired AIVATEstimator semantics. We deliberately do NOT fit beta in-sample: only the
        # ~handful of diverged hands carry signal, so an in-sample lstsq overfits the realized
        # stack-offs and can INFLATE the diff variance (that was the +278 se blow-up). Non-
        # diverged hands have ΔX=0 and D=0 -> drop out, so the se reflects only diverged hands.
        VT = np.array([estimator._hand_variates(r) for r in turn_recs], float)   # [n,3]=c1,c2,c3
        VR = np.array([estimator._hand_variates(r) for r in river_recs], float)
        D = np.array(diffs, float)                              # per-hand raw diff (chips)
        dCV = (VT - VR)[:, 1:].sum(axis=1)                      # Δc2 + Δc3 (Δc1 structurally 0)
        adiff = (D - dCV) * (1000.0 / 2.0)                     # per-hand AIVAT diff (mbb)
        rawmbb = D * (1000.0 / 2.0)
        amd = float(adiff.mean())
        ase = float(adiff.std(ddof=1) / len(adiff) ** 0.5) if len(adiff) > 1 else 0.0
        # Guard (M2): if the CVs INCREASE the diff variance on this sample (poorly correlated --
        # e.g. turn deviations that rarely reach all-in, so Δc3 is just noise), they HURT. Report
        # the raw paired diff as the estimate rather than a worse se mislabeled "AIVAT".
        if adiff.var() <= rawmbb.var() or rawmbb.var() <= 1e-12:
            vr = (1.0 - adiff.var() / rawmbb.var()) if rawmbb.var() > 1e-12 else 0.0
            print(f"  AIVAT ({lb} - {la}) = {amd:+.1f} +/- {ase:.1f} mbb/hand  (PAIRED control "
                  f"variate, var -{vr * 100:.0f}% vs raw; all-in/runout luck collapsed)")
        else:
            print(f"  AIVAT ({lb} - {la}): control variates INFLATE the diff variance on this "
                  f"sample (would be {amd:+.1f} +/- {ase:.1f}); too few all-in/showdown "
                  f"deviations for the CVs to grip -> use the RAW paired diff above as the "
                  f"estimate ({md_mbb:+.1f} +/- {se_mbb:.1f}).")
    _turn_stats_report(bot)
    # Per-deviation attribution (the low-variance dev signal): the aggregate (turn-river) is
    # noise-dominated by a few big-swing hands, so show WHERE the EV comes from -- which
    # diverged hands win vs lose. A few big losers => fixable spots, not a broad failure.
    if _div_hands:
        win = [x for x in _div_hands if x[2] > 0]
        lose = [x for x in _div_hands if x[2] < 0]
        sw, sl = sum(x[2] for x in win), sum(x[2] for x in lose)
        print(f"\n  PER-DEVIATION ({len(_div_hands)} diverged): {len(win)} win (+{sw:.0f}) / "
              f"{len(lose)} lose ({sl:.0f}) mbb-total -> net {sw + sl:+.0f} over {n} hands")
        srt = sorted(_div_hands, key=lambda x: x[2])
        print("    LOSERS  (hand/seat:mbb): " +
              "  ".join(f"#{h}/s{s}:{d:+.0f}" for h, s, d in srt[:6]))
        print("    WINNERS (hand/seat:mbb): " +
              "  ".join(f"#{h}/s{s}:{d:+.0f}" for h, s, d in srt[-6:][::-1]))
    print("\n  GATE: paired diff > 0 beyond ~2se => turn solving ADDS EV over river-only.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bot', choices=['turn', 'river', 'blueprint'], default='turn')
    p.add_argument('--db', default=None)
    p.add_argument('--hands', type=int, default=6000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--aivat', action='store_true',
                   help="Report AIVAT-corrected mbb/hand (c1 preflop-equity + c2 river-runout + "
                        "c3 all-in-EV control variates) alongside raw -- cuts the card-luck variance.")
    p.add_argument('--n-buckets', type=int, default=20)
    p.add_argument('--leaf-rivers', type=int, default=4)
    p.add_argument('--turn-iters', type=int, default=80)
    p.add_argument('--turn-budget', type=float, default=10.0,
                   help="wall-clock cap per turn solve incl. leaf-build (s); raise for an "
                        "OFFLINE gate so high-SPR turns actually solve instead of timing out.")
    p.add_argument('--turn-max-spr', type=float, default=10.0,
                   help="skip turns above this SPR (bigger tree -> leaf-build timeout). Keep "
                        "it in the band that solves within --turn-budget.")
    p.add_argument('--multivalued-leaf', action='store_true',
                   help="Phase 5b #2 EXPERIMENT: Modicum multi-valued turn leaf (opponent picks "
                        "worst of 4 bias continuations) instead of the single blueprint leaf. ~4x "
                        "leaf cost -> OFFLINE only (raise --turn-budget).")
    p.add_argument('--opponent', choices=['blueprint', 'maxbet', 'jam', 'widejam'],
                   default='blueprint',
                   help="paired-gate villain. 'maxbet' builds pots -> LOW-SPR turns -> the turn "
                        "solver actually fires (a passive 'blueprint' rarely triggers it).")
    p.add_argument('--river-iters', type=int, default=200)    # served strategy_api.py value
    p.add_argument('--river-budget', type=float, default=5.0)  # served strategy_api.py value
    p.add_argument('--max-steps', type=int, default=400)
    p.add_argument('--paired', action='store_true',
                   help="#4 CRN-paired diff (low variance); ignores --bot.")
    p.add_argument('--measure', choices=['turn', 'river'], default='turn',
                   help="paired diff to measure: 'turn' = (river+turn) - (river-only) = the TURN "
                        "solver's gain; 'river' = (river-only) - (blueprint) = the RIVER solver's "
                        "gain. Same metric -> apples-to-apples comparison.")
    args = p.parse_args()

    path = args.db or resolve_blueprint_path()
    if args.paired:
        run_paired(path, args.hands, args.seed, args)
        return
    rawdb = BlueprintDB(path, read_only=True)
    fb = FrozenBlueprint(rawdb)
    menu = db_menu_mode(rawdb)
    bot = _build_bot(args.bot, rawdb, fb, args)
    opponent = BlueprintStrategy(fb)
    strat_fn = BlueprintStrategy(fb).range_model_fn()

    estimator = aivat_db = None
    if args.aivat:
        from src.evaluation.aivat import AIVATEstimator
        aivat_db = BlueprintDB(path, read_only=True)
        estimator = AIVATEstimator(aivat_db, seed=args.seed)

    print(f"bot={args.bot} vs blueprint | {os.path.basename(path)} | menu={menu} | "
          f"hands={args.hands} seed={args.seed} aivat={args.aivat} | turn(n={args.n_buckets},"
          f"rivers={args.leaf_rivers},it={args.turn_iters})", flush=True)
    print("  (driven through advance_bot_turns -> continual-re-solving 1a/1b/1c are LIVE)")
    if args.bot != 'blueprint':
        print(f"  served policy: safe_gadget=True anchor=auto purify=0.01 "
              f"river_iters={args.river_iters} river_budget={args.river_budget} (rng seeded)")
    print(flush=True)

    random.seed(args.seed)               # deck + opponent + solver RNG seeded for reproducibility
    sess = GameSession.new('n0', 'bp', strategy_fn=strat_fn, menu_mode=menu,
                           max_raises_per_street=float('inf'))
    deltas = []
    records = []
    prev_net = 0.0
    bot_seat = 1 - sess.data['human_seat']
    t0 = time.time()
    for h in range(args.hands):
        if h > 0:
            sess.start_next_hand()
            bot_seat = 1 - sess.data['human_seat']
        rec = _play_and_record(sess, bot, opponent, bot_seat, args.max_steps, args.aivat)
        net = sess.data['human_net']
        bot_chips = -(net - prev_net)               # bot's chips this hand
        prev_net = net
        deltas.append(bot_chips)
        if args.aivat and rec is not None:
            rec['result'] = bot_chips               # AIVAT result is in chips
            records.append(rec)
        if (h + 1) % max(1, args.hands // 10) == 0:
            n = len(deltas)
            mbb = (sum(deltas) / n / BIG_BLIND * 1000.0) if n else 0.0
            print(f"  hand {h + 1:5d}: bot {mbb:+8.1f} mbb/hand  ({time.time() - t0:.0f}s)",
                  flush=True)
    rawdb.close()

    n = len(deltas)
    mean_chips = sum(deltas) / n if n else 0.0
    var = sum((dlt - mean_chips) ** 2 for dlt in deltas) / n if n else 0.0
    mbb = mean_chips / BIG_BLIND * 1000.0
    stderr = (math.sqrt(var / n) / BIG_BLIND * 1000.0) if n else 0.0
    print(f"\n  RESULT bot={args.bot}: raw {mbb:+.1f} +/- {stderr:.1f} mbb/hand "
          f"over {n} hands ({time.time() - t0:.0f}s)")
    if estimator is not None:
        est = estimator.estimate(records)
        print(f"  RESULT bot={args.bot}: AIVAT {est['aivat_mbb']:+.1f} +/- "
              f"{est['aivat_stderr_mbb']:.1f} mbb/hand  (var -{est['var_reduction'] * 100:.0f}%; "
              f"raw se {est['raw_stderr_mbb']:.1f})")
        aivat_db.close()
    _turn_stats_report(bot)
    print("\n  GATE: prefer --paired for the decision (low-variance turn-river diff). "
          "Unpaired arms need ~5-10k hands EACH to resolve a ~200 mbb effect.")


if __name__ == '__main__':
    main()
