# backend/bot/src/cfr/blueprint_trainer.py
import os
import random
from .poker_game import PokerGame, STARTING_STACK
from .information_set import InformationSet
from .keys import action_char as _action_char, make_info_set_key
from ..bot.game_adapter import GameAdapter


def _format_duration(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class BlueprintTrainer:
    """
    Blueprint CFR Trainer with Monte Carlo CFR+ (External Sampling) and stack constraints.
    """

    def __init__(self, menu_mode='control'):
        """
        menu_mode : 'control' (default) trains on the current postflop menu with the
                    voluntary all-in node (byte-identical to before -- the A/B
                    baseline). 'capped' trains the Fix-#4 redesign: the 5-size capped
                    postflop menu (incl. overbet2=2.0x) with the voluntary all-in node
                    DROPPED (all-in emerges only when a sized tier clamps to stack).
                    This is an ABSTRACTION choice -- a 'capped' blueprint is
                    incompatible with a 'control' one and vice versa (the C
                    measurement trains one of each). The mode is stamped into the DB
                    metadata so a resume can't silently mix arms (mirrors the
                    single/parallel mode guard).
        """
        self.menu_mode = menu_mode
        self.info_sets = {}
        # Keys created/mutated since the last checkpoint. Checkpoints persist
        # only these (every visited info set is mutated -- regrets or strategy --
        # so marking on touch is complete), turning the checkpoint from an
        # O(all info sets) full-table rewrite into O(touched-since-last). The DB
        # always holds the latest row for every key, so resume is unaffected.
        self._dirty = set()
        # Parallel/worker mode only (discount disabled): the regret update happens
        # at traverser nodes, the average-strategy accumulation at opponent nodes.
        # Tracking the two separately lets a worker export exactly the right delta
        # and lets the master advance the correct discount clock per key. Populated
        # only when discount_enabled is False (see cfr) so single-thread pays nothing.
        self._dirty_regret = set()
        self._dirty_strategy = set()
        # When False, cfr() skips the per-iteration alpha/gamma decay and just
        # accumulates raw (CFR+-floored) contributions. Parallel workers run with
        # discount disabled; the master applies a block discount once per merge
        # round (see parallel_trainer.merge_round). Single-thread keeps it True
        # so its behaviour is unchanged / bit-identical.
        self.discount_enabled = True
        from ..abstractions.sizing import postflop_menu_for, is_capped_mode
        if menu_mode == 'control':
            self.game = PokerGame()                  # default menu + voluntary all-in
        elif is_capped_mode(menu_mode):              # 'capped' | 'capped_no2'
            self.game = PokerGame(postflop_menu=postflop_menu_for(menu_mode),
                                  voluntary_allin=False)
        else:
            raise ValueError(
                f"menu_mode must be 'control', 'capped', or 'capped_no2', "
                f"got {menu_mode!r}")
        self.game_adapter = GameAdapter()
        self.deck = self.create_deck()

        # Discount exponents (Linear-CFR-style; cf. Brown & Sandholm 2019).
        # alpha discounts the cumulative regrets; gamma discounts the cumulative
        # AVERAGE strategy so later (better-converged) iterations dominate the
        # blueprint.
        #
        # NOTE: this is CFR+ with a Linear-CFR-style discount, NOT the canonical
        # Discounted CFR (DCFR) scheme. Differences, on purpose: (1) regrets are
        # floored at 0 (CFR+) and the discount is applied to those non-negative
        # regrets -- there is no separate negative-regret exponent beta;
        # (2) the per-step decay is ((t-1)/t)**alpha (Linear CFR at alpha=1),
        # not DCFR's t**a/(t**a+1) multiplier; (3) the alpha/gamma clocks advance
        # on each role's own visit counts, not a single global t. It is a valid,
        # convergent regret-minimiser; just don't expect canonical-DCFR behaviour
        # when tuning alpha.
        self.alpha = 1.5
        # gamma = 2.0 weights iteration t's strategy contribution by ~(t/T)**2, so the
        # served AVERAGE is heavily recent-weighted -- it tracks the current iterate (avg
        # vs iterate TV ~0.09 on the 25M run) instead of being a stable uniform time-
        # average. Late in training the abstract iterate PURIFIES (avg max action-prob
        # rises 0.71 -> 0.85 across 0.5M -> 28M), and a more deterministic strategy loses
        # GTO mixing -> MORE exploitable -> the U-shaped BR/LBR curve (rises late). It is
        # NOT a code bug (the parallel gamma-clock was audited clean; H1 averaging is
        # correct -- Kuhn-verified in tests/h1_kuhn_check.py). A less aggressive discount
        # retains the earlier, more-mixed play. ===> NEXT RETRAIN: use gamma = 1.0 (true
        # Linear-CFR) and BR-sweep to confirm the tail flattens. Keep serving the BEST
        # snapshot (not the last) regardless. See analysis/training_curve + the U-curve
        # probe (2026-06-10).
        self.gamma = 2.0

        # Cumulative EV gauge, persisted across resumes. `ev_sum` / `ev_count`
        # accumulate the per-iteration sampled root value over the blueprint's
        # ENTIRE lifetime (not just the current session), so the printed mean
        # is stable across resume boundaries instead of resetting to a noisy
        # session-only average. NOTE: this is a convergence gauge, not the
        # blueprint's true EV -- it averages the value of the still-evolving
        # current strategy (including early iterations). For the real strength
        # of the average strategy, use the evaluation harness (best_response.py).
        self.ev_sum = 0.0
        self.ev_count = 0

    def create_deck(self):
        suits = ['H', 'D', 'C', 'S']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        return [suit + rank for rank in ranks for suit in suits]

    def deal_random_hand(self):
        shuffled_deck = self.deck.copy()
        random.shuffle(shuffled_deck)
        p0_cards = shuffled_deck[0:2]
        p1_cards = shuffled_deck[2:4]
        community_cards = shuffled_deck[4:9]
        return p0_cards, p1_cards, community_cards

    def _postflop_strength(self, player, street, community_cards):
        """Acting player's postflop strength bucket for this street, computed
        lazily and memoized per iteration. street 1/2/3 -> board of 3/4/5 cards.
        Only called for the player about to act at a reached node, so the
        expensive river bucket is never computed for a hand that ends earlier."""
        key = (player, street)
        val = self._postflop_memo.get(key)
        if val is None:
            cards = self._p0_cards if player == 0 else self._p1_cards
            ca = self.game_adapter.card_abstractions
            val = ca.get_bucket(cards, community_cards[:street + 2])
            self._postflop_memo[key] = val
        return val

    def _rollout_avg(self, p0, p1, comm):
        """One self-play hand sampling the AVERAGE (served) strategy for both
        seats; returns the P0-perspective utility. Mirrors cfr()'s forward pass
        (state threading, street transitions, pattern chars) but follows a SINGLE
        sampled path and reads get_average_strategy -- the strategy actually served
        at inference -- instead of the current regret-matched one. Untrained keys
        fall back to uniform. Assumes the per-hand caches (_p0_preflop/_p1_preflop,
        _p0_cards/_p1_cards, _postflop_memo) are already set for this hand."""
        g = self.game
        g._calc_cache.clear()
        street, history, spot = 0, [], 3
        p0_inv = p1_inv = 0.0
        p0_stack, p1_stack = STARTING_STACK - 1, STARTING_STACK - 2
        pat = ''
        st = g.init_node_state(0, spot)

        def term(s, state):
            return g.get_utility(p0, p1, comm, history, min(s, 3), spot,
                                 p0_inv, p1_inv,
                                 _final_pot=state['pot'],
                                 _p0_total=p0_inv + state['c'][0])

        for _ in range(60):
            if street > 3 or g.is_terminal(history, street):
                return term(street, st)
            cp = g._acting_player(len(history), street)
            stack_cp = p0_stack if cp == 0 else p1_stack
            legal = g.state_legal_actions(street, st, cp, stack_cp)
            if not legal:
                if street >= 3:
                    return term(street, st)
                p0_inv += st['c'][0]
                p1_inv += st['c'][1]
                p0_stack = STARTING_STACK - p0_inv
                p1_stack = STARTING_STACK - p1_inv
                spot, street, history, pat = st['pot'], street + 1, [], ''
                st = g.init_node_state(street, spot)
                continue
            pos = 'ip' if cp == 0 else 'oop'
            pf = self._p0_preflop if cp == 0 else self._p1_preflop
            strength = self._postflop_strength(cp, street, comm) if street > 0 else None
            key = make_info_set_key(street, pos, pf, strength, pat)
            iset = self.info_sets.get(key)
            if iset is None:
                probs = [1.0 / len(legal)] * len(legal)
            else:
                # get_average_strategy returns a numpy array aligned to `legal`.
                probs = list(iset.get_average_strategy(legal))
            action = random.choices(legal, weights=probs)[0]
            cost = g.state_action_cost(action, street, st, cp, stack_cp)
            child = g.advance_node_state(st, action, street, cp, stack_cp,
                                         (p0_inv, p1_inv))
            if cp == 0:
                p0_stack -= cost
            else:
                p1_stack -= cost
            st = child
            history = history + [action]
            pat += _action_char(action)
        return term(street, st)

    def evaluate_served_ev(self, n=4000, seed=12345):
        """Mean P0-perspective EV (chips) of the SERVED (average) strategy, over n
        self-play rollouts. THIS is the iterate CFR guarantees converges -- unlike
        the per-iteration EV gauge (EV(cum)/EV(round)/EV(sess)), which is the
        CURRENT regret-matched strategy and need NOT converge (it can cycle at a
        large value forever while the average settles; see the 2026-06-03 diagnosis).
        This settles to a small STABLE constant -- the button's game-value edge, not
        literally 0 (HU poker is asymmetric: P0=button/SB plays the better seat).
        CAVEAT: this is a SEAT-BALANCE / convergence sanity check, NOT a strength
        metric. A seat-lopsided or unconverged served strategy reads large, but two
        equally-bad symmetric strategies also self-play near the constant -- so a
        small value does NOT prove strength. For exploitability use LBR/BR.

        Deterministically seeded so the estimate is paired across checkpoints
        (low-variance trend), and RNG-isolated: the global random state is saved
        and restored, so calling this mid-training does not perturb the training
        hand stream. Clobbers the per-hand caches, which is safe between iterations
        (every _run_iteration resets them)."""
        rng_state = random.getstate()
        try:
            random.seed(seed)
            ca = self.game_adapter.card_abstractions
            total = 0.0
            for _ in range(n):
                p0, p1, comm = self.deal_random_hand()
                self._p0_preflop = ca.get_bucket(p0, None)
                self._p1_preflop = ca.get_bucket(p1, None)
                self._p0_cards, self._p1_cards = p0, p1
                self._postflop_memo = {}
                total += self._rollout_avg(p0, p1, comm)
            return total / n if n else 0.0
        finally:
            random.setstate(rng_state)

    def cfr(self, p0_cards, p1_cards, community_cards, history,
            street, updating_player, depth=0, iteration=0, starting_pot=None,
            p0_invested=0.0, p1_invested=0.0, bet_pattern='',
            p0_stack=None, p1_stack=None, st=None):
        """
        External-sampling Monte Carlo CFR+ (with Linear-CFR-style regret discounting).

        Perspective convention: cfr() ALWAYS returns the value from P0's fixed
        perspective. get_utility() returns P0's perspective; terminal and
        street-transition results pass straight through. At a decision node the
        value is converted to the acting player's perspective for the regret
        computation, then converted back to P0's perspective before returning.

        - Traverser (updating_player): explores every action, updates regrets.
        - Opponent: accumulates the average strategy, samples a single action.

        Lever A: `st` is the threaded within-street betting state (pot /
        contributions / to-call / legal actions), so the hot path never replays
        `history` for chip math. It's bit-identical to the history-based
        functions (validated by tests/test_lever_a_oracle.py + the seed-compare).
        history is still maintained for is_terminal / _acting_player / info-set keys.
        """

        if starting_pot is None:
            starting_pot = 3  # SB(1) + BB(2)
        if p0_stack is None:
            p0_stack = STARTING_STACK - 1  # P0 posted SB
        if p1_stack is None:
            p1_stack = STARTING_STACK - 2  # P1 posted BB
        if st is None:
            st = self.game.init_node_state(street, starting_pot)

        def _terminal_value(s):
            # P0-perspective utility using the threaded pot + P0 total (no replay).
            return self.game.get_utility(
                p0_cards, p1_cards, community_cards, history, min(s, 3), starting_pot,
                p0_invested, p1_invested,
                _final_pot=st['pot'], _p0_total=p0_invested + st['c'][0])

        if depth > 50:
            print(f"WARNING: Max depth reached at street {street}, history {history}")
            return _terminal_value(street)

        if street > 3:
            return _terminal_value(3)

        if self.game.is_terminal(history, street):
            return _terminal_value(street)

        current_player = self.game._acting_player(len(history), street)
        stack_cp = p0_stack if current_player == 0 else p1_stack

        legal_actions = self.game.state_legal_actions(street, st, current_player, stack_cp)

        if not legal_actions:
            if street < 3:
                p0_this, p1_this = st['c'][0], st['c'][1]
                # Recompute stacks from total invested to avoid drift across streets
                new_p0_stack = STARTING_STACK - (p0_invested + p0_this)
                new_p1_stack = STARTING_STACK - (p1_invested + p1_this)
                return self.cfr(p0_cards, p1_cards, community_cards, [],
                                street + 1, updating_player,
                                depth + 1, iteration, st['pot'],
                                p0_invested + p0_this, p1_invested + p1_this,
                                bet_pattern='',
                                p0_stack=new_p0_stack, p1_stack=new_p1_stack, st=None)
            else:
                return _terminal_value(street)

        # Build info set key
        position = 'ip' if current_player == 0 else 'oop'
        preflop_bucket = self._p0_preflop if current_player == 0 else self._p1_preflop
        strength = (self._postflop_strength(current_player, street, community_cards)
                    if street > 0 else None)
        info_set_key = make_info_set_key(
            street, position, preflop_bucket, strength, bet_pattern)

        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = InformationSet()
        info_set = self.info_sets[info_set_key]
        # This visit mutates the info set (regret update at a traverser node, or
        # average-strategy accumulation at an opponent node), so mark it dirty for
        # the next checkpoint.
        self._dirty.add(info_set_key)

        strategy = info_set.get_strategy(legal_actions)

        p_inv = (p0_invested, p1_invested)

        def child_step(action):
            """(new_p0_stack, new_p1_stack, child_state) after taking `action` --
            cost + state from the threaded state (no history replay)."""
            cost = self.game.state_action_cost(action, street, st, current_player, stack_cp)
            child_st = self.game.advance_node_state(
                st, action, street, current_player, stack_cp, p_inv)
            if current_player == 0:
                return p0_stack - cost, p1_stack, child_st
            return p0_stack, p1_stack - cost, child_st

        if current_player == updating_player:
            # --- Traverser node: explore every action, update regrets. ---
            if not self.discount_enabled:
                # Worker mode: record that this key got a regret update this round
                # so it is exported and the master advances its regret clock.
                self._dirty_regret.add(info_set_key)
            # First visit of this iteration: bump the discount clock and discount
            # the prior cumulative regret ONCE. An info set can be reached more
            # than once per traversal (different lines collapse onto the same
            # key), so decaying inside the per-action loop would over-discount.
            if self.discount_enabled and info_set.last_visited_iteration != iteration:
                info_set.visit_count += 1
                info_set.last_visited_iteration = iteration
                t = info_set.visit_count
                if t > 1:
                    decay = ((t - 1) / t) ** self.alpha
                    for a in info_set.cumulative_regrets:
                        info_set.cumulative_regrets[a] *= decay

            # child values, all in P0's perspective
            action_values = []
            for i, action in enumerate(legal_actions):
                next_history = history + [action]
                next_pattern = bet_pattern + _action_char(action)
                new_p0_stack, new_p1_stack, child_st = child_step(action)
                action_values.append(self.cfr(
                    p0_cards, p1_cards, community_cards, next_history,
                    street, updating_player,
                    depth + 1, iteration, starting_pot,
                    p0_invested, p1_invested, next_pattern,
                    new_p0_stack, new_p1_stack, st=child_st))

            # Convert to the acting player's own perspective for regret matching.
            sign = 1.0 if current_player == 0 else -1.0
            own_values = [sign * v for v in action_values]
            node_value = sum(strategy[i] * own_values[i]
                             for i in range(len(legal_actions)))

            # Accumulate this visit's regret. The discount was already applied once
            # at the first visit above; no reach weighting — external sampling
            # handles opponent reach.
            #
            # CFR+ write-floor: cumulative regret is floored at 0 on EVERY write, in
            # BOTH single-thread and parallel-worker mode (canonical CFR+).
            #
            # REVERTED Fix #2 (2026-06-04). Fix #2 had workers store the RAW unfloored
            # sum so the master could cancel opposite-signed deltas before one per-round
            # floor (to remove a small upward jam bias). But raw storage BREAKS CFR+
            # re-activation: an action driven negative can't pop back the instant it gets
            # one good iteration, so within a chunk it stays suppressed and the strategy
            # COLLAPSES onto whatever dominated early. In practice that produced a
            # blueprint that opens `xlarge` (5 BB) with 100% of hands and never folds the
            # button. A 500k-iter A/B (scripts/ab_fix2_revert.py) confirmed it: per-worker
            # flooring -> pf_0 folds 74%; raw -> 1%. The +0.0014 jam bias Fix #2 chased is
            # negligible and is covered structurally by Fix #4 (the capped menu drops the
            # voluntary all-in node). So we floor per worker = canonical data-parallel
            # CFR+ -- the path that trained the sane blueprint_par_20260529_233056.
            for i, action in enumerate(legal_actions):
                regret = own_values[i] - node_value
                prior = info_set.cumulative_regrets.get(action, 0.0)
                info_set.cumulative_regrets[action] = max(0.0, prior + regret)

            # Return value back in P0's perspective.
            return sign * node_value

        else:
            # --- Opponent node: accumulate avg strategy, sample one action. ---
            if not self.discount_enabled:
                # Worker mode: record that this key got a strategy update this round
                # so it is exported and the master advances its strategy clock.
                self._dirty_strategy.add(info_set_key)
            # gamma discount: discount the prior average-strategy sum ONCE per
            # iteration (its own clock, separate from the regret clock) before
            # adding this visit's contribution, so later iterations dominate.
            # Same once-per-iteration guard as the regret discount above: an info
            # set can recur within a traversal via different lines.
            if self.discount_enabled and info_set.last_strategy_iteration != iteration:
                info_set.strategy_visit_count += 1
                info_set.last_strategy_iteration = iteration
                s = info_set.strategy_visit_count
                if s > 1 and self.gamma:
                    decay = ((s - 1) / s) ** self.gamma
                    for a in info_set.cumulative_strategy:
                        info_set.cumulative_strategy[a] *= decay
            info_set.accumulate_strategy(legal_actions, strategy)
            sampled_action = random.choices(legal_actions, weights=strategy)[0]
            next_history = history + [sampled_action]
            next_pattern = bet_pattern + _action_char(sampled_action)
            new_p0_stack, new_p1_stack, child_st = child_step(sampled_action)
            return self.cfr(
                p0_cards, p1_cards, community_cards, next_history,
                street, updating_player,
                depth + 1, iteration, starting_pot,
                p0_invested, p1_invested, next_pattern,
                new_p0_stack, new_p1_stack, st=child_st)

    def _run_iteration(self, actual_iteration):
        """One MCCFR+ iteration: deal a hand, set up the per-iteration bucket
        cache, run a full traversal for this iteration's updating player, and
        fold the sampled root value into the lifetime EV gauge. Returns the
        sampled root utility. Shared by the single-thread loop and the parallel
        worker chunk so both run identical per-iteration logic."""
        p0_cards, p1_cards, community_cards = self.deal_random_hand()

        ca = self.game_adapter.card_abstractions
        # Preflop buckets are cheap (dict lookup) -- compute eagerly.
        self._p0_preflop = ca.get_bucket(p0_cards, None)
        self._p1_preflop = ca.get_bucket(p1_cards, None)
        # Postflop buckets (esp. river, ~990-hand equity) are expensive and only
        # the ACTING player's bucket at the CURRENT street is ever read for a key.
        # Compute them lazily on first use inside cfr(), memoized per iteration.
        self._p0_cards = p0_cards
        self._p1_cards = p1_cards
        self._postflop_memo = {}

        updating_player = actual_iteration % 2
        self.game._calc_cache.clear()
        util = self.cfr(
            p0_cards, p1_cards, community_cards, [],
            0, updating_player, 0, actual_iteration, 3)
        self.ev_sum += util
        self.ev_count += 1
        return util

    def train_blueprint(self, iterations, db=None, start_iteration=0, checkpoint_every=10000):
        """Main training loop."""
        import time

        LOG_EVERY = 10000

        total_target = start_iteration + iterations
        print(f"Starting blueprint CFR training")
        print(f"  Target iterations : {total_target:,}")
        print(f"  Starting from     : {start_iteration:,}")
        print(f"  Remaining         : {iterations:,}")
        print(f"  Checkpoint every  : {checkpoint_every:,}")
        print(f"  Starting stack    : {STARTING_STACK}")
        print()

        # H2: resume monotonicity guard. On a resumed run the loaded info sets carry
        # discount clocks (last_visited/last_strategy_iteration) up to the prior run's
        # final iteration. If start_iteration is BEHIND those, the once-per-iteration decay
        # guard (`clock != iteration`) mis-fires and the Linear-CFR weighting is silently
        # corrupted. The correct resume passes start_iteration = resume_from_db() (= the
        # stored total_iterations, which exceeds every stored clock). Refuse otherwise.
        if self.discount_enabled and self.info_sets:
            max_clock = max(max(iset.last_visited_iteration, iset.last_strategy_iteration)
                            for iset in self.info_sets.values())
            if start_iteration < max_clock:
                raise ValueError(
                    f"resume start_iteration={start_iteration} is behind the loaded "
                    f"discount clocks (max={max_clock}); pass start_iteration="
                    f"resume_from_db(...) so the iteration counter stays monotonic.")

        session_ev_sum = 0.0
        t_start = time.time()

        for i in range(iterations):
            actual_iteration = start_iteration + i
            t_iter = time.time()
            util = self._run_iteration(actual_iteration)
            session_ev_sum += util

            if (i + 1) % LOG_EVERY == 0:
                iter_ms = (time.time() - t_iter) * 1000
                now = time.time()
                elapsed_total = now - t_start
                iters_done = i + 1
                iters_per_sec = iters_done / elapsed_total if elapsed_total > 0 else 0
                remaining_iters = iterations - iters_done
                eta_sec = remaining_iters / iters_per_sec if iters_per_sec > 0 else 0
                eta_str = _format_duration(eta_sec)
                elapsed_str = _format_duration(elapsed_total)

                # Cumulative EV (lifetime, stable across resumes) plus this
                # session's EV so a resume's local progress is still visible.
                cum_ev = self.ev_sum / self.ev_count if self.ev_count else 0.0
                session_ev = session_ev_sum / iters_done

                print(f"  iter {actual_iteration + 1:>9,} / {total_target:,} | "
                      f"EV(cum): {cum_ev:+.5f} | EV(sess): {session_ev:+.5f} | "
                      f"info sets: {len(self.info_sets):>7,} | "
                      f"{iters_per_sec:>6.1f} it/s | "
                      f"iter_ms: {iter_ms:.1f} | "
                      f"elapsed: {elapsed_str} | "
                      f"ETA: {eta_str}")

            if db is not None and (i + 1) % checkpoint_every == 0:
                self.checkpoint_to_db(db, actual_iteration)
                # Served-EV (seat-balance/convergence gauge, NOT strength -> use LBR) +
                # the strategy-shape collapse probe (BUG-014: open one size with 100% of
                # hands, never fold). Exception-wrapped: a gauge bug must not kill the run
                # (the DB is already checkpointed above).
                try:
                    served_ev = self.evaluate_served_ev()
                    print(f"  EV(served, avg strategy): {served_ev:+.4f}  <- served "
                          f"self-play value (seat-balance check, NOT strength -> use LBR)")
                    from .strategy_shape import strategy_shape_report, format_shape_line
                    print(format_shape_line(strategy_shape_report(
                        lambda k: (self.info_sets[k].cumulative_strategy
                                   if k in self.info_sets else None))))
                except Exception as e:           # noqa: BLE001 -- gauges must never crash training
                    print(f"  [gauge/probe skipped: {type(e).__name__}: {e}]")

        total_elapsed = _format_duration(time.time() - t_start)
        print(f"\nTraining completed in {total_elapsed}.")
        # Return the lifetime cumulative EV (stable across resumes), not just
        # this session's average.
        return self.ev_sum / self.ev_count if self.ev_count else 0.0

    def checkpoint_to_db(self, db, iteration):
        # Atomic: info sets + all metadata land together or not at all, so an
        # interrupt mid-checkpoint can't leave info sets ahead of total_iterations
        # (which would double-count regrets on resume).
        #
        # Incremental: persist ONLY the info sets touched since the last
        # checkpoint (the dirty set). INSERT OR REPLACE leaves untouched rows as
        # previously written, and every row was written when it was first dirtied,
        # so the on-disk blueprint is byte-identical to a full rewrite while the
        # write cost is O(touched) instead of O(all). Guarded by
        # tests/test_checkpoint_dirty.py (DB == in-memory) + the seed-compare.
        dirty = {k: self.info_sets[k] for k in self._dirty if k in self.info_sets}
        db.save_checkpoint(dirty, {
            'total_iterations': iteration + 1,
            'alpha': self.alpha,
            'gamma': self.gamma,
            # Action-abstraction arm (control vs Fix-#4 capped). Stamped so a resume
            # is refused across menus (resume_from_db menu-mode guard).
            'menu_mode': self.menu_mode,
            # Lifetime EV accumulators so the cumulative mean survives a resume.
            'ev_sum': self.ev_sum,
            'ev_count': self.ev_count,
        })
        self._dirty.clear()
        print(f"Checkpoint: {len(dirty)} info sets written "
              f"({len(self.info_sets)} total) at iteration {iteration + 1}")

    def resume_from_db(self, db, mode=None):
        self.info_sets = db.load_all_to_memory()
        # Every loaded row is already on disk; nothing is dirty until mutated.
        self._dirty.clear()
        start_iteration = db.get_metadata('total_iterations', 0)
        # Restore lifetime EV accumulators (absent on pre-EV-persistence DBs).
        self.ev_sum = db.get_metadata('ev_sum', 0.0)
        self.ev_count = db.get_metadata('ev_count', 0)
        # Refuse to silently change the discount schedule mid-blueprint:
        # the average strategy is only valid under one consistent (alpha, gamma).
        # Pre-schedule DBs (no stored value) are skipped for backward compat.
        for name in ('alpha', 'gamma'):
            stored = db.get_metadata(name, None)
            if stored is not None and abs(float(stored) - getattr(self, name)) > 1e-9:
                raise ValueError(
                    f"Resume {name} mismatch: this blueprint was trained with "
                    f"{name}={stored}, but the run is configured with {getattr(self, name)}. "
                    f"Changing the discount schedule mid-blueprint corrupts the average "
                    f"strategy. Resume without overriding {name} (or pass the stored value).")
        # Cross-mode resume guard (mode = 'single' | 'parallel' | None to skip).
        # Single-thread and parallel runs use the SAME visit_count field with
        # incompatible clocks (it counts iterations in single-thread but merge-
        # ROUNDS in parallel), so resuming across modes corrupts the Linear-CFR
        # discount -- either neutering it or triggering a one-shot regret-wiping
        # decay. Prefer the stamped mode; for a pre-stamp DB (e.g. an in-flight
        # run started before this guard) fall back to the on-disk filename tag
        # ('blueprint_par_*' = parallel, written by run_blueprint_trainer). Only
        # a KNOWN, DIFFERENT mode is refused -- an unknown mode is allowed, like
        # the alpha/gamma guards above.
        if mode is not None:
            stored_mode = db.get_metadata('training_mode', None)
            if stored_mode is None:
                base = os.path.basename(getattr(db, 'db_path', '') or '')
                if base.startswith('blueprint_par'):
                    stored_mode = 'parallel'
                # A plain 'blueprint_*' name is ambiguous (single-thread runs and
                # legacy names share it), so leave stored_mode None and allow it.
            if stored_mode is not None and stored_mode != mode:
                raise ValueError(
                    f"Resume mode mismatch: this blueprint was trained in "
                    f"{stored_mode!r} mode, but the run is configured for {mode!r}. "
                    f"Single-thread and parallel runs keep incompatible per-info-set "
                    f"clocks (visit_count = iterations vs merge-rounds), so resuming "
                    f"across modes corrupts the discount. Resume in {stored_mode!r} "
                    f"mode (match/omit the workers arg), or start a fresh run.")
        # Menu-mode guard: a 'capped' blueprint has a DIFFERENT action abstraction
        # (5-size menu + no voluntary all-in node) than a 'control' one, so resuming
        # one as the other would mix incompatible info-set keys (e.g. an 'overbet2'
        # char that the control menu never produces). Refuse a known mismatch; a
        # pre-stamp DB (no stored menu_mode) is assumed 'control' (the only mode that
        # existed before this flag), so resuming it as 'capped' is refused while
        # resuming it as 'control' is allowed.
        stored_menu = db.get_metadata('menu_mode', 'control')
        if stored_menu != self.menu_mode:
            raise ValueError(
                f"Resume menu-mode mismatch: this blueprint was trained with "
                f"menu_mode={stored_menu!r}, but the run is configured for "
                f"{self.menu_mode!r}. The capped and control menus are different "
                f"action abstractions (5-size + emergent all-in vs 4-size + voluntary "
                f"all-in) and produce incompatible info-set keys. Resume with "
                f"menu_mode={stored_menu!r}, or start a fresh run.")
        print(f"Resumed: {len(self.info_sets)} info sets, continuing from iteration {start_iteration}")
        return start_iteration
