# backend/bot/src/cfr/blueprint_trainer.py
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

    def __init__(self):
        self.info_sets = {}
        self.game = PokerGame()
        self.game_adapter = GameAdapter()
        self.deck = self.create_deck()
        self.BET_MULTIPLIERS = {'small': 0.33, 'medium': 0.66, 'large': 1.00}

        # DCFR discount exponents (Brown & Sandholm 2019). alpha discounts the
        # cumulative regrets; gamma discounts the cumulative AVERAGE strategy so
        # that later (better-converged) iterations dominate the blueprint.
        self.alpha = 1.5
        self.gamma = 2.0

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

    def _action_stack_cost(self, action, history, street, starting_pot, p0_prev, p1_prev, current_player):
        """Chips the current player must put in to take this action (for stack tracking)."""
        return self.game._action_cost(
            action, street, history, starting_pot, current_player, p0_prev, p1_prev)

    def cfr(self, p0_cards, p1_cards, community_cards, history,
            street, updating_player, depth=0, iteration=0, starting_pot=None,
            p0_invested=0.0, p1_invested=0.0, bet_pattern='',
            p0_stack=None, p1_stack=None):
        """
        External-sampling Monte Carlo CFR+ (with DCFR regret discounting).

        Perspective convention: cfr() ALWAYS returns the value from P0's fixed
        perspective. get_utility() returns P0's perspective; terminal and
        street-transition results pass straight through. At a decision node the
        value is converted to the acting player's perspective for the regret
        computation, then converted back to P0's perspective before returning.

        - Traverser (updating_player): explores every action, updates regrets.
        - Opponent: accumulates the average strategy, samples a single action.
        """

        if depth > 50:
            print(f"WARNING: Max depth reached at street {street}, history {history}")
            return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                         history, min(street, 3), starting_pot,
                                         p0_invested, p1_invested)

        if starting_pot is None:
            starting_pot = 3  # SB(1) + BB(2)

        if p0_stack is None:
            p0_stack = STARTING_STACK - 1  # P0 posted SB
        if p1_stack is None:
            p1_stack = STARTING_STACK - 2  # P1 posted BB

        if street > 3:
            return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                         history, 3, starting_pot,
                                         p0_invested, p1_invested)

        if self.game.is_terminal(history, street):
            return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                         history, street, starting_pot,
                                         p0_invested, p1_invested)

        current_player = self.game._acting_player(len(history), street)

        legal_actions = self.game.get_legal_actions(
            street, history, starting_pot, current_player,
            p0_stack, p1_stack, p0_invested, p1_invested)

        if not legal_actions:
            current_pot = self.game.calculate_current_pot(
                starting_pot, history, street, p0_invested, p1_invested)
            if street < 3:
                p0_this = self.game.get_player_contribution_this_round(
                    history, street, starting_pot, 0, p0_invested, p1_invested)
                p1_this = self.game.get_player_contribution_this_round(
                    history, street, starting_pot, 1, p0_invested, p1_invested)
                # Recompute stacks from total invested to avoid drift across streets
                new_p0_stack = STARTING_STACK - (p0_invested + p0_this)
                new_p1_stack = STARTING_STACK - (p1_invested + p1_this)

                return self.cfr(p0_cards, p1_cards, community_cards, [],
                                street + 1, updating_player,
                                depth + 1, iteration, current_pot,
                                p0_invested + p0_this, p1_invested + p1_this,
                                bet_pattern='',
                                p0_stack=new_p0_stack, p1_stack=new_p1_stack)
            else:
                return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                             history, street, starting_pot,
                                             p0_invested, p1_invested)

        # Build info set key
        position = 'ip' if current_player == 0 else 'oop'
        preflop_bucket = self._p0_preflop if current_player == 0 else self._p1_preflop
        strength = (self._p0_postflop[street] if current_player == 0
                    else self._p1_postflop[street]) if street > 0 else None
        info_set_key = make_info_set_key(
            street, position, preflop_bucket, strength, bet_pattern)

        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = InformationSet()
        info_set = self.info_sets[info_set_key]

        strategy = info_set.get_strategy(legal_actions)

        def child_stacks(action):
            """Return (new_p0_stack, new_p1_stack) after taking action."""
            cost = self._action_stack_cost(
                action, history, street, starting_pot, p0_invested, p1_invested, current_player)
            if current_player == 0:
                return p0_stack - cost, p1_stack
            else:
                return p0_stack, p1_stack - cost

        if current_player == updating_player:
            # --- Traverser node: explore every action, update regrets. ---
            # First visit of this iteration: bump the DCFR clock and discount
            # the prior cumulative regret ONCE. An info set can be reached more
            # than once per traversal (different lines collapse onto the same
            # key), so decaying inside the per-action loop would over-discount.
            if info_set.last_visited_iteration != iteration:
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
                new_p0_stack, new_p1_stack = child_stacks(action)
                action_values.append(self.cfr(
                    p0_cards, p1_cards, community_cards, next_history,
                    street, updating_player,
                    depth + 1, iteration, starting_pot,
                    p0_invested, p1_invested, next_pattern,
                    new_p0_stack, new_p1_stack))

            # Convert to the acting player's own perspective for regret matching.
            sign = 1.0 if current_player == 0 else -1.0
            own_values = [sign * v for v in action_values]
            node_value = sum(strategy[i] * own_values[i]
                             for i in range(len(legal_actions)))

            # Accumulate this visit's regret (CFR+ floors at 0). The DCFR
            # discount was already applied once at the first visit above; no
            # reach weighting — external sampling handles opponent reach.
            for i, action in enumerate(legal_actions):
                regret = own_values[i] - node_value
                prior = info_set.cumulative_regrets.get(action, 0.0)
                info_set.cumulative_regrets[action] = max(0.0, prior + regret)

            # Return value back in P0's perspective.
            return sign * node_value

        else:
            # --- Opponent node: accumulate avg strategy, sample one action. ---
            # DCFR gamma: discount the prior average-strategy sum ONCE per
            # iteration (its own clock, separate from the regret clock) before
            # adding this visit's contribution, so later iterations dominate.
            # Same once-per-iteration guard as the regret discount above: an info
            # set can recur within a traversal via different lines.
            if info_set.last_strategy_iteration != iteration:
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
            new_p0_stack, new_p1_stack = child_stacks(sampled_action)
            return self.cfr(
                p0_cards, p1_cards, community_cards, next_history,
                street, updating_player,
                depth + 1, iteration, starting_pot,
                p0_invested, p1_invested, next_pattern,
                new_p0_stack, new_p1_stack)

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

        expected_value = 0
        t_start = time.time()

        for i in range(iterations):
            actual_iteration = start_iteration + i
            t_iter = time.time()
            p0_cards, p1_cards, community_cards = self.deal_random_hand()

            ca = self.game_adapter.card_abstractions
            self._p0_preflop = ca.get_bucket(p0_cards, None)
            self._p1_preflop = ca.get_bucket(p1_cards, None)
            self._p0_postflop = {
                1: ca.get_bucket(p0_cards, community_cards[:3]),
                2: ca.get_bucket(p0_cards, community_cards[:4]),
                3: ca.get_bucket(p0_cards, community_cards[:5]),
            }
            self._p1_postflop = {
                1: ca.get_bucket(p1_cards, community_cards[:3]),
                2: ca.get_bucket(p1_cards, community_cards[:4]),
                3: ca.get_bucket(p1_cards, community_cards[:5]),
            }

            updating_player = actual_iteration % 2
            self.game._calc_cache.clear()
            util = self.cfr(
                p0_cards, p1_cards, community_cards, [],
                0, updating_player, 0, actual_iteration, 3)
            expected_value += util

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

                print(f"  iter {actual_iteration + 1:>9,} / {total_target:,} | "
                      f"EV: {expected_value / iters_done:+.5f} | "
                      f"info sets: {len(self.info_sets):>7,} | "
                      f"{iters_per_sec:>6.1f} it/s | "
                      f"iter_ms: {iter_ms:.1f} | "
                      f"elapsed: {elapsed_str} | "
                      f"ETA: {eta_str}")

            if db is not None and (i + 1) % checkpoint_every == 0:
                self.checkpoint_to_db(db, actual_iteration)

        total_elapsed = _format_duration(time.time() - t_start)
        print(f"\nTraining completed in {total_elapsed}.")
        return expected_value / iterations

    def checkpoint_to_db(self, db, iteration):
        db.save_batch(self.info_sets)
        db.set_metadata('total_iterations', iteration + 1)
        db.set_metadata('alpha', self.alpha)
        db.set_metadata('gamma', self.gamma)
        print(f"Checkpoint: {len(self.info_sets)} info sets saved at iteration {iteration + 1}")

    def resume_from_db(self, db):
        self.info_sets = db.load_all_to_memory()
        start_iteration = db.get_metadata('total_iterations', 0)
        print(f"Resumed: {len(self.info_sets)} info sets, continuing from iteration {start_iteration}")
        return start_iteration
