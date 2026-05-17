# backend/bot/src/cfr/blueprint_trainer.py
import random
from .poker_game import PokerGame
from .information_set import InformationSet
from ..bot.game_adapter import GameAdapter

# Precomputed action → betting-pattern character mapping (avoids per-node method call)
_ACTION_CHARS = {
    'check': 'k', 'call': 'c', 'fold': 'f',
    'bet_small': 's', 'bet_medium': 'm', 'bet_large': 'l',
    'raise_small': 's', 'raise_medium': 'm', 'raise_large': 'l',
}
_STREET_NAMES = ['preflop', 'flop', 'turn', 'river']


def _format_duration(seconds):
    """Convert seconds to a human-readable h:mm:ss string."""
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
    Blueprint CFR Trainer - focused on CFR algorithm like my Leduc Trainer
    Uses PokerGame for training simulation (separate from PyPokerEngine)
    """

    def __init__(self):
        self.info_sets = {}  # Like Leduc trainer
        self.game = PokerGame()  # My own game logic for training
        self.game_adapter = GameAdapter()  # For creating info set keys

        # Create deck for dealing
        self.deck = self.create_deck()
        self.BET_MULTIPLIERS = {'small': 0.33, 'medium': 0.66, 'large': 1.00}

        # DCFR hyperparameters (Brown & Sandholm 2019)
        self.alpha = 1.5  # Regret decay: discounts early noisy regret accumulation
        self.beta = 0.0   # Strategy decay: 0 = standard reach-weighted sum (recommended)

    def create_deck(self):
        """Create standard 52-card deck"""
        suits = ['H', 'D', 'C', 'S']
        ranks = ['2', '3', '4', '5', '6', '7',
                 '8', '9', 'T', 'J', 'Q', 'K', 'A']
        return [suit + rank for rank in ranks for suit in suits]

    def deal_random_hand(self):
        """Deal random cards for training iteration"""
        shuffled_deck = self.deck.copy()
        random.shuffle(shuffled_deck)

        p0_cards = shuffled_deck[0:2]
        p1_cards = shuffled_deck[2:4]
        community_cards = shuffled_deck[4:9]

        return p0_cards, p1_cards, community_cards

    def cfr(self, p0_cards, p1_cards, community_cards, history, p0_reach, p1_reach, street, updating_player, depth=0, iteration=0, starting_pot=None, p0_invested=0.0, p1_invested=0.0, bet_pattern=''):
        """
        Monte Carlo CFR+ with External Sampling
        - Updating player: explores all actions
        - Opponent: samples single action based on strategy

        p0_invested / p1_invested track each player's cumulative chip investment
        across all streets completed before this one, so get_utility can compute
        the correct net gain/loss from P0's perspective.
        """

        # Depth limiting to prevent infinite recursion
        if depth > 50:
            print(
                f"WARNING: Max depth reached at street {street}, history {history}")
            return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                         history, min(street, 3), starting_pot,
                                         p0_invested, p1_invested)

        # Initialize accumulated pot if not provided
        if starting_pot is None:
            starting_pot = 3  # Starting pot: SB(1) + BB(2)

        # Check for terminal states
        if street > 3:
            return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                         history, 3, starting_pot,
                                         p0_invested, p1_invested)

        if self.game.is_terminal(history, street):
            return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                         history, street, starting_pot,
                                         p0_invested, p1_invested)

        # Determine current player (preflop: SB/0 first; postflop: BB/1 first)
        current_player = self.game._acting_player(len(history), street)

        # Get legal actions for current situation
        legal_actions = self.game.get_legal_actions(
            street, history, starting_pot, current_player)

        # If no legal actions, advance to next street or terminal
        if not legal_actions:
            current_pot = self.game.calculate_current_pot(
                starting_pot, history, street)
            if street < 3:
                # Accumulate this street's investments before advancing
                p0_this = self.game.get_player_contribution_this_round(
                    history, street, starting_pot, 0)
                p1_this = self.game.get_player_contribution_this_round(
                    history, street, starting_pot, 1)
                return self.cfr(p0_cards, p1_cards, community_cards, [],
                                p0_reach, p1_reach, street + 1, updating_player,
                                depth + 1, iteration, current_pot,
                                p0_invested + p0_this, p1_invested + p1_this,
                                bet_pattern='')
            else:
                return self.game.get_utility(p0_cards, p1_cards, community_cards,
                                             history, street, starting_pot,
                                             p0_invested, p1_invested)

        # Build info set key directly — avoids dict allocation and O(depth) history scan
        position = 'ip' if current_player == 0 else 'oop'
        if street == 0:
            card_bucket = self._p0_preflop if current_player == 0 else self._p1_preflop
            info_set_key = f"{card_bucket}_{position}_{bet_pattern}"
        else:
            starting = self._p0_preflop if current_player == 0 else self._p1_preflop
            strength = (self._p0_postflop[street] if current_player == 0
                        else self._p1_postflop[street])
            info_set_key = f"{starting}_{strength}_{position}_{_STREET_NAMES[street]}_{bet_pattern}"

        # Get or create information set
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = InformationSet()

        info_set = self.info_sets[info_set_key]

        # Update visit tracking
        if info_set.last_visited_iteration != iteration:
            info_set.visit_count += 1
            info_set.last_visited_iteration = iteration

        # Get current strategy
        reach_prob = p0_reach if current_player == 0 else p1_reach
        strategy = info_set.get_strategy(legal_actions, reach_prob, iteration, self.beta)

        if current_player == updating_player:
            # UPDATING PLAYER: Explore all actions.
            # Traverser reach is multiplied by strategy[i] so child info sets accumulate
            # average strategy weighted by π_i × σ(I, a) — the true reach probability.
            action_utilities = {}
            node_utility = 0

            for i, action in enumerate(legal_actions):
                next_history = history + [action]
                next_pattern = bet_pattern + _ACTION_CHARS[action]
                if current_player == 0:
                    action_utilities[action] = -self.cfr(
                        p0_cards, p1_cards, community_cards, next_history,
                        p0_reach * strategy[i], p1_reach, street, updating_player,
                        depth + 1, iteration, starting_pot,
                        p0_invested, p1_invested, next_pattern)
                else:
                    action_utilities[action] = -self.cfr(
                        p0_cards, p1_cards, community_cards, next_history,
                        p0_reach, p1_reach * strategy[i], street, updating_player,
                        depth + 1, iteration, starting_pot,
                        p0_invested, p1_invested, next_pattern)
                node_utility += strategy[i] * action_utilities[action]

            # Update regrets (CFR+ floor + DCFR temporal decay)
            # Decay relative to this info set's own visit count, not the global iteration.
            # An info set discovered late in training shouldn't get near-zero decay.
            t = info_set.visit_count
            regret_decay = ((t - 1) / t) ** self.alpha if t > 1 else 0.0
            opponent_reach = p1_reach if current_player == 0 else p0_reach

            for i, action in enumerate(legal_actions):
                regret = action_utilities[action] - node_utility
                prior = info_set.cumulative_regrets.get(action, 0)
                info_set.cumulative_regrets[action] = max(
                    0, regret_decay * prior + opponent_reach * regret)

            return node_utility

        else:
            # OPPONENT: Sample single action based on strategy
            sampled_action = random.choices(legal_actions, weights=strategy)[0]
            sampled_prob = strategy[legal_actions.index(sampled_action)]
            next_history = history + [sampled_action]
            next_pattern = bet_pattern + _ACTION_CHARS[sampled_action]

            if current_player == 0:
                return -self.cfr(
                    p0_cards, p1_cards, community_cards, next_history,
                    p0_reach * sampled_prob, p1_reach, street, updating_player,
                    depth + 1, iteration, starting_pot,
                    p0_invested, p1_invested, next_pattern)
            else:
                return -self.cfr(
                    p0_cards, p1_cards, community_cards, next_history,
                    p0_reach, p1_reach * sampled_prob, street, updating_player,
                    depth + 1, iteration, starting_pot,
                    p0_invested, p1_invested, next_pattern)

    def train_blueprint(self, iterations, db=None, start_iteration=0, checkpoint_every=10000):
        """Main training loop. Pass db to enable periodic checkpointing and resume support."""
        import time

        LOG_EVERY = 1

        total_target = start_iteration + iterations
        print(f"Starting blueprint CFR training")
        print(f"  Target iterations : {total_target:,}")
        print(f"  Starting from     : {start_iteration:,}")
        print(f"  Remaining         : {iterations:,}")
        print(f"  Checkpoint every  : {checkpoint_every:,}")
        print()

        expected_value = 0
        t_start = time.time()

        for i in range(iterations):
            actual_iteration = start_iteration + i
            t_iter = time.time()
            p0_cards, p1_cards, community_cards = self.deal_random_hand()

            # Precompute card buckets once per deal — cards never change within an iteration
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
            util = self.cfr(p0_cards, p1_cards,
                            community_cards, [], 1.0, 1.0, 0, updating_player, 0, actual_iteration, 3)
            print(f"  util raw: {util:.2f}") # TEMP
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
        """Write all current info sets to SQLite and record progress."""
        db.save_batch(self.info_sets)
        db.set_metadata('total_iterations', iteration + 1)
        db.set_metadata('alpha', self.alpha)
        db.set_metadata('beta', self.beta)
        print(f"Checkpoint: {len(self.info_sets)} info sets saved at iteration {iteration + 1}")

    def resume_from_db(self, db):
        """Load info sets from SQLite and return the iteration to start from."""
        self.info_sets = db.load_all_to_memory()
        start_iteration = db.get_metadata('total_iterations', 0)
        print(f"Resumed: {len(self.info_sets)} info sets, continuing from iteration {start_iteration}")
        return start_iteration

