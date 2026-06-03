# backend/bot/src/cfr/poker_game.py
from ..abstractions.hand_evaluator import HandEvaluator
from ..abstractions.sizing import preflop_open_chips, PREFLOP_RAISE_MULT, POSTFLOP_BET_MULT

STARTING_STACK = 200


# ----------------------------------------------------------------------
# Custom (unrestricted) bet/raise support
# ----------------------------------------------------------------------
# A human may bet an arbitrary, off-grid chip amount. Such an action is encoded
# as 'bet_custom_<total>' / 'raise_custom_<total>', where <total> is the player's
# raise-to TOTAL street commitment after the action (the same quantity the sized
# bet/raise branches produce). This keeps `history` a flat list of strings — the
# engine can still recover every chip amount by name — so the chip-conservation
# fuzz / property tests stay valid. The bot's own actions remain on the abstract
# grid; only the human can produce a custom action. Off-grid amounts are mapped
# onto the trained grid for blueprint lookup via cfr/translation.py.

def _is_custom(action):
    return isinstance(action, str) and '_custom_' in action


def _custom_total(action):
    """Raise-to street total encoded in a custom action ('bet_custom_37.5' -> 37.5)."""
    return float(action.rsplit('_', 1)[1])


def make_custom_action(is_raise, total):
    """Build the canonical custom action string for a raise-to `total`."""
    return f"{'raise' if is_raise else 'bet'}_custom_{total:g}"


class PokerGame:
    """
    Simplified poker game for CFR training - like LeducPoker class
    This is separate from PyPokerEngine gameplay
    """

    def __init__(self, postflop_menu=None, voluntary_allin=True,
                 max_raises_per_street=2):
        """
        max_raises_per_street : raises allowed per street beyond the opening bet
                          (default 2 → 1 bet + 2 raises = 3 aggressions, the cap the
                          blueprint is TRAINED under — keep this default everywhere in
                          training/eval). LIVE play (`GameSession`) passes
                          ``float('inf')`` to UNCAP re-raises so a human can 5-bet/
                          6-bet+ any amount on any street; the bound is then the stack
                          (aggression closes when a tier clamps to all-in). The bot
                          stays within its trained tree automatically — it has ~0
                          average-strategy mass on beyond-cap raises so it won't
                          propose them; a faced all-in 5-bet is answered by the
                          near-terminal equity guard (`RiverSubgameSolver._facing_allin_guard`),
                          and a faced NON-jam deep raise (money behind, no trained key)
                          falls to the blueprint/translation stopgap until the
                          depth-limited deep-raise solver (Phase 4) lands. inf works
                          unchanged in every `>= cap+1` / `< cap` comparison below.
        postflop_menu   : the postflop bet/raise size dict (name -> pot fraction).
                          Default None -> POSTFLOP_BET_MULT (the current 4-size menu;
                          control arm, byte-identical to before). Pass
                          sizing.POSTFLOP_BET_MULT_CAPPED for the Fix-#4 capped menu.
        voluntary_allin : when True (default) every aggression node offers 'allin' as
                          a free-standing action (current behaviour). When False
                          (Fix #4) the voluntary anchor is suppressed and all-in only
                          EMERGES when a sized tier clamps to the stack -- the
                          proposal/response split. The emergent shove still maps to
                          char 'a' (key consistency: two physically identical all-ins
                          share one key regardless of which tier produced them).
        Both are an ABSTRACTION choice: a blueprint trained under one menu/flag is
        incompatible with another. They are constructor args (not the sizing SOT) so
        the control and capped arms can be trained from the same code for the C
        measurement.
        """
        self.streets = ['preflop', 'flop', 'turn', 'river']
        # Default 2 = 1 bet + 2 raises = 3 aggressions (the trained cap). LIVE play
        # passes float('inf') to uncap re-raises (see the constructor docstring).
        self.max_raises_per_street = max_raises_per_street
        self.hand_evaluator = HandEvaluator()
        self.BET_MULTIPLIERS = dict(postflop_menu if postflop_menu is not None
                                    else POSTFLOP_BET_MULT)
        self.voluntary_allin = voluntary_allin
        self._calc_cache = {}

    def _acting_player(self, action_index, street):
        """Preflop: SB (0) acts first. Postflop: BB (1) acts first (they are OOP)."""
        offset = 1 if street > 0 else 0
        return (action_index + offset) & 1     # & 1 == % 2 for non-negative ints

    # ------------------------------------------------------------------
    # Legal action generation
    # ------------------------------------------------------------------

    def get_legal_actions(self, street, history, starting_pot, current_player,
                          p0_stack=None, p1_stack=None, p0_prev=0.0, p1_prev=0.0):
        """Generate legal actions, respecting stack sizes."""
        key = ('legal', street, tuple(history), starting_pot, current_player,
               p0_stack, p1_stack, p0_prev, p1_prev)
        if key in self._calc_cache:
            return self._calc_cache[key]

        if self.is_round_complete(history):
            result = []
        elif 'fold' in history:
            result = []
        else:
            bet_and_raise_count = sum(1 for a in history if a.startswith(('bet_', 'raise_')))
            if 'allin' in history:
                # After allin only fold/call are available
                result = ['fold', 'call']
            elif bet_and_raise_count >= self.max_raises_per_street + 1:
                result = ['fold', 'call']
            elif street == 0:
                result = self.get_preflop_legal_actions(
                    street, history, starting_pot, current_player,
                    p0_stack, p1_stack, p0_prev, p1_prev)
            else:
                result = self.get_postflop_legal_actions(
                    street, history, starting_pot, current_player,
                    p0_stack, p1_stack, p0_prev, p1_prev)

        self._calc_cache[key] = result
        return result

    def _action_cost(self, action, street, history, starting_pot,
                     current_player, p0_prev=0.0, p1_prev=0.0):
        """Chips `current_player` must put in to take `action` given `history`."""
        if action in ('check', 'fold'):
            return 0.0
        if action == 'call':
            return self.get_call_amount_from_history(
                street, history, starting_pot, p0_prev, p1_prev)
        if action == 'allin':
            return self._allin_amount(
                history, street, starting_pot, current_player, p0_prev, p1_prev)
        if action.startswith('bet_'):
            return self.calculate_bet_amount(
                action, street, starting_pot, history, p0_prev, p1_prev)
        if action.startswith('raise_'):
            return self.calculate_raise_amount(
                action, street, starting_pot, history, len(history), p0_prev, p1_prev)
        return 0.0

    def custom_bet_bounds(self, street, history, starting_pot, current_player,
                          p0_stack, p1_stack, p0_prev=0.0, p1_prev=0.0):
        """
        Legal raise-to TOTAL (this street) range [min_total, max_total) for an
        unrestricted custom bet/raise by `current_player`, or None when no custom
        bet/raise is legal here (aggression closed, or only an all-in fits).

        min_total = min legal bet (BB) or min legal raise; max_total = the all-in
        street total. A request equal to max_total is an all-in and must be sent
        as 'allin', not a custom action. All quantities are CHIPS.
        """
        legal = self.get_legal_actions(
            street, history, starting_pot, current_player,
            p0_stack, p1_stack, p0_prev, p1_prev)
        # Aggression must be available at this node (a sized bet/raise survived,
        # or the only aggressive option is the shove).
        if not any(a.startswith(('bet_', 'raise_')) for a in legal) and 'allin' not in legal:
            return None

        contribution = self.get_player_contribution_this_round(
            history, street, starting_pot, current_player, p0_prev, p1_prev)
        remaining = p0_stack if current_player == 0 else p1_stack
        max_total = contribution + remaining        # all-in street total

        to_call = self.get_call_amount_from_history(
            street, history, starting_pot, p0_prev, p1_prev)
        if to_call > 0:
            min_total = self.get_min_raise(street, history, starting_pot, p0_prev, p1_prev)
        else:
            min_total = contribution + 2.0          # min bet = 1 BB

        if min_total >= max_total:
            return None                              # only an all-in fits
        return (float(min_total), float(max_total))

    def _apply_stack_constraints(self, actions, player_remaining,
                                 street, history, starting_pot,
                                 current_player, p0_prev=0.0, p1_prev=0.0):
        """
        Replace any sized bet/raise the player cannot afford with 'allin'.

        A sized bet/raise whose exact chip cost meets or exceeds the player's
        remaining stack is not a distinct action — it is simply an all-in. Using
        the exact cost (not a pot-multiple heuristic) guarantees no action ever
        commits more chips than the player actually has.
        """
        if player_remaining is None:
            return actions

        needs_allin = False
        filtered = []

        for action in actions:
            if not (action.startswith('bet_') or action.startswith('raise_')):
                filtered.append(action)
                continue

            cost = self._action_cost(action, street, history, starting_pot,
                                     current_player, p0_prev, p1_prev)
            if cost >= player_remaining:
                needs_allin = True
            else:
                filtered.append(action)

        # Voluntary all-in (the all-in ANCHOR) + short-stack shove safety net.
        # Any node that reaches this function permits aggression, so the player may
        # always SHOVE, provided the shove is a genuine raise (commits strictly more
        # than a call). This covers two cases at once:
        #   * jamming over an affordable sized bet/raise (small/large/overbet) — a
        #     distinct action, so the blueprint learns jams and carries shove mass;
        #   * the old safety net (every sized raise fell below the min legal raise,
        #     so none survived) — the player can still go all-in.
        # The 3-aggression cap is enforced upstream (get_legal_actions counts only
        # bet_/raise_, and 'allin in history' closes betting), and all-in is not a
        # bet_/raise_, so offering it here never exceeds the cap.
        #
        # Fix #4: when voluntary_allin is False the FREE-STANDING anchor is
        # suppressed -- all-in is NOT offered just because a shove would be a legal
        # raise. It is still forced when a sized tier CLAMPS (cost >= remaining, the
        # `needs_allin` set in the loop above), so a low-SPR jam still emerges; only
        # the high-SPR "jam into a small pot" stray option goes away.
        if not needs_allin and self.voluntary_allin:
            call_amount = self.get_call_amount_from_history(
                street, history, starting_pot, p0_prev, p1_prev)
            allin_amount = self._allin_amount(
                history, street, starting_pot, current_player, p0_prev, p1_prev)
            if allin_amount > call_amount:
                needs_allin = True

        if needs_allin and 'allin' not in filtered:
            # Insert allin before the first bet/raise in the list, or at end
            insert_at = next(
                (i for i, a in enumerate(filtered) if a.startswith(('bet_', 'raise_'))),
                len(filtered)
            )
            filtered.insert(insert_at, 'allin')

        return filtered

    def get_preflop_legal_actions(self, street, history, starting_pot, current_player,
                                  p0_stack=None, p1_stack=None, p0_prev=0.0, p1_prev=0.0):
        """Preflop actions with pot calculation"""

        raise_count = sum(1 for a in history if a.startswith('raise_'))
        player_remaining = (p0_stack if current_player == 0 else p1_stack)

        if not history:
            actions = ['fold', 'call', 'bet_small', 'bet_medium', 'bet_large', 'bet_xlarge']
            return self._apply_stack_constraints(
                actions, player_remaining, street, history, starting_pot,
                current_player, p0_prev, p1_prev)

        # SB limped (call): BB may check or raise (a raise over the BB is modelled
        # as bet_*, since the blinds are already posted).
        if len(history) == 1 and history[0] == 'call':
            actions = ['check', 'bet_small', 'bet_medium', 'bet_large', 'bet_xlarge']
            return self._apply_stack_constraints(
                actions, player_remaining, street, history, starting_pot,
                current_player, p0_prev, p1_prev)

        # Everything else facing a bet/raise (3-bet, 4-bet, limp-then-raise) is
        # handled uniformly by the pot-relative branch below -- there is no
        # special absolute-3-bet case, so the sizes scale with the open and never
        # collapse below the minimum legal raise.
        last_action = history[-1]

        if last_action == 'check':
            actions = ['check', 'bet_small', 'bet_medium', 'bet_large', 'bet_xlarge']
            return self._apply_stack_constraints(
                actions, player_remaining, street, history, starting_pot,
                current_player, p0_prev, p1_prev)

        elif last_action.startswith(('bet_', 'raise_')):
            actions = ['fold', 'call']
            if raise_count < self.max_raises_per_street:
                min_raise = self.get_min_raise(street, history, starting_pot, p0_prev, p1_prev)
                pot_now = self.calculate_current_pot(starting_pot, history, street, p0_prev, p1_prev)
                call_amount = self.get_call_amount_from_history(
                    street, history, starting_pot, p0_prev, p1_prev)
                pot_after_call = pot_now + call_amount
                preflop_multipliers = self.get_preflop_bet_amounts('pot_relative', pot_after_call)

                for size_name in ['small', 'medium', 'large']:
                    raise_amount = preflop_multipliers[size_name] + call_amount
                    if raise_amount >= min_raise:
                        actions.append(f'raise_{size_name}')
            return self._apply_stack_constraints(
                actions, player_remaining, street, history, starting_pot,
                current_player, p0_prev, p1_prev)

        elif last_action == 'call':
            return []

        return ['fold', 'call']

    def get_postflop_legal_actions(self, street, history, starting_pot, current_player,
                                   p0_stack=None, p1_stack=None, p0_prev=0.0, p1_prev=0.0):
        """Postflop actions with pot calculation"""

        current_pot = self.calculate_current_pot(starting_pot, history, street, p0_prev, p1_prev)
        player_remaining = (p0_stack if current_player == 0 else p1_stack)

        if not history:
            actions = ['check']
            for size_name, multiplier in self.BET_MULTIPLIERS.items():
                if multiplier * current_pot >= 2:
                    actions.append(f'bet_{size_name}')
            return self._apply_stack_constraints(
                actions, player_remaining, street, history, starting_pot,
                current_player, p0_prev, p1_prev)

        if len(history) >= 2 and history[-2:] == ['check', 'check']:
            return []

        last_action = history[-1]

        if last_action == 'check':
            actions = ['check']
            for size_name, multiplier in self.BET_MULTIPLIERS.items():
                if multiplier * current_pot >= 2:
                    actions.append(f'bet_{size_name}')
            return self._apply_stack_constraints(
                actions, player_remaining, street, history, starting_pot,
                current_player, p0_prev, p1_prev)

        elif last_action.startswith('bet_') or last_action.startswith('raise_'):
            actions = ['fold', 'call']
            raise_count = sum(1 for a in history if a.startswith('raise_'))
            if raise_count < self.max_raises_per_street:
                min_raise = self.get_min_raise(street, history, starting_pot, p0_prev, p1_prev)
                call_amount = self.get_call_amount_from_history(
                    street, history, starting_pot, p0_prev, p1_prev)
                pot_after_call = current_pot + call_amount

                for size_name, multiplier in self.BET_MULTIPLIERS.items():
                    raise_amount = multiplier * pot_after_call + call_amount
                    if raise_amount >= min_raise:
                        actions.append(f'raise_{size_name}')
            return self._apply_stack_constraints(
                actions, player_remaining, street, history, starting_pot,
                current_player, p0_prev, p1_prev)

        elif last_action == 'call':
            return []

        return ['fold', 'call']

    # ------------------------------------------------------------------
    # Terminal / round-complete checks
    # ------------------------------------------------------------------

    def is_round_complete(self, history):
        """Check if betting round is complete."""
        if not history:
            return False
        if 'fold' in history:
            return True
        # Preflop limp: SB calls BB, then BB checks to end preflop
        if len(history) >= 2 and history[-2:] == ['call', 'check']:
            return True
        # Both checked
        if len(history) >= 2 and history[-2:] == ['check', 'check']:
            return True
        # Bet/raise followed by call
        if len(history) >= 2 and history[-1] == 'call':
            prev_action = history[-2]
            if prev_action.startswith(('bet_', 'raise_')):
                return True
        # Allin followed by call or fold
        if len(history) >= 2 and history[-2] == 'allin' and history[-1] in ('call', 'fold'):
            return True
        return False

    def is_terminal(self, history, street):
        """Check if game is completely over."""
        if 'fold' in history:
            return True
        # Allin called = showdown (no more streets needed)
        if len(history) >= 2 and history[-2] == 'allin' and history[-1] == 'call':
            return True
        if street == 3 and self.is_round_complete(history):
            return True
        return False

    # ------------------------------------------------------------------
    # Pot / contribution calculations (now handle 'allin')
    # ------------------------------------------------------------------

    def _allin_amount(self, history_before, street, starting_pot, player, p0_prev, p1_prev):
        """Compute how many chips 'player' puts in when going allin."""
        prev_invested = p0_prev if player == 0 else p1_prev
        contrib_before = self.get_player_contribution_this_round(
            history_before, street, starting_pot, player, p0_prev, p1_prev)
        remaining = STARTING_STACK - prev_invested - contrib_before
        return max(0.0, remaining)

    def calculate_current_pot(self, starting_pot, history, street, p0_prev=0.0, p1_prev=0.0):
        """
        Central function to calculate current pot size from street start and history.
        Memoized: same inputs always produce the same result.
        """
        key = ('pot', starting_pot, tuple(history), street, p0_prev, p1_prev)
        cached = self._calc_cache.get(key)
        if cached is not None:
            return cached

        current_pot = starting_pot
        for i, action in enumerate(history):
            if action in ['check', 'fold']:
                continue
            elif action == 'allin':
                player = self._acting_player(i, street)
                amount = self._allin_amount(history[:i], street, starting_pot, player, p0_prev, p1_prev)
                current_pot += amount
            elif action == 'call':
                current_pot += self.get_call_amount_from_history(
                    street, history[:i], starting_pot, p0_prev, p1_prev)
            elif action.startswith('bet_'):
                current_pot += self.calculate_bet_amount(
                    action, street, starting_pot, history[:i], p0_prev, p1_prev)
            elif action.startswith('raise_'):
                current_pot += self.calculate_raise_amount(
                    action, street, starting_pot, history[:i], i, p0_prev, p1_prev)

        self._calc_cache[key] = current_pot
        return current_pot

    def calculate_bet_amount(self, action, street, starting_pot, history_before,
                             p0_prev=0.0, p1_prev=0.0):
        """Calculate the actual bet amount for bet actions"""
        current_player = self._acting_player(len(history_before), street)
        if _is_custom(action):
            current_contribution = self.get_player_contribution_this_round(
                history_before, street, starting_pot, current_player, p0_prev, p1_prev)
            return _custom_total(action) - current_contribution
        size = action.split('_')[1]
        if street == 0:
            action_type = self.get_preflop_action_type(history_before)
            bet_amounts = self.get_preflop_bet_amounts(action_type, starting_pot)
            target_amount = bet_amounts[size]
            current_contribution = self.get_player_contribution_this_round(
                history_before, street, starting_pot, current_player, p0_prev, p1_prev)
            return target_amount - current_contribution
        else:
            pot_before_bet = self.calculate_current_pot(
                starting_pot, history_before, street, p0_prev, p1_prev)
            return self.BET_MULTIPLIERS[size] * pot_before_bet

    def calculate_raise_amount(self, action, street, starting_pot, history_before, action_index,
                               p0_prev=0.0, p1_prev=0.0):
        """Calculate the additional amount needed for raise actions"""
        current_player = self._acting_player(action_index, street)

        if _is_custom(action):
            current_contribution = self.get_player_contribution_this_round(
                history_before, street, starting_pot, current_player, p0_prev, p1_prev)
            return _custom_total(action) - current_contribution

        size = action.split('_')[1]
        if street == 0:
            action_type = self.get_preflop_action_type(history_before)
            if action_type != 'pot_relative':
                bet_amounts = self.get_preflop_bet_amounts(action_type, starting_pot)
                target_amount = bet_amounts[size]
            else:
                pot_before_raise = self.calculate_current_pot(
                    starting_pot, history_before, street, p0_prev, p1_prev)
                call_amount = self.get_call_amount_from_history(
                    street, history_before, starting_pot, p0_prev, p1_prev)
                pot_after_call = pot_before_raise + call_amount
                preflop_multipliers = self.get_preflop_bet_amounts('pot_relative', pot_after_call)
                target_amount = preflop_multipliers[size] + call_amount
        else:
            pot_before_raise = self.calculate_current_pot(
                starting_pot, history_before, street, p0_prev, p1_prev)
            call_amount = self.get_call_amount_from_history(
                street, history_before, starting_pot, p0_prev, p1_prev)
            pot_after_call = pot_before_raise + call_amount
            target_amount = self.BET_MULTIPLIERS[size] * pot_after_call + call_amount

        current_contribution = self.get_player_contribution_this_round(
            history_before, street, starting_pot, current_player, p0_prev, p1_prev)
        return target_amount - current_contribution

    # ==================================================================
    # Lever A: state-threaded fast path (bit-identical to the history-based
    # functions above; validated exhaustively by tests/test_lever_a_oracle.py).
    # The CFR hot path threads this within-street state down the recursion
    # instead of replaying `history` to re-derive pot/contribution/to-call/legal
    # actions. State dict keys: pot (total), c=[c0,c1] (this-street contribs),
    # bet_to (current match level this street), prev_bet_to (for min-raise),
    # num_br (bet/raise count this street), last_kind/prev_kind
    # ('start'|'check'|'call'|'aggr'|'fold'), allin_seen.
    # ==================================================================

    def init_node_state(self, street, starting_pot):
        base = {'prev_kind': 'none', 'allin_seen': False}
        if street == 0:
            return {'pot': float(starting_pot), 'c': [1.0, 2.0], 'bet_to': 2.0,
                    'prev_bet_to': 0.0, 'num_br': 0, 'last_kind': 'start', **base}
        return {'pot': float(starting_pot), 'c': [0.0, 0.0], 'bet_to': 0.0,
                'prev_bet_to': 0.0, 'num_br': 0, 'last_kind': 'start', **base}

    @staticmethod
    def _ns_round_complete(st):
        # Mirrors is_round_complete: a call closing a bet/raise, check-check, or
        # limp(call)-check ends the street.
        if st['last_kind'] == 'call' and st['num_br'] >= 1:
            return True
        if st['last_kind'] == 'check' and st['prev_kind'] in ('check', 'call'):
            return True
        return False

    @staticmethod
    def _ns_to_call(st, cp):
        return max(0.0, st['bet_to'] - st['c'][cp])

    @staticmethod
    def _ns_min_raise(st):
        # Mirrors _compute_min_raise (preflop seeds the sequence with BB=2; the
        # unified increment formula reproduces every case via prev_bet_to).
        if st['num_br'] == 0:
            return 2.0
        return st['bet_to'] + (st['bet_to'] - st['prev_bet_to'])

    def _ns_sized_total(self, size, street, st, cp):
        """Raise-TO street total for a sized bet/raise (reuses engine sizing)."""
        to_call = self._ns_to_call(st, cp)
        if street == 0:
            if st['num_br'] == 0:
                return self.get_preflop_bet_amounts('open', st['pot'])[size]
            amts = self.get_preflop_bet_amounts('pot_relative', st['pot'] + to_call)
            return amts[size] + to_call
        mult = self.BET_MULTIPLIERS[size]
        if to_call > 0:
            return mult * (st['pot'] + to_call) + to_call
        return mult * st['pot']

    def state_action_cost(self, action, street, st, cp, stack_cp):
        """Chips cp adds for `action` (== _action_cost), from threaded state."""
        if action in ('check', 'fold'):
            return 0.0
        if action == 'call':
            return self._ns_to_call(st, cp)
        if action == 'allin':
            return stack_cp
        if _is_custom(action):
            return _custom_total(action) - st['c'][cp]
        size = action.split('_')[1]
        return self._ns_sized_total(size, street, st, cp) - st['c'][cp]

    def state_legal_actions(self, street, st, cp, stack_cp):
        """Legal actions from threaded state (== get_legal_actions). Top-level
        order matches get_legal_actions: round-complete -> all-in -> aggression
        cap (these return directly) -> normal (with the stack-constraint pass)."""
        if self._ns_round_complete(st):
            return []
        if st['allin_seen']:
            return ['fold', 'call']
        if st['num_br'] >= self.max_raises_per_street + 1:
            return ['fold', 'call']
        lk, num_br = st['last_kind'], st['num_br']
        if street == 0:
            if lk == 'start':
                actions = ['fold', 'call', 'bet_small', 'bet_medium', 'bet_large', 'bet_xlarge']
            elif lk == 'call' and num_br == 0:                 # SB limped -> BB option
                actions = ['check', 'bet_small', 'bet_medium', 'bet_large', 'bet_xlarge']
            else:
                actions = self._ns_fold_call_raises(street, st, cp)
        else:
            if lk in ('start', 'check'):
                actions = ['check']
                for size in self.BET_MULTIPLIERS:             # incl. 'overbet'
                    if self.BET_MULTIPLIERS[size] * st['pot'] >= 2:
                        actions.append(f'bet_{size}')
            else:
                actions = self._ns_fold_call_raises(street, st, cp)
        return self._ns_stack_constrain(actions, street, st, cp, stack_cp)

    def _ns_fold_call_raises(self, street, st, cp):
        actions = ['fold', 'call']
        raise_count = max(0, st['num_br'] - 1)
        if raise_count < self.max_raises_per_street:
            mr = self._ns_min_raise(st)
            # Postflop raises include 'overbet'; preflop 3-bet/4-bet stay 3 sizes
            # (mirrors get_postflop_legal_actions vs get_preflop_legal_actions).
            sizes = self.BET_MULTIPLIERS if street > 0 else ('small', 'medium', 'large')
            for size in sizes:
                if self._ns_sized_total(size, street, st, cp) >= mr:
                    actions.append(f'raise_{size}')
        return actions

    def _ns_stack_constrain(self, actions, street, st, cp, stack_cp):
        needs_allin = False
        filtered = []
        for a in actions:
            if not a.startswith(('bet_', 'raise_')):
                filtered.append(a)
                continue
            if self.state_action_cost(a, street, st, cp, stack_cp) >= stack_cp:
                needs_allin = True
            else:
                filtered.append(a)
        # Voluntary all-in + short-stack safety net (mirror of
        # _apply_stack_constraints): offer the shove whenever it is a genuine raise
        # (more than a call), including over an affordable sized bet (the anchor).
        # Fix #4: gated on voluntary_allin -- when False the free-standing anchor is
        # suppressed; the emergent clamp (needs_allin set in the loop above when a
        # sized cost >= stack) still forces a low-SPR shove. Mirrors
        # _apply_stack_constraints exactly so the two legal-action paths never drift.
        if not needs_allin and self.voluntary_allin and stack_cp > self._ns_to_call(st, cp):
            needs_allin = True
        if needs_allin and 'allin' not in filtered:
            i = next((j for j, a in enumerate(filtered) if a.startswith(('bet_', 'raise_'))),
                     len(filtered))
            filtered.insert(i, 'allin')
        return filtered

    def advance_node_state(self, st, action, street, cp, stack_cp, p_inv):
        """New state after cp takes `action`. p_inv = cross-street prior per
        player (the all-in match level uses the engine's clean STARTING_STACK -
        prev, matching get_call_amount_from_history, to stay bit-identical)."""
        new = {'pot': st['pot'], 'c': list(st['c']), 'bet_to': st['bet_to'],
               'prev_bet_to': st['prev_bet_to'], 'num_br': st['num_br'],
               'last_kind': st['last_kind'], 'prev_kind': st['last_kind'],
               'allin_seen': st['allin_seen'] or action == 'allin'}
        cost = self.state_action_cost(action, street, st, cp, stack_cp)
        new['pot'] += cost
        if action == 'check':
            new['last_kind'] = 'check'
        elif action == 'fold':
            new['last_kind'] = 'fold'
        elif action == 'call':
            new['c'][cp] += cost
            new['last_kind'] = 'call'
        else:                                                  # bet/raise/allin
            new['prev_bet_to'] = st['bet_to']
            if action == 'allin':
                # Engine uses += (_allin_amount); the match level is the clean
                # STARTING_STACK - prev (get_call_amount_from_history's branch).
                new['c'][cp] += cost
                match_level = STARTING_STACK - p_inv[cp]
            else:
                # Engine ASSIGNS the raiser's contribution to the clean raise-to
                # TOTAL (get_player_contribution_this_round). Assigning -- rather
                # than c_before + cost -- avoids a float-associativity ULP that
                # compounds across a multi-raise chain.
                if _is_custom(action):
                    total = _custom_total(action)
                else:
                    total = self._ns_sized_total(action.split('_')[1], street, st, cp)
                new['c'][cp] = total
                match_level = total
            new['bet_to'] = max(st['bet_to'], match_level)
            new['num_br'] += 1
            new['last_kind'] = 'aggr'
        return new

    def _raw_cached(self, cards, board):
        """Memoized 7-card rank for `cards` on `board` (lower = stronger). The two
        showdown ranks are fixed per hand but the same terminal recurs across many
        CFR branches, so caching by (cards, board) avoids recomputing them. Pure
        function of the inputs -> bit-identical. Shares the per-iteration
        _calc_cache (cleared each training iteration with everything else).
        phevaluator scores are >= 1, so None is a safe 'absent' sentinel."""
        key = ('rank', tuple(cards), tuple(board))
        v = self._calc_cache.get(key)
        if v is None:
            v = self.hand_evaluator.get_raw_hand_value(cards, board)
            self._calc_cache[key] = v
        return v

    def get_utility(self, p0_cards, p1_cards, community_cards, history, street, starting_pot,
                    p0_prev_invested=0.0, p1_prev_invested=0.0,
                    _final_pot=None, _p0_total=None):
        """Calculate utility from P0's perspective. `_final_pot` / `_p0_total`
        are optional threaded values (Lever A) -- when given, the final pot and
        P0's total are used directly instead of replaying `history`; bit-identical
        to the replay (validated). Omitted by the eval harness/tests -> replay."""

        if _final_pot is not None:
            final_pot = _final_pot
        else:
            final_pot = self.calculate_current_pot(
                starting_pot, history, street, p0_prev_invested, p1_prev_invested)

        if _p0_total is not None:
            p0_total = _p0_total
        else:
            p0_this = self.get_player_contribution_this_round(
                history, street, starting_pot, 0, p0_prev_invested, p1_prev_invested)
            p0_total = p0_prev_invested + p0_this

        if 'fold' in history:
            folder_index = next(i for i, action in enumerate(history)
                                if action == 'fold')
            folder_player = self._acting_player(folder_index, street)

            if folder_player == 0:
                return -p0_total
            else:
                return final_pot - p0_total

        else:  # Showdown (allin+call or river complete)
            # When allin occurred, run out remaining board cards
            if 'allin' in history:
                community_for_eval = community_cards[:5]
            else:
                community_for_eval = community_cards[:self.get_community_cards_count(street)]

            p0_raw = self._raw_cached(p0_cards, community_for_eval)
            p1_raw = self._raw_cached(p1_cards, community_for_eval)

            if p0_raw < p1_raw:
                return final_pot - p0_total
            elif p1_raw < p0_raw:
                return -p0_total
            else:
                return (final_pot / 2) - p0_total

    def get_community_cards_count(self, street):
        if street < 0:
            return 0
        elif street >= 3:
            return 5
        else:
            return [0, 3, 4, 5][street]

    # ------------------------------------------------------------------
    # Contribution / call helpers
    # ------------------------------------------------------------------

    def get_player_contribution_this_round(self, history, street, starting_pot, current_player,
                                           p0_prev=0.0, p1_prev=0.0):
        """Calculate player's contribution this round. Memoized."""
        key = ('contrib', tuple(history), street, starting_pot, current_player, p0_prev, p1_prev)
        cached = self._calc_cache.get(key)
        if cached is not None:
            return cached

        if street == 0:
            contribution = 1.0 if current_player == 0 else 2.0
        else:
            contribution = 0.0

        for i, action in enumerate(history):
            action_player = self._acting_player(i, street)

            if action_player == current_player:
                if action == 'call':
                    contribution += self.get_call_amount_from_history(
                        street, history[:i], starting_pot, p0_prev, p1_prev)

                elif action == 'allin':
                    amount = self._allin_amount(history[:i], street, starting_pot,
                                                current_player, p0_prev, p1_prev)
                    contribution += amount

                elif action.startswith(('bet_', 'raise_')):
                    if _is_custom(action):
                        contribution = _custom_total(action)
                    elif street == 0:
                        action_type = self.get_preflop_action_type(history[:i])
                        if action_type != 'pot_relative':
                            bet_amounts = self.get_preflop_bet_amounts(action_type, starting_pot)
                            size = action.split('_')[1]
                            contribution = bet_amounts[size]
                        else:
                            pot_before = self.calculate_current_pot(
                                starting_pot, history[:i], street, p0_prev, p1_prev)
                            call_amount = self.get_call_amount_from_history(
                                street, history[:i], starting_pot, p0_prev, p1_prev)
                            pot_after_call = pot_before + call_amount
                            size = action.split('_')[1]
                            preflop_multipliers = self.get_preflop_bet_amounts(
                                'pot_relative', pot_after_call)
                            contribution = preflop_multipliers[size] + call_amount
                    else:
                        pot_before = self.calculate_current_pot(
                            starting_pot, history[:i], street, p0_prev, p1_prev)
                        call_amount = self.get_call_amount_from_history(
                            street, history[:i], starting_pot, p0_prev, p1_prev)
                        pot_after_call = pot_before + call_amount
                        size = action.split('_')[1]
                        # This is the raise-to TOTAL for the street, not an
                        # increment — assign it (matching the preflop branches).
                        # Using += double-counts a player's earlier bet/raise
                        # when they bet then re-raise the same street.
                        contribution = self.BET_MULTIPLIERS[size] * pot_after_call + call_amount

        self._calc_cache[key] = contribution
        return contribution

    def get_call_amount_from_history(self, street, history, starting_pot,
                                     p0_prev=0.0, p1_prev=0.0):
        """
        Return the extra chips the next-to-act player must put in to call.
        `history` is the sequence BEFORE the call. Memoized.
        """
        key = ('call', street, tuple(history), starting_pot, p0_prev, p1_prev)
        cached = self._calc_cache.get(key)
        if cached is not None:
            return cached

        if not history:
            result = 1.0 if street == 0 else 0.0
            self._calc_cache[key] = result
            return result

        current_player = self._acting_player(len(history), street)

        last_bet_amt = 0
        for i in range(len(history) - 1, -1, -1):
            act = history[i]
            if act == 'allin':
                bet_player = self._acting_player(i, street)
                # `last_bet_amt` must be the all-in player's TOTAL street
                # commitment AFTER going all-in (matching the bet/raise branch
                # below, which also produces a raise-to total). An all-in
                # player has no chips left, so their total committed is
                # STARTING_STACK minus their cross-street prior investment.
                # Earlier code used `_allin_amount` (just the chips the all-in
                # added) which is an increment, not a total — produces a 0
                # call cost whenever the caller had already put in more this
                # street than the all-in's increment.
                bet_player_prev = p0_prev if bet_player == 0 else p1_prev
                last_bet_amt = STARTING_STACK - bet_player_prev
                break
            elif act.startswith(('bet_', 'raise_')):
                if _is_custom(act):
                    last_bet_amt = _custom_total(act)
                elif street == 0:
                    action_type = self.get_preflop_action_type(history[:i])
                    if action_type != 'pot_relative':
                        size = act.split('_')[1]
                        last_bet_amt = self.get_preflop_bet_amounts(
                            action_type, starting_pot)[size]
                    else:
                        pot_before = self.calculate_current_pot(
                            starting_pot, history[:i], street, p0_prev, p1_prev)
                        raiser_call = self.get_call_amount_from_history(
                            street, history[:i], starting_pot, p0_prev, p1_prev)
                        pot_after_call = pot_before + raiser_call
                        size = act.split('_')[1]
                        preflop_multipliers = self.get_preflop_bet_amounts(
                            'pot_relative', pot_after_call)
                        last_bet_amt = preflop_multipliers[size] + raiser_call
                else:
                    pot_before = self.calculate_current_pot(
                        starting_pot, history[:i], street, p0_prev, p1_prev)
                    raiser_call = self.get_call_amount_from_history(
                        street, history[:i], starting_pot, p0_prev, p1_prev)
                    pot_after_call = pot_before + raiser_call
                    size = act.split('_')[1]
                    last_bet_amt = self.BET_MULTIPLIERS[size] * pot_after_call + raiser_call
                break

        player_contrib = self.get_player_contribution_this_round(
            history, street, starting_pot, current_player, p0_prev, p1_prev)

        result = max(0.0, last_bet_amt - player_contrib)
        self._calc_cache[key] = result
        return result

    # ------------------------------------------------------------------
    # Min-raise / preflop helpers (unchanged in logic, no allin in these)
    # ------------------------------------------------------------------

    def get_min_raise(self, street, history, starting_pot, p0_prev=0.0, p1_prev=0.0):
        """Calculate minimum raise. Memoized."""
        key = ('minraise', street, tuple(history), starting_pot, p0_prev, p1_prev)
        if key in self._calc_cache:
            return self._calc_cache[key]
        result = self._compute_min_raise(street, history, starting_pot, p0_prev, p1_prev)
        self._calc_cache[key] = result
        return result

    def _compute_min_raise(self, street, history, starting_pot, p0_prev=0.0, p1_prev=0.0):
        if not history:
            return 2.0

        bet_amounts = [2.0] if street == 0 else []

        for i, action in enumerate(history):
            if action.startswith(('bet_', 'raise_')):
                if _is_custom(action):
                    bet_amounts.append(_custom_total(action))
                elif street == 0:
                    action_type = self.get_preflop_action_type(history[:i])
                    if action_type != 'pot_relative':
                        bet_amounts_dict = self.get_preflop_bet_amounts(action_type, starting_pot)
                        size = action.split('_')[1]
                        bet_amounts.append(bet_amounts_dict[size])
                    else:
                        pot_before = self.calculate_current_pot(
                            starting_pot, history[:i], street, p0_prev, p1_prev)
                        call_amount = self.get_call_amount_from_history(
                            street, history[:i], starting_pot, p0_prev, p1_prev)
                        pot_after_call = pot_before + call_amount
                        size = action.split('_')[1]
                        preflop_multipliers = self.get_preflop_bet_amounts('pot_relative', pot_after_call)
                        bet_amounts.append(preflop_multipliers[size] + call_amount)
                else:
                    pot_before = self.calculate_current_pot(
                        starting_pot, history[:i], street, p0_prev, p1_prev)
                    call_amount = self.get_call_amount_from_history(
                        street, history[:i], starting_pot, p0_prev, p1_prev)
                    pot_after_call = pot_before + call_amount
                    size = action.split('_')[1]
                    bet_amounts.append(self.BET_MULTIPLIERS[size] * pot_after_call + call_amount)

        if len(bet_amounts) >= 2:
            min_raise_increment = bet_amounts[-1] - bet_amounts[-2]
            return bet_amounts[-1] + min_raise_increment
        elif len(bet_amounts) == 1:
            return bet_amounts[-1] + bet_amounts[-1]

        return 2.0

    def get_preflop_action_type(self, history):
        # Only the OPEN (first-in raise) is sized absolutely (BB-anchored). Every
        # subsequent raise -- 3-bet, 4-bet, ... -- is pot-relative (unified). The
        # old separate absolute '3bet' tier collapsed below the min-raise vs a
        # large open; pot-relative scales so all sizes stay legal.
        if not history:
            return 'open'
        bet_raise_count = sum(1 for a in history if a.startswith(('bet_', 'raise_')))
        return 'open' if bet_raise_count == 0 else 'pot_relative'

    def get_preflop_bet_amounts(self, action_type, current_pot):
        # Centralised in abstractions/sizing.py (single source of truth).
        if action_type == 'open':
            return preflop_open_chips()                     # raise-TO totals, chips
        # pot_relative (3-bet / 4-bet+): fraction of pot-after-call.
        return {k: m * current_pot for k, m in PREFLOP_RAISE_MULT.items()}
