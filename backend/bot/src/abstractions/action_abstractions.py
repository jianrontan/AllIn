# backend/bot/abstractions/action_abstractions.py
from .sizing import preflop_open_chips, PREFLOP_RAISE_MULT, POSTFLOP_BET_MULT


class ActionAbstraction:
    """
    Maps between PyPokerEngine actions and the CFR action vocabulary
    (check/call/fold + small/medium/large bet sizing + allin).
    """

    def abstract_action_history(self, pypoker_actions, game_state, street='preflop'):
        """Convert PyPokerEngine action history to simple format."""
        abstracted_history = ""

        for action in pypoker_actions:
            if action['action'] == 'fold':
                abstracted_history += 'f'
            elif action['action'] == 'call':
                abstracted_history += 'c'
            elif action['action'] in ['bet', 'raise']:
                bet_category = self.categorize_bet_size(
                    action, game_state, pypoker_actions, street)
                abstracted_history += bet_category[0]
            elif action['action'] == 'check':
                abstracted_history += 'k'

        return abstracted_history

    def categorize_bet_size(self, action, game_state, action_history=None, street='preflop'):
        """
        Determine if bet is small/medium/large/allin.
        Thresholds aligned with training (poker_game.py _apply_stack_constraints).
        """
        bet_amount = action.get('amount', 0)
        pot_size = game_state.get('pot_size', 1)
        big_blind = game_state.get('big_blind', 2)
        player_stack = game_state.get('player_stack', float('inf'))

        # Remaining stack = player_stack (already reflects current holdings)
        # All-in: bet consumes (nearly) all remaining chips
        if player_stack < float('inf') and bet_amount >= player_stack * 0.95:
            return 'allin'

        # `pot_size` from the caller is the pot BEFORE this bet (pre-bet pot).
        pre_bet_pot_size = pot_size if pot_size > 0 else 1

        # Classify by nearest abstraction size, with boundaries at the midpoints
        # of the sizes in abstractions/sizing.py (so this can't drift from the
        # trained grid). NOTE: this opponent-bet categorisation is the
        # PyPokerEngine path; the LIVE bot maps off-grid bets via
        # cfr/translation.py (pseudo-harmonic), not this function.
        def _bucket(value, sizes):
            # Nearest size by value. Handles the 4-size sets (xlarge open / overbet);
            # the size-name's first letter is its pattern char (s/m/l/o/x), which the
            # caller uses via bet_category[0].
            return min(sizes.items(), key=lambda kv: abs(kv[1] - value))[0]

        if street == 'preflop':
            bet_raise_count = sum(
                1 for a in (action_history or [])
                if a.get('action', '').upper() in ['BET', 'RAISE'])
            if bet_raise_count == 0:                          # open: absolute chips
                return _bucket(bet_amount, preflop_open_chips())
            # 3-bet / 4-bet+: pot-relative (fraction of the pre-bet pot, approx).
            return _bucket(bet_amount / pre_bet_pot_size, PREFLOP_RAISE_MULT)
        return _bucket(bet_amount / pre_bet_pot_size, POSTFLOP_BET_MULT)

    def pypoker_to_cfr_actions(self, pypoker_valid_actions, game_state, round_state):
        """
        Convert PyPokerEngine valid_actions into the CFR action set, mirroring
        the training-side legal-action rules (poker_game.get_legal_actions).

        For each sized bet/raise the EXACT chip cost is computed (via the same
        training sizing in _calculate_target_amount). A size is offered only
        when it is a legal raise (>= min) and affordable; an unaffordable size
        collapses to 'allin'. Using exact costs — not a pot-multiple heuristic —
        keeps inference consistent with training.
        """
        player_remaining = game_state.get('player_stack', float('inf'))
        player_contribution = game_state.get('player_contribution', 0)

        cfr_actions = []
        for action_info in pypoker_valid_actions:
            action_type = action_info['action']

            if action_type in ('fold', 'call', 'check'):
                cfr_actions.append(action_type)

            elif action_type in ('bet', 'raise'):
                amount = action_info.get('amount')
                if isinstance(amount, dict):
                    min_amt, max_amt = amount.get('min'), amount.get('max')
                else:
                    min_amt = max_amt = amount
                # PyPokerEngine signals "raise not possible" with min == -1
                if min_amt is None or min_amt < 0:
                    continue

                needs_allin = False
                added_sized = False
                # Use the SAME size set + action-name kind as the engine: preflop
                # OPEN -> bet_* on the 4-size BB ladder (incl xlarge); preflop
                # 3-bet/4-bet -> raise_* (3 sizes); postflop -> bet_/raise_ incl
                # overbet. (PyPokerEngine reports an open as a 'raise' over the BB,
                # but the blueprint stores opens as bet_*, so force the kind here.)
                street_name = round_state.get('street', 'preflop')
                if street_name == 'preflop':
                    ah = round_state.get('action_histories', {}).get('preflop', [])
                    br = sum(1 for a in ah if a.get('action', '').upper() in ('BET', 'RAISE'))
                    if br == 0:
                        kind, size_names = 'bet', list(preflop_open_chips())
                    else:
                        kind, size_names = 'raise', ['small', 'medium', 'large']
                else:
                    kind, size_names = action_type, list(POSTFLOP_BET_MULT)
                for size_name in size_names:
                    target = self._calculate_target_amount(
                        size_name, kind, game_state, round_state)
                    cost = target - player_contribution  # chips added now
                    if cost <= 0 or target < min_amt:
                        # below the minimum legal raise — not a distinct size
                        continue
                    if cost >= player_remaining or (max_amt is not None and target >= max_amt):
                        needs_allin = True
                    else:
                        cfr_actions.append(f"{kind}_{size_name}")
                        added_sized = True

                # Short-stack safety net: PyPokerEngine offered a legal raise,
                # but no abstract size fits — the player can still shove.
                if not added_sized and not needs_allin:
                    needs_allin = True

                if needs_allin:
                    cfr_actions.append('allin')

        return list(dict.fromkeys(cfr_actions))

    def cfr_to_pypoker_action(self, cfr_action, valid_actions, round_state, game_state):
        """Convert CFR action back to PyPokerEngine format."""
        if cfr_action == 'fold':
            return 'fold', 0

        elif cfr_action == 'check':
            return 'check', 0

        elif cfr_action == 'call':
            for action in valid_actions:
                if action['action'] == 'call':
                    return 'call', action['amount']
            return 'call', 0

        elif cfr_action == 'allin':
            # Go all-in: use maximum valid raise/bet amount
            for action in valid_actions:
                if action['action'] in ('raise', 'bet'):
                    if isinstance(action.get('amount'), dict):
                        return action['action'], int(action['amount']['max'])
            # Fallback: call if can't bet/raise
            for action in valid_actions:
                if action['action'] == 'call':
                    return 'call', action['amount']
            return 'check', 0

        elif cfr_action.startswith('bet_') or cfr_action.startswith('raise_'):
            action_type = 'raise' if cfr_action.startswith('raise_') else 'bet'
            size_name = cfr_action.split('_')[1]

            for action in valid_actions:
                if action['action'] == action_type:
                    if isinstance(action.get('amount'), dict):
                        min_amt = action['amount']['min']
                        max_amt = action['amount']['max']
                        target_amount = self._calculate_target_amount(
                            size_name, action_type, game_state, round_state)
                        final_amount = max(min_amt, min(target_amount, max_amt))
                        return action_type, int(final_amount)
                    else:
                        return action_type, action['amount']

            for action in valid_actions:
                if action['action'] == 'call':
                    return 'call', action['amount']
            return 'check', 0

        return 'check', 0

    def _calculate_target_amount(self, size_name, action_type, game_state, round_state):
        """Calculate target bet/raise amount following training structure."""
        pot_size = game_state.get('pot_size', 3)
        current_bet = game_state.get('current_bet', 0)
        player_contrib = game_state.get('player_contribution', 0)
        big_blind = game_state.get('big_blind', 2)
        street = round_state.get('street', 'preflop')

        # Sizes from abstractions/sizing.py (single source of truth).
        if street == 'preflop':
            action_history = round_state.get('action_histories', {}).get('preflop', [])
            bet_raise_count = sum(1 for a in action_history
                                  if a.get('action', '').upper() in ['BET', 'RAISE'])
            if bet_raise_count == 0:                       # open: absolute BB ladder
                return preflop_open_chips()[size_name]
            multipliers = PREFLOP_RAISE_MULT               # 3-bet / 4-bet+: pot-relative
        else:
            multipliers = POSTFLOP_BET_MULT
        if action_type == 'raise':
            to_call = current_bet - player_contrib
            pot_after_call = pot_size + to_call
            return multipliers[size_name] * pot_after_call + to_call
        return multipliers[size_name] * pot_size
