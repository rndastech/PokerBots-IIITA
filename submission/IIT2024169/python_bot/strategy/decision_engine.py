'''
DecisionEngine — central brain of the bot.

Combines equity, board texture, opponent model, bounty EV, and
Bayesian bounty inference into a single action decision.

Improvements over v1:
  - mc_samples passed in from TimeManager (adaptive computation)
  - 4-tier EHS framework (matching postflop logic to P4-grade structure)
  - Correct bounty pot-odds formula (fixed from v1 bug)
  - Position-aware betting (in-position probes, OOP check-call)
  - Check-raise bluff path vs. ROCK/nit opponents
  - River call discipline (fold weak hands to large river bets)
  - Slowplay gated on opponent type
  - BountyInferencer used to suppress bluffing when opponent is bounty-motivated
  - Turn/river bet sizing escalation
  - Opponent call-shift capped to avoid cancelling bounty advantage
'''
import random
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction


class DecisionEngine:
    def __init__(self, preflop_table, monte_carlo, bounty_ev,
                 opponent_tracker, opponent_classifier, board_analyzer,
                 bounty_inf, mixed_strategy, bet_sizer):
        self.preflop_table  = preflop_table
        self.monte_carlo    = monte_carlo
        self.bounty_ev      = bounty_ev
        self.tracker        = opponent_tracker
        self.classifier     = opponent_classifier
        self.board_analyzer = board_analyzer
        self.bounty_inf     = bounty_inf
        self.mixed_strategy = mixed_strategy
        self.bet_sizer      = bet_sizer

    # =========================================================================
    # PUBLIC ENTRY POINT
    # =========================================================================

    def get_action(self, game_state, round_state, active, mc_samples=150):
        '''
        Main decision function. Called by player.py each action.

        Args:
            game_state:  Engine GameState
            round_state: Engine RoundState
            active:      Our player index (0 or 1)
            mc_samples:  Number of MC simulations (set by TimeManager)

        Returns:
            A legal engine Action object.
        '''
        legal_actions = round_state.legal_actions()
        street        = round_state.street
        my_cards      = round_state.hands[active]
        board_cards   = round_state.deck[:street]

        my_pip    = round_state.pips[active]
        opp_pip   = round_state.pips[1 - active]
        my_stack  = round_state.stacks[active]

        # Total chips committed by both players (= pot)
        effective_pot = 800 - (round_state.stacks[0] + round_state.stacks[1])

        call_cost = opp_pip - my_pip
        can_raise = RaiseAction in legal_actions

        min_raise, max_raise = 0, 0
        if can_raise:
            min_raise, max_raise = round_state.raise_bounds()

        bounty_rank = round_state.bounties[active]

        # ── Position ──────────────────────────────────────────────────────────
        # In this HU engine, player 0 is always the button preflop and acts
        # second on every postflop street, so active == 0 is always IP.
        is_in_position = (active == 0)

        # ── Opponent model ────────────────────────────────────────────────────
        stats       = self.tracker.get_stats()
        archetype   = self.classifier.classify(stats)
        adjustments = self.classifier.get_adjustments(archetype, stats)
        adjustments = self._blend_adjustments(adjustments, stats)

        opp_type          = archetype
        bluff_freq_mult   = adjustments['bluff_freq_mult']
        min_value_equity  = adjustments['min_value_equity']
        call_equity_shift = adjustments['call_equity_shift']     # capped ±0.06 below
        bet_size_mult     = adjustments['bet_size_mult']
        trap_more         = adjustments['trap_more']
        bluff_less        = adjustments['bluff_less']
        fold_adj          = adjustments['fold_to_raise_adjust']

        # Cap call_equity_shift so it can't cancel bounty advantage entirely
        call_equity_shift = max(-0.06, min(0.06, call_equity_shift))

        # ── Bounty inference: is opponent bounty-motivated this board? ────────
        opp_bounty_prob = self.bounty_inf.opponent_bounty_hit_probability(board_cards)
        opp_bounty_motivated = self.bounty_ev.should_suppress_bluff(opp_bounty_prob)

        # ── PREFLOP ───────────────────────────────────────────────────────────
        if street == 0:
            return self._decide_preflop(
                my_cards=my_cards,
                active=active,
                call_cost=call_cost,
                my_pip=my_pip,
                opp_pip=opp_pip,
                my_stack=my_stack,
                min_raise=min_raise,
                max_raise=max_raise,
                can_raise=can_raise,
                bounty_rank=bounty_rank,
                legal_actions=legal_actions,
                adjustments=adjustments,
                effective_pot=effective_pot,
            )

        # ── POST-FLOP EQUITY ──────────────────────────────────────────────────
        if mc_samples == 0 or not board_cards:
            # Emergency / no-board fallback — use preflop equity from table
            equity = self.preflop_table.get_equity(my_cards[0], my_cards[1])
            board_features = self.board_analyzer._empty_board()
        else:
            equity = self.monte_carlo.estimate(my_cards, board_cards,
                                               num_simulations=mc_samples)
            board_features = self.board_analyzer.analyze(my_cards, board_cards)

        # ── Bounty equity boost ───────────────────────────────────────────────
        bounty_boost = self.bounty_ev.equity_boost_for_ongoing(
            my_cards, board_cards, bounty_rank
        )
        equity = min(1.0, equity + bounty_boost)

        # Draws: extra implied-odds bonus
        if board_features.get('flush_draw') and street < 5:
            equity = min(1.0, equity + 0.05)
        elif board_features.get('straight_draw_outs', 0) >= 4 and street < 5:
            equity = min(1.0, equity + 0.03)

        # ── Board signals ─────────────────────────────────────────────────────
        board_wet   = board_features.get('wet_board', False)
        has_draw    = board_features.get('has_draw', False)
        made_strength = self._made_strength(board_features)
        can_stack_off = self._can_stack_off(board_features, street)

        # ── Pot odds — bounty-corrected ───────────────────────────────────────
        if call_cost > 0:
            bounty_pot_odds = self.bounty_ev.adjusted_pot_odds(
                pot=effective_pot,
                call_amount=call_cost,
                my_cards=my_cards,
                board_cards=board_cards,
                bounty_rank=bounty_rank,
            )
        else:
            bounty_pot_odds = 0.0

        # ── Dispatch ──────────────────────────────────────────────────────────
        if call_cost > 0:
            return self._decide_facing_bet(
                equity=equity,
                bounty_pot_odds=bounty_pot_odds,
                effective_pot=effective_pot,
                call_cost=call_cost,
                my_pip=my_pip,
                my_stack=my_stack,
                street=street,
                board_features=board_features,
                board_wet=board_wet,
                made_strength=made_strength,
                can_stack_off=can_stack_off,
                is_in_position=is_in_position,
                min_raise=min_raise,
                max_raise=max_raise,
                legal_actions=legal_actions,
                can_raise=can_raise,
                call_equity_shift=call_equity_shift,
                bluff_freq_mult=bluff_freq_mult,
                bluff_less=bluff_less or opp_bounty_motivated,
                fold_adj=fold_adj,
                trap_more=trap_more,
                opp_type=opp_type,
                round_state=round_state,
                active=active,
            )
        else:
            return self._decide_free_option(
                equity=equity,
                effective_pot=effective_pot,
                my_pip=my_pip,
                my_stack=my_stack,
                street=street,
                board_features=board_features,
                board_wet=board_wet,
                has_draw=has_draw,
                made_strength=made_strength,
                is_in_position=is_in_position,
                min_raise=min_raise,
                max_raise=max_raise,
                legal_actions=legal_actions,
                can_raise=can_raise,
                min_value_equity=min_value_equity,
                bluff_freq_mult=bluff_freq_mult,
                bluff_less=bluff_less or opp_bounty_motivated,
                trap_more=trap_more,
                opp_type=opp_type,
            )

    # =========================================================================
    # PREFLOP
    # =========================================================================

    def _decide_preflop(self, my_cards, active, call_cost, my_pip, opp_pip,
                        my_stack, min_raise, max_raise, can_raise,
                        bounty_rank, legal_actions, adjustments, effective_pot):
        '''Delegate to GTO preflop table.'''
        action_type, amount = self.preflop_table.get_preflop_action(
            my_cards=my_cards,
            my_bounty_rank=bounty_rank,
            is_button=(active == 0),
            continue_cost=call_cost,
            my_pip=my_pip,
            opp_pip=opp_pip,
            my_stack=my_stack,
            min_raise=min_raise,
            max_raise=max_raise,
            pot_size=effective_pot,
            opponent_adjustments=adjustments,
        )
        return self._to_action(action_type, amount, legal_actions,
                               min_raise, max_raise)

    # =========================================================================
    # FACING A BET
    # =========================================================================

    def _decide_facing_bet(self, equity, bounty_pot_odds, effective_pot,
                           call_cost, my_pip, my_stack, street, board_features,
                           board_wet, made_strength, can_stack_off,
                           is_in_position, min_raise, max_raise, legal_actions,
                           can_raise, call_equity_shift, bluff_freq_mult,
                           bluff_less, fold_adj, trap_more, opp_type,
                           round_state, active):
        '''4-tier decision logic when opponent has bet/raised.'''

        # ── POSTFLOP JAM FILTER (from GODBOT) ─────────────────────────────────
        # Facing massive overbets or high stack pressure → only continue with
        # monsters (sets, straights, flushes+). Prevents the #1 leak.
        # Street penalty: opponent bets on later streets are more polarized
        street_penalty = {3: 0.0, 4: 0.08, 5: 0.14}.get(street, 0.0)

        # Needed equity to call profitably (bounty-corrected + opponent shift)
        needed_equity = bounty_pot_odds + call_equity_shift + street_penalty

        # ── Stack pressure ────────────────────────────────────────────────────
        stack_pressure   = call_cost / max(1, my_stack)
        pot_commit_ratio = call_cost / max(1, effective_pot + call_cost)

        # ── Raise-count on this street (cap aggression escalation) ────────────
        raise_count      = self._street_raise_count(round_state)
        facing_raise_back = self._facing_raise_after_we_bet(round_state, active)

        # ── EHS TIER: MONSTER (> 0.80) ────────────────────────────────────────
        if equity > 0.80:
            if can_raise:
                # Slowplay very occasionally vs LAG only (they'll fire again)
                if trap_more and random.random() < 0.25:
                    return CallAction()
                size = self.bet_sizer.get_bet_size(
                    equity=equity, pot=effective_pot + call_cost,
                    min_raise=min_raise, max_raise=max_raise,
                    board_wet=board_wet, street=street,
                    opponent_type=opp_type, my_pip=my_pip,
                )
                return RaiseAction(size)
            return CallAction()

        # ── EHS TIER: STRONG (0.60–0.80) ──────────────────────────────────────
        elif equity > 0.60:
            if can_raise and equity > 0.70:
                raise_freq = 0.45
                if random.random() < raise_freq:
                    size = self.bet_sizer.get_bet_size(
                        equity=equity, pot=effective_pot + call_cost,
                        min_raise=min_raise, max_raise=max_raise,
                        board_wet=board_wet, street=street,
                        opponent_type=opp_type, my_pip=my_pip,
                    )
                    return RaiseAction(size)
            # Fold discipline: if we've been raised back multiple times on rough boards
            if facing_raise_back and not can_stack_off:
                if street >= 4 and stack_pressure >= 0.18:
                    return self._safe_fold(legal_actions)
                if raise_count >= 3:
                    return self._safe_fold(legal_actions)
            if CallAction in legal_actions:
                return CallAction()

        # ── EHS TIER: MEDIUM (0.40–0.60) ──────────────────────────────────────
        elif equity > 0.40:
            # Defensive folds on later streets with weak made hands
            if stack_pressure >= 0.55 and made_strength == 'weak':
                # Allow flush draws to continue cheaply on the flop
                if board_features.get('flush_draw') and equity >= needed_equity + 0.06 and street == 3:
                    return CallAction()
                return self._safe_fold(legal_actions)
            if street >= 4 and made_strength == 'weak' and equity < needed_equity + 0.06:
                return self._safe_fold(legal_actions)
            if facing_raise_back:
                if street == 3 and not can_stack_off and equity < needed_equity + 0.12:
                    return self._safe_fold(legal_actions)
                if street >= 4 and not can_stack_off:
                    if stack_pressure >= 0.16 or raise_count >= 2:
                        return self._safe_fold(legal_actions)
            if raise_count >= 3 and not can_stack_off:
                return self._safe_fold(legal_actions)

            # River call discipline: fold weak hands to large bets
            if street == 5 and made_strength != 'strong':
                if board_features.get('wet_board') or board_features.get('board_paired'):
                    if equity < needed_equity + 0.16:
                        return self._safe_fold(legal_actions)
                if call_cost >= max(16, effective_pot // 3) and equity < 0.50:
                    return self._safe_fold(legal_actions)

            # Value raise with medium+ equity
            if (equity > needed_equity + 0.14 and can_raise and
                    made_strength in ('strong', 'medium') and
                    not facing_raise_back and street == 3):
                size = self.bet_sizer.get_bet_size(
                    equity=equity, pot=effective_pot,
                    min_raise=min_raise, max_raise=max_raise,
                    board_wet=board_wet, street=street,
                    opponent_type=opp_type, my_pip=my_pip,
                )
                noisy = self.mixed_strategy.add_noise_to_sizing(size, min_raise, max_raise)
                return RaiseAction(noisy)

            if equity > needed_equity:
                if CallAction in legal_actions:
                    return CallAction()
            return self._safe_fold(legal_actions)

        # ── EHS TIER: WEAK (≤ 0.40) ───────────────────────────────────────────
        else:
            # Check-raise bluff vs. ROCK/nit on flop only
            if (can_raise and not bluff_less and street == 3 and
                    opp_type == 'ROCK' and equity > 0.22 and
                    pot_commit_ratio < 0.28 and raise_count == 0):
                if random.random() < 0.18:
                    size = self.bet_sizer.get_bluff_size(
                        pot=effective_pot, min_raise=min_raise, max_raise=max_raise,
                        my_pip=my_pip, opponent_type=opp_type,
                    )
                    return RaiseAction(size)

            # Semi-bluff raise on flop draw if pot-committed
            if (can_raise and not bluff_less and street == 3 and
                    board_features.get('has_draw') and equity > 0.30 and
                    pot_commit_ratio < 0.30 and
                    self.mixed_strategy.should_bluff(bounty_pot_odds,
                                                     bluff_freq_mult * 0.5)):
                size = self.bet_sizer.get_bluff_size(
                    pot=effective_pot, min_raise=min_raise, max_raise=max_raise,
                    my_pip=my_pip, opponent_type=opp_type,
                )
                noisy = self.mixed_strategy.add_noise_to_sizing(size, min_raise, max_raise)
                return RaiseAction(noisy)

            return self._safe_fold(legal_actions)

    # =========================================================================
    # FREE OPTION (Check or Bet)
    # =========================================================================

    def _decide_free_option(self, equity, effective_pot, my_pip, my_stack,
                            street, board_features, board_wet, has_draw,
                            made_strength, is_in_position, min_raise, max_raise,
                            legal_actions, can_raise, min_value_equity,
                            bluff_freq_mult, bluff_less, trap_more, opp_type):
        '''4-tier decision logic when no one has bet (check or bet).'''

        # ── MONSTER (> 0.80) ──────────────────────────────────────────────────
        if equity > 0.80:
            if can_raise:
                # Slowplay vs LAG only (they'll bet into us)
                if trap_more and is_in_position and random.random() < 0.30:
                    return CheckAction()
                size = self.bet_sizer.get_bet_size(
                    equity=equity, pot=effective_pot,
                    min_raise=min_raise, max_raise=max_raise,
                    board_wet=board_wet, street=street,
                    opponent_type=opp_type, my_pip=my_pip,
                )
                noisy = self.mixed_strategy.add_noise_to_sizing(size, min_raise, max_raise)
                return RaiseAction(noisy)
            return CheckAction()

        # ── STRONG (0.60–0.80) ────────────────────────────────────────────────
        elif equity > 0.60:
            if can_raise:
                bet_freq = 0.68 + (0.08 if opp_type == 'CALLING_STATION' else 0.0)
                if random.random() < bet_freq:
                    size = self.bet_sizer.get_bet_size(
                        equity=equity, pot=effective_pot,
                        min_raise=min_raise, max_raise=max_raise,
                        board_wet=board_wet, street=street,
                        opponent_type=opp_type, my_pip=my_pip,
                    )
                    noisy = self.mixed_strategy.add_noise_to_sizing(size, min_raise, max_raise)
                    return RaiseAction(noisy)
            return CheckAction()

        # ── MEDIUM (0.40–0.60) ────────────────────────────────────────────────
        elif equity > 0.40 + min_value_equity:
            if can_raise:
                # Probe when in-position; check-call when OOP
                base_freq = 0.32 if is_in_position else 0.14
                if opp_type == 'CALLING_STATION':
                    base_freq += 0.12   # always look for thin value vs. stations
                if random.random() < base_freq:
                    size = self.bet_sizer.get_bet_size(
                        equity=equity, pot=effective_pot,
                        min_raise=min_raise, max_raise=max_raise,
                        board_wet=board_wet, street=street,
                        opponent_type=opp_type, my_pip=my_pip,
                    )
                    noisy = self.mixed_strategy.add_noise_to_sizing(size, min_raise, max_raise)
                    return RaiseAction(noisy)
            return CheckAction()

        # ── WEAK (≤ 0.40 + threshold) ─────────────────────────────────────────
        else:
            if can_raise and not bluff_less:
                # Board-representation bluffs (e.g. monotone = represent flush)
                bluff_freq = 0.0
                if opp_type == 'ROCK':
                    bluff_freq += 0.22
                elif opp_type == 'TAG':
                    bluff_freq += 0.12
                if board_features.get('board_monotone') and equity > 0.20:
                    bluff_freq += 0.14  # increased from 0.10 — better flush repr
                if street == 3 and has_draw and equity > 0.25:
                    bluff_freq += 0.08

                bluff_freq = min(bluff_freq * bluff_freq_mult, 0.40)
                if bluff_freq > 0 and random.random() < bluff_freq:
                    size = self.bet_sizer.get_bluff_size(
                        pot=effective_pot, min_raise=min_raise, max_raise=max_raise,
                        my_pip=my_pip, opponent_type=opp_type,
                    )
                    noisy = self.mixed_strategy.add_noise_to_sizing(size, min_raise, max_raise)
                    return RaiseAction(noisy)
            return CheckAction()

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _made_strength(self, board_features):
        if (board_features.get('set') or
                board_features.get('two_pair_or_better') or
                board_features.get('flush_made')):
            return 'strong'
        if board_features.get('overpair') or board_features.get('top_pair'):
            return 'medium'
        return 'weak'

    def _can_stack_off(self, board_features, street):
        if board_features.get('flush_made') or board_features.get('set'):
            return True
        if (street == 3 and
                board_features.get('two_pair_or_better') and
                not board_features.get('board_monotone') and
                not board_features.get('board_paired')):
            return True
        return False

    def _street_raise_count(self, round_state):
        count = 0
        state = round_state
        while state.previous_state is not None and state.previous_state.street == state.street:
            prev  = state.previous_state
            actor = prev.button % 2
            if (state.pips[actor] > prev.pips[actor] and
                    state.pips[actor] > state.pips[1 - actor]):
                count += 1
            state = prev
        return count

    def _facing_raise_after_we_bet(self, round_state, active):
        prev = round_state.previous_state
        if prev is None or prev.street != round_state.street:
            return False
        opp = 1 - active
        return (round_state.pips[opp] > prev.pips[opp] and
                prev.pips[active] > prev.pips[opp])

    def _safe_fold(self, legal_actions):
        if FoldAction in legal_actions:
            return FoldAction()
        if CheckAction in legal_actions:
            return CheckAction()
        # Should not happen, but absolute safety fallback
        return list(legal_actions)[0]()

    def _blend_adjustments(self, adjustments, stats):
        """
        Soften exploitative deviations until the tracker has enough evidence.
        This keeps the bot closer to a general default strategy early on.
        """
        baseline = self.classifier.get_adjustments('UNKNOWN', stats)
        hands_played = stats.get('hands_played', 0)
        min_hands = getattr(self.classifier, 'MIN_HANDS', 15)
        confidence = max(0.0, min(1.0, (hands_played - min_hands) / 35.0))

        blended = {}
        for key, value in adjustments.items():
            base_value = baseline.get(key, value)
            if isinstance(value, (int, float)) and isinstance(base_value, (int, float)):
                blended[key] = base_value + (value - base_value) * confidence
            else:
                blended[key] = value if confidence >= 0.65 else base_value
        return blended

    def _to_action(self, action_type, amount, legal_actions, min_raise, max_raise):
        '''Convert (action_type, amount) string pair to engine Action.'''
        if action_type == 'raise' and RaiseAction in legal_actions:
            amount = int(max(min_raise, min(amount, max_raise)))
            return RaiseAction(amount)
        elif action_type == 'call':
            if CallAction in legal_actions:
                return CallAction()
            if CheckAction in legal_actions:
                return CheckAction()
        elif action_type == 'check':
            if CheckAction in legal_actions:
                return CheckAction()
            if FoldAction in legal_actions:
                return FoldAction()
        elif action_type == 'fold':
            if FoldAction in legal_actions:
                return FoldAction()
            if CheckAction in legal_actions:
                return CheckAction()
        # Fallback
        if CheckAction in legal_actions:
            return CheckAction()
        if CallAction in legal_actions:
            return CallAction()
        if FoldAction in legal_actions:
            return FoldAction()
        return CheckAction()
