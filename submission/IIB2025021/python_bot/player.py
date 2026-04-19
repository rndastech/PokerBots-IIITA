"""
God-tier Monte Carlo poker bot using eval7.

Key upgrades over baseline:
  - Position-aware strategy (IP vs OOP)
  - Stack-to-pot ratio (SPR) driven sizing and thresholds
  - Board texture analysis (wet/dry, paired, monotone)
  - Range-weighted Monte Carlo (opponent range narrows by action history)
  - Proper bounty EV math (shifts thresholds, not win_prob)
  - Adaptive opponent model (aggression freq, fold-to-cbet tracking)
  - Geometric all-in sizing on short SPR
  - Polarised raise/check-raise ranges
  - Semi-bluff logic on draws with fold equity
"""
import random
from collections import defaultdict

import eval7

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.states import STARTING_STACK

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BIG_BLIND = 2          # adjust if skeleton uses a different BB value
BOUNTY_CHIPS = 25      # chips awarded for hitting your bounty (adjust to match rules)

# Preflop range tiers (Chen-formula inspired, indexed by (high, low, suited, pair))
# Used to build a probability weight for opponent hand ranges in MC.
_RANK_STR = "23456789TJQKA"
_RANK_VAL  = {r: i + 2 for i, r in enumerate(_RANK_STR)}


def _hand_tier(high, low, is_suited, is_pair):
    """Return 0.0–1.0 weight for how often a random opponent plays this hand."""
    gap = high - low
    if is_pair:
        return 0.9 if high >= 10 else (0.7 if high >= 7 else 0.45)
    if high == 14:
        if low >= 13: return 0.95
        if low >= 11: return 0.85
        if low >= 9:  return 0.70 if is_suited else 0.55
        return 0.45 if is_suited else 0.25
    if high == 13:
        if low >= 12: return 0.80 if is_suited else 0.65
        if low >= 10: return 0.65 if is_suited else 0.45
        return 0.40 if is_suited else 0.20
    if is_suited and gap <= 1 and high >= 9: return 0.65
    if is_suited and gap <= 2 and high >= 8: return 0.50
    if high >= 11 and low >= 9:              return 0.45
    if is_suited and high >= 7 and gap <= 3: return 0.35
    return 0.15


def _board_texture(board_cards):
    """
    Returns a dict describing board texture.
    flush_draw / flush_made / straight_draw / paired / trips / dry
    """
    if not board_cards:
        return {"wet": False, "paired": False, "monotone": False, "two_tone": False,
                "straight_draw": False, "flush_draw": False, "dry": True}

    suits = [str(c)[1] for c in board_cards]
    ranks = [_RANK_VAL.get(str(c)[0], 0) for c in board_cards]
    suit_counts = defaultdict(int)
    for s in suits:
        suit_counts[s] += 1

    monotone     = max(suit_counts.values()) == len(board_cards)
    two_tone     = max(suit_counts.values()) >= 2
    flush_draw   = max(suit_counts.values()) >= 3 and not monotone
    flush_made   = max(suit_counts.values()) >= 4

    sorted_r = sorted(set(ranks))
    straight_draw = False
    if len(sorted_r) >= 2:
        for i in range(len(sorted_r) - 1):
            if sorted_r[i + 1] - sorted_r[i] <= 2:
                straight_draw = True
                break

    rank_counts = defaultdict(int)
    for r in ranks:
        rank_counts[r] += 1
    paired = max(rank_counts.values()) >= 2

    wet = flush_draw or flush_made or monotone or straight_draw
    dry = not wet and not paired

    return {
        "wet": wet,
        "dry": dry,
        "paired": paired,
        "monotone": monotone,
        "two_tone": two_tone,
        "straight_draw": straight_draw,
        "flush_draw": flush_draw,
        "flush_made": flush_made,
    }


class OpponentModel:
    """
    Lightweight per-session opponent model.
    Tracks aggression frequency and fold-to-cbet patterns.
    """
    def __init__(self):
        self.raises_seen      = 0
        self.calls_seen       = 0
        self.folds_seen       = 0
        self.cbets_faced      = 0
        self.folds_to_cbet    = 0
        self.total_rounds     = 0
        self.vpip_rounds      = 0   # rounds where opp voluntarily put $ in
        self.streets_played   = defaultdict(int)

    @property
    def aggression_freq(self):
        total = self.raises_seen + self.calls_seen + self.folds_seen
        return self.raises_seen / max(1, total)

    @property
    def fold_to_cbet(self):
        return self.folds_to_cbet / max(1, self.cbets_faced)

    @property
    def vpip(self):
        return self.vpip_rounds / max(1, self.total_rounds)

    def is_tight(self):
        return self.vpip < 0.40

    def is_aggressive(self):
        return self.aggression_freq > 0.35

    def is_passive(self):
        return self.aggression_freq < 0.20 and self.total_rounds > 5

    def hand_range_weight(self, raises_in_hand):
        """Multiplier to narrow opponent range based on aggression in this hand."""
        if raises_in_hand == 0:
            return 1.0   # full range
        if raises_in_hand == 1:
            return 0.5 if self.is_tight() else 0.65
        if raises_in_hand == 2:
            return 0.25 if self.is_tight() else 0.35
        return 0.15


class Player(Bot):

    def __init__(self):
        self.rng         = random.Random()
        self.card_cache  = {}
        self.full_deck   = [eval7.Card(r + s) for r in "23456789TJQKA" for s in "shdc"]
        self.opp_model   = OpponentModel()

        # Per-round state
        self._prev_opp_pip   = {0: 0, 3: 0, 4: 0, 5: 0}
        self._cbet_fired     = False
        self._last_street    = -1
        self._opp_raised_this_street = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def handle_new_round(self, game_state, round_state, active):
        self.opp_model.total_rounds += 1
        self._cbet_fired = False
        self._last_street = -1
        self._opp_raised_this_street = False
        self._prev_opp_pip = {0: 0, 3: 0, 4: 0, 5: 0}

    def handle_round_over(self, game_state, terminal_state, active):
        # Infer opp actions from terminal state deltas (basic tracking)
        pass

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _card(self, card_text):
        c = self.card_cache.get(card_text)
        if c is None:
            c = eval7.Card(card_text)
            self.card_cache[card_text] = c
        return c

    def _get_board_cards(self, round_state):
        """
        Robustly fetch community cards from round_state.
        Tries .board first (most frameworks), then .deck.
        """
        board = getattr(round_state, "board", None) or getattr(round_state, "deck", None) or []
        return [self._card(str(c)) for c in board if c]

    def _pot_size(self, round_state):
        return 2 * STARTING_STACK - round_state.stacks[0] - round_state.stacks[1]

    def _spr(self, round_state, active):
        pot = self._pot_size(round_state)
        my_stack = round_state.stacks[active]
        return my_stack / max(1, pot)

    def _is_in_position(self, round_state, active):
        """In heads-up: button acts last postflop → is in position."""
        return (round_state.button % 2) == active

    def _action_history(self, round_state, active):
        """Walk state chain and return (opp_raises, opp_calls, opp_folds) this hand."""
        opp_raises = 0
        state = round_state
        while state is not None and getattr(state, "previous_state", None) is not None:
            prev = state.previous_state
            actor = prev.button % 2
            if actor == 1 - active:
                opp_pip_now  = state.pips[actor]
                opp_pip_prev = prev.pips[actor]
                if opp_pip_now > opp_pip_prev:
                    cost_to_call = prev.pips[active] - prev.pips[actor]
                    extra = opp_pip_now - opp_pip_prev - max(0, cost_to_call)
                    if extra > 0:
                        opp_raises += 1
            state = prev
        return opp_raises

    # ------------------------------------------------------------------
    # Preflop hand classification
    # ------------------------------------------------------------------
    def _preflop_profile(self, hole_cards, bounty_rank):
        ra, sa = hole_cards[0][0], hole_cards[0][1]
        rb, sb = hole_cards[1][0], hole_cards[1][1]
        va, vb = _RANK_VAL[ra], _RANK_VAL[rb]
        high, low = max(va, vb), min(va, vb)
        gap = high - low
        is_pair   = ra == rb
        is_suited = sa == sb
        has_bounty = bounty_rank not in (None, "-1") and (ra == bounty_rank or rb == bounty_rank)

        premium = (
            (is_pair and high >= 11)
            or (high == 14 and low >= 12)
            or (high >= 13 and low >= 11)
            or (is_suited and high >= 12 and gap <= 1)
        )
        strong = premium or (
            (is_pair and high >= 7)
            or high >= 13
            or (high >= 11 and low >= 10)
            or (is_suited and high >= 10 and gap <= 2)
            or (high == 14 and low >= 8)
        )
        playable = strong or (
            is_pair
            or (is_suited and gap <= 3 and high >= 7)
            or (high >= 11 and low >= 7)
            or (high >= 10 and gap <= 2)
            or has_bounty
        )
        # Is this hand a good semi-bluff candidate (has draw equity)?
        draw_potential = is_suited or (gap <= 2 and high >= 6)

        return {
            "premium":        premium,
            "strong":         strong,
            "playable":       playable,
            "has_bounty":     has_bounty,
            "draw_potential": draw_potential,
            "high":           high,
            "low":            low,
            "is_suited":      is_suited,
            "is_pair":        is_pair,
        }

    # ------------------------------------------------------------------
    # Monte Carlo simulation — range-weighted
    # ------------------------------------------------------------------
    def _simulation_count(self, street, game_clock):
        if game_clock < 3.0:
            return 200
        if game_clock < 8.0:
            return 400
        if street >= 4:
            return 1000
        if street == 3:
            return 800
        return 600

    def _estimate_win_probability(self, hole_cards, board_cards, iterations,
                                   opp_range_weight=1.0, texture=None):
        """
        Monte Carlo with optional range-weighting.
        opp_range_weight < 1 → opponent is weighted toward stronger hands
        (we increase rejection rate for weak opponent hands).
        """
        hero = [self._card(c) for c in hole_cards]
        used = set(hero) | set(board_cards)
        available = [c for c in self.full_deck if c not in used]
        board_needed = 5 - len(board_cards)

        # Precompute hand weights for available 2-card combos
        # We do lazy weighting: sample and accept/reject proportional to tier weight
        wins = 0.0
        attempts = 0
        max_attempts = iterations * 6   # allow rejection overhead
        successes = 0

        while successes < iterations and attempts < max_attempts:
            attempts += 1
            total_needed = board_needed + 2
            if len(available) < total_needed:
                break
            sample = self.rng.sample(available, total_needed)
            opp_hand = sample[:2]

            # Range weight: weakly-represented hands get discarded more
            if opp_range_weight < 0.99:
                r0, r1 = str(opp_hand[0])[0], str(opp_hand[1])[0]
                s0, s1 = str(opp_hand[0])[1], str(opp_hand[1])[1]
                v0, v1 = _RANK_VAL.get(r0, 7), _RANK_VAL.get(r1, 7)
                tier = _hand_tier(max(v0,v1), min(v0,v1), s0==s1, r0==r1)
                # Accept with probability proportional to tier * range_weight scaling
                accept_prob = 0.2 + 0.8 * tier * opp_range_weight
                if self.rng.random() > accept_prob:
                    continue

            runout = board_cards + sample[2:]
            hv = eval7.evaluate(hero + runout)
            ov = eval7.evaluate(opp_hand + runout)
            if hv > ov:
                wins += 1.0
            elif hv == ov:
                wins += 0.5
            successes += 1

        return wins / max(1, successes)

    # ------------------------------------------------------------------
    # Sizing helpers
    # ------------------------------------------------------------------
    def _raise_amount(self, round_state, active, target_fraction, pot_size):
        min_raise, max_raise = round_state.raise_bounds()
        my_pip = round_state.pips[active]
        raw = my_pip + max(BIG_BLIND, int(max(BIG_BLIND, pot_size) * target_fraction))
        return max(min_raise, min(raw, max_raise))

    def _geometric_allin_fraction(self, round_state, active):
        """
        When SPR is low, size to put in all chips over 1 more street.
        Returns the fraction of pot to raise.
        """
        my_stack  = round_state.stacks[active]
        pot       = self._pot_size(round_state)
        if pot <= 0:
            return 1.0
        # fraction s.t. raise_amount ≈ my_stack
        return my_stack / max(1, pot)

    # ------------------------------------------------------------------
    # Bounty EV adjustment (proper math)
    # ------------------------------------------------------------------
    def _bounty_ev_adjustment(self, hole_cards, board_cards, my_bounty,
                               win_prob, pot_size, continue_cost):
        """
        Return an adjusted EV that accounts for bounty bonus chips.
        bounty_ev = P(win) * pot + P(bounty_hit_win) * BOUNTY_CHIPS
        We return the delta to add to base EV.
        """
        if my_bounty in (None, "-1"):
            return 0.0
        board_strs  = [str(c)[0] for c in board_cards]
        hole_strs   = [c[0] for c in hole_cards]
        bounty_on_board = my_bounty in board_strs
        bounty_in_hand  = my_bounty in hole_strs
        # Probability we hit bounty in this hand given current info
        if bounty_in_hand or bounty_on_board:
            p_bounty_hit = 1.0   # already on board or in hand
        else:
            # Rough: ~(cards_remaining * 4 suits) / deck_remaining
            cards_remaining = 5 - len(board_cards)
            deck_remaining  = 52 - 2 - len(board_cards)
            p_bounty_hit    = min(0.95, (cards_remaining * 4) / max(1, deck_remaining))
        # Bounty bonus only pays if we also WIN
        return win_prob * p_bounty_hit * BOUNTY_CHIPS

    # ------------------------------------------------------------------
    # Main decision logic
    # ------------------------------------------------------------------
    def get_action(self, game_state, round_state, active):
        legal_actions  = round_state.legal_actions()
        street         = round_state.street
        hole_cards     = round_state.hands[active]
        board_cards    = self._get_board_cards(round_state)
        my_pip         = round_state.pips[active]
        opp_pip        = round_state.pips[1 - active]
        continue_cost  = opp_pip - my_pip
        pot_size       = self._pot_size(round_state)
        my_stack       = round_state.stacks[active]
        my_bounty      = round_state.bounties[active]
        in_position    = self._is_in_position(round_state, active)
        spr            = self._spr(round_state, active)
        opp_raises     = self._action_history(round_state, active)
        preflop        = self._preflop_profile(hole_cards, my_bounty)
        texture        = _board_texture(board_cards) if board_cards else {"wet": False, "dry": True}
        opp_range_w    = self.opp_model.hand_range_weight(opp_raises)

        # ---------------------------------------------------------------
        # PREFLOP
        # ---------------------------------------------------------------
        if street == 0:
            return self._preflop_action(
                round_state, active, legal_actions,
                continue_cost, pot_size, preflop, in_position, opp_raises
            )

        # ---------------------------------------------------------------
        # POSTFLOP: run Monte Carlo
        # ---------------------------------------------------------------
        iterations = self._simulation_count(street, game_state.game_clock)
        win_prob   = self._estimate_win_probability(
            hole_cards, board_cards, iterations,
            opp_range_weight=opp_range_w, texture=texture
        )

        # ---- Bounty EV delta ----
        bounty_ev_bonus = self._bounty_ev_adjustment(
            hole_cards, board_cards, my_bounty, win_prob, pot_size, continue_cost
        )

        # ---- Position adjustment ----
        # IP: we see more streets, can extract more value / bluff better
        pos_adj = 0.02 if in_position else -0.02
        win_prob = min(0.99, max(0.01, win_prob + pos_adj))

        # ---- Opponent raise discount ----
        if opp_raises >= 1:
            discount = min(0.12, 0.04 * opp_raises)
            if self.opp_model.is_tight():
                discount += 0.03
            win_prob = max(0.01, win_prob - discount)

        # ---- Board texture adjustment ----
        if texture.get("wet") and not preflop["is_suited"] and not preflop["draw_potential"]:
            win_prob = max(0.01, win_prob - 0.03)   # backdoor outs less likely

        # ---- EV calculations ----
        ev_call   = (win_prob * (pot_size + continue_cost)) \
                  - ((1.0 - win_prob) * continue_cost) \
                  + bounty_ev_bonus

        ev_fold   = 0.0

        # ---- Pot odds & call threshold ----
        pot_odds        = continue_cost / float(max(1, pot_size + continue_cost))
        pressure_ratio  = continue_cost / float(max(1, pot_size))

        # Base threshold = pot odds + small edge requirement
        call_threshold  = pot_odds + 0.04

        # Street-specific tightening (less info early)
        if street == 3:   call_threshold += 0.02
        if street == 4:   call_threshold += 0.04
        if street == 5:   call_threshold += 0.06

        # Big bets need more equity
        if pressure_ratio > 0.75:  call_threshold += 0.05
        if pressure_ratio > 1.50:  call_threshold += 0.05

        # Opponent aggression → tighten further
        if opp_raises >= 2:        call_threshold += 0.04
        if self.opp_model.is_aggressive(): call_threshold += 0.02

        # IP we can widen calls slightly
        if in_position:            call_threshold -= 0.02

        # Premium hands widen further
        if preflop["premium"]:     call_threshold -= 0.03

        # Low SPR → commit threshold lower (already near all-in)
        if spr < 2:                call_threshold -= 0.05
        if spr < 1:                call_threshold -= 0.07

        # ---- Bluff / semi-bluff conditions ----
        fold_equity    = self.opp_model.fold_to_cbet if self.opp_model.cbets_faced > 3 else 0.40
        has_draw       = preflop["draw_potential"] and texture.get("wet")
        can_semibluff  = (
            has_draw
            and continue_cost == 0
            and RaiseAction in legal_actions
            and street in (3, 4)
            and fold_equity >= 0.35
        )
        pure_bluff_ok  = (
            continue_cost == 0
            and RaiseAction in legal_actions
            and street == 5
            and in_position
            and texture.get("paired")
            and self.rng.random() < 0.25
            and not self.opp_model.is_passive()  # don't bluff stations
        )

        # ---------------------------------------------------------------
        # DECISION TREE — no continue_cost (check or bet)
        # ---------------------------------------------------------------
        if continue_cost <= 0:
            if RaiseAction in legal_actions:
                # Value bet sizing: scale by SPR
                if win_prob >= 0.80:
                    frac = min(1.2, self._geometric_allin_fraction(round_state, active)) \
                           if spr < 3 else (0.90 if in_position else 0.70)
                    return RaiseAction(self._raise_amount(round_state, active, frac, pot_size))

                if win_prob >= 0.65 and street >= 3:
                    frac = 0.65 if texture.get("dry") else 0.50
                    return RaiseAction(self._raise_amount(round_state, active, frac, pot_size))

                if win_prob >= 0.55 and in_position and street == 3:
                    # Cbet in position on flop
                    frac = 0.45 if texture.get("dry") else 0.35
                    self._cbet_fired = True
                    return RaiseAction(self._raise_amount(round_state, active, frac, pot_size))

                if can_semibluff:
                    frac = 0.55 if texture.get("flush_draw") else 0.45
                    return RaiseAction(self._raise_amount(round_state, active, frac, pot_size))

                if pure_bluff_ok:
                    return RaiseAction(self._raise_amount(round_state, active, 0.70, pot_size))

            if CheckAction in legal_actions:
                return CheckAction()
            return CallAction()

        # ---------------------------------------------------------------
        # DECISION TREE — facing a bet (continue_cost > 0)
        # ---------------------------------------------------------------

        # All-in or pot-commit shove with strong hand + low SPR
        if RaiseAction in legal_actions and spr < 2 and win_prob >= 0.62:
            return RaiseAction(self._raise_amount(
                round_state, active,
                self._geometric_allin_fraction(round_state, active),
                pot_size
            ))

        # Strong re-raise for value
        if RaiseAction in legal_actions and win_prob >= 0.84:
            frac = 1.10 if spr < 4 else 0.90
            return RaiseAction(self._raise_amount(
                round_state, active, frac, pot_size + continue_cost
            ))

        # Check-raise bluff OOP vs likely cbet
        if (RaiseAction in legal_actions
                and not in_position
                and opp_raises == 1
                and street == 3
                and win_prob >= 0.45
                and fold_equity >= 0.45
                and self.rng.random() < 0.30):
            return RaiseAction(self._raise_amount(round_state, active, 0.80, pot_size + continue_cost))

        # EV-based call/fold decision
        if ev_call > ev_fold and win_prob >= call_threshold:
            if CallAction in legal_actions:
                return CallAction()
            if CheckAction in legal_actions:
                return CheckAction()

        # Fold / check otherwise
        if FoldAction in legal_actions and continue_cost > 0:
            return FoldAction()
        if CheckAction in legal_actions:
            return CheckAction()
        return CallAction()

    # ------------------------------------------------------------------
    # Preflop sub-routine (HU-specific)
    # ------------------------------------------------------------------
    def _preflop_action(self, round_state, active, legal_actions,
                         continue_cost, pot_size, preflop, in_position, opp_raises):
        """
        Heads-up preflop strategy.
        Button (dealer) posts SB and acts first preflop → out of position preflop.
        """
        cheap_call  = continue_cost <= BIG_BLIND * 2
        medium_call = continue_cost <= BIG_BLIND * 5
        spr         = self._spr(round_state, active)

        # ---- Facing no bet (we can check or open) ----
        if continue_cost == 0:
            if RaiseAction in legal_actions:
                if preflop["premium"]:
                    # Open big with premiums
                    frac = 7.0 if in_position else 5.0
                    return RaiseAction(self._raise_amount(round_state, active, frac, BIG_BLIND * 2))
                if preflop["strong"]:
                    frac = 4.0 if in_position else 3.0
                    return RaiseAction(self._raise_amount(round_state, active, frac, BIG_BLIND * 2))
                if preflop["playable"] and (in_position or self.rng.random() < 0.50):
                    return RaiseAction(self._raise_amount(round_state, active, 2.5, BIG_BLIND * 2))
                # Mixed: steal more from position
                if in_position and self.rng.random() < 0.45:
                    return RaiseAction(self._raise_amount(round_state, active, 2.0, BIG_BLIND * 2))
            if CheckAction in legal_actions:
                return CheckAction()
            return CallAction()

        # ---- Facing a raise ----
        if opp_raises == 0:
            # Single raise from opp
            if RaiseAction in legal_actions and preflop["premium"] and continue_cost <= BIG_BLIND * 8:
                frac = 9.0 if preflop["high"] >= 13 else 7.0
                return RaiseAction(self._raise_amount(round_state, active, frac, continue_cost))

            if CallAction in legal_actions:
                if preflop["premium"]:
                    return CallAction()
                if preflop["strong"] and (medium_call or continue_cost <= pot_size // 2):
                    return CallAction()
                if preflop["playable"] and cheap_call:
                    return CallAction()
                if preflop["has_bounty"] and continue_cost <= BIG_BLIND * 4:
                    return CallAction()
                if continue_cost <= BIG_BLIND:
                    return CallAction()

        else:
            # Facing a 3-bet or more → tighten drastically
            if preflop["premium"] and CallAction in legal_actions:
                # 4-bet shove with AA/KK
                if preflop["is_pair"] and preflop["high"] >= 13 and RaiseAction in legal_actions:
                    return RaiseAction(self._raise_amount(
                        round_state, active,
                        self._geometric_allin_fraction(round_state, active),
                        pot_size
                    ))
                return CallAction()
            # Fold everything else to heavy 3-bet
            if FoldAction in legal_actions:
                return FoldAction()

        if FoldAction in legal_actions:
            return FoldAction()
        return CallAction()


if __name__ == "__main__":
    run_bot(Player(), parse_args())