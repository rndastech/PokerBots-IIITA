'''
Bounty-adjusted EV calculator for IIITA Pokerbots Bounty Hold'em.

Engine bounty rules (config.py):
    Win + bounty hit:  delta = pot × 1.5 + 10
    Win + no bounty:   delta = pot
    Split + only I hit: delta = pot × 0.25 + 10

Key formula improvement over previous version:
    Correct breakeven equity when bounty is active:
    equity_needed = call_cost / ((pot + call_cost) × BountyMult_EV + BountyConstant_EV)

    Where:
        BountyMult_EV     = P(hit) × 1.5 + P(no_hit) × 1.0
        BountyConstant_EV = P(hit) × 10

    This gives a strictly lower breakeven equity than the standard
    call_cost / (pot + call_cost) formula — we call more profitably
    when bounties are active.
'''
from math import comb


BOUNTY_RATIO    = 1.5
BOUNTY_CONSTANT = 10
RANK_ORDER      = '23456789TJQKA'


class BountyEV:
    '''Calculates bounty-adjusted expected values and pot odds.'''

    def __init__(self, bounty_ratio=BOUNTY_RATIO, bounty_constant=BOUNTY_CONSTANT):
        self.bounty_ratio    = bounty_ratio
        self.bounty_constant = bounty_constant

    # ─────────────────────────────────────────────────────────────────────────
    # Status checks
    # ─────────────────────────────────────────────────────────────────────────
    def bounty_in_hand(self, my_cards, bounty_rank):
        '''True if bounty rank appears in hole cards.'''
        if not bounty_rank or bounty_rank == '-1':
            return False
        return any(c[0] == bounty_rank for c in my_cards)

    def bounty_on_board(self, board_cards, bounty_rank):
        '''True if bounty rank appears on the board.'''
        if not bounty_rank or bounty_rank == '-1':
            return False
        return any(c[0] == bounty_rank for c in board_cards)

    def bounty_already_hit(self, my_cards, board_cards, bounty_rank):
        '''True if bounty is guaranteed (rank in hole OR board).'''
        return (self.bounty_in_hand(my_cards, bounty_rank) or
                self.bounty_on_board(board_cards, bounty_rank))

    # ─────────────────────────────────────────────────────────────────────────
    # Probability
    # ─────────────────────────────────────────────────────────────────────────
    def bounty_hit_probability(self, my_cards, board_cards, bounty_rank):
        '''
        Probability of hitting bounty by the river (including current board).

        Returns:
            float: 0.0 – 1.0
        '''
        if not bounty_rank or bounty_rank == '-1':
            return 0.0

        if self.bounty_already_hit(my_cards, board_cards, bounty_rank):
            return 1.0

        cards_to_come = 5 - len(board_cards)
        if cards_to_come <= 0:
            return 0.0

        seen = list(my_cards) + list(board_cards)
        bounty_seen      = sum(1 for c in seen if c[0] == bounty_rank)
        bounty_remaining = 4 - bounty_seen
        total_unseen     = 52 - len(seen)

        if bounty_remaining <= 0 or total_unseen <= 0:
            return 0.0

        non_bounty_unseen = total_unseen - bounty_remaining
        if non_bounty_unseen < cards_to_come:
            return 1.0

        # P(miss all) = C(non_bounty, cards_to_come) / C(total, cards_to_come)
        p_miss = comb(non_bounty_unseen, cards_to_come) / comb(total_unseen, cards_to_come)
        return 1.0 - p_miss

    # ─────────────────────────────────────────────────────────────────────────
    # EV multipliers
    # ─────────────────────────────────────────────────────────────────────────
    def bounty_ev_multiplier(self, my_cards, board_cards, bounty_rank):
        '''
        Expected pot-size multiplier from the bounty system.
        Returns 1.5 if bounty hit, otherwise probability-weighted blend.
        '''
        if not bounty_rank or bounty_rank == '-1':
            return 1.0

        p = self.bounty_hit_probability(my_cards, board_cards, bounty_rank)
        return p * self.bounty_ratio + (1.0 - p) * 1.0

    def bounty_constant_ev(self, my_cards, board_cards, bounty_rank):
        '''Expected flat chip bonus from bounty (+10 if hit).'''
        if not bounty_rank or bounty_rank == '-1':
            return 0.0
        p = self.bounty_hit_probability(my_cards, board_cards, bounty_rank)
        return p * self.bounty_constant

    # ─────────────────────────────────────────────────────────────────────────
    # Core: Adjusted pot odds (FIXED vs. previous version)
    # ─────────────────────────────────────────────────────────────────────────
    def adjusted_pot_odds(self, pot, call_amount, my_cards, board_cards, bounty_rank):
        '''
        Bounty-corrected equity needed to call profitably.

        Formula:
            effective_pot = (pot + call_amount) × BountyMult_EV + BountyConstant_EV
            equity_needed = call_amount / effective_pot

        This accounts for BOTH the multiplier AND the flat constant,
        giving a strictly lower breakeven than standard pot odds when
        bounty is active.

        Args:
            pot: current pot size (before call)
            call_amount: chips to call
            my_cards, board_cards, bounty_rank: for bounty probability

        Returns:
            float: required equity to call (lower = easier to call)
        '''
        if call_amount <= 0:
            return 0.0

        mult    = self.bounty_ev_multiplier(my_cards, board_cards, bounty_rank)
        const   = self.bounty_constant_ev(my_cards, board_cards, bounty_rank)

        effective_pot = (pot + call_amount) * mult + const
        if effective_pot <= 0:
            return 1.0

        return call_amount / effective_pot

    def standard_pot_odds(self, pot, call_amount):
        '''Simple pot odds without bounty (fallback).'''
        if call_amount <= 0:
            return 0.0
        denom = pot + call_amount
        return call_amount / denom if denom > 0 else 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Aggression / bet sizing adjustments
    # ─────────────────────────────────────────────────────────────────────────
    def aggression_boost(self, my_cards, board_cards, bounty_rank):
        '''
        Multiplier for bet sizing when bounty is active.

        When bounty is confirmed in hand → grow the pot we will collect at 1.5x.
        When bounty might hit → proportional boost.

        Returns:
            float: ≥ 1.0 (multiply target bet fraction by this)
        '''
        if not bounty_rank or bounty_rank == '-1':
            return 1.0

        if self.bounty_in_hand(my_cards, bounty_rank):
            return 1.18   # guaranteed win × 1.5 + 10: always grow the pot
        if self.bounty_already_hit(my_cards, board_cards, bounty_rank):
            return 1.18

        p = self.bounty_hit_probability(my_cards, board_cards, bounty_rank)
        if p > 0.50:
            return 1.10
        elif p > 0.25:
            return 1.05
        return 1.0

    def equity_boost_for_ongoing(self, my_cards, board_cards, bounty_rank):
        '''
        Small flat equity bonus to add to computed EHS when bounty is active,
        to account for the option value of winning more chips.

        Bounty in hand (guaranteed hit): up to 0.08
        Board-only or probabilistic: up to 0.06
        '''
        if not bounty_rank or bounty_rank == '-1':
            return 0.0
        if self.bounty_in_hand(my_cards, bounty_rank):
            # Guaranteed hit — be more aggressive to grow the pot
            return 0.08
        p = self.bounty_hit_probability(my_cards, board_cards, bounty_rank)
        # Scale: 0.06 if guaranteed on board, proportional otherwise
        return min(0.06, p * 0.06)

    def should_suppress_bluff(self, opp_bounty_hit_prob):
        '''
        If opponent is bounty-motivated (high prob of their rank on board),
        they will hero-call to protect their pot multiplier.
        Return True if we should suppress bluffing.
        '''
        return opp_bounty_hit_prob > 0.55
