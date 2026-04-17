'''
Bayesian inference of opponent's bounty rank.
Uses showdown information to narrow down possible bounty ranks.

Information rules (from engine.py L524-534):
  - When we WIN: opponent's bounty hit/miss is revealed
  - When we LOSE: opponent's bounty hit is MASKED (#)
  - When SPLIT: both bounty hits revealed
'''

ALL_RANKS = set('23456789TJQKA')
ROUNDS_PER_BOUNTY = 25


class BountyInferencer:
    '''Tracks and narrows down possible opponent bounty ranks.'''

    def __init__(self):
        self.possible_ranks = set(ALL_RANKS)
        self.current_period = 0
        self.observations = 0

    def check_reset(self, round_num):
        '''
        Check if bounty period has changed and reset inference.
        Bounties reset on rounds 1, 26, 51, 76, etc.
        '''
        period = (round_num - 1) // ROUNDS_PER_BOUNTY
        if period != self.current_period:
            self.possible_ranks = set(ALL_RANKS)
            self.current_period = period
            self.observations = 0

    def update(self, terminal_state, active):
        '''
        Update beliefs based on round outcome.

        Args:
            terminal_state: TerminalState object
            active: our player index
        '''
        my_delta = terminal_state.deltas[active]
        bounty_hits = terminal_state.bounty_hits

        if bounty_hits is None:
            return

        i_won = my_delta > 0
        is_split = my_delta == 0

        # Get opponent's bounty hit (True, False, or unavailable)
        opp_hit = bounty_hits[1 - active]

        # Get previous state to access cards
        prev = terminal_state.previous_state
        if prev is None:
            return

        # We only get useful info when we can see opponent's cards
        # AND their bounty hit status is revealed (we won or split)
        if not (i_won or is_split):
            return  # opponent hit is masked when we lose

        # Check if opponent cards are visible (showdown, not fold)
        opp_cards = prev.hands[1 - active] if hasattr(prev, 'hands') else []
        if not opp_cards:
            return

        board_cards = prev.deck[:prev.street] if hasattr(prev, 'deck') else []
        if isinstance(board_cards, list) and board_cards:
            # Get all ranks visible to opponent
            opp_card_ranks = set()
            for c in opp_cards:
                if isinstance(c, str) and len(c) >= 1:
                    opp_card_ranks.add(c[0])
                elif hasattr(c, 'rank'):
                    rank_names = '23456789TJQKA'
                    opp_card_ranks.add(rank_names[c.rank])

            board_ranks = set()
            for c in board_cards:
                if isinstance(c, str) and len(c) >= 1:
                    board_ranks.add(c[0])
                elif hasattr(c, 'rank'):
                    rank_names = '23456789TJQKA'
                    board_ranks.add(rank_names[c.rank])

            visible_ranks = opp_card_ranks | board_ranks

            if opp_hit:
                # Opponent bounty rank IS one of the visible ranks
                self.possible_ranks &= visible_ranks
            else:
                # Opponent bounty rank is NOT any of the visible ranks
                self.possible_ranks -= visible_ranks

            # Safety: don't let possible_ranks become empty
            if not self.possible_ranks:
                self.possible_ranks = set(ALL_RANKS)

            self.observations += 1

    def get_possible_ranks(self):
        '''Return current set of possible opponent bounty ranks.'''
        return set(self.possible_ranks)

    def get_num_possible(self):
        '''How many ranks remain possible?'''
        return len(self.possible_ranks)

    def opponent_bounty_hit_probability(self, board_cards):
        '''
        Estimate probability that opponent has hit their bounty on current board.

        Args:
            board_cards: list of current community card strings

        Returns:
            float: probability 0.0-1.0
        '''
        if not self.possible_ranks or not board_cards:
            return 1.0 / 13.0  # uniform prior, ~7.7%

        board_ranks = set()
        for c in board_cards:
            if isinstance(c, str) and len(c) >= 1:
                board_ranks.add(c[0])

        # Probability = |possible_ranks ∩ board_ranks| / |possible_ranks|
        # But we also need to consider opponent's unknown hole cards
        # Simplify: just check if any possible rank is on the board
        matching = self.possible_ranks & board_ranks
        return len(matching) / len(self.possible_ranks) if self.possible_ranks else 0.0

    def is_rank_narrowed(self):
        '''Have we narrowed down the opponent's bounty rank significantly?'''
        return len(self.possible_ranks) <= 4 and self.observations >= 2
