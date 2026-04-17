'''
Board texture analyzer.
Detects draws, paired boards, wet/dry textures, and hand-board interactions.
'''

RANK_ORDER = '23456789TJQKA'
RANK_TO_VALUE = {r: i for i, r in enumerate(RANK_ORDER)}


def _rank_val(rank_char):
    '''Convert rank character to numeric value. 2=0, A=12.'''
    return RANK_TO_VALUE.get(rank_char, 0)


class BoardTextureAnalyzer:
    '''Analyzes the community cards + hero cards for strategic features.'''

    def analyze(self, my_cards, board_cards):
        '''
        Analyze the board texture relative to hero's hand.

        Args:
            my_cards: list of 2 card strings, e.g., ['Ah', 'Kd']
            board_cards: list of card strings (community cards)

        Returns:
            dict with boolean/int features describing the board
        '''
        if not board_cards:
            return self._empty_board()

        my_ranks = [c[0] for c in my_cards]
        my_suits = [c[1] for c in my_cards]
        board_ranks = [c[0] for c in board_cards]
        board_suits = [c[1] for c in board_cards]
        all_ranks = my_ranks + board_ranks
        all_suits = my_suits + board_suits

        features = {}

        # === Flush detection ===
        suit_counts = {}
        for s in all_suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1

        my_suit_counts = {}
        for s in all_suits:
            my_suit_counts[s] = my_suit_counts.get(s, 0) + 1

        features['flush_made'] = any(v >= 5 for v in suit_counts.values())
        features['flush_draw'] = any(
            suit_counts.get(s, 0) >= 4 and s in my_suits
            for s in suit_counts
        )

        # Nut flush draw: flush draw where hero has the Ace of that suit
        features['nut_flush_draw'] = False
        if features['flush_draw'] and not features['flush_made']:
            for s in my_suits:
                if suit_counts.get(s, 0) >= 4:
                    # Check if hero has the Ace of this suit
                    for c in my_cards:
                        if c[0] == 'A' and c[1] == s:
                            features['nut_flush_draw'] = True

        # Board monotone: 3+ board cards of same suit
        board_suit_counts = {}
        for s in board_suits:
            board_suit_counts[s] = board_suit_counts.get(s, 0) + 1
        features['board_monotone'] = any(v >= 3 for v in board_suit_counts.values())

        # === Straight detection ===
        all_values = sorted(set(_rank_val(r) for r in all_ranks))
        # Add low ace (value -1 mapped to position before 2)
        if 12 in all_values:  # Ace
            all_values = [-1] + all_values  # A-2-3-4-5 straight

        features['straight_draw_outs'] = self._count_straight_outs(
            my_cards, board_cards, all_values
        )

        # Board connected: 3+ board cards within 5-rank span
        board_values = sorted(set(_rank_val(r) for r in board_ranks))
        features['board_connected'] = False
        if len(board_values) >= 3:
            for i in range(len(board_values) - 2):
                if board_values[i + 2] - board_values[i] <= 4:
                    features['board_connected'] = True
                    break

        # === Pair / made hand detection ===
        board_rank_counts = {}
        for r in board_ranks:
            board_rank_counts[r] = board_rank_counts.get(r, 0) + 1

        features['board_paired'] = any(v >= 2 for v in board_rank_counts.values())

        # Overcards: hero cards above the highest board card
        board_high = max(_rank_val(r) for r in board_ranks) if board_ranks else -1
        features['overcards'] = sum(
            1 for c in my_cards if _rank_val(c[0]) > board_high
        )

        # Top pair: hero's card matches the highest board card
        board_high_rank = max(board_ranks, key=lambda r: _rank_val(r)) if board_ranks else None
        features['top_pair'] = board_high_rank in my_ranks if board_high_rank else False

        # Overpair: hero has a pocket pair above all board cards
        features['overpair'] = (
            my_ranks[0] == my_ranks[1]
            and _rank_val(my_ranks[0]) > board_high
        )

        # Two pair or better detection
        all_rank_counts = {}
        for r in all_ranks:
            all_rank_counts[r] = all_rank_counts.get(r, 0) + 1

        hero_pair_ranks = {
            r for r in set(my_ranks) if all_rank_counts.get(r, 0) >= 2
        }
        features['two_pair_or_better'] = (
            len(hero_pair_ranks) >= 2  # two pair using two distinct ranks
            or any(all_rank_counts.get(r, 0) >= 3 for r in set(my_ranks))  # trips/set
        )

        # Set: hero has pocket pair and board has matching rank
        features['set'] = (
            my_ranks[0] == my_ranks[1]
            and my_ranks[0] in board_ranks
        )

        # === Composite features ===
        features['wet_board'] = features['board_monotone'] or features['board_connected']
        features['has_draw'] = features['flush_draw'] or features['straight_draw_outs'] >= 4

        return features

    def _count_straight_outs(self, my_cards, board_cards, all_values):
        '''
        Estimate straight draw outs.
        Returns 0 (no draw), 4 (gutshot), or 8 (open-ended).
        '''
        if len(all_values) < 4:
            return 0

        my_values = set(_rank_val(c[0]) for c in my_cards)
        board_values_set = set(_rank_val(c[0]) for c in board_cards)
        combined = set(all_values)

        best_outs = 0

        # Check all possible 5-card straight windows
        for bottom in range(-1, 9):  # A-low through T-high
            window = set(range(bottom, bottom + 5))
            have = combined & window
            missing = window - combined
            # Need at least one of my cards to be in the window
            my_in_window = my_values & window

            if len(have) == 4 and len(missing) == 1 and len(my_in_window) >= 1:
                # Check if the missing card is at the ends (OESD) or middle (gutshot)
                missing_val = list(missing)[0]
                if missing_val == bottom or missing_val == bottom + 4:
                    best_outs = max(best_outs, 4)  # gutshot (one end)
                else:
                    best_outs = max(best_outs, 4)  # gutshot (middle)

        # Open-ended: check if we have 4 consecutive with room on both sides
        sorted_vals = sorted(combined)
        for i in range(len(sorted_vals) - 3):
            if sorted_vals[i + 3] - sorted_vals[i] == 3:
                # 4 consecutive cards
                low_end = sorted_vals[i] - 1
                high_end = sorted_vals[i + 3] + 1
                my_in_seq = my_values & set(sorted_vals[i:i + 4])
                if len(my_in_seq) >= 1:
                    if 0 <= low_end <= 12 and 0 <= high_end <= 12:
                        best_outs = max(best_outs, 8)  # OESD
                    else:
                        best_outs = max(best_outs, 4)  # one-ended

        return best_outs

    def _empty_board(self):
        '''Return default features when no board cards are dealt.'''
        return {
            'flush_made': False,
            'flush_draw': False,
            'nut_flush_draw': False,
            'board_monotone': False,
            'straight_draw_outs': 0,
            'board_connected': False,
            'board_paired': False,
            'overcards': 0,
            'top_pair': False,
            'overpair': False,
            'two_pair_or_better': False,
            'set': False,
            'wet_board': False,
            'has_draw': False,
        }
