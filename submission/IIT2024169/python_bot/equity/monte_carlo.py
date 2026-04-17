'''
Monte Carlo equity estimator for post-flop decisions.
Uses eval7 for fast hand evaluation via Cython.
'''
import random
import eval7


# Pre-build all 52 card objects once for reuse
ALL_CARDS = [eval7.Card(r + s) for r in '23456789TJQKA' for s in 'shdc']
CARD_STR_TO_OBJ = {str(c): c for c in ALL_CARDS}


def _str_to_card(card_str):
    '''Convert a string like "Ah" to an eval7.Card object.'''
    if isinstance(card_str, eval7.Card):
        return card_str
    return CARD_STR_TO_OBJ.get(card_str, eval7.Card(card_str))


class MonteCarloEquity:
    '''Estimates hand equity using Monte Carlo simulation with eval7.'''

    def estimate(self, my_cards, board_cards, num_simulations=500):
        '''
        Estimate equity of my hand against a random opponent hand.

        Args:
            my_cards: list of 2 card strings, e.g., ['Ah', 'Kd']
            board_cards: list of 0-5 card strings (community cards so far)
            num_simulations: number of rollouts (500 for flop, 300 for turn)

        Returns:
            float: equity 0.0-1.0
        '''
        # Convert to eval7.Card objects
        my_hand = [_str_to_card(c) for c in my_cards]
        board = [_str_to_card(c) for c in board_cards]

        # Cards remaining to deal on the board
        cards_to_deal = 5 - len(board)

        # Build deck of remaining cards
        used = set(str(c) for c in my_hand + board)
        remaining = [c for c in ALL_CARDS if str(c) not in used]

        # On river (5 board cards known) — do exhaustive enumeration
        if cards_to_deal == 0:
            return self._exhaustive_river(my_hand, board, remaining)

        wins = 0
        ties = 0
        total = 0

        for _ in range(num_simulations):
            # Sample cards for opponent hand + remaining board
            sampled = random.sample(remaining, 2 + cards_to_deal)
            opp_hand = sampled[:2]
            future_board = sampled[2:]

            full_board = board + future_board
            my_score = eval7.evaluate(my_hand + full_board)
            opp_score = eval7.evaluate(opp_hand + full_board)

            if my_score > opp_score:
                wins += 1
            elif my_score == opp_score:
                ties += 1
            total += 1

        if total == 0:
            return 0.5
        return (wins + 0.5 * ties) / total

        # Deleted estimate_vs_range because we are transitioning to absolute EV

    def _exhaustive_river(self, my_hand, board, remaining):
        '''
        On river, enumerate all possible opponent hands for exact equity.
        C(remaining, 2) combinations — typically ~990.
        '''
        wins = 0
        ties = 0
        total = 0

        my_score = eval7.evaluate(my_hand + board)

        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                opp_hand = [remaining[i], remaining[j]]
                opp_score = eval7.evaluate(opp_hand + board)

                if my_score > opp_score:
                    wins += 1
                elif my_score == opp_score:
                    ties += 1
                total += 1

        if total == 0:
            return 0.5
        return (wins + 0.5 * ties) / total
