'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot


class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        self.rank_value = {
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "T": 10,
            "J": 11,
            "Q": 12,
            "K": 13,
            "A": 14,
        }

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        pass

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        continue_cost = opp_pip - my_pip
        my_bounty = round_state.bounties[active]

        rank_a = my_cards[0][0]
        rank_b = my_cards[1][0]
        suit_a = my_cards[0][1]
        suit_b = my_cards[1][1]

        value_a = self.rank_value[rank_a]
        value_b = self.rank_value[rank_b]
        high = max(value_a, value_b)
        low = min(value_a, value_b)
        gap = high - low

        is_pair = rank_a == rank_b
        is_suited = suit_a == suit_b
        has_bounty_rank = rank_a == my_bounty or rank_b == my_bounty

        strong_preflop = is_pair or high >= 13 or (high >= 11 and low >= 10) or (is_suited and gap <= 1 and high >= 10)
        medium_preflop = high >= 11 or (is_suited and gap <= 2) or (low >= 8)

        # Bounty-card hands are worth contesting a little wider because bounty wins pay extra.
        if has_bounty_rank:
            medium_preflop = True

        if street == 0:
            if continue_cost == 0:
                # Big blind option: attack limps with a larger raise to pressure weak ranges.
                if RaiseAction in legal_actions and (medium_preflop or has_bounty_rank):
                    min_raise, max_raise = round_state.raise_bounds()
                    target = max(min_raise, my_pip + 10)
                    return RaiseAction(min(target, max_raise))
                if CheckAction in legal_actions:
                    return CheckAction()

            if continue_cost > 0:
                if RaiseAction in legal_actions and (strong_preflop or (medium_preflop and continue_cost <= 6)):
                    min_raise, max_raise = round_state.raise_bounds()
                    target = my_pip + max(10, continue_cost * 3)
                    return RaiseAction(min(max(target, min_raise), max_raise))

                if not medium_preflop and continue_cost >= 10 and FoldAction in legal_actions:
                    return FoldAction()

                if CallAction in legal_actions and (medium_preflop or continue_cost <= 4):
                    return CallAction()

                if FoldAction in legal_actions:
                    return FoldAction()

        # Postflop: keep a low-variance line and realize equity with calls/checks.
        if continue_cost == 0 and CheckAction in legal_actions:
            return CheckAction()

        if CallAction in legal_actions:
            return CallAction()
        if CheckAction in legal_actions:
            return CheckAction()
        if FoldAction in legal_actions:
            return FoldAction()
        return CallAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
