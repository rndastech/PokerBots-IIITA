'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random


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
        pass

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
        #my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        #game_clock = game_state.game_clock  # the total number of seconds your bot has left to play this game
        #round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        #my_cards = round_state.hands[active]  # your cards
        #big_blind = bool(active)  # True if you are the big blind
        #my_bounty = round_state.bounties[active]  # your current bounty rank
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
        #my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        #street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        #my_cards = previous_state.hands[active]  # your cards
        #opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed
        
        my_bounty_hit = terminal_state.bounty_hits[active]  # True if you hit bounty
        opponent_bounty_hit = terminal_state.bounty_hits[1-active] # True if opponent hit bounty
        bounty_rank = previous_state.bounties[active]  # your bounty rank

        # The following is a demonstration of accessing illegal information (will not work)
        opponent_bounty_rank = previous_state.bounties[1-active]  # attempting to grab opponent's bounty rank

        if my_bounty_hit:
            print("I hit my bounty of " + bounty_rank + "!")
        if opponent_bounty_hit:
            print("Opponent hit their bounty of " + opponent_bounty_rank + "!")

    def monte_carlo_strength(self,my_cards, board_cards, iterations=80):
        deck = [r+s for r in "23456789TJQKA" for s in "shdc"]
        for c in my_cards + board_cards:
            if c in deck:
                deck.remove(c)

        rank_order = "23456789TJQKA"

        def card_rank(card):
            return rank_order.index(card[0])

        def evaluate(cards):
            ranks = [card_rank(c) for c in cards]
            counts = {}
            for r in ranks:
                counts[r] = counts.get(r, 0) + 1

            values = sorted(counts.values(), reverse=True)

            if 4 in values:
                return 7
            if 3 in values and 2 in values:
                return 6
            if 3 in values:
                return 5
            if values.count(2) >= 2:
                return 4
            if 2 in values:
                return 3
            return max(ranks)

        wins = 0

        for _ in range(iterations):
            sample = random.sample(deck, 2 + (5 - len(board_cards)))
            opp_cards = sample[:2]
            future_board = board_cards + sample[2:]

            my_score = evaluate(my_cards + future_board)
            opp_score = evaluate(opp_cards + future_board)

            if my_score > opp_score:
                wins += 1

        return wins / iterations


    def get_action(self, game_state, round_state, active):
        
        if not hasattr(self, "opp_aggression"):
            self.opp_aggression = 0
            self.opp_passive = 0

        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:street]

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1-active]

        continue_cost = opp_pip - my_pip
        pot = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)

        if continue_cost > 6:
            self.opp_aggression += 1
        else:
            self.opp_passive += 1

        strength = self.monte_carlo_strength(my_cards, board_cards, 80)

        if self.opp_aggression > self.opp_passive:
            strength -= 0.05
        else:
            strength += 0.05

        pot_odds = continue_cost / (pot + 1) if continue_cost > 0 else 0

        bluff_chance = 0.05
        if self.opp_passive > self.opp_aggression:
            bluff_chance = 0.1

        bluff = random.random() < bluff_chance

        if strength > 0.8 and RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()
            return RaiseAction(max_raise)

        if strength > 0.6:
            if RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                if random.random() < 0.5:
                    return RaiseAction((min_raise + max_raise) // 2)
                return RaiseAction(min_raise)
            return CallAction() if continue_cost > 0 else CheckAction()

        if strength > 0.4:
            if continue_cost == 0:
                return CheckAction()
            if pot_odds < strength:
                return CallAction()
            return FoldAction()

        if strength > 0.25:
            if continue_cost == 0:
                return CheckAction()
            if pot_odds < 0.2:
                return CallAction()
            return FoldAction()

        if bluff and RaiseAction in legal_actions and pot < 80:
            min_raise, max_raise = round_state.raise_bounds()
            return RaiseAction(min_raise)

        if continue_cost == 0:
            return CheckAction()

        if continue_cost < 4:
            return CallAction()

        return FoldAction()

    
if __name__ == '__main__':
    run_bot(Player(), parse_args())
