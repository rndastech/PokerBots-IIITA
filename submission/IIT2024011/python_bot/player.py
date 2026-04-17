import eval7
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

class Player(Bot):
    def handle_new_round(self, game_state, round_state, active):
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        street = round_state.street  # 0=Preflop, 3=Flop, 4=Turn, 5=River
        
        my_cards_str = round_state.hands[active]
        board_cards_str = round_state.deck

        # ====================================================================
        # CORE MATH & AWARENESS 
        # ====================================================================
        
        # Level 2: Pot Odds & Sizing Math
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1-active]
        pot = (400 - my_stack) + (400 - opp_stack) # Assuming 400 starting stack
        continue_cost = round_state.pips[1-active] - round_state.pips[active]
        
        # Level 3: Position Awareness (Button acts last post-flop)
        is_button = (active == round_state.button % 2)

        # Level 4: Bounty Mechanics
        my_bounty_rank = round_state.bounties[active]
        bounty_hit = False
        if my_bounty_rank != '-1':
            all_visible_cards = my_cards_str + board_cards_str
            if any(card[0] == my_bounty_rank for card in all_visible_cards):
                bounty_hit = True

        # ====================================================================
        # PRE-FLOP STRATEGY (Street 0)
        # ====================================================================
        if street == 0:
            rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
            r1, r2 = rank_map[my_cards_str[0][0]], rank_map[my_cards_str[1][0]]
            high_card, low_card = max(r1, r2), min(r1, r2)
            
            # Good hand: Pair, two big cards, or Ace with decent kicker
            is_strong = (r1 == r2) or (low_card >= 10) or (high_card == 14 and low_card >= 8)

            if bounty_hit:
                is_strong = True # Never fold a bounty pre-flop!

            if is_strong:
                if RaiseAction in legal_actions:
                    min_raise, max_raise = round_state.raise_bounds()
                    # Value Bet: Raise min + half the pot
                    target_raise = min(max_raise, min_raise + int(pot * 0.5))
                    return RaiseAction(target_raise)
                if CallAction in legal_actions:
                    return CallAction()
                return CheckAction()
            else:
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()

        # ====================================================================
        # POST-FLOP STRATEGY (Streets 3, 4, 5)
        # ====================================================================
        else:
            # Level 1: Eval7 Integration
            my_eval7_cards = [eval7.Card(c) for c in my_cards_str]
            board_eval7_cards = [eval7.Card(c) for c in board_cards_str]

            # Evaluate hand strength vs the board
            my_value = eval7.evaluate(my_eval7_cards + board_eval7_cards)
            board_value = eval7.evaluate(board_eval7_cards)

            hand_type = eval7.handtype(my_value) # e.g., "Pair", "Flush"
            improved_board = my_value > board_value # Did our hole cards actually help?

            # Classify hand strength
            very_strong = improved_board and hand_type in ["Two Pair", "Trips", "Straight", "Flush", "Full House", "Quads", "Straight Flush"]
            decent_hand = improved_board and hand_type == "Pair"

            # Execute Logic
            if very_strong:
                if RaiseAction in legal_actions:
                    min_raise, max_raise = round_state.raise_bounds()
                    # Extract maximum value: Bet 75% of the pot
                    target_raise = min(max_raise, min_raise + int(pot * 0.75))
                    return RaiseAction(target_raise)
                if CallAction in legal_actions:
                    return CallAction()

            elif decent_hand or bounty_hit:
                # We have a Pair OR we hit our bounty. Play defensively but don't easily fold.
                if continue_cost <= pot * 0.5: # Only call if pot odds are reasonable
                    if CallAction in legal_actions:
                        return CallAction()
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()

            else:
                # We have absolute trash.
                # Secret Bluffing Mechanic: If we act last and opponent checked, try to steal the pot!
                if is_button and continue_cost == 0 and RaiseAction in legal_actions:
                    min_raise, _ = round_state.raise_bounds()
                    return RaiseAction(min_raise) # Small bet to scare them

                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()

if __name__ == '__main__':
    run_bot(Player(), parse_args())