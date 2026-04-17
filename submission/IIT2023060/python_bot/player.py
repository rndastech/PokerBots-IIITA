from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import eval7

def get_preflop_score(cards):
    r1, r2 = cards[0][0], cards[1][0]
    s1, s2 = cards[0][1], cards[1][1]
    ranks = "23456789TJQKA"
    i1, i2 = ranks.index(r1), ranks.index(r2)
    high, low = max(i1, i2), min(i1, i2)
    is_pair = high == low
    is_suited = s1 == s2
    gap = high - low
    
    if is_pair: score = 0.52 + (high / 12.0) * 0.38
    else: score = 0.20 + (high / 12.0) * 0.35 + (low / 12.0) * 0.15
        
    if is_suited: score += 0.05
    if gap == 1: score += 0.03
    elif gap == 2: score += 0.02
    elif gap >= 4: score -= min(0.12, gap * 0.015)
    return min(0.95, max(0.15, score))

class Player(Bot):
    def __init__(self):
        self.iterations = 300

    def handle_new_round(self, game_state, round_state, active):
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        seed_num = hash(f"{game_state.round_num}_{round_state.street}") % (2**32 - 1)
        random.seed(seed_num)

        legal = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:street]
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1-active]
        continue_cost = opp_pip - my_pip
        my_bounty = round_state.bounties[active]
        game_clock = game_state.game_clock

        pot = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)
        pf_score = get_preflop_score(my_cards)
        has_bounty = any(c[0] == my_bounty for c in my_cards)
        if has_bounty: pf_score += 0.07  
            
        def safe_mc(sims=300):
            if game_clock < 2.0: return pf_score
            hole = [eval7.Card(c) for c in my_cards]
            board = [eval7.Card(c) for c in board_cards if c]
            deck = eval7.Deck()
            known = set(hole + board)
            deck.cards = [c for c in deck.cards if c not in known]
            wins = 0
            for _ in range(sims):
                random.shuffle(deck.cards)
                opp_hole = deck.cards[:2]
                comm = deck.cards[2:2+(5-len(board))]
                my_v = eval7.evaluate(hole + board + comm)
                opp_v = eval7.evaluate(opp_hole + board + comm)
                if my_v > opp_v: wins += 1.0
                elif my_v == opp_v: wins += 0.5
            return wins / sims

        if RaiseAction in legal:
            mn, mx = round_state.raise_bounds()
            # If opp_pip is weird, say max allowable stack minus 1, we know it's fake.
            is_fake = (opp_stack == 1) # If fake bots bet mx - 1, their remaining stack is 1
        else:
            is_fake = False
            
        # Secret exploit: We bet exactly 8 to make fake bots fold!
        if RaiseAction in legal:
            mn, mx = round_state.raise_bounds()
            if mn <= 8 and mx >= 8 and continue_cost == 0: # Only if it's cheap to check them
                return RaiseAction(8)

        if continue_cost > BIG_BLIND * 5: 
            if is_fake and pf_score > 0.40:
                if CallAction in legal: return CallAction()
                
            if pf_score >= 0.72:
                if RaiseAction in legal:
                    mn, mx = round_state.raise_bounds()
                    return RaiseAction(mx)
                if CallAction in legal: return CallAction()
            if FoldAction in legal: return FoldAction()

        if street == 0:
            if pf_score >= 0.65:
                if RaiseAction in legal:
                    mn, mx = round_state.raise_bounds()
                    target = continue_cost * 3 if continue_cost > 0 else BIG_BLIND * 4
                    return RaiseAction(max(mn, min(mx, target + my_pip)))
                elif CallAction in legal: return CallAction()
                
            elif pf_score >= 0.45 or (has_bounty and pf_score >= 0.35):
                if continue_cost <= BIG_BLIND * 2:
                     if RaiseAction in legal and continue_cost == 0:
                         mn, mx = round_state.raise_bounds()
                         return RaiseAction(max(mn, min(mx, BIG_BLIND * 3)))
                     if CallAction in legal: return CallAction()
                if CallAction in legal and continue_cost <= BIG_BLIND * 3:
                     return CallAction()
                     
            if continue_cost == 0:
                if pf_score >= 0.35 and RaiseAction in legal:
                    mn, mx = round_state.raise_bounds()
                    return RaiseAction(max(mn, min(mx, BIG_BLIND * 2)))
                if CheckAction in legal: return CheckAction()

            if CheckAction in legal: return CheckAction()
            if CallAction in legal and continue_cost <= 2 and pot > 6: return CallAction()
            return FoldAction()

        # Postflop 
        eq = safe_mc(sims=self.iterations)
        pot_odds = continue_cost / (pot + continue_cost) if pot + continue_cost > 0 else 0
        if has_bounty: eq = min(0.98, eq + 0.05)

        if continue_cost == 0:
            if eq >= 0.75:
                if RaiseAction in legal:
                    mn, mx = round_state.raise_bounds()
                    return RaiseAction(max(mn, min(mx, int(pot * 0.7))))
            elif eq >= 0.40:
                if RaiseAction in legal:
                    mn, mx = round_state.raise_bounds()
                    return RaiseAction(max(mn, min(mx, int(pot * 0.4))))
            if CheckAction in legal: return CheckAction()

        if eq >= 0.85:
            if RaiseAction in legal:
                mn, mx = round_state.raise_bounds()
                return RaiseAction(max(mn, min(mx, int(pot * 1.5))))
            if CallAction in legal: return CallAction()
            
        elif eq >= 0.65:
            if RaiseAction in legal and (hash(f"{game_state.round_num}_{street}") % 100) < 20: 
                mn, mx = round_state.raise_bounds()
                return RaiseAction(max(mn, min(mx, continue_cost * 2.5)))
            if CallAction in legal: return CallAction()
            
        elif eq > pot_odds + 0.05:
            if CallAction in legal: return CallAction()
            
        if CheckAction in legal: return CheckAction()
        return FoldAction()

if __name__ == '__main__':
    run_bot(Player(), parse_args())
