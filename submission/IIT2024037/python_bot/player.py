'''
The Extractor (Phase 1) — Preflop & Profiling Engine
=====================================================
A dynamically adjusting Heads-Up bot that profiles the opponent
and calculates mathematical Expected Value (EV).
'''
import eval7
import random

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import STARTING_STACK, BIG_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

# --- O(1) Preflop Tier System for Heads-Up ---
# 1 = Premium (3-bet/Jam), 2 = Strong (Raise/Call 3-bet), 
# 3 = Playable (Raise/Call), 4 = Marginal (Call cheap), 5 = Trash (Fold unless free)
def get_preflop_tier(card1, card2):
    rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    r1, r2 = rank_map[card1[0]], rank_map[card2[0]]
    suited = card1[1] == card2[1]
    hi, lo = max(r1, r2), min(r1, r2)
    
    if hi == lo: 
        return 1 if hi >= 8 else 2 # Pocket pairs are huge in HU
    if hi == 14: 
        return 1 if lo >= 10 else (2 if suited else 3) # Any Ace is playable
    if hi >= 10 and lo >= 9: 
        return 2 # High Broadway
    if suited and (hi - lo <= 2) and hi >= 6: 
        return 3 # Suited connectors
    if hi >= 9 or (suited and hi >= 7): 
        return 4
    return 5

# --- Opponent Memory Tracker ---
class OpponentModel:
    def __init__(self):
        self.hands_played = 0
        self.vpip_count = 0  # Voluntarily put chips in (calls/raises)
        self.pfr_count = 0   # Preflop raises
        self.folds_to_steal = 0
        self.steal_attempts_faced = 0

    def update_vpip(self): self.vpip_count += 1
    def update_pfr(self): self.pfr_count += 1
    
    @property
    def vpip(self): return self.vpip_count / max(1, self.hands_played)
    
    @property
    def pfr(self): return self.pfr_count / max(1, self.hands_played)
    
    @property
    def is_nit(self): return self.hands_played > 20 and self.vpip < 0.40
    
    @property
    def is_maniac(self): return self.hands_played > 20 and self.pfr > 0.60


# --- Fast Monte Carlo Simulator ---
_FULL_DECK = [eval7.Card(r + s) for r in '23456789TJQKA' for s in 'shdc']

def calculate_equity(hole_cards_str, board_cards_str, iterations=50):
    """
    Runs a fast Monte Carlo simulation to find our exact win probability (0.0 to 1.0).
    """
    my_cards = [eval7.Card(c) for c in hole_cards_str]
    board = [eval7.Card(c) for c in board_cards_str]
    used_cards = set(my_cards + board)
    
    # Create the remaining deck
    deck = [c for c in _FULL_DECK if c not in used_cards]
    needed_board = 5 - len(board)
    
    wins = 0.0
    for _ in range(iterations):
        # Faster than random.shuffle
        draw = random.sample(deck, needed_board + 2)
        sim_board = board + draw[:needed_board]
        opp_cards = draw[needed_board:]
        
        my_val = eval7.evaluate(my_cards + sim_board)
        opp_val = eval7.evaluate(opp_cards + sim_board)
        
        if my_val > opp_val:
            wins += 1.0
        elif my_val == opp_val:
            wins += 0.5
            
    return wins / iterations

class Player(Bot):
    def __init__(self):
        self.opp = OpponentModel()

    def handle_new_round(self, game_state, round_state, active):
        self.opp.hands_played += 1
        self.bounty_rank = round_state.bounties[active]

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:street]
        
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1-active]
        
        continue_cost = opp_pip - my_pip
        pot_size = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)

        min_raise, max_raise = 0, 0
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()

        # Is the bounty card in our hand?
        has_bounty = (my_cards[0][0] == self.bounty_rank or my_cards[1][0] == self.bounty_rank)

        # Clock Safety
        if game_state.game_clock < 2.0:
            if CheckAction in legal_actions: return CheckAction()
            if FoldAction in legal_actions: return FoldAction()
            return CallAction()

        # ==========================================
        # PREFLOP STRATEGY (O(1) Matrix + Exploit)
        # ==========================================
        if street == 0:
            tier = get_preflop_tier(my_cards[0], my_cards[1])
            if has_bounty: tier -= 1 # Bump up a tier if we have the bounty
            
            # Update Opponent Stats
            if continue_cost > BIG_BLIND: self.opp.update_pfr()
            if continue_cost > 0: self.opp.update_vpip()

            # --- Facing a Raise ---
            if continue_cost > 0:
                if tier <= 2: # Premium/Strong
                    if RaiseAction in legal_actions and not self.opp.is_nit:
                        return RaiseAction(min(max_raise, my_pip + continue_cost * 3))
                    if CallAction in legal_actions: return CallAction()
                
                if tier == 3 and CallAction in legal_actions:
                    # Don't call big raises with marginal hands unless they are a maniac
                    if continue_cost < (my_stack // 4) or self.opp.is_maniac:
                        return CallAction()
                
                if FoldAction in legal_actions: return FoldAction()

            # --- Free to Act / Opening ---
            if continue_cost == 0:
                # Exploit: If they are a Nit, steal their blinds relentlessly with any two cards
                if self.opp.is_nit and RaiseAction in legal_actions:
                    return RaiseAction(min_raise)

                if tier <= 4 and RaiseAction in legal_actions:
                    # Standard open is 2.5x to 3x
                    sz = min(max_raise, min_raise + BIG_BLIND)
                    return RaiseAction(sz)
                
                if CheckAction in legal_actions: return CheckAction()
                if FoldAction in legal_actions: return FoldAction()

            if CallAction in legal_actions: return CallAction()

        # ==========================================
        # POSTFLOP STRATEGY (Placeholder for Phase 2)
        # ==========================================
# ==========================================
        # POSTFLOP STRATEGY (The EV Exploit Engine)
        # ==========================================
        if street > 0:
            # 1. Budget our clock time. If we are running out of time, run fewer sims.
            sim_count = 50 if game_state.game_clock > 15.0 else 20
            
            # 2. Calculate our mathematical Win Probability (Equity)
            equity = calculate_equity(my_cards, board_cards, iterations=sim_count)
            
            # If we hold the bounty, the pot is technically worth more to us.
            if has_bounty:
                equity = min(0.99, equity + 0.08)

            # 3. Calculate Pot Odds (How much it costs vs what we can win)
            pot_odds = continue_cost / max(1, pot_size + continue_cost)

            # --- Facing a Bet ---
            if continue_cost > 0:
                # Value Raise: If we have a monster, extract chips.
                if equity > 0.70 and RaiseAction in legal_actions:
                    # Size our raise based on how deep the stacks are
                    sz = min(max_raise, my_pip + continue_cost + int(pot_size * 0.75))
                    return RaiseAction(sz)

                # Positive Expected Value (+EV) Call: 
                # If our chance of winning is greater than the pot odds, math says we MUST call.
                # Example: Bet is 1/3 pot (Pot odds 20%). If our equity is > 20%, we call.
                if equity > (pot_odds + 0.03):  # +3% buffer for safety
                    if CallAction in legal_actions: return CallAction()

                # Mathematical Fold: We are drawing dead or not getting the right price.
                if FoldAction in legal_actions: return FoldAction()

            # --- Free to Act / Checked to Us ---
            if continue_cost == 0:
                # Value Bet: We are likely ahead, build the pot.
                if equity > 0.60 and RaiseAction in legal_actions:
                    # Bet 60% of the pot for pure value
                    sz = min(max_raise, min_raise + int(pot_size * 0.60))
                    return RaiseAction(sz)

                # The "Nit" Exploit (Bluffing): 
                # If the opponent folds too much preflop (is_nit) and they checked to us, 
                # AND we have bad cards (equity < 0.35), we bluff 50% of the pot.
                if equity < 0.35 and self.opp.is_nit and RaiseAction in legal_actions:
                    bluff_sz = min(max_raise, min_raise + int(pot_size * 0.50))
                    return RaiseAction(bluff_sz)

                # Showdown Value / Checking
                if CheckAction in legal_actions: return CheckAction()

            # Fallbacks
            if CallAction in legal_actions: return CallAction()
            if FoldAction in legal_actions: return FoldAction()
    
if __name__ == '__main__':
    run_bot(Player(), parse_args())