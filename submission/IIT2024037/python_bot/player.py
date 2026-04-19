import eval7
import random

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import STARTING_STACK, BIG_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

BOUNTY_RATIO = 1.5
BOUNTY_CONSTANT = 10

# ==========================================
# PRECOMPUTED PREFLOP EQUITY TABLE (O(1) Speed)
# ==========================================
RANK_VALUE = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

PREFLOP_EQUITY = {}
def _build_preflop_table():
    pair_eq = {
        14: 0.852, 13: 0.824, 12: 0.799, 11: 0.773, 10: 0.750,
        9: 0.720, 8: 0.691, 7: 0.664, 6: 0.636, 5: 0.605,
        4: 0.577, 3: 0.546, 2: 0.512
    }
    for r, eq in pair_eq.items():
        PREFLOP_EQUITY[(r, r, True, False)] = eq
        PREFLOP_EQUITY[(r, r, True, True)] = eq
    suited_eq = {
        (14,13):0.670,(14,12):0.662,(14,11):0.654,(14,10):0.646,
        (14,9):0.621,(14,8):0.610,(14,7):0.600,(14,6):0.589,
        (14,5):0.585,(14,4):0.575,(14,3):0.567,(14,2):0.558,
        (13,12):0.636,(13,11):0.627,(13,10):0.618,(13,9):0.594,
        (13,8):0.577,(13,7):0.566,(13,6):0.553,(13,5):0.544,
        (13,4):0.534,(13,3):0.525,(13,2):0.515,
        (12,11):0.604,(12,10):0.595,(12,9):0.573,(12,8):0.553,
        (12,7):0.541,(12,6):0.529,(12,5):0.519,(12,4):0.508,
        (12,3):0.498,(12,2):0.487,
        (11,10):0.575,(11,9):0.555,(11,8):0.535,(11,7):0.521,
        (11,6):0.508,(11,5):0.498,(11,4):0.487,(11,3):0.477,
        (11,2):0.466,
        (10,9):0.539,(10,8):0.521,(10,7):0.505,(10,6):0.491,
        (10,5):0.479,(10,4):0.468,(10,3):0.458,(10,2):0.446,
        (9,8):0.507,(9,7):0.492,(9,6):0.475,(9,5):0.462,
        (9,4):0.449,(9,3):0.437,(9,2):0.426,
        (8,7):0.483,(8,6):0.465,(8,5):0.450,(8,4):0.436,
        (8,3):0.423,(8,2):0.412,
        (7,6):0.456,(7,5):0.442,(7,4):0.427,(7,3):0.413,(7,2):0.400,
        (6,5):0.432,(6,4):0.417,(6,3):0.401,(6,2):0.387,
        (5,4):0.414,(5,3):0.398,(5,2):0.383,
        (4,3):0.389,(4,2):0.374,
        (3,2):0.371,
    }
    for (h, l), eq in suited_eq.items():
        PREFLOP_EQUITY[(h, l, False, True)] = eq
    for (h, l), eq in suited_eq.items():
        PREFLOP_EQUITY[(h, l, False, False)] = eq - 0.035

_build_preflop_table()

def get_preflop_equity(my_cards):
    r1 = RANK_VALUE.get(my_cards[0][0], 5)
    r2 = RANK_VALUE.get(my_cards[1][0], 5)
    high, low = max(r1, r2), min(r1, r2)
    is_pair = r1 == r2
    is_suited = (len(my_cards[0]) > 1 and len(my_cards[1]) > 1
                 and my_cards[0][1] == my_cards[1][1])
    return PREFLOP_EQUITY.get((high, low, is_pair, is_suited), 0.42)

_FULL_DECK =[eval7.Card(r + s) for r in '23456789TJQKA' for s in 'shdc']

# ==========================================
# MAIN BOT CLASS
# ==========================================
class Player(Bot):
    '''
    The Iron Turtle (Defensive Counter-Bot)
    Designed specifically to defensively exploit aggressive, profiling bots 
    by neutralizing their sizing tells and trapping them.
    '''
    def __init__(self):
        self.equity_cache = {}
        self.bounty_rank = None

    def handle_new_round(self, game_state, round_state, active):
        self.bounty_rank = round_state.bounties[active]

    def handle_round_over(self, game_state, terminal_state, active):
        # We clear the cache intermittently to preserve memory, though identical hands benefit from persistence
        if game_state.round_num % 50 == 0:
            self.equity_cache.clear()

    def calculate_equity(self, hole_cards_str, board_cards_str, iterations=150):
        # Cache results to completely dodge clock timeouts
        key = (tuple(hole_cards_str), tuple(board_cards_str))
        if key in self.equity_cache:
            return self.equity_cache[key]
            
        my_cards = [eval7.Card(c) for c in hole_cards_str]
        board = [eval7.Card(c) for c in board_cards_str]
        used_cards = set(my_cards + board)
        
        deck =[c for c in _FULL_DECK if c not in used_cards]
        needed_board = 5 - len(board)
        wins = 0.0
        
        for _ in range(iterations):
            draw = random.sample(deck, needed_board + 2)
            sim_board = board + draw[:needed_board]
            opp_cards = draw[needed_board:]
            
            my_val = eval7.evaluate(my_cards + sim_board)
            opp_val = eval7.evaluate(opp_cards + sim_board)
            
            if my_val > opp_val:
                wins += 1.0
            elif my_val == opp_val:
                wins += 0.5
                
        eq = wins / iterations
        self.equity_cache[key] = eq
        return eq

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:street]
        
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1-active]
        
        cost = opp_pip - my_pip
        pot_size = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)

        min_raise, max_raise = 0, 0
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()

        has_bounty = (my_cards[0][0] == self.bounty_rank or my_cards[1][0] == self.bounty_rank)
        board_has_bounty = any(c[0] == self.bounty_rank for c in board_cards) if board_cards else False
        bounty_active = has_bounty or board_has_bounty
        
        # Calculate actual game value
        eff_pot = pot_size * BOUNTY_RATIO + BOUNTY_CONSTANT if bounty_active else pot_size
        pot_odds = cost / max(1, eff_pot + cost)

        # Emergency Clock Safety
        if game_state.game_clock < 2.0:
            if CheckAction in legal_actions: return CheckAction()
            if FoldAction in legal_actions: return FoldAction()
            return CallAction()

        # ==========================================
        # PREFLOP STRATEGY (Tight-Aggressive Defense)
        # ==========================================
        if street == 0:
            equity = get_preflop_equity(my_cards)
            
            # Bounty inflates preflop value defensively
            if has_bounty:
                equity += 0.06

            if cost > 0:
                # Facing a raise
                if cost > STARTING_STACK * 0.4:  # All-in or huge 4-bet jam
                    if equity > 0.62:
                        if CallAction in legal_actions: return CallAction()
                    if FoldAction in legal_actions: return FoldAction()

                if equity >= 0.65:
                    # Premium: 3-bet for value
                    if RaiseAction in legal_actions:
                        sz = min(max_raise, my_pip + int(cost * 2.5))
                        return RaiseAction(sz)
                    if CallAction in legal_actions: return CallAction()
                
                elif equity >= 0.48:
                    # Strong: Defend by calling
                    if CallAction in legal_actions: return CallAction()
                    
                elif equity >= 0.42 and cost <= BIG_BLIND * 3:
                    # Playable: Defend against standard opens
                    if CallAction in legal_actions: return CallAction()
                    
                if FoldAction in legal_actions: return FoldAction()

            else:
                # Free to act (We are the SB/Opening)
                if equity >= 0.60:
                    if RaiseAction in legal_actions:
                        # Standard 2.5x open
                        sz = min(max_raise, min_raise + BIG_BLIND)
                        return RaiseAction(sz)
                elif equity >= 0.45:
                    if RaiseAction in legal_actions:
                        return RaiseAction(min_raise)
                elif equity >= 0.38:
                    # Limp / check
                    if CheckAction in legal_actions: return CheckAction()
                    
                if FoldAction in legal_actions: return FoldAction()
                if CheckAction in legal_actions: return CheckAction()

        # ==========================================
        # POSTFLOP STRATEGY (Defensive Exploit Engine)
        # ==========================================
        if street > 0:
            # Scale MC iterations smoothly based on clock to prevent lagouts
            sim_count = 150 if game_state.game_clock > 15.0 else (80 if game_state.game_clock > 5.0 else 30)
            
            equity = self.calculate_equity(my_cards, board_cards, iterations=sim_count)
            
            if has_bounty:
                # Postflop bounty inflation 
                equity = min(0.99, equity + 0.08)

            bet_ratio = cost / max(1, pot_size) if cost > 0 else 0

            # --- Facing a Bet ---
            if cost > 0:
                # 1. Overbets (Pure Value from opponent. We completely dodge their trap)
                if bet_ratio >= 1.0:
                    req_eq = 0.85
                    if equity >= req_eq:
                        if CallAction in legal_actions: return CallAction()
                    if FoldAction in legal_actions: return FoldAction()

                # 2. Large Bets (Strong Value or River Bluffs)
                elif bet_ratio >= 0.7:
                    # Opponent bluffs big on the river, so we call slightly wider on street 5 to catch them.
                    req_eq = 0.75 if street < 5 else 0.65
                    if equity >= req_eq:
                        if CallAction in legal_actions: return CallAction()
                    if FoldAction in legal_actions: return FoldAction()

                # 3. Standard/Small Bets (Thin Value, Probes, or Bluffs)
                else:
                    if equity >= 0.85:
                        # Monster: Protect and extract
                        if RaiseAction in legal_actions:
                            sz = min(max_raise, my_pip + cost + int(pot_size * 0.6))
                            return RaiseAction(sz)
                        if CallAction in legal_actions: return CallAction()
                        
                    # Defensive Call Margin: Pot Odds + Range Disadvantage Buffer
                    req_eq = pot_odds + 0.08
                    if equity >= req_eq:
                        if CallAction in legal_actions: return CallAction()
                        
                    if FoldAction in legal_actions: return FoldAction()

            # --- Free to Act (Checked to Us) ---
            if cost == 0:
                if equity >= 0.80:
                    # Solid value bet. Keep it manageable to induce calls.
                    if RaiseAction in legal_actions:
                        sz = min(max_raise, min_raise + int(pot_size * 0.45))
                        return RaiseAction(sz)
                        
                elif equity >= 0.65:
                    # Thin value / Probe. Mix bets and checks to stay unpredictable to their profiler.
                    if random.random() < 0.4 and RaiseAction in legal_actions:
                        sz = min(max_raise, min_raise + int(pot_size * 0.30))
                        return RaiseAction(sz)
                    if CheckAction in legal_actions: return CheckAction()

                # If marginal or weak, check safely
                if CheckAction in legal_actions: return CheckAction()

        # Fallbacks
        if CheckAction in legal_actions: return CheckAction()
        if CallAction in legal_actions: return CallAction()
        if FoldAction in legal_actions: return FoldAction()

if __name__ == '__main__':
    run_bot(Player(), parse_args())