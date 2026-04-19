'''
Poker Bot — Ultimate Strategy V1.0 (IEC2025036)
=================================================
A hybrid bot merging the speed and robustness of earlier TAG
versions with an advanced Bayesian Opponent Model, robust
Monte Carlo Engine, Draw Detection, and Pot Odds based decision scaling.

Features:
- O(1) Preflop Equity lookups from 50k iter cached weights
- Scaled Post-Flop Monte Carlo EV simulation based on clock remaining
- Advanced Continuous Bayesian Opponent Tracker (EMA decay)
- Bounty & Position EV Modifiers
- Check-Raise trapping limits
'''

import json
import os
import hashlib
import random

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

try:
    import eval7
except ImportError:
    pass

# Canonical format matching the train script
def get_canonical_name(rank1, rank2, suit1, suit2):
    ranks = 'AKQJT98765432'
    r1_idx = ranks.index(rank1)
    r2_idx = ranks.index(rank2)
    if r1_idx > r2_idx:
        rank1, rank2 = rank2, rank1
    if rank1 == rank2:
        return f"{rank1}{rank2}"
    return f"{rank1}{rank2}{'s' if suit1 == suit2 else 'o'}"

_RANK_ORDER: dict[str, int] = {
    'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
    '9': 9, '8': 8, '7': 7, '6': 6, '5': 5,
    '4': 4, '3': 3, '2': 2,
}

class BetaStat:
    """Bayesian beta distribution stat."""
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    def observe(self, success):
        if success: self.alpha += 1
        else: self.beta += 1
    def decay(self, factor):
        self.alpha *= factor
        self.beta *= factor
    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)
    @property
    def mean_complement(self):
        return self.beta / (self.alpha + self.beta)

class BayesianOpponentModel:
    def __init__(self):
        # Priors
        self.vpip = BetaStat(2, 4)
        self.pfr = BetaStat(2, 6)
        self.fold_to_cbet = BetaStat(3, 3)
        self.aggression = BetaStat(3, 3)
        self.check_raise_freq = BetaStat(1, 9)
        self.rounds = 0

    def apply_decay(self, factor=0.97):
        self.vpip.decay(factor)
        self.pfr.decay(factor)
        self.fold_to_cbet.decay(factor)
        self.aggression.decay(factor)
        self.check_raise_freq.decay(factor)
        
    @property
    def adjustments(self):
        return {
            'equity_discount': 0.85 if self.pfr.mean > 0.50 else (1.05 if self.pfr.mean < 0.20 else 1.0),
            'bluff_freq': min(0.40, self.fold_to_cbet.mean * 0.6),
            'steal_freq': min(0.80, self.vpip.mean_complement * 0.9),
            'is_passive': self.aggression.mean < 0.35,
            'is_aggressive': self.aggression.mean > 0.60,
        }

class Player(Bot):
    def __init__(self):
        self.bounty_rank = None
        self.model = BayesianOpponentModel()
        self.preflop_equities = {}
        self.curr_opp_pfr = False # Track if opp raised THIS hand
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(dir_path, "preflop_weights.json")
        try:
            with open(weights_path, "r") as f:
                self.preflop_equities = json.load(f)
        except Exception:
            pass

    def handle_new_round(self, game_state, round_state, active):
        self.bounty_rank = round_state.bounties[active]
        self.curr_opp_pfr = False

    def handle_round_over(self, game_state, terminal_state, active):
        previous_state = terminal_state.previous_state
        if previous_state is None:
            return
            
        opp_contribution = STARTING_STACK - previous_state.stacks[1 - active]
        my_contribution = STARTING_STACK - previous_state.stacks[active]
        street = previous_state.street
        
        self.model.rounds += 1
        
        # We need to know if they raised preflop for the model
        # But for the current hand simulation, we track it in get_action
        opp_pfr = (opp_contribution > BIG_BLIND * 2)
        
        if opp_contribution > BIG_BLIND:
            self.model.vpip.observe(True)
        else:
            self.model.vpip.observe(False)
            
        self.model.pfr.observe(opp_pfr)
            
        folded = (my_contribution > opp_contribution and my_contribution > BIG_BLIND)
        if street > 0:
            if opp_contribution > my_contribution:
                self.model.aggression.observe(True)
            self.model.fold_to_cbet.observe(folded)
        
        self.model.apply_decay(0.97)

    def _pseudo_random(self, salt=""):
        h = hashlib.md5(f"{self.model.rounds}_{salt}".encode())
        return int(h.hexdigest(), 16) % 100

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:street]
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        pot_size = (STARTING_STACK - my_stack) + (STARTING_STACK - round_state.stacks[1 - active])
        continue_cost = opp_pip - my_pip
        
        min_raise, max_raise = 0, 0
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()

        # Update current hand state
        if street == 0 and opp_pip > BIG_BLIND:
            self.curr_opp_pfr = True

        has_bounty = (my_cards[0][0] == self.bounty_rank or my_cards[1][0] == self.bounty_rank)
        is_oop = (street == 0 and active == 1) or (street > 0 and active == 0)

        # Basic fallback for time
        if game_state.game_clock < 2.5:
            if continue_cost == 0 and CheckAction in legal_actions: return CheckAction()
            if CallAction in legal_actions and continue_cost <= BIG_BLIND * 2: return CallAction()
            if FoldAction in legal_actions: return FoldAction()
            return CheckAction()

        equity = 0.5
        if street == 0:
            canonical = get_canonical_name(my_cards[0][0], my_cards[1][0], my_cards[0][1], my_cards[1][1])
            equity = self.preflop_equities.get(canonical, 0.45)
            if not is_oop: equity += 0.02
        else:
            hole = [eval7.Card(c) for c in my_cards]
            board = [eval7.Card(c) for c in board_cards]
            equity = self._postflop_equity(hole, board, game_state.game_clock)

        # Bounty Bonus
        bounty_ev = 0
        if has_bounty:
            bounty_on_board = any(c[0] == self.bounty_rank for c in board_cards)
            if bounty_on_board:
                bounty_ev = (pot_size * 0.4 + 10) * equity
            else:
                remaining_cards = 5 - street
                if remaining_cards > 0:
                    outs = sum(1 for c in my_cards if c[0] == self.bounty_rank)
                    if outs > 0:
                        bounty_ev = (pot_size * 0.4 + 10) * equity * (3 / 47.0 * remaining_cards)
                        
        effective_equity = equity + (bounty_ev / max(1, pot_size))

        draw_boost = 0.0
        if street > 0 and street < 5:
            fd, oesd, gut = self._detect_draws(my_cards, board_cards)
            if fd: draw_boost += 0.14 if street == 3 else 0.07
            if oesd: draw_boost += 0.12 if street == 3 else 0.06
            if gut: draw_boost += 0.05 if street == 3 else 0.02
        
        effective_equity += draw_boost

        adj = self.model.adjustments
        pot_odds = continue_cost / max(1.0, float(pot_size + continue_cost))
        
        if adj['is_aggressive'] and continue_cost > 0:
            effective_equity *= 1.12
        elif adj['is_passive'] and continue_cost > 0:
            effective_equity *= 0.82

        if street == 0:
            return self._preflop_decision(effective_equity, continue_cost, pot_odds, legal_actions, min_raise, max_raise, my_pip, adj)
        else:
            return self._postflop_decision(street, effective_equity, continue_cost, pot_odds, legal_actions, min_raise, max_raise, my_pip, pot_size, adj, is_oop)

    def _preflop_decision(self, eq, cost, odds, legal, min_r, max_r, my_pip, adj):
        if eq >= 0.58:
            if RaiseAction in legal:
                sz = my_pip + max(cost * 3, 6)
                if eq > 0.65: sz = my_pip + max(cost * 4, 12)
                return RaiseAction(min(max(sz, min_r), max_r))
            if CallAction in legal: return CallAction()
        elif eq >= 0.48:
            if cost <= BIG_BLIND and RaiseAction in legal and self._pseudo_random("steal") < adj['steal_freq'] * 100:
                return RaiseAction(min(max_r, max(min_r, my_pip + 6)))
            if eq >= odds + 0.03 and CallAction in legal:
                return CallAction()
            if cost == 0 and CheckAction in legal: return CheckAction()
        
        if cost == 0 and CheckAction in legal: return CheckAction()
        if FoldAction in legal: return FoldAction()
        return CallAction()

    def _postflop_decision(self, street, eq, cost, odds, legal, min_r, max_r, my_pip, pot, adj, is_oop):
        # Street-specific tightness
        # On turn/river, we need much more than just raw pot odds
        call_buffer = 0.04
        if street == 4: call_buffer = 0.08
        if street == 5: call_buffer = 0.12

        # Small pot exception
        if cost <= 4: call_buffer -= 0.03

        # Check-Raise trap
        if cost > 0 and is_oop and eq > 0.72 and adj['is_aggressive'] and self._pseudo_random("cr") < 20:
            if RaiseAction in legal:
                sz = my_pip + max(cost * 3, pot // 2)
                return RaiseAction(min(max(sz, min_r), max_r))
                
        # Pure Value/Bluff logic
        if eq > odds + 0.18:
            if RaiseAction in legal:
                sz = pot // 2 if eq < 0.78 else pot
                return RaiseAction(min(max(my_pip + int(sz), min_r), max_r))
            if CallAction in legal: return CallAction()
            
        elif eq > odds + call_buffer:
            if CallAction in legal: return CallAction()
            
        if cost == 0 and CheckAction in legal:
            # Semi-bluff / stab
            if RaiseAction in legal and self._pseudo_random("stab") < adj['bluff_freq'] * 100:
                sz = min(max_r, max(min_r, my_pip + int(pot * 0.45)))
                return RaiseAction(sz)
            return CheckAction()
            
        if FoldAction in legal: return FoldAction()
        if CheckAction in legal: return CheckAction()
        return CallAction()

    def _detect_draws(self, hole, board):
        all_c = hole + board
        suits = [c[1] for c in all_c]
        ranks = sorted(list(set([_RANK_ORDER[c[0]] for c in all_c])))
        
        fd = any(suits.count(s) >= 4 for s in 'cdhs')
        
        oesd, gut = False, False
        if len(ranks) >= 4:
            for i in range(len(ranks)-3):
                subset = ranks[i:i+4]
                if subset[-1] - subset[0] == 3:
                    oesd = True
                elif subset[-1] - subset[0] == 4:
                    gut = True
        return fd, oesd, gut

    def _postflop_equity(self, hole, board, clock):
        if clock < 4.0: return self._fast_hand_strength(hole, board)
        
        iters = 80 if clock < 15.0 else (180 if clock < 30.0 else 300)
        wins, ties = 0, 0
        known = set(hole + board)
        deck = [eval7.Card(r+s) for r in 'AKQJT98765432' for s in 'cdhs' if eval7.Card(r+s) not in known]
        
        # Preflop-Aware Range Filtering
        # If opp raised preflop, filter out bottom 50% of hands (heuristic)
        opp_range = deck
        if self.curr_opp_pfr:
             # Basic heuristic: if they raised, they have at least top-tier cards
             # We simulate them having a pair or T+ high card mostly. 
             # In a real engine we'd use a table. Here we'll just filter slightly during picking.
             pass

        for _ in range(iters):
            random.shuffle(deck)
            
            # Simple heuristic: if they raised preflop, skip the 2nd card if it's very low
            # This simulates a "strength filtered" random range
            o = deck[:2]
            if self.curr_opp_pfr and _RANK_ORDER[str(o[0])[0]] < 8 and _RANK_ORDER[str(o[1])[0]] < 8:
                # Re-pick once to bias towards higher cards
                o = deck[2:4]
            
            rem = 5 - len(board)
            f_board = board + deck[4:4+rem]
            mv = eval7.evaluate(hole + f_board)
            ov = eval7.evaluate(o + f_board)
            if mv > ov: wins += 1
            elif mv == ov: ties += 1
        return (wins + ties/2.0) / iters

    def _fast_hand_strength(self, hole, board):
        rank = eval7.evaluate(hole + board)
        ht = eval7.handtype(rank)
        m = { 'Straight Flush': 0.99, 'Quads': 0.98, 'Full House': 0.95, 
              'Flush': 0.90, 'Straight': 0.85, 'Trips': 0.80, 
              'Two Pair': 0.72, 'Pair': 0.55, 'High Card': 0.30 }
        return m.get(ht, 0.30)

if __name__ == '__main__':
    run_bot(Player(), parse_args())
