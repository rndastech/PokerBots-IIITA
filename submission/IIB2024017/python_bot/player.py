'''
Advanced TAG Pokerbot (God-Tier Version)
Roll No: IIB2024017 
Mridankan Mandal
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import eval7

def bounty_hit_prob_future(bounty_rank, visible_hole, visible_board, remaining_board_count):
    visible = visible_hole + visible_board
    if bounty_rank in {c[0] for c in visible}: return 1.0
    if remaining_board_count <= 0: return 0.0
    unknown = 52 - len(visible)
    bounty_left = 4
    p_none = 1.0
    for i in range(remaining_board_count):
        p_none *= max(0, unknown - bounty_left - i) / (unknown - i)
    return 1.0 - p_none

def bounty_expected_multiplier(my_bounty, hero_str, board_str, street):
    visible_hole_ranks = {c[0] for c in hero_str}
    visible_board_ranks = {c[0] for c in board_str}
    remaining = 5 - len(board_str)
    if my_bounty in visible_hole_ranks or my_bounty in visible_board_ranks: return 1.5, 10.0
    p_hit = bounty_hit_prob_future(my_bounty, hero_str, board_str, remaining)
    return 1.0 + 0.5 * p_hit, 10.0 * p_hit

def opp_bounty_expected_multiplier(hero_str, board_str, street):
    visible_hole_ranks = {c[0] for c in hero_str}
    visible_board_ranks = {c[0] for c in board_str}
    remaining = 5 - len(board_str)
    total_p = 0.0
    for rank in '23456789TJQKA':
        if rank in visible_board_ranks:
            p_hit = 1.0
        else:
            cards_removed = 2 + len(board_str)
            rank_left = 4 - (1 if rank in visible_hole_ranks else 0)
            hidden = 52 - cards_removed
            drawn = 2 + remaining
            p_none = 1.0
            for i in range(drawn):
                denom = hidden - i
                if denom <= 0:
                    p_none = 0.0
                    break
                p_none *= max(0, denom - rank_left - i) / denom
            p_hit = 1.0 - p_none
        total_p += p_hit
    avg_p_hit = total_p / 13.0
    return 1.0 + 0.5 * avg_p_hit, 10.0 * avg_p_hit

def marginal_payoffs(pot, my_contrib, continue_cost, my_bm, my_bb, opp_bm, opp_bb):
    opp_contrib = pot - my_contrib
    win_marginal = opp_contrib * my_bm + my_bb + my_contrib
    lose_marginal = (my_contrib + continue_cost) * opp_bm + opp_bb - my_contrib
    return win_marginal, lose_marginal

def _rough_made_strength(hero_str, board_str):
    hero_ranks = [c[0] for c in hero_str]
    board_ranks = [c[0] for c in board_str]
    rank_value = {c: i + 2 for i, c in enumerate('23456789TJQKA')}
    if hero_ranks[0] == hero_ranks[1]:
        v = rank_value[hero_ranks[0]]
        return min(0.85, 0.55 + v * 0.015)
    top_board = max([rank_value[r] for r in board_ranks]) if board_ranks else 0
    have_tp = any(rank_value[h] == top_board for h in hero_ranks)
    have_board_pair = len(set(board_ranks)) < len(board_ranks)
    if have_tp: return 0.6
    if have_board_pair: return 0.45
    high = max(rank_value[h] for h in hero_ranks)
    return 0.30 + 0.01 * high

class Player(Bot):
    def __init__(self):
        self.w2f_count = 0  # Went-to-Flop count (proxy for VPIP)
        self.round_count = 0
        self.opp_folds = 0
        self.bluff_opportunities = 0
        self.is_baseline = None

    def handle_new_round(self, game_state, round_state, active):
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        self.round_count += 1
        previous_state = terminal_state.previous_state
        if previous_state.street > 0:
            self.w2f_count += 1

        # Fold Equity Tracking: Did the hand end before showdown?
        if previous_state.street < 5:
            # Did they fold? (We won the delta!)
            my_delta = terminal_state.deltas[active]
            if my_delta > 0:
                self.opp_folds += 1
            self.bluff_opportunities += 1

    def get_preflop_tier(self, my_cards):
        c1, c2 = my_cards
        r1, r2 = c1.rank, c2.rank
        if r1 < r2: r1, r2 = r2, r1
        suited = (c1.s[1] == c2.s[1])

        # Tier S: AA, KK, QQ, JJ, AKs, AKo
        if (r1 == r2 and r1 >= 9) or (r1 == 12 and r2 == 11):
            return 'S'
        # Tier A: TT, 99, AQs, AQo, AJs, AJo, KQs
        if (r1 == r2 and r1 >= 7) or (r1 == 12 and r2 >= 9) or (r1 == 11 and r2 == 10 and suited):
            return 'A'
        # Tier B: 88, 77, ATs, KJs, QJs, JTs
        if (r1 == r2 and r1 >= 5) or (r1 == 12 and r2 == 8 and suited) or (r1 == 11 and r2 == 9 and suited) or (r1 == 10 and r2 == 9 and suited) or (r1 == 9 and r2 == 8 and suited):
            return 'B'
        # Trash
        return 'C'

    def get_action(self, game_state, round_state, active):
        try:
            return self._get_action(game_state, round_state, active)
        except Exception:
            if CheckAction in round_state.legal_actions():
                return CheckAction()
            return FoldAction()

    def _get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = ground_eval7_cards(round_state.hands[active])
        board_cards = ground_eval7_cards(round_state.deck[:street])
        
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        my_stack = round_state.stacks[active]
        
        continue_cost = opp_pip - my_pip
        my_bounty = round_state.bounties[active]
        my_contribution = STARTING_STACK - my_stack
        opp_contribution = STARTING_STACK - round_state.stacks[1-active]

        # Baseline Fingerprinting
        if self.is_baseline is None:
            # Force a limp when playing as Small Blind (active == 0 usually starts pip 1) if we haven't classified
            if street == 0 and my_pip == 1 and opp_pip == 2 and CallAction in legal_actions:
                return CallAction()
            
            # Observe opponent raise amount
            if street == 0:
                if opp_pip in [14, 26]:
                    self.is_baseline = True
                elif opp_pip not in [1, 2, 11, 400]:
                    self.is_baseline = False

        # Baseline Destroyer Trap Execution
        if self.is_baseline:
            if street == 0:
                if opp_pip >= 390:
                    # They shoved premium_preflop. Fold unless we have S-tier.
                    tier = self.get_preflop_tier(my_cards)
                    if tier == 'S' and CallAction in legal_actions: return CallAction()
                    if FoldAction in legal_actions: return FoldAction()
                    return CheckAction()
                
                if RaiseAction in legal_actions and my_pip < 250:
                    _, max_raise = round_state.raise_bounds()
                    return RaiseAction(min(max_raise, max(4, 250)))  # Fixed bet size to extract strong_preflop
                
                if CallAction in legal_actions: return CallAction()
                return CheckAction()
            else:
                # Post-flop. Pot is extremely inflated, baseline range is capped at non-premium.
                if RaiseAction in legal_actions:
                    min_raise, max_raise = round_state.raise_bounds()
                    # Bet enough to force fold criteria (135 perfectly forces >pot/5)
                    target = my_pip + 135
                    return RaiseAction(min(max_raise, max(target, min_raise)))
                
                if CheckAction in legal_actions: return CheckAction()
                if FoldAction in legal_actions: return FoldAction()
                return CallAction()

        # Exact Hypergeometric Bounty Math (BEAST Integration)
        my_cards_str = [c.s for c in my_cards]
        board_cards_str = [c.s for c in board_cards]
        remaining = 5 - len(board_cards)
        p_hit = bounty_hit_prob_future(my_bounty, my_cards_str, board_cards_str, remaining)
        
        # Original Dynamic CFR/VPIP Setup
        vpip_eff = self.w2f_count / max(1, self.round_count)
        fold_eff = self.opp_folds / max(1, self.bluff_opportunities)

        # Original Bounty Threat Inference (Crucial for the Trap payload)
        if street > 0:
            opp_comm = opp_contribution / max(STARTING_STACK, 1)
            opp_p_hit = min(0.65, 0.2 + (opp_comm * 0.8)) # softer inference
        else:
            opp_p_hit = 0.46 * vpip_eff

        B_RATIO = 1.5
        B_CONST = 10.0

        # Preflop Nash Strategy overrides standard logic
        if street == 0:
            tier = self.get_preflop_tier(my_cards)
            if RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                
                # Value Trapping strategy vs BEAST
                if tier == 'S':
                    target = opp_pip + int(max(continue_cost * 2, 7))
                    return RaiseAction(min(max_raise, max(min_raise, target)))
                elif tier == 'A':
                    target = opp_pip + int(max(continue_cost * 2.5, 12))
                    return RaiseAction(min(max_raise, max(min_raise, target)))
                elif tier == 'B' and continue_cost <= BIG_BLIND * 2:
                    return RaiseAction(min_raise)
            
            if CallAction in legal_actions:
                if tier in ['S', 'A']: return CallAction()
                if tier == 'B' and continue_cost <= BIG_BLIND: return CallAction()
            
            if CheckAction in legal_actions: return CheckAction()
            return FoldAction()

        # Post-Flop Evaluation
        if vpip_eff < 0.2:
            opp_range_str = "88+,A8s+,K9s+,QJs,AJo+,KQo"
        elif vpip_eff < 0.5:
            opp_range_str = "22+,A2s+,K2s+,Q2s+,J8s+,T8s+,98s,A2o+,K8o+,Q9o+,JTo"
        else:
            opp_range_str = "22+,A2+,K2+,Q2+,J2+,T2+,92+,82+,72+,62+,52+,42+,32+"

        opp_range = eval7.HandRange(opp_range_str)
        
        # Dynamic Time Resource Allocation
        clock = game_state.game_clock
        if clock < 2.5:
            # High-Speed Fallback
            equity = _rough_made_strength(my_cards_str, board_cards_str)
        else:
            if clock > 45.0: iterations = 400
            elif clock > 20.0: iterations = 200
            elif clock > 5.0: iterations = 100
            else: iterations = 25
            equity = eval7.py_hand_vs_range_monte_carlo(my_cards, opp_range, board_cards, iterations)

        # Opponent Haircut based on aggression and commitment
        if street > 0:
            opp_comm = opp_contribution / max(STARTING_STACK, 1)
            if opp_comm >= 0.30: equity -= 0.18
            elif opp_comm >= 0.10: equity -= 0.10
            elif opp_comm >= 0.05: equity -= 0.05
            
            # extra penalty if we face a big bet right now
            if continue_cost > 30: equity -= 0.08
            elif continue_cost > 10: equity -= 0.04
        
        equity = max(0.01, equity)

        # Restored Legacy EV Calculus Action Thresholds
        base_win = opp_contribution
        bounty_dead_equity = p_hit * (opp_contribution * (B_RATIO - 1.0) + B_CONST)
        win_gain = base_win + bounty_dead_equity
        
        curr_loss_base = my_contribution + continue_cost
        loss_drain = curr_loss_base + (opp_p_hit * (curr_loss_base * (B_RATIO - 1.0) + B_CONST))
        
        ev_call = equity * win_gain - (1.0 - equity) * loss_drain
        
        pot_size_curr = my_contribution + opp_contribution

        # Hyper-exploit Bluffs
        if street > 0 and opp_pip == 0 and equity < 0.40:
            # Overbet bluff when opponent checks a medium pot
            if RaiseAction in legal_actions and pot_size_curr < 150:
                min_raise, max_raise = round_state.raise_bounds()
                bluff_target = int(pot_size_curr * 0.65)
                # 25% time bluff to prevent extreme exposure
                if random.random() < 0.25:
                    return RaiseAction(min(max_raise, max(min_raise, bluff_target)))
            
        # Standard Value Thresholding
        if ev_call > -0.1:
            if RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                
                if equity > 0.65:
                    spr = my_stack / max(1, pot_size_curr)
                    if spr < 0.5:
                        raise_amount = max_raise
                    else:
                        # Value Extract: BEAST accepts any bet under 30-40% pot with huge range!
                        # Exploit by betting exactly 35%
                        target = int(pot_size_curr * 0.35) 
                        if opp_pip > 0: target += opp_pip
                        raise_amount = min(max_raise, max(min_raise, target))
                        
                    return RaiseAction(raise_amount)
                elif equity > 0.55:
                    return RaiseAction(min_raise)
            
            if CheckAction in legal_actions and equity < 0.6:
                return CheckAction()
                
            return CallAction()
        else:
            if CheckAction in legal_actions:
                return CheckAction()
                
            return FoldAction()

def ground_eval7_cards(str_cards):
    return [eval7.Card(card) for card in str_cards]

def get_rank_idx(rank_str):
    char_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    return char_map[rank_str]

if __name__ == '__main__':
    run_bot(Player(), parse_args())
