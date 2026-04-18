from __future__ import annotations

import random
import time
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import eval7

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import STARTING_STACK
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot


CARD_RANKINGS = '23456789TJQKA'
RANK_STRENGTHS = {c: i + 2 for i, c in enumerate(CARD_RANKINGS)}
BOUNTY_MULTIPLIER = 1.5
BOUNTY_BASE_AMOUNT = 10

BASE_STRATEGY_CONFIG = {
    'random_seed': 199441361,
    'preflop_simulation_count': 2048,
    'monte_carlo_scaling': 1.35,
    'equity_discount_factor': 0.58,
    'aggression_trigger_upper': 0.52,
    'aggression_trigger_lower': 0.43,
    'aggression_increment': 0.18,
    'bounty_influence_factor': 1.25,
    'preflop_call_tolerance': 0.035,
    'river_bluff_frequency': 0.58,
    'weak_hand_bluff_rate': 0.32,
    'value_betting_threshold': 0.84,
    'semi_bluff_lower_bound': 0.54,
    'semi_bluff_upper_bound': 0.78,
    'semi_bluff_fold_threshold': 0.48,
    'postflop_call_tolerance': 0.088,
}

AGGRESSIVE_STRATEGY_CONFIG = {
    'preflop_call_tolerance': -0.01,
    'postflop_call_tolerance': 0.025,
    'value_betting_threshold': 0.73,
    'semi_bluff_lower_bound': 0.48,
    'semi_bluff_upper_bound': 0.71,
    'semi_bluff_fold_threshold': 0.55,
    'river_bluff_frequency': 0.75,
    'weak_hand_bluff_rate': 0.18,
    'aggression_trigger_upper': 0.58,
    'aggression_trigger_lower': 0.22,
    'aggression_increment': 0.12,
}


def initialize_strategy_parameters():
    return dict(BASE_STRATEGY_CONFIG)


def normalize_hand_representation(card_strings):
    '''Convert 2-card hand to standard notation (e.g. "AKs", "77", "T9o").'''
    first_rank, first_suit = card_strings[0][0], card_strings[0][1]
    second_rank, second_suit = card_strings[1][0], card_strings[1][1]
    first_value, second_value = RANK_STRENGTHS[first_rank], RANK_STRENGTHS[second_rank]
    
    if first_value < second_value:
        first_rank, second_rank, first_suit, second_suit = second_rank, first_rank, second_suit, first_suit
        first_value, second_value = second_value, first_value
    
    if first_rank == second_rank:
        return first_rank + second_rank
    
    return first_rank + second_rank + ('s' if first_suit == second_suit else 'o')


def construct_cards_from_notation(hand_notation):
    '''Generate eval7 Card objects from hand notation.'''
    rank1, rank2 = hand_notation[0], hand_notation[1]
    if len(hand_notation) == 2:
        return [eval7.Card(rank1 + 's'), eval7.Card(rank2 + 'h')]
    if hand_notation[2] == 's':
        return [eval7.Card(rank1 + 's'), eval7.Card(rank2 + 's')]
    return [eval7.Card(rank1 + 's'), eval7.Card(rank2 + 'h')]


def calculate_preflop_win_rates(simulations_per_hand=512):
    '''Precompute win rates for all possible starting hands vs random opponents.'''
    hand_variations = []
    for i, rank1 in enumerate(CARD_RANKINGS):
        for j, rank2 in enumerate(CARD_RANKINGS):
            if i == j:
                hand_variations.append(rank1 + rank2)
            elif i < j:
                high, low = rank2, rank1
                hand_variations.append(high + low + 's')
                hand_variations.append(high + low + 'o')
    
    complete_deck = list(eval7.Deck().cards)
    win_rate_table = {}
    
    for hand_type in hand_variations:
        player_cards = construct_cards_from_notation(hand_type)
        used_cards = {str(c) for c in player_cards}
        available_cards = [c for c in complete_deck if str(c) not in used_cards]
        victories, draws = 0, 0
        
        for _ in range(simulations_per_hand):
            random.shuffle(available_cards)
            opponent_cards = available_cards[:2]
            community_cards = available_cards[2:7]
            player_score = eval7.evaluate(player_cards + community_cards)
            opponent_score = eval7.evaluate(opponent_cards + community_cards)
            
            if player_score > opponent_score:
                victories += 1
            elif player_score == opponent_score:
                draws += 1
        
        win_rate_table[hand_type] = (victories + 0.5 * draws) / simulations_per_hand
    
    return win_rate_table


def compute_bounty_hit_probability(target_rank, visible_hole_cards, visible_community, remaining_community):
    '''Calculate probability of bounty rank appearing in remaining cards.'''
    all_visible = visible_hole_cards + visible_community
    if target_rank in {c[0] for c in all_visible}:
        return 1.0
    if remaining_community <= 0:
        return 0.0
    
    unknown_cards = 52 - len(all_visible)
    bounty_cards_remaining = 4
    probability_miss = 1.0
    
    for i in range(remaining_community):
        probability_miss *= max(0, unknown_cards - bounty_cards_remaining - i) / (unknown_cards - i)
    
    return 1.0 - probability_miss


def calculate_hand_strength(player_cards, community_cards, trial_count):
    '''Monte Carlo calculation of hand equity vs random opponent.'''
    excluded_cards = {str(c) for c in player_cards}
    for community_card in community_cards:
        excluded_cards.add(str(community_card))
    
    remaining_deck = [c for c in eval7.Deck().cards if str(c) not in excluded_cards]
    cards_to_come = 5 - len(community_cards)
    wins, ties = 0, 0
    
    # Early exit optimization for obvious strength/weakness
    if trial_count >= 100:
        quick_wins = quick_ties = 0
        for _ in range(min(20, trial_count // 5)):
            random.shuffle(remaining_deck)
            opponent_cards = remaining_deck[:2]
            remaining_community = remaining_deck[2:2 + cards_to_come]
            full_board = community_cards + remaining_community if cards_to_come else community_cards
            player_score = eval7.evaluate(player_cards + full_board)
            opponent_score = eval7.evaluate(opponent_cards + full_board)
            
            if player_score > opponent_score:
                quick_wins += 1
            elif player_score == opponent_score:
                quick_ties += 1
        
        quick_equity = (quick_wins + 0.5 * quick_ties) / min(20, trial_count // 5)
        if quick_equity > 0.85:
            return min(0.95, quick_equity + 0.05)
        elif quick_equity < 0.15:
            return max(0.05, quick_equity - 0.05)
    
    for _ in range(trial_count):
        random.shuffle(remaining_deck)
        opponent_cards = remaining_deck[:2]
        remaining_community = remaining_deck[2:2 + cards_to_come]
        full_board = community_cards + remaining_community if cards_to_come else community_cards
        player_score = eval7.evaluate(player_cards + full_board)
        opponent_score = eval7.evaluate(opponent_cards + full_board)
        
        if player_score > opponent_score:
            wins += 1
        elif player_score == opponent_score:
            ties += 1
    
    return (wins + 0.5 * ties) / trial_count


def generate_all_hand_combinations():
    '''Create list of all possible 2-card combinations.'''
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    all_cards = [r + s for r in ranks for s in suits]
    combinations = []
    
    for i in range(52):
        for j in range(i + 1, 52):
            combinations.append((all_cards[i], all_cards[j]))
    
    return combinations


def build_range_filters(preflop_win_rates):
    '''Create filtered hand ranges based on equity percentages.'''
    all_combinations = generate_all_hand_combinations()
    
    def combination_strength(combo):
        hand_key = normalize_hand_representation(list(combo))
        return preflop_win_rates.get(hand_key, 0.5)
    
    sorted_combinations = sorted(all_combinations, key=lambda c: -combination_strength(c))
    range_filters = {}
    
    for percentage in (0.05, 0.10, 0.20, 0.40, 1.00):
        count = max(1, int(round(len(sorted_combinations) * percentage)))
        range_filters[percentage] = sorted_combinations[:count]
    
    return range_filters


def calculate_equity_vs_hand_range(player_cards, community_cards, opponent_range, simulations):
    '''Monte Carlo equity against specific opponent hand range.'''
    player_card_strings = {str(c) for c in player_cards}
    community_card_strings = {str(c) for c in community_cards}
    excluded = player_card_strings | community_card_strings
    
    valid_combinations = [(a, b) for (a, b) in opponent_range if a not in excluded and b not in excluded]
    
    if not valid_combinations:
        return calculate_hand_strength(player_cards, community_cards, simulations)
    
    full_deck = eval7.Deck().cards
    remaining_cards = [c for c in full_deck if str(c) not in excluded]
    cards_to_come = 5 - len(community_cards)
    wins, ties = 0, 0
    
    for _ in range(simulations):
        first_card, second_card = random.choice(valid_combinations)
        opponent_cards = [eval7.Card(first_card), eval7.Card(second_card)]
        opponent_card_strings = {first_card, second_card}
        
        available_for_community = [c for c in remaining_cards if str(c) not in opponent_card_strings]
        random.shuffle(available_for_community)
        
        remaining_community = available_for_community[:cards_to_come] if cards_to_come else []
        complete_board = community_cards + remaining_community if cards_to_come else community_cards
        
        player_score = eval7.evaluate(player_cards + complete_board)
        opponent_score = eval7.evaluate(opponent_cards + complete_board)
        
        if player_score > opponent_score:
            wins += 1
        elif player_score == opponent_score:
            ties += 1
    
    return (wins + 0.5 * ties) / simulations


def convert_string_to_cards(card_strings):
    return [eval7.Card(s) for s in card_strings]


def categorize_hand_strength(card_strings):
    '''Classify hand into strength categories.'''
    rank1, rank2 = card_strings[0][0], card_strings[1][0]
    suit1, suit2 = card_strings[0][1], card_strings[1][1]
    value1, value2 = RANK_STRENGTHS[rank1], RANK_STRENGTHS[rank2]
    
    if value1 < value2:
        value1, value2 = value2, value1
        rank1, rank2 = rank2, rank1
        suit1, suit2 = suit2, suit1
    
    is_pair = value1 == value2
    is_suited = suit1 == suit2
    
    # Elite hands
    if is_pair and value1 >= 11:
        return 'elite'
    if value1 == 14 and value2 == 13:
        return 'elite'
    if value1 == 14 and value2 == 12 and is_suited:
        return 'elite'
    
    # Powerful hands
    if is_pair and value1 >= 9:
        return 'powerful'
    if value1 == 14 and value2 == 12:
        return 'powerful'
    if value1 == 14 and value2 == 11 and is_suited:
        return 'powerful'
    if value1 == 13 and value2 == 12 and is_suited:
        return 'powerful'
    
    # Moderate hands
    if is_pair:
        return 'moderate'
    if value1 == 14:
        return 'moderate'
    if value1 == 13 and value2 >= 9:
        return 'moderate'
    if value1 == 13 and is_suited:
        return 'moderate'
    if value1 == 12 and value2 >= 9:
        return 'moderate'
    if value1 == 12 and is_suited:
        return 'moderate'
    if value1 == 11 and value2 >= 9:
        return 'moderate'
    if value1 == 11 and is_suited:
        return 'moderate'
    if value1 == 10 and value2 >= 8:
        return 'moderate'
    if is_suited and (value1 - value2) <= 2 and value1 >= 6:
        return 'moderate'
    
    # Playable hands
    if value1 >= 10:
        return 'playable'
    if is_suited and (value1 - value2) <= 3:
        return 'playable'
    
    return 'weak'


class AdversaryTracker:
    '''Track opponent behavior patterns for strategic adjustments.'''

    def __init__(self):
        self.preflop_total_actions = 0
        self.preflop_raise_count = 0
        self.preflop_call_count = 0
        self.preflop_fold_count = 0
        self.postflop_total_actions = 0
        self.postflop_raise_count = 0
        self.postflop_bet_count = 0
        self.postflop_check_count = 0
        self.postflop_fold_count = 0
        self.postflop_call_count = 0
        self.voluntarily_put_money_in_pot = 0
        self.total_hands_played = 0
        self.showdown_wins = []
        self.last_was_aggressive = False
        self.fold_to_bet_attempts = 0
        self.fold_to_bet_successes = 0
        
        # Enhanced tracking capabilities
        self.observed_ranges = []
        self.betting_patterns = defaultdict(list)
        self.street_activity = defaultdict(list)
        self.similar_style_detected = False
        self.style_confidence = 0.0

    def calculate_fold_frequency(self):
        total = self.preflop_fold_count + self.preflop_call_count + self.preflop_raise_count
        return (self.preflop_fold_count + 3) / (total + 6)

    def calculate_raise_frequency(self):
        total = self.preflop_total_actions
        return (self.preflop_raise_count + 1) / (total + 4)

    def calculate_postflop_aggression(self):
        total = self.postflop_total_actions
        return (self.postflop_bet_count + self.postflop_raise_count + 2) / (total + 6)

    def calculate_fold_to_bet_rate(self):
        total = self.fold_to_bet_attempts
        return (self.fold_to_bet_successes + 2) / (total + 5)
    
    def identify_similar_strategy(self):
        if self.preflop_total_actions < 20:
            return False
        
        estimated_vpip = self.calculate_raise_frequency() + 0.3
        preflop_raise = self.calculate_raise_frequency()
        
        # Detect balanced, similar playing style
        if 0.25 <= preflop_raise <= 0.35 and 0.4 <= self.calculate_postflop_aggression() <= 0.6:
            self.style_confidence = min(1.0, self.style_confidence + 0.1)
        else:
            self.style_confidence = max(0.0, self.style_confidence - 0.05)
        
        return self.style_confidence > 0.6


@dataclass(frozen=True)
class StrategyProfile:
    call_tolerance_preflop: float
    call_tolerance_postflop: float
    value_betting_equity_minimum: float
    semi_bluff_equity_lower: float
    semi_bluff_equity_upper: float
    semi_bluff_fold_threshold: float
    river_probe_frequency: float
    bluff_frequency_weak_equity: float
    aggression_upper_trigger: float
    aggression_lower_trigger: float
    aggression_step_size: float


class PokerPlayer(Bot):
    def __init__(self):
        self.card_strengths = RANK_STRENGTHS
        self.strategy_config = initialize_strategy_parameters()
        random.seed(self.strategy_config['random_seed'])
        
        try:
            self.preflop_win_rates = calculate_preflop_win_rates(
                simulations_per_hand=self.strategy_config['preflop_simulation_count']
            )
        except Exception:
            self.preflop_win_rates = {}
        
        try:
            self.hand_range_filters = build_range_filters(self.preflop_win_rates)
        except Exception:
            self.hand_range_filters = {1.00: generate_all_hand_combinations()}
        
        self.opponent_tracker = AdversaryTracker()
        self.hand_history = []
        self.previous_opponent_investment = 0
        self.previous_player_investment = 0
        self.last_street = -1
        self.most_recent_action = None
        self.cumulative_profit = 0
        
        # Performance enhancements
        self.equity_calculation_cache = {}
        self.last_board_state = None
        self.cached_result = None
        self.mirror_adjustment = 1.0

    def determine_simulation_count(self, time_remaining, current_street):
        scaling = self.strategy_config['monte_carlo_scaling']
        if time_remaining < 2.5:
            return 0 if scaling >= 1.0 else int(60 * scaling)
        if time_remaining < 6.0:
            return int(160 * scaling)
        if time_remaining < 15.0:
            return int(300 * scaling)
        if time_remaining < 30.0:
            return int(450 * scaling)
        return int(650 * scaling)

    def lookup_preflop_equity(self, hand_cards):
        hand_key = normalize_hand_representation(hand_cards)
        return self.preflop_win_rates.get(hand_key, 0.5)

    def estimate_opponent_hand_range(self, opponent_investment, is_facing_bet, street):
        commitment_ratio = opponent_investment / STARTING_STACK
        if not is_facing_bet:
            return 1.00
        elif commitment_ratio >= 0.80:
            base_range = 0.05
        elif commitment_ratio >= 0.40:
            base_range = 0.10
        elif commitment_ratio >= 0.15:
            base_range = 0.20
        elif commitment_ratio >= 0.06:
            base_range = 0.40
        else:
            base_range = 1.00

        if is_facing_bet:
            aggression_factor = (0.45 * self.opponent_tracker.calculate_raise_frequency() + 
                              0.55 * self.opponent_tracker.calculate_postflop_aggression())
            if aggression_factor >= 0.56:
                base_range *= 1.45
            elif aggression_factor <= 0.38:
                base_range *= 0.70

            fold_frequency = self.opponent_tracker.calculate_fold_frequency()
            if fold_frequency >= 0.55:
                base_range *= 0.85
            elif fold_frequency <= 0.35:
                base_range *= 1.12

        base_range = max(0.05, min(1.00, base_range))
        available_ranges = (0.05, 0.10, 0.20, 0.40, 1.00)
        return min(available_ranges, key=lambda r: abs(r - base_range))

    def calculate_monte_carlo_equity(self, player_hand, community_cards, simulations, 
                                   opponent_investment=0, is_facing_bet=False):
        board_hash = hash(tuple(community_cards))
        cache_identifier = (tuple(player_hand), board_hash, simulations)
        
        if cache_identifier in self.equity_calculation_cache and simulations < 200:
            return self.equity_calculation_cache[cache_identifier]
        
        player_cards = convert_string_to_cards(player_hand)
        board_cards = convert_string_to_cards(community_cards)
        
        if simulations <= 0:
            if not board_cards:
                return self.lookup_preflop_equity(player_hand)
            return self._estimate_hand_strength(player_hand, community_cards)
        
        range_percentage = self.estimate_opponent_hand_range(opponent_investment, is_facing_bet, len(community_cards))
        opponent_combos = self.hand_range_filters.get(range_percentage) or self.hand_range_filters.get(1.00)
        
        if opponent_combos is None:
            result = calculate_hand_strength(player_cards, board_cards, simulations)
        else:
            result = calculate_equity_vs_hand_range(player_cards, board_cards, opponent_combos, simulations)
        
        if simulations < 200:
            self.equity_calculation_cache[cache_identifier] = result
            
        return result

    def _estimate_hand_strength(self, hole_cards, community_cards):
        hole_ranks = [c[0] for c in hole_cards]
        board_ranks = [c[0] for c in community_cards]
        
        if hole_ranks[0] == hole_ranks[1]:
            pair_value = RANK_STRENGTHS[hole_ranks[0]]
            return min(0.85, 0.55 + pair_value * 0.015)
        
        highest_board = max(RANK_STRENGTHS[r] for r in board_ranks) if board_ranks else 0
        has_top_pair = any(RANK_STRENGTHS[h] == highest_board for h in hole_ranks)
        board_is_paired = len(set(board_ranks)) < len(board_ranks)
        
        if has_top_pair:
            return 0.6
        if board_is_paired:
            return 0.45
        
        highest_hole = max(RANK_STRENGTHS[h] for h in hole_ranks)
        return 0.30 + 0.01 * highest_hole

    def calculate_bounty_multiplier(self, player_bounty_rank, hole_cards, community_cards, street):
        visible_hole_ranks = {c[0] for c in hole_cards}
        visible_board_ranks = {c[0] for c in community_cards}
        remaining_cards = 5 - len(community_cards)
        
        if player_bounty_rank in visible_hole_ranks or player_bounty_rank in visible_board_ranks:
            return BOUNTY_MULTIPLIER, float(BOUNTY_BASE_AMOUNT)
        
        hit_probability = compute_bounty_hit_probability(
            player_bounty_rank, hole_cards, community_cards, remaining_cards
        )
        
        multiplier = 1.0 + (BOUNTY_MULTIPLIER - 1.0) * hit_probability
        bonus = BOUNTY_BASE_AMOUNT * hit_probability
        return multiplier, bonus

    def estimate_opponent_bounty_multiplier(self, hole_cards, community_cards, street):
        visible_hole_ranks = {c[0] for c in hole_cards}
        visible_board_ranks = {c[0] for c in community_cards}
        remaining_cards = 5 - len(community_cards)
        total_probability = 0.0
        
        for rank in CARD_RANKINGS:
            if rank in visible_board_ranks:
                hit_probability = 1.0
            else:
                cards_seen = 2 + len(community_cards)
                rank_remaining = 4 - (1 if rank in visible_hole_ranks else 0)
                unknown_cards = 52 - cards_seen
                opponent_hole_cards = 2
                future_board_cards = remaining_cards
                cards_to_draw = opponent_hole_cards + future_board_cards
                probability_no_hit = 1.0
                
                for i in range(cards_to_draw):
                    denominator = unknown_cards - i
                    if denominator <= 0:
                        probability_no_hit = 0.0
                        break
                    probability_no_hit *= max(0, denominator - rank_remaining - i) / denominator
                
                hit_probability = 1.0 - probability_no_hit
            
            total_probability += hit_probability
        
        average_hit_probability = total_probability / 13.0
        multiplier = 1.0 + (BOUNTY_MULTIPLIER - 1.0) * average_hit_probability
        bonus = BOUNTY_BASE_AMOUNT * average_hit_probability
        return multiplier, bonus

    def compute_marginal_outcomes(self, pot_size, player_contribution, cost_to_continue, 
                                 player_bounty_mult, player_bounty_bonus, 
                                 opponent_bounty_mult, opponent_bounty_bonus):
        opponent_contribution = pot_size - player_contribution
        win_marginal = opponent_contribution * player_bounty_mult + player_bounty_bonus + player_contribution
        lose_marginal = (player_contribution + cost_to_continue) * opponent_bounty_mult + opponent_bounty_bonus - player_contribution
        return win_marginal, lose_marginal

    def calculate_minimum_equity(self, continue_cost, pot, opp_bounty_mult, opp_bounty_bonus, 
                               my_bounty_mult, my_bounty_bonus, my_contribution):
        win_marginal, lose_marginal = self.compute_marginal_outcomes(
            pot, my_contribution, continue_cost, my_bounty_mult, my_bounty_bonus, 
            opp_bounty_mult, opp_bounty_bonus
        )
        total_marginal = win_marginal + lose_marginal
        if total_marginal <= 0:
            return 1.0
        return max(0.0, lose_marginal / total_marginal)

    def determine_bet_amount(self, hand_equity, pot, player_bet, opponent_bet, 
                           player_stack, opponent_stack, min_allowed, max_allowed, 
                           street, aggression_modifier):
        # Adjust for mirror matches
        mirror_bonus = 0.0
        if self.opponent_tracker.identify_similar_strategy():
            if hand_equity > 0.70:
                mirror_bonus = 0.05
            elif hand_equity > 0.50:
                mirror_bonus = -0.03
        
        if hand_equity > 0.85:
            pot_fraction = 1.15
        elif hand_equity > 0.72:
            pot_fraction = 0.85
        elif hand_equity > 0.60:
            pot_fraction = 0.65
        else:
            pot_fraction = 0.52
        
        pot_fraction += aggression_modifier + mirror_bonus
        pot_fraction = max(0.35, min(1.35, pot_fraction))
        
        target_total = opponent_bet + int(max(pot, 4) * pot_fraction)
        target_total = max(target_total, min_allowed)
        target_total = min(target_total, max_allowed)
        return target_total

    def assess_opponent_tightness(self, player_investment, opponent_investment, 
                                continue_cost, street, is_facing_bet):
        if not is_facing_bet:
            return 0.01
        
        commitment_level = opponent_investment / STARTING_STACK
        
        if street == 0:
            if commitment_level >= 0.85:
                return 0.20
            if commitment_level >= 0.40:
                return 0.12
            if commitment_level >= 0.15:
                return 0.06
            if commitment_level >= 0.06:
                return 0.04
            return 0.02
        else:
            if commitment_level >= 0.80:
                return 0.16
            if commitment_level >= 0.40:
                return 0.10
            if commitment_level >= 0.20:
                return 0.07
            if commitment_level >= 0.08:
                return 0.03
            return 0.02

    def determine_tournament_phase(self, game_state):
        hands_remaining = 500 if game_state.round_num <= 500 else 1000
        rounds_left = max(0, hands_remaining - game_state.round_num)
        chip_advantage = game_state.bankroll
        
        if chip_advantage >= max(140, int(rounds_left * 0.36)):
            return 'conservative'
        if chip_advantage <= -max(200, int(rounds_left * 0.42)):
            return 'aggressive'
        return 'standard'

    def create_strategy_profile(self, game_state, street):
        preflop_raise_freq = self.opponent_tracker.calculate_raise_frequency()
        
        if self.opponent_tracker.preflop_total_actions < 12:
            passive_weight = 0.35
        else:
            if preflop_raise_freq <= 0.34:
                passive_weight = 0.95
            elif preflop_raise_freq >= 0.50:
                passive_weight = 0.0
            else:
                passive_weight = max(0.0, min(0.95, (0.50 - preflop_raise_freq) / 0.16))

        def blend_strategies(base_key, attack_key):
            base_value = self.strategy_config[base_key]
            attack_value = AGGRESSIVE_STRATEGY_CONFIG[attack_key]
            return base_value + (attack_value - base_value) * passive_weight

        profile = StrategyProfile(
            call_tolerance_preflop=blend_strategies('preflop_call_tolerance', 'preflop_call_tolerance'),
            call_tolerance_postflop=blend_strategies('postflop_call_tolerance', 'postflop_call_tolerance'),
            value_betting_equity_minimum=blend_strategies('value_betting_threshold', 'value_betting_threshold'),
            semi_bluff_equity_lower=blend_strategies('semi_bluff_lower_bound', 'semi_bluff_lower_bound'),
            semi_bluff_equity_upper=blend_strategies('semi_bluff_upper_bound', 'semi_bluff_upper_bound'),
            semi_bluff_fold_threshold=blend_strategies('semi_bluff_fold_threshold', 'semi_bluff_fold_threshold'),
            river_probe_frequency=blend_strategies('river_bluff_frequency', 'river_bluff_frequency'),
            bluff_frequency_weak_equity=blend_strategies('weak_hand_bluff_rate', 'weak_hand_bluff_rate'),
            aggression_upper_trigger=blend_strategies('aggression_trigger_upper', 'aggression_trigger_upper'),
            aggression_lower_trigger=blend_strategies('aggression_trigger_lower', 'aggression_trigger_lower'),
            aggression_step_size=blend_strategies('aggression_increment', 'aggression_increment'),
        )

        # Phase-based adjustments
        phase = self.determine_tournament_phase(game_state)
        if phase == 'conservative':
            profile = StrategyProfile(
                call_tolerance_preflop=profile.call_tolerance_preflop + 0.015,
                call_tolerance_postflop=profile.call_tolerance_postflop + 0.018,
                value_betting_equity_minimum=min(0.93, profile.value_betting_equity_minimum + 0.016),
                semi_bluff_equity_lower=min(0.86, profile.semi_bluff_equity_lower + 0.016),
                semi_bluff_equity_upper=min(0.93, profile.semi_bluff_equity_upper + 0.012),
                semi_bluff_fold_threshold=min(0.91, profile.semi_bluff_fold_threshold + 0.022),
                river_probe_frequency=max(0.0, profile.river_probe_frequency * 0.78),
                bluff_frequency_weak_equity=max(0.0, profile.bluff_frequency_weak_equity * 0.68),
                aggression_upper_trigger=profile.aggression_upper_trigger + 0.025,
                aggression_lower_trigger=min(0.66, profile.aggression_lower_trigger + 0.025),
                aggression_step_size=profile.aggression_step_size * 0.88,
            )
        elif phase == 'aggressive':
            profile = StrategyProfile(
                call_tolerance_preflop=profile.call_tolerance_preflop - 0.004,
                call_tolerance_postflop=profile.call_tolerance_postflop - 0.004,
                value_betting_equity_minimum=max(0.54, profile.value_betting_equity_minimum - 0.004),
                semi_bluff_equity_lower=max(0.28, profile.semi_bluff_equity_lower - 0.008),
                semi_bluff_equity_upper=min(0.93, profile.semi_bluff_equity_upper + 0.008),
                semi_bluff_fold_threshold=max(0.23, profile.semi_bluff_fold_threshold - 0.008),
                river_probe_frequency=min(0.96, profile.river_probe_frequency * 1.025),
                bluff_frequency_weak_equity=min(0.52, profile.bluff_frequency_weak_equity * 1.035),
                aggression_upper_trigger=max(0.28, profile.aggression_upper_trigger - 0.015),
                aggression_lower_trigger=max(0.07, profile.aggression_lower_trigger - 0.015),
                aggression_step_size=min(0.32, profile.aggression_step_size * 1.025),
            )

        if street == 5:
            profile = StrategyProfile(
                call_tolerance_preflop=profile.call_tolerance_preflop,
                call_tolerance_postflop=profile.call_tolerance_postflop,
                value_betting_equity_minimum=max(0.54, profile.value_betting_equity_minimum - 0.012),
                semi_bluff_equity_lower=max(0.28, profile.semi_bluff_equity_lower - 0.012),
                semi_bluff_equity_upper=profile.semi_bluff_equity_upper,
                semi_bluff_fold_threshold=profile.semi_bluff_fold_threshold,
                river_probe_frequency=profile.river_probe_frequency,
                bluff_frequency_weak_equity=profile.bluff_frequency_weak_equity,
                aggression_upper_trigger=profile.aggression_upper_trigger,
                aggression_lower_trigger=profile.aggression_lower_trigger,
                aggression_step_size=profile.aggression_step_size,
            )

        return profile

    def handle_new_round(self, game_state, round_state, active_player):
        self.most_recent_action = None
        self.last_street = 0
        self.previous_player_investment = 0
        self.previous_opponent_investment = 0
        self.opponent_tracker.total_hands_played += 1
        
        # Periodic cache cleanup
        if self.opponent_tracker.total_hands_played % 50 == 0:
            self.equity_calculation_cache.clear()

    def handle_round_over(self, game_state, terminal_state, active_player):
        profit_loss = terminal_state.deltas[active_player]
        self.cumulative_profit += profit_loss

    def get_action(self, game_state, round_state, active_player):
        available_actions = round_state.legal_actions()
        current_street = round_state.street
        hole_cards = round_state.hands[active_player]
        community_cards = round_state.deck[:current_street] if current_street > 0 else []
        player_current_bet = round_state.pips[active_player]
        opponent_current_bet = round_state.pips[1 - active_player]
        player_remaining_stack = round_state.stacks[active_player]
        opponent_remaining_stack = round_state.stacks[1 - active_player]
        cost_to_continue = opponent_current_bet - player_current_bet
        player_bounty_rank = round_state.bounties[active_player]
        player_total_investment = STARTING_STACK - player_remaining_stack
        opponent_total_investment = STARTING_STACK - opponent_remaining_stack
        current_pot = player_total_investment + opponent_total_investment
        time_remaining = game_state.game_clock

        self._monitor_opponent_behavior(round_state, active_player, current_street)
        is_facing_bet = cost_to_continue > 0
        min_raise_amount = max_raise_amount = 0
        
        if RaiseAction in available_actions:
            min_raise_amount, max_raise_amount = round_state.raise_bounds()

        # Early game simplification
        if time_remaining < 1.2:
            if CheckAction in available_actions:
                return CheckAction()
            if cost_to_continue <= 2 and CallAction in available_actions:
                return CallAction()
            if FoldAction in available_actions:
                return FoldAction()
            return CallAction()

        simulation_count = self.determine_simulation_count(time_remaining, current_street)
        
        if current_street == 0:
            base_equity = self.lookup_preflop_equity(hole_cards)
        else:
            base_equity = self.calculate_monte_carlo_equity(
                hole_cards, community_cards, simulation_count,
                opponent_investment=opponent_total_investment,
                is_facing_bet=is_facing_bet,
            )

        player_bounty_mult, player_bounty_bonus = self.calculate_bounty_multiplier(
            player_bounty_rank, hole_cards, community_cards, current_street
        )
        opponent_bounty_mult, opponent_bounty_bonus = self.estimate_opponent_bounty_multiplier(
            hole_cards, community_cards, current_street
        )

        if current_street == 0:
            tightness_adjustment = self.assess_opponent_tightness(
                player_total_investment, opponent_total_investment, cost_to_continue, current_street, is_facing_bet
            )
            tightness_adjustment *= self.strategy_config['equity_discount_factor']
        else:
            tightness_adjustment = 0.0

        strategy_profile = self.create_strategy_profile(game_state, current_street)
        opponent_fold_tendency = self.opponent_tracker.calculate_fold_to_bet_rate()
        aggression_modifier = 0.0
        
        if opponent_fold_tendency > strategy_profile.aggression_upper_trigger:
            aggression_modifier += strategy_profile.aggression_step_size
        elif opponent_fold_tendency < strategy_profile.aggression_lower_trigger:
            aggression_modifier -= strategy_profile.aggression_step_size

        bounty_adjustment = (self.strategy_config['bounty_influence_factor'] * 
                           (0.02 * (player_bounty_mult - 1.0) + (player_bounty_bonus / 450.0)))

        # Preflop decision logic
        if current_street == 0:
            hole_card_ranks = [c[0] for c in hole_cards]
            has_bounty_card = player_bounty_rank in hole_card_ranks
            raw_equity = base_equity
            equity_vs_opponent = max(0.05, raw_equity - tightness_adjustment)
            
            # Mirror match dynamic adjustment
            mirror_bonus = 0.0
            if self.opponent_tracker.identify_similar_strategy():
                if raw_equity > 0.65:
                    mirror_bonus = 0.02
                elif raw_equity < 0.35:
                    mirror_bonus = -0.01
                    
            effective_equity = equity_vs_opponent + bounty_adjustment + mirror_bonus
            hand_category = categorize_hand_strength(hole_cards)
            opponent_commitment = opponent_total_investment / STARTING_STACK
            minimum_equity_needed = self.calculate_minimum_equity(
                cost_to_continue, current_pot, opponent_bounty_mult, opponent_bounty_bonus,
                player_bounty_mult, player_bounty_bonus, player_total_investment
            )

            # Opening decisions
            if not is_facing_bet:
                if RaiseAction in available_actions:
                    if hand_category == 'elite':
                        target = opponent_current_bet + max(7, int(current_pot * 1.7))
                    elif hand_category == 'powerful':
                        target = opponent_current_bet + max(6, int(current_pot * 1.3))
                    elif hand_category in ('moderate', 'playable'):
                        target = opponent_current_bet + max(5, int(current_pot * 1.0))
                    elif has_bounty_card:
                        target = opponent_current_bet + max(5, int(current_pot * 0.9))
                    else:
                        target = None
                    if target is not None:
                        return RaiseAction(min(max(target, min_raise_amount), max_raise_amount))
                if CheckAction in available_actions:
                    return CheckAction()
                if CallAction in available_actions:
                    return CallAction()

            # Facing significant aggression
            if opponent_commitment >= 0.65 or cost_to_continue >= player_remaining_stack * 0.55:
                if hand_category == 'elite' and CallAction in available_actions:
                    return CallAction()
                if FoldAction in available_actions:
                    return FoldAction()
                if CheckAction in available_actions:
                    return CheckAction()
                return CallAction()

            # Medium aggression response
            if opponent_current_bet >= 25 or opponent_commitment >= 0.12:
                if RaiseAction in available_actions and hand_category == 'elite':
                    _, max_reraise = round_state.raise_bounds()
                    all_in_target = min(opponent_current_bet + int((current_pot + cost_to_continue) * 2.4), max_reraise)
                    return RaiseAction(max(all_in_target, min_raise_amount))
                if CallAction in available_actions and hand_category in ('elite', 'powerful'):
                    return CallAction()
                if CallAction in available_actions and hand_category == 'moderate' and cost_to_continue <= current_pot * 0.58:
                    return CallAction()
                if FoldAction in available_actions:
                    return FoldAction()
                if CallAction in available_actions:
                    return CallAction()
                return CheckAction() if CheckAction in available_actions else CallAction()

            # Small aggression response
            if opponent_current_bet >= 5:
                if RaiseAction in available_actions and hand_category == 'elite':
                    target = opponent_current_bet + max(min_raise_amount - opponent_current_bet, int(cost_to_continue * 3.2))
                    return RaiseAction(min(max(target, min_raise_amount), max_raise_amount))
                if RaiseAction in available_actions and hand_category == 'powerful':
                    target = opponent_current_bet + max(min_raise_amount - opponent_current_bet, int(cost_to_continue * 2.8))
                    return RaiseAction(min(max(target, min_raise_amount), max_raise_amount))
                if CallAction in available_actions and hand_category in ('elite', 'powerful', 'moderate'):
                    return CallAction()
                if CallAction in available_actions and has_bounty_card and hand_category == 'playable' and cost_to_continue <= 9:
                    return CallAction()
                if FoldAction in available_actions:
                    return FoldAction()
                if CheckAction in available_actions:
                    return CheckAction()
                return CallAction()

            # Minimal aggression
            if cost_to_continue <= 2:
                if RaiseAction in available_actions and hand_category == 'elite':
                    target = opponent_current_bet + max(min_raise_amount - opponent_current_bet, 7)
                    return RaiseAction(min(max(target, min_raise_amount), max_raise_amount))
                if CallAction in available_actions and hand_category != 'weak':
                    return CallAction()
                if CallAction in available_actions and has_bounty_card:
                    return CallAction()
                if FoldAction in available_actions:
                    return FoldAction()
                return CallAction() if CallAction in available_actions else CheckAction()

            # Standard calling decision
            if CallAction in available_actions and effective_equity >= minimum_equity_needed + strategy_profile.call_tolerance_preflop and hand_category != 'weak':
                return CallAction()
            if FoldAction in available_actions:
                return FoldAction()
            if CheckAction in available_actions:
                return CheckAction()
            return CallAction() if CallAction in available_actions else CheckAction()

        # Postflop decision logic
        raw_equity = base_equity
        equity_vs_opponent = max(0.02, raw_equity - tightness_adjustment)
        effective_equity = equity_vs_opponent + bounty_adjustment
        minimum_equity_needed = self.calculate_minimum_equity(
            cost_to_continue, current_pot, opponent_bounty_mult, opponent_bounty_bonus,
            player_bounty_mult, player_bounty_bonus, player_total_investment
        )
        opponent_commitment = opponent_total_investment / STARTING_STACK

        # In position betting
        if not is_facing_bet:
            if RaiseAction in available_actions and current_pot >= 4:
                if raw_equity >= 0.66:
                    target = self.determine_bet_amount(
                        raw_equity, current_pot, player_current_bet, opponent_current_bet,
                        player_remaining_stack, opponent_remaining_stack, min_raise_amount, max_raise_amount,
                        current_street, aggression_modifier
                    )
                    return RaiseAction(target)
                if raw_equity >= 0.50:
                    fraction = 0.58 + aggression_modifier
                    fraction = max(0.42, min(0.82, fraction))
                    target = opponent_current_bet + max(min_raise_amount - opponent_current_bet, int(max(current_pot, 4) * fraction))
                    return RaiseAction(min(max(target, min_raise_amount), max_raise_amount))
                if raw_equity >= 0.36 and current_street == 3:
                    if random.random() < strategy_profile.river_probe_frequency:
                        fraction = 0.48
                        target = opponent_current_bet + max(min_raise_amount - opponent_current_bet, int(max(current_pot, 4) * fraction))
                        return RaiseAction(min(max(target, min_raise_amount), max_raise_amount))
                if raw_equity < 0.28 and current_pot >= 10 and random.random() < strategy_profile.bluff_frequency_weak_equity:
                    target = opponent_current_bet + max(min_raise_amount - opponent_current_bet, int(max(current_pot, 4) * 0.48))
                    return RaiseAction(min(max(target, min_raise_amount), max_raise_amount))
            if CheckAction in available_actions:
                return CheckAction()

        # Value betting
        if RaiseAction in available_actions and effective_equity >= strategy_profile.value_betting_equity_minimum and opponent_commitment < 0.55:
            target = self.determine_bet_amount(
                raw_equity, current_pot + cost_to_continue, player_current_bet, opponent_current_bet,
                player_remaining_stack, opponent_remaining_stack, min_raise_amount, max_raise_amount,
                current_street, aggression_modifier
            )
            return RaiseAction(target)

        # Semi-bluffing
        if (RaiseAction in available_actions and 
            strategy_profile.semi_bluff_equity_lower <= effective_equity < strategy_profile.semi_bluff_equity_upper and 
            opponent_fold_tendency > strategy_profile.semi_bluff_fold_threshold and 
            cost_to_continue <= current_pot * 0.65 and opponent_commitment < 0.38):
            target = opponent_current_bet + max(min_raise_amount - opponent_current_bet, int((current_pot + cost_to_continue) * 0.78))
            return RaiseAction(min(max(target, min_raise_amount), max_raise_amount))

        # Calling decisions
        pot_odds = cost_to_continue / max(1, (current_pot + cost_to_continue))
        has_made_hand = self._has_minimum_hand(hole_cards, community_cards)
        is_large_bet = cost_to_continue >= current_pot * 0.55
        
        if CallAction in available_actions:
            if effective_equity >= minimum_equity_needed + strategy_profile.call_tolerance_postflop:
                if is_large_bet and not has_made_hand and raw_equity < 0.56:
                    pass
                else:
                    return CallAction()
            if cost_to_continue <= max(3, current_pot * 0.16) and raw_equity >= 0.30:
                return CallAction()
            if current_street == 5 and opponent_fold_tendency < 0.27 and effective_equity >= pot_odds and has_made_hand:
                return CallAction()

        if FoldAction in available_actions:
            return FoldAction()
        if CheckAction in available_actions:
            return CheckAction()
        return CallAction()

    def _has_minimum_hand(self, hole_cards, community_cards):
        if not community_cards:
            return hole_cards[0][0] == hole_cards[1][0]
        
        hole_ranks = [c[0] for c in hole_cards]
        board_ranks = [c[0] for c in community_cards]
        
        if hole_ranks[0] == hole_ranks[1]:
            return True
        if hole_ranks[0] in board_ranks or hole_ranks[1] in board_ranks:
            return True
        
        hole_suits = [c[1] for c in hole_cards]
        board_suits = [c[1] for c in community_cards]
        
        for suit in set(hole_suits):
            if hole_suits.count(suit) + board_suits.count(suit) >= 4:
                return True
        
        try:
            all_ranks = sorted(set(RANK_STRENGTHS[r] for r in hole_ranks + board_ranks))
            for start in range(2, 11):
                straight_window = set(range(start, start + 5))
                if len(straight_window & set(all_ranks)) >= 4:
                    return True
        except Exception:
            pass
        
        return False

    def _monitor_opponent_behavior(self, round_state, active_player, street):
        opponent_bet = round_state.pips[1 - active_player]
        player_bet = round_state.pips[active_player]
        
        # Track betting patterns
        if opponent_bet > self.previous_opponent_investment:
            bet_amount = opponent_bet - self.previous_opponent_investment
            pot_size = (player_bet + opponent_bet + 
                       (STARTING_STACK - round_state.stacks[active_player]) + 
                       (STARTING_STACK - round_state.stacks[1 - active_player]))
            if pot_size > 0:
                bet_fraction = bet_amount / pot_size
                self.opponent_tracker.betting_patterns[street].append(bet_fraction)
        
        if street != self.last_street:
            self.last_street = street
            self.previous_opponent_investment = opponent_bet
            self.previous_player_investment = player_bet
            return
        
        if opponent_bet > self.previous_opponent_investment:
            if street == 0:
                self.opponent_tracker.preflop_total_actions += 1
                self.opponent_tracker.preflop_raise_count += 1
            else:
                self.opponent_tracker.postflop_total_actions += 1
                if self.previous_player_investment > self.previous_opponent_investment:
                    self.opponent_tracker.postflop_raise_count += 1
                else:
                    self.opponent_tracker.postflop_bet_count += 1
            self.opponent_tracker.last_was_aggressive = True
        
        self.previous_opponent_investment = opponent_bet
        self.previous_player_investment = player_bet


if __name__ == '__main__':
    run_bot(PokerPlayer(), parse_args())
