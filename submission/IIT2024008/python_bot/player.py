import random
import eval7
from collections import Counter
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import STARTING_STACK, BIG_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from __future__ import annotations
from dataclasses import dataclass


class Test_1(Bot):
    def __init__(self):
        self.rank_map = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        self.range_cache = {}
        self.opp = {
            'showdowns': 0,
            'fold_seen': 0,
            'big_bets_seen': 0,
            'small_bets_seen': 0,
            'recent_deltas': []
        }
        self.round_id = 0
        self.decision_points = 0
        self.opp_bet_raise_events = 0
        self.opp_check_events = 0
        self.hand_opp_checks = 0
        self.we_bet_or_raised_this_hand = False
        self.our_bet_attempts = 0

# --- Core Constants & Mappings ---
CARD_RANKS = '23456789TJQKA'
RANK_MAPPING = {c: i + 2 for i, c in enumerate(CARD_RANKS)}
MULTIPLIER_RATIO = 1.5
MULTIPLIER_BASE = 10

# Baseline tactical weights (adaptive, no external runtime training)
BASE_TACTICAL_WEIGHTS = {
    'seed': 199441360,
    'preflop_samples': 1711,
    'mc_scale': 1.1960216944729805,
    'haircut_scale': 0.6162629155139829,
    'aggro_up_threshold': 0.547331376895446,
    'aggro_down_threshold': 0.45,
    'aggro_step': 0.16365262733737607,
    'bounty_bump_scale': 1.1862543175502915,
    'preflop_call_margin': 0.04052781931372232,
    'river_probe_raise_prob': 0.5376180747833821,
    'low_equity_bluff_prob': 0.3449330274551519,
    'postflop_value_raise_eq': 0.8630728810070052,
    'semi_raise_eq_low': 0.5576964681334766,
    'semi_raise_eq_high': 0.767999096314951,
    'semi_raise_fold_prior': 0.5162509809675553,
    'postflop_call_margin': 0.09518729724315873,
}

# Overrides for highly aggressive scenarios
AGGRESSIVE_TACTICAL_WEIGHTS = {
    'preflop_call_margin': 0.00,
    'postflop_call_margin': 0.03,
    'postflop_value_raise_eq': 0.75,
    'semi_raise_eq_low': 0.50,
    'semi_raise_eq_high': 0.73,
    'semi_raise_fold_prior': 0.58,
    'river_probe_raise_prob': 0.70,
    'low_equity_bluff_prob': 0.20,
    'aggro_up_threshold': 0.60,
    'aggro_down_threshold': 0.25,
    'aggro_step': 0.10,
}


def load_tactical_weights():
    """Initializes the baseline model dictionary."""
    return dict(BASE_TACTICAL_WEIGHTS)


def get_normalized_hand_signature(card_strings):
    """
    Compresses a 2-card hand into a standardized string signature.
    Examples: 'AKs' (suited), '77' (pocket pair), 'T9o' (offsuit).
    """
    rank1, suit1 = card_strings[0][0], card_strings[0][1]
    rank2, suit2 = card_strings[1][0], card_strings[1][1]
    val1, val2 = RANK_MAPPING[rank1], RANK_MAPPING[rank2]
    
    # Ensure highest rank is always first
    if val1 < val2:
        rank1, rank2, suit1, suit2, val1, val2 = rank2, rank1, suit2, suit1, val2, val1
        
    if rank1 == rank2:
        return rank1 + rank2
    return rank1 + rank2 + ('s' if suit1 == suit2 else 'o')


def construct_cards_from_signature(signature):
    """Parses a signature (e.g., 'AKs') back into eval7 Card objects."""
    r1 = signature[0]
    r2 = signature[1]
    if len(signature) == 2:
        return [eval7.Card(r1 + 's'), eval7.Card(r2 + 'h')]
    if signature[2] == 's':
        return [eval7.Card(r1 + 's'), eval7.Card(r2 + 's')]
    return [eval7.Card(r1 + 's'), eval7.Card(r2 + 'h')]


def _calculate_initial_mc_winrates(iterations_per_hand=480):
    """
    Executes Monte-Carlo simulations preflop against a random distribution 
    to populate base equities for all 169 starting hands.
    """
    signatures = []
    for i, r1 in enumerate(CARD_RANKS):
        for j, r2 in enumerate(CARD_RANKS):
            if i == j:
                signatures.append(r1 + r2)
            elif i < j:
                hi, lo = r2, r1
                signatures.append(hi + lo + 's')
                signatures.append(hi + lo + 'o')
                
    full_deck = list(eval7.Deck().cards)
    equity_matrix = {}
    
    for sig in signatures:
        hero_hand = construct_cards_from_signature(sig)
        unavailable_cards = set(str(c) for c in hero_hand)
        available_deck = [c for c in full_deck if str(c) not in unavailable_cards]
        
        victories = ties = 0
        for _ in range(iterations_per_hand):
            random.shuffle(available_deck)
            villain_hand = available_deck[:2]
            community_cards = available_deck[2:7]
            
            hero_score = eval7.evaluate(hero_hand + community_cards)
            villain_score = eval7.evaluate(villain_hand + community_cards)
            
            if hero_score > villain_score:
                victories += 1
            elif hero_score == villain_score:
                ties += 1
                
        equity_matrix[sig] = (victories + 0.5 * ties) / iterations_per_hand
    return equity_matrix


def calc_target_appearance_probability(target_rank, known_hole, known_board, upcoming_cards):
    """
    Calculates the statistical likelihood of our target bounty rank appearing.
    Returns 1.0 immediately if the rank is already visible.
    """
    revealed_cards = known_hole + known_board
    if target_rank in {c[0] for c in revealed_cards}:
        return 1.0
    if upcoming_cards <= 0:
        return 0.0
        
    hidden_pool = 52 - len(revealed_cards)
    targets_remaining = 4
    prob_missing_all = 1.0
    
    for i in range(upcoming_cards):
        prob_missing_all *= max(0, hidden_pool - targets_remaining - i) / (hidden_pool - i)
        
    return 1.0 - prob_missing_all


def quick_monte_carlo_eval(hero_hand, community_cards, iterations):
    """Fast Monte Carlo equity estimation against a random opponent hand."""
    unavailable = {str(c) for c in hero_hand}
    for c in community_cards:
        unavailable.add(str(c))
        
    available_deck = [c for c in eval7.Deck().cards if str(c) not in unavailable]
    cards_to_deal = 5 - len(community_cards)
    victories = ties = 0
    
    for _ in range(iterations):
        random.shuffle(available_deck)
        villain_hand = available_deck[:2]
        runout = available_deck[2:2 + cards_to_deal]
        
        final_board = community_cards + runout if cards_to_deal else community_cards
        hero_score = eval7.evaluate(hero_hand + final_board)
        villain_score = eval7.evaluate(villain_hand + final_board)
        
        if hero_score > villain_score:
            victories += 1
        elif hero_score == villain_score:
            ties += 1
            
    return (victories + 0.5 * ties) / iterations


def _get_all_possible_hole_pairs():
    """Generates all 1326 possible two-card combinations."""
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    deck = [r + s for r in ranks for s in suits]
    pairs = []
    for i in range(52):
        for j in range(i + 1, 52):
            pairs.append((deck[i], deck[j]))
    return pairs


def _map_winrates_to_percentiles(equity_matrix):
    """Buckets hole card pairs into top percentile ranges for opponent modeling."""
    all_pairs = _get_all_possible_hole_pairs()

    def get_pair_winrate(pair):
        sig = get_normalized_hand_signature(list(pair))
        return equity_matrix.get(sig, 0.5)

    sorted_pairs = sorted(all_pairs, key=lambda c: -get_pair_winrate(c))
    percentile_map = {}
    
    for threshold in (0.05, 0.10, 0.20, 0.40, 1.00):
        cutoff_idx = max(1, int(round(len(sorted_pairs) * threshold)))
        percentile_map[threshold] = sorted_pairs[:cutoff_idx]
        
    return percentile_map


def simulate_winrate_against_distribution(hero_hand, community_cards, distribution_pool, iterations):
    """Monte Carlo simulation targeting a specific range distribution for the opponent."""
    hero_set = {str(c) for c in hero_hand}
    board_set = {str(c) for c in community_cards}
    unavailable = hero_set | board_set
    
    valid_villain_pairs = [(a, b) for (a, b) in distribution_pool if a not in unavailable and b not in unavailable]
    if not valid_villain_pairs:
        return quick_monte_carlo_eval(hero_hand, community_cards, iterations)

    full_deck_raw = eval7.Deck().cards
    clean_deck = [c for c in full_deck_raw if str(c) not in unavailable]
    cards_to_deal = 5 - len(community_cards)
    victories = ties = 0
    
    for _ in range(iterations):
        card_a, card_b = random.choice(valid_villain_pairs)
        villain_hand = [eval7.Card(card_a), eval7.Card(card_b)]
        villain_set = {card_a, card_b}
        
        runout_deck = [c for c in clean_deck if str(c) not in villain_set]
        random.shuffle(runout_deck)
        
        runout = runout_deck[:cards_to_deal] if cards_to_deal else []
        final_board = community_cards + runout if cards_to_deal else community_cards
        
        hero_score = eval7.evaluate(hero_hand + final_board)
        villain_score = eval7.evaluate(villain_hand + final_board)
        
        if hero_score > villain_score:
            victories += 1
        elif hero_score == villain_score:
            ties += 1
            
    return (victories + 0.5 * ties) / iterations


def convert_strings_to_cards(card_strings):
    """Helper to convert raw strings to eval7 Card objects."""
    return [eval7.Card(s) for s in card_strings]


def categorize_starting_hand(card_strings):
    """
    Evaluates preflop hand strength into distinct strategic buckets:
    premium | strong | medium | openable | trash
    """
    rank1 = card_strings[0][0]
    rank2 = card_strings[1][0]
    suit1 = card_strings[0][1]
    suit2 = card_strings[1][1]
    
    val1 = RANK_MAPPING[rank1]
    val2 = RANK_MAPPING[rank2]
    
    if val1 < val2:
        val1, val2 = val2, val1
        
    is_suited = suit1 == suit2
    is_pair = val1 == val2
    
    if is_pair and val1 >= 11: return 'premium'
    if val1 == 14 and val2 == 13: return 'premium'
    if val1 == 14 and val2 == 12 and is_suited: return 'premium'
    
    if is_pair and val1 >= 9: return 'strong'
    if val1 == 14 and val2 == 12: return 'strong'
    if val1 == 14 and val1 == 11 and is_suited: return 'strong'
    if val1 == 13 and val2 == 12 and is_suited: return 'strong'
    
    if is_pair: return 'medium'
    if val1 == 14: return 'medium'
    if val1 == 13 and val2 >= 9: return 'medium'
    if val1 == 13 and is_suited: return 'medium'
    if val1 == 12 and val2 >= 9: return 'medium'
    if val1 == 12 and is_suited: return 'medium'
    if val1 == 11 and val2 >= 9: return 'medium'
    if val1 == 11 and is_suited: return 'medium'
    if val1 == 10 and val2 >= 8: return 'medium'
    if is_suited and (val1 - val2) <= 2 and val1 >= 6: return 'medium'
    
    if val1 >= 10: return 'openable'
    if is_suited and (val1 - val2) <= 3: return 'openable'
    
    return 'trash'


class AdversaryTracker:
    """Monitors opponent behaviors to detect tilt or highly aggressive patterns."""

    def __init__(self):
        self.preflop_actions = 0
        self.preflop_raises = 0
        self.preflop_calls = 0
        self.preflop_folds = 0
        self.postflop_actions = 0
        self.postflop_raises = 0
        self.postflop_bets = 0
        self.postflop_checks = 0
        self.postflop_folds = 0
        self.postflop_calls = 0
        self.vpip_rounds = 0
        self.rounds_played = 0
        self.showdowns_won_at = []
        self.last_action_aggressive = False
        self.bet_fold_freq_num = 0
        self.bet_fold_freq_den = 0

    def calc_fold_prior(self):
        total = self.preflop_folds + self.preflop_calls + self.preflop_raises
        return (self.preflop_folds + 3) / (total + 6)

    def calc_preflop_aggression(self):
        total = self.preflop_actions
        return (self.preflop_raises + 1) / (total + 4)

    def calc_postflop_aggression(self):
        total = self.postflop_actions
        return (self.postflop_bets + self.postflop_raises + 2) / (total + 6)

    def calc_fold_to_bet_prior(self):
        total = self.bet_fold_freq_den
        return (self.bet_fold_freq_num + 2) / (total + 5)


@dataclass(frozen=True)
class ActionThresholds:
    """Data container for decision-making bounds and margins."""
    preflop_call_margin: float
    postflop_call_margin: float
    value_raise_eq: float
    semi_raise_eq_low: float
    semi_raise_eq_high: float
    semi_raise_fold_prior: float
    river_probe_raise_prob: float
    low_equity_bluff_prob: float
    aggro_up_threshold: float
    aggro_down_threshold: float
    aggro_step: float


class SmartPokerBot(Bot):
    def __init__(self):
        self.rank_mapping = RANK_MAPPING
        self.tactics = load_tactical_weights()
        random.seed(self.tactics['seed'])
        
        try:
            self.preflop_winrates = _calculate_initial_mc_winrates(iterations_per_hand=self.tactics['preflop_samples'])
        except Exception:
            self.preflop_winrates = {}
            
        try:
            self.distribution_ranges = _map_winrates_to_percentiles(self.preflop_winrates)
        except Exception:
            self.distribution_ranges = {1.00: _get_all_possible_hole_pairs()}
            
        self.adversary = AdversaryTracker()
        self.round_history = []
        self.last_villain_pip = 0
        self.last_hero_pip = 0
        self.last_street_processed = -1
        self.hero_last_action = None
        self.current_bankroll = 0

    def determine_mc_iterations(self, game_clock, street):
        """Scales our Monte Carlo sampling based on remaining computation time."""
        scale_factor = self.tactics['mc_scale']
        if game_clock < 2.5:
            return 0 if scale_factor >= 1.0 else int(40 * scale_factor)
        if game_clock < 6.0:
            return int(120 * scale_factor)
        if game_clock < 15.0:
            return int(240 * scale_factor)
        if game_clock < 30.0:
            return int(380 * scale_factor)
        return int(520 * scale_factor)

    def query_preflop_equity(self, card_strings):
        sig = get_normalized_hand_signature(card_strings)
        return self.preflop_winrates.get(sig, 0.5)

    def infer_villain_range(self, villain_invested, facing_bet, street):
        """Deduces the opponent's likely holding percentile based on their investment."""
        commitment_ratio = villain_invested / STARTING_STACK
        if not facing_bet:
            return 1.00
            
        if commitment_ratio >= 0.80: base = 0.05
        elif commitment_ratio >= 0.40: base = 0.10
        elif commitment_ratio >= 0.15: base = 0.20
        elif commitment_ratio >= 0.06: base = 0.40
        else: base = 1.00

        if facing_bet:
            aggression_factor = 0.45 * self.adversary.calc_preflop_aggression() + 0.55 * self.adversary.calc_postflop_aggression()
            if aggression_factor >= 0.56:
                base *= 1.45
            elif aggression_factor <= 0.38:
                base *= 0.70

            fold_prior = self.adversary.calc_fold_prior()
            if fold_prior >= 0.55:
                base *= 0.85
            elif fold_prior <= 0.35:
                base *= 1.12

        base = max(0.05, min(1.00, base))
        available_buckets = (0.05, 0.10, 0.20, 0.40, 1.00)
        return min(available_buckets, key=lambda b: abs(b - base))

    def evaluate_hand_strength(self, hero_strs, board_strs, iterations, villain_invested=0, facing_bet=False):
        """Unified entry point for determining current hand strength."""
        hero_cards = convert_strings_to_cards(hero_strs)
        board_cards = convert_strings_to_cards(board_strs)
        
        if iterations <= 0:
            if not board_cards:
                return self.query_preflop_equity(hero_strs)
            return self._heuristic_board_strength(hero_strs, board_strs)
            
        range_percentile = self.infer_villain_range(villain_invested, facing_bet, len(board_strs))
        valid_combos = self.distribution_ranges.get(range_percentile) or self.distribution_ranges.get(1.00)
        
        if valid_combos is None:
            return quick_monte_carlo_eval(hero_cards, board_cards, iterations)
        return simulate_winrate_against_distribution(hero_cards, board_cards, valid_combos, iterations)

    def _heuristic_board_strength(self, hero_strs, board_strs):
        """Fast approximation for hand strength when we run completely out of time."""
        h_ranks = [c[0] for c in hero_strs]
        b_ranks = [c[0] for c in board_strs]
        
        if h_ranks[0] == h_ranks[1]:
            val = RANK_MAPPING[h_ranks[0]]
            return min(0.85, 0.55 + val * 0.015)
            
        highest_board = max(RANK_MAPPING[r] for r in b_ranks) if b_ranks else 0
        has_top_pair = any(RANK_MAPPING[h] == highest_board for h in h_ranks)
        board_is_paired = len(set(b_ranks)) < len(b_ranks)
        
        if has_top_pair: return 0.6
        if board_is_paired: return 0.45
        
        max_hole_val = max(RANK_MAPPING[h] for h in h_ranks)
        return 0.30 + 0.01 * max_hole_val

    def hero_bounty_impact(self, hero_bounty, hero_strs, board_strs, street):
        visible_h_ranks = {c[0] for c in hero_strs}
        visible_b_ranks = {c[0] for c in board_strs}
        cards_to_come = 5 - len(board_strs)
        
        if hero_bounty in visible_h_ranks or hero_bounty in visible_b_ranks:
            return MULTIPLIER_RATIO, float(MULTIPLIER_BASE)
            
        hit_prob = calc_target_appearance_probability(hero_bounty, hero_strs, board_strs, cards_to_come)
        multiplier = 1.0 + (MULTIPLIER_RATIO - 1.0) * hit_prob
        addition = MULTIPLIER_BASE * hit_prob
        return multiplier, addition

    def villain_bounty_impact(self, hero_strs, board_strs, street):
        visible_h_ranks = {c[0] for c in hero_strs}
        visible_b_ranks = {c[0] for c in board_strs}
        cards_to_come = 5 - len(board_strs)
        
        cumulative_prob = 0.0
        for rank in CARD_RANKS:
            if rank in visible_b_ranks:
                hit_prob = 1.0
            else:
                cards_removed = 2 + len(board_strs)
                targets_left = 4 - (1 if rank in visible_h_ranks else 0)
                hidden_cards = 52 - cards_removed
                villain_hole_count = 2
                draws_total = villain_hole_count + cards_to_come
                
                prob_missing_all = 1.0
                for i in range(draws_total):
                    denominator = hidden_cards - i
                    if denominator <= 0:
                        prob_missing_all = 0.0
                        break
                    prob_missing_all *= max(0, denominator - targets_left - i) / denominator
                hit_prob = 1.0 - prob_missing_all
            cumulative_prob += hit_prob
            
        avg_hit_prob = cumulative_prob / 13.0
        multiplier = 1.0 + (MULTIPLIER_RATIO - 1.0) * avg_hit_prob
        addition = MULTIPLIER_BASE * avg_hit_prob
        return multiplier, addition

    def _calc_marginal_payoffs(self, total_pot, hero_invested, cost_to_play, hero_bm, hero_bb, villain_bm, villain_bb):
        villain_invested = total_pot - hero_invested
        win_value = villain_invested * hero_bm + hero_bb + hero_invested
        lose_value = (hero_invested + cost_to_play) * villain_bm + villain_bb - hero_invested
        return win_value, lose_value

    def determine_pot_odds(self, cost_to_play, total_pot, villain_bm, villain_bb, hero_bm, hero_bb, hero_invested):
        win_value, lose_value = self._calc_marginal_payoffs(
            total_pot, hero_invested, cost_to_play, hero_bm, hero_bb, villain_bm, villain_bb
        )
        total_denom = win_value + lose_value
        if total_denom <= 0:
            return 1.0
        return max(0.0, lose_value / total_denom)

    def formulate_raise_size(self, winrate, total_pot, hero_pip, villain_pip, hero_stack, villain_stack, min_raise, max_raise, street, aggro_modifier):
        if winrate > 0.85: fraction = 1.1
        elif winrate > 0.72: fraction = 0.8
        elif winrate > 0.60: fraction = 0.6
        else: fraction = 0.5
        
        fraction += aggro_modifier
        fraction = max(0.35, min(1.3, fraction))
        
        target_amount = villain_pip + int(max(total_pot, 4) * fraction)
        target_amount = max(target_amount, min_raise)
        target_amount = min(target_amount, max_raise)
        return target_amount

    def gauge_villain_tightness(self, hero_invested, villain_invested, cost_to_play, street, facing_bet):
        if not facing_bet:
            return 0.01
        commitment_ratio = villain_invested / STARTING_STACK
        if street == 0:
            if commitment_ratio >= 0.85: return 0.22
            if commitment_ratio >= 0.40: return 0.14
            if commitment_ratio >= 0.15: return 0.08
            if commitment_ratio >= 0.06: return 0.05
            return 0.02
        else:
            if commitment_ratio >= 0.80: return 0.18
            if commitment_ratio >= 0.40: return 0.12
            if commitment_ratio >= 0.20: return 0.08
            if commitment_ratio >= 0.08: return 0.04
            return 0.02

    def determine_macro_strategy(self, game_state):
        """Analyzes overall tournament state to decide if we need to play tight or loose."""
        horizon_length = 500 if game_state.round_num <= 500 else 1000
        rounds_remaining = max(0, horizon_length - game_state.round_num)
        lead_delta = game_state.bankroll
        
        if lead_delta >= max(140, int(rounds_remaining * 0.36)):
            return 'protect'
        if lead_delta <= -max(200, int(rounds_remaining * 0.42)):
            return 'chase'
        return 'neutral'

    def construct_action_thresholds(self, game_state, street):
        preflop_aggro = self.adversary.calc_preflop_aggression()
        if self.adversary.preflop_actions < 12:
            passivity = 0.35
        else:
            if preflop_aggro <= 0.34: passivity = 0.95
            elif preflop_aggro >= 0.50: passivity = 0.0
            else: passivity = max(0.0, min(0.95, (0.50 - preflop_aggro) / 0.16))

        def blend_metrics(base_key, attack_key):
            base_val = self.tactics[base_key]
            attack_val = AGGRESSIVE_TACTICAL_WEIGHTS[attack_key]
            return base_val + (attack_val - base_val) * passivity

        bounds = ActionThresholds(
            preflop_call_margin=blend_metrics('preflop_call_margin', 'preflop_call_margin'),
            postflop_call_margin=blend_metrics('postflop_call_margin', 'postflop_call_margin'),
            value_raise_eq=blend_metrics('postflop_value_raise_eq', 'postflop_value_raise_eq'),
            semi_raise_eq_low=blend_metrics('semi_raise_eq_low', 'semi_raise_eq_low'),
            semi_raise_eq_high=blend_metrics('semi_raise_eq_high', 'semi_raise_eq_high'),
            semi_raise_fold_prior=blend_metrics('semi_raise_fold_prior', 'semi_raise_fold_prior'),
            river_probe_raise_prob=blend_metrics('river_probe_raise_prob', 'river_probe_raise_prob'),
            low_equity_bluff_prob=blend_metrics('low_equity_bluff_prob', 'low_equity_bluff_prob'),
            aggro_up_threshold=blend_metrics('aggro_up_threshold', 'aggro_up_threshold'),
            aggro_down_threshold=blend_metrics('aggro_down_threshold', 'aggro_down_threshold'),
            aggro_step=blend_metrics('aggro_step', 'aggro_step'),
        )

        macro_mode = self.determine_macro_strategy(game_state)
        if macro_mode == 'protect':
            bounds = ActionThresholds(
                preflop_call_margin=bounds.preflop_call_margin + 0.012,
                postflop_call_margin=bounds.postflop_call_margin + 0.016,
                value_raise_eq=min(0.92, bounds.value_raise_eq + 0.014),
                semi_raise_eq_low=min(0.85, bounds.semi_raise_eq_low + 0.014),
                semi_raise_eq_high=min(0.92, bounds.semi_raise_eq_high + 0.01),
                semi_raise_fold_prior=min(0.90, bounds.semi_raise_fold_prior + 0.02),
                river_probe_raise_prob=max(0.0, bounds.river_probe_raise_prob * 0.80),
                low_equity_bluff_prob=max(0.0, bounds.low_equity_bluff_prob * 0.70),
                aggro_up_threshold=bounds.aggro_up_threshold + 0.02,
                aggro_down_threshold=min(0.65, bounds.aggro_down_threshold + 0.02),
                aggro_step=bounds.aggro_step * 0.90,
            )
        elif macro_mode == 'chase':
            bounds = ActionThresholds(
                preflop_call_margin=bounds.preflop_call_margin - 0.003,
                postflop_call_margin=bounds.postflop_call_margin - 0.003,
                value_raise_eq=max(0.55, bounds.value_raise_eq - 0.003),
                semi_raise_eq_low=max(0.30, bounds.semi_raise_eq_low - 0.006),
                semi_raise_eq_high=min(0.92, bounds.semi_raise_eq_high + 0.006),
                semi_raise_fold_prior=max(0.25, bounds.semi_raise_fold_prior - 0.006),
                river_probe_raise_prob=min(0.95, bounds.river_probe_raise_prob * 1.02),
                low_equity_bluff_prob=min(0.50, bounds.low_equity_bluff_prob * 1.03),
                aggro_up_threshold=max(0.30, bounds.aggro_up_threshold - 0.01),
                aggro_down_threshold=max(0.08, bounds.aggro_down_threshold - 0.01),
                aggro_step=min(0.30, bounds.aggro_step * 1.02),
            )

        if street == 5:
            bounds = ActionThresholds(
                preflop_call_margin=bounds.preflop_call_margin,
                postflop_call_margin=bounds.postflop_call_margin,
                value_raise_eq=max(0.55, bounds.value_raise_eq - 0.01),
                semi_raise_eq_low=max(0.30, bounds.semi_raise_eq_low - 0.01),
                semi_raise_eq_high=bounds.semi_raise_eq_high,
                semi_raise_fold_prior=bounds.semi_raise_fold_prior,
                river_probe_raise_prob=bounds.river_probe_raise_prob,
                low_equity_bluff_prob=bounds.low_equity_bluff_prob,
                aggro_up_threshold=bounds.aggro_up_threshold,
                aggro_down_threshold=bounds.aggro_down_threshold,
                aggro_step=bounds.aggro_step,
            )

        return bounds

    def handle_new_round(self, game_state, round_state, active):
        self.hero_last_action = None
        self.last_street_processed = 0
        self.last_hero_pip = 0
        self.last_villain_pip = 0
        self.adversary.rounds_played += 1

    def handle_round_over(self, game_state, terminal_state, active):
        hero_net = terminal_state.deltas[active]
        self.current_bankroll += hero_net

    def get_action(self, game_state, round_state, active):
        available_actions = round_state.legal_actions()
        current_street = round_state.street
        hero_cards = round_state.hands[active]
        board_cards = round_state.deck[:current_street] if current_street > 0 else []
        
        hero_pip = round_state.pips[active]
        villain_pip = round_state.pips[1 - active]
        hero_stack = round_state.stacks[active]
        villain_stack = round_state.stacks[1 - active]
        
        cost_to_play = villain_pip - hero_pip
        hero_bounty = round_state.bounties[active]
        hero_invested = STARTING_STACK - hero_stack
        villain_invested = STARTING_STACK - villain_stack
        total_pot = hero_invested + villain_invested
        game_clock = game_state.game_clock

        self._record_villain_tendencies(round_state, active, current_street)
        facing_bet = cost_to_play > 0
        min_raise = max_raise = 0
        
        if RaiseAction in available_actions:
            min_raise, max_raise = round_state.raise_bounds()

        # Emergency escape hatch for clock preservation
        if game_clock < 1.2:
            if CheckAction in available_actions: return CheckAction()
            if cost_to_play <= 2 and CallAction in available_actions: return CallAction()
            if FoldAction in available_actions: return FoldAction()
            return CallAction()

        sim_iterations = self.determine_mc_iterations(game_clock, current_street)
        
        if current_street == 0:
            raw_winrate = self.query_preflop_equity(hero_cards)
        else:
            raw_winrate = self.evaluate_hand_strength(
                hero_cards, board_cards, sim_iterations,
                villain_invested=villain_invested,
                facing_bet=facing_bet,
            )

        hero_bm, hero_bb = self.hero_bounty_impact(hero_bounty, hero_cards, board_cards, current_street)
        villain_bm, villain_bb = self.villain_bounty_impact(hero_cards, board_cards, current_street)

        if current_street == 0:
            tight_modifier = self.gauge_villain_tightness(
                hero_invested, villain_invested, cost_to_play, current_street, facing_bet
            )
            tight_modifier *= self.tactics['haircut_scale']
        else:
            tight_modifier = 0.0

        bounds = self.construct_action_thresholds(game_state, current_street)
        fold_likelihood = self.adversary.calc_fold_to_bet_prior()
        aggression_shift = 0.0
        
        if fold_likelihood > bounds.aggro_up_threshold:
            aggression_shift += bounds.aggro_step
        elif fold_likelihood < bounds.aggro_down_threshold:
            aggression_shift -= bounds.aggro_step

        bounty_inflation = self.tactics['bounty_bump_scale'] * (0.02 * (hero_bm - 1.0) + (hero_bb / 450.0))

        # --- PREFLOP LOGIC ---
        if current_street == 0:
            h_ranks = [c[0] for c in hero_cards]
            holds_bounty = hero_bounty in h_ranks
            winrate_vs_range = max(0.05, raw_winrate - tight_modifier)
            effective_winrate = winrate_vs_range + bounty_inflation
            
            hand_category = categorize_starting_hand(hero_cards)
            villain_commitment_ratio = villain_invested / STARTING_STACK
            req_odds = self.determine_pot_odds(cost_to_play, total_pot, villain_bm, villain_bb, hero_bm, hero_bb, hero_invested)

            if not facing_bet:
                if RaiseAction in available_actions:
                    if hand_category == 'premium': target_amt = villain_pip + max(6, int(total_pot * 1.6))
                    elif hand_category == 'strong': target_amt = villain_pip + max(5, int(total_pot * 1.2))
                    elif hand_category in ('medium', 'openable'): target_amt = villain_pip + max(4, int(total_pot * 0.9))
                    elif holds_bounty: target_amt = villain_pip + max(4, int(total_pot * 0.8))
                    else: target_amt = None
                    
                    if target_amt is not None:
                        return RaiseAction(min(max(target_amt, min_raise), max_raise))
                if CheckAction in available_actions: return CheckAction()
                if CallAction in available_actions: return CallAction()

            if villain_commitment_ratio >= 0.65 or cost_to_play >= hero_stack * 0.55:
                if hand_category == 'premium' and CallAction in available_actions: return CallAction()
                if FoldAction in available_actions: return FoldAction()
                if CheckAction in available_actions: return CheckAction()
                return CallAction()

            if villain_pip >= 25 or villain_commitment_ratio >= 0.12:
                if RaiseAction in available_actions and hand_category == 'premium':
                    _, top_raise = round_state.raise_bounds()
                    shove_amt = min(villain_pip + int((total_pot + cost_to_play) * 2.2), top_raise)
                    return RaiseAction(max(shove_amt, min_raise))
                if CallAction in available_actions and hand_category in ('premium', 'strong'): return CallAction()
                if CallAction in available_actions and hand_category == 'medium' and cost_to_play <= total_pot * 0.55: return CallAction()
                if FoldAction in available_actions: return FoldAction()
                if CallAction in available_actions: return CallAction()
                return CheckAction() if CheckAction in available_actions else CallAction()

            if villain_pip >= 5:
                if RaiseAction in available_actions and hand_category == 'premium':
                    target_amt = villain_pip + max(min_raise - villain_pip, int(cost_to_play * 3.0))
                    return RaiseAction(min(max(target_amt, min_raise), max_raise))
                if RaiseAction in available_actions and hand_category == 'strong':
                    target_amt = villain_pip + max(min_raise - villain_pip, int(cost_to_play * 2.6))
                    return RaiseAction(min(max(target_amt, min_raise), max_raise))
                if CallAction in available_actions and hand_category in ('premium', 'strong', 'medium'): return CallAction()
                if CallAction in available_actions and holds_bounty and hand_category == 'openable' and cost_to_play <= 8: return CallAction()
                if FoldAction in available_actions: return FoldAction()
                if CheckAction in available_actions: return CheckAction()
                return CallAction()

            if cost_to_play <= 2:
                if RaiseAction in available_actions and hand_category == 'premium':
                    target_amt = villain_pip + max(min_raise - villain_pip, 6)
                    return RaiseAction(min(max(target_amt, min_raise), max_raise))
                if CallAction in available_actions and hand_category != 'trash': return CallAction()
                if CallAction in available_actions and holds_bounty: return CallAction()
                if FoldAction in available_actions: return FoldAction()
                return CallAction() if CallAction in available_actions else CheckAction()

            if CallAction in available_actions and effective_winrate >= req_odds + bounds.preflop_call_margin and hand_category != 'trash':
                return CallAction()
            if FoldAction in available_actions: return FoldAction()
            if CheckAction in available_actions: return CheckAction()
            return CallAction() if CallAction in available_actions else CheckAction()

        # --- POSTFLOP LOGIC ---
        winrate_vs_range = max(0.02, raw_winrate - tight_modifier)
        effective_winrate = winrate_vs_range + bounty_inflation
        req_odds = self.determine_pot_odds(cost_to_play, total_pot, villain_bm, villain_bb, hero_bm, hero_bb, hero_invested)
        villain_commitment_ratio = villain_invested / STARTING_STACK

        if not facing_bet:
            if RaiseAction in available_actions and total_pot >= 4:
                if raw_winrate >= 0.68:
                    target_amt = self.formulate_raise_size(raw_winrate, total_pot, hero_pip, villain_pip, hero_stack, villain_stack, min_raise, max_raise, current_street, aggression_shift)
                    return RaiseAction(target_amt)
                if raw_winrate >= 0.52:
                    raise_frac = 0.55 + aggression_shift
                    raise_frac = max(0.4, min(0.8, raise_frac))
                    target_amt = villain_pip + max(min_raise - villain_pip, int(max(total_pot, 4) * raise_frac))
                    return RaiseAction(min(max(target_amt, min_raise), max_raise))
                if raw_winrate >= 0.38 and current_street == 3:
                    if random.random() < bounds.river_probe_raise_prob:
                        target_amt = villain_pip + max(min_raise - villain_pip, int(max(total_pot, 4) * 0.45))
                        return RaiseAction(min(max(target_amt, min_raise), max_raise))
                if raw_winrate < 0.30 and total_pot >= 10 and random.random() < bounds.low_equity_bluff_prob:
                    target_amt = villain_pip + max(min_raise - villain_pip, int(max(total_pot, 4) * 0.45))
                    return RaiseAction(min(max(target_amt, min_raise), max_raise))
            if CheckAction in available_actions:
                return CheckAction()

        if RaiseAction in available_actions and effective_winrate >= bounds.value_raise_eq and villain_commitment_ratio < 0.55:
            target_amt = self.formulate_raise_size(raw_winrate, total_pot + cost_to_play, hero_pip, villain_pip, hero_stack, villain_stack, min_raise, max_raise, current_street, aggression_shift)
            return RaiseAction(target_amt)

        if (RaiseAction in available_actions and bounds.semi_raise_eq_low <= effective_winrate < bounds.semi_raise_eq_high and fold_likelihood > bounds.semi_raise_fold_prior and cost_to_play <= total_pot * 0.6 and villain_commitment_ratio < 0.35):
            target_amt = villain_pip + max(min_raise - villain_pip, int((total_pot + cost_to_play) * 0.75))
            return RaiseAction(min(max(target_amt, min_raise), max_raise))

        pot_odds = cost_to_play / max(1, (total_pot + cost_to_play))
        hit_pair_plus = self._check_pair_plus_made_hand(hero_cards, board_cards)
        heavy_bet = cost_to_play >= total_pot * 0.55
        
        if CallAction in available_actions:
            if effective_winrate >= req_odds + bounds.postflop_call_margin:
                if heavy_bet and not hit_pair_plus and raw_winrate < 0.58:
                    pass # Fold heavily against large aggression if we purely missed the board
                else:
                    return CallAction()
            if cost_to_play <= max(3, total_pot * 0.15) and raw_winrate >= 0.32:
                return CallAction()
            if current_street == 5 and fold_likelihood < 0.25 and effective_winrate >= pot_odds and hit_pair_plus:
                return CallAction()

        if FoldAction in available_actions: return FoldAction()
        if CheckAction in available_actions: return CheckAction()
        return CallAction()

    def _check_pair_plus_made_hand(self, hero_strs, board_strs):
        if not board_strs:
            return hero_strs[0][0] == hero_strs[1][0]
        h_ranks = [c[0] for c in hero_strs]
        b_ranks = [c[0] for c in board_strs]
        
        if h_ranks[0] == h_ranks[1]: return True
        if h_ranks[0] in b_ranks or h_ranks[1] in b_ranks: return True
        
        h_suits = [c[1] for c in hero_strs]
        b_suits = [c[1] for c in board_strs]
        for s in set(h_suits):
            if h_suits.count(s) + b_suits.count(s) >= 4:
                return True
                
        try:
            combined_ranks = sorted(set(RANK_MAPPING[r] for r in h_ranks + b_ranks))
            for start in range(2, 11):
                straight_window = set(range(start, start + 5))
                if len(straight_window & set(combined_ranks)) >= 4:
                    return True
        except Exception:
            pass
        return False

    def _record_villain_tendencies(self, round_state, active, street):
        villain_pip = round_state.pips[1 - active]
        hero_pip = round_state.pips[active]
        
        if street != self.last_street_processed:
            self.last_street_processed = street
            self.last_villain_pip = villain_pip
            self.last_hero_pip = hero_pip
            return
            
        if villain_pip > self.last_villain_pip:
            if street == 0:
                self.adversary.preflop_actions += 1
                self.adversary.preflop_raises += 1
            else:
                self.adversary.postflop_actions += 1
                if self.last_hero_pip > self.last_villain_pip:
                    self.adversary.postflop_raises += 1
                else:
                    self.adversary.postflop_bets += 1
            self.adversary.last_action_aggressive = True
            
        self.last_villain_pip = villain_pip
        self.last_hero_pip = hero_pip


if __name__ == '__main__':
    run_bot(SmartPokerBot(), parse_args())
