"""Exploit-oriented Bounty Hold'em bot for `submission/IIT2024008/python_bot`.

This module keeps all static tuning data in one place, estimates equity with
Monte Carlo sampling, and updates lightweight in-memory opponent statistics
during the match. The strategy is still the same as before; this file is just
structured so the fixed parameters are easier to find and maintain.
"""



from __future__ import annotations
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import STARTING_STACK , BIG_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
import random
from dataclasses import dataclass
import eval7 , typing_extensions


def load_static_profiles():
    target_mapping = '23456789TJQKA'
    ranks_currV = {c: i + 2 for i, c in enumerate(target_mapping)}
    bounty_ratio = 1.5
    bit_check = 10
    default_model = {
        # Static profile: adaptive but not runtime-trained from any external file.
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
    attack_profile = {
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
    return target_mapping, ranks_currV, bounty_ratio, bit_check, default_model, attack_profile


target_mapping, ranks_currV, BOUNTY_RATIO, bit_check, DEFAULT_MODEL, ATTACK_PROFILE = load_static_profiles()


def load_model():
    return dict(DEFAULT_MODEL)


def canonical_hand_key(cards_str):
    '''Collapse a 2-card hand into a canonical key (e.g. "AKs", "77", "T9o").'''
    r1, s1 = cards_str[0][0], cards_str[0][1]
    r2, s2 = cards_str[1][0], cards_str[1][1]
    v1, v2 = ranks_currV[r1], ranks_currV[r2]
    if v1 < v2:
        r1, r2, s1, s2, v1, v2 = r2, r1, s2, s1, v2, v1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ('s' if s1 == s2 else 'o')


def build_hand_from_key(key):
    '''Return two eval7.Cards representing a concrete instance of the class.'''
    r1 = key[0]
    r2 = key[1]
    if len(key) == 2:
        return [eval7.Card(r1 + 's'), eval7.Card(r2 + 'h')]
    if key[2] == 's':
        return [eval7.Card(r1 + 's'), eval7.Card(r2 + 's')]
    return [eval7.Card(r1 + 's'), eval7.Card(r2 + 'h')]


def _precompute_preflop_equity(samples_per_hand=480):
    '''Monte-Carlo preflop equity vs a random hand for all 169 hand classes.'''
    keys = []
    for i, r1 in enumerate(target_mapping):
        for j, r2 in enumerate(target_mapping):
            if i == j:
                keys.append(r1 + r2)
            elif i < j:
                hi, lo = r2, r1
                keys.append(hi + lo + 's')
                keys.append(hi + lo + 'o')
    full_deck = list(eval7.Deck().cards)
    table = {}
    for key in keys:
        hero = build_hand_from_key(key)
        dead = set(str(c) for c in hero)
        remaining = [c for c in full_deck if str(c) not in dead]
        wins = ties = 0
        for _ in range(samples_per_hand):
            random.shuffle(remaining)
            opp = remaining[:2]
            board = remaining[2:7]
            s1 = eval7.evaluate(hero + board)
            s2 = eval7.evaluate(opp + board)
            if s1 > s2:
                wins += 1
            elif s1 == s2:
                ties += 1
        table[key] = (wins + 0.5 * ties) / samples_per_hand
    return table


def bounty_hit_prob_future(bounty_rank, visible_hole, visible_board, remaining_board_count):
    '''Probability my bounty rank appears in (hole + full 5-card board),
    given what we can already see. Returns 1.0 when already hit.'''
    visible = visible_hole + visible_board
    if bounty_rank in {c[0] for c in visible}:
        return 1.0
    if remaining_board_count <= 0:
        return 0.0
    unknown = 52 - len(visible)
    bounty_left = 4
    p_none = 1.0
    for i in range(remaining_board_count):
        p_none *= max(0, unknown - bounty_left - i) / (unknown - i)
    return 1.0 - p_none


def fast_equity(hero, board, samples):
    '''Monte Carlo equity vs a random opponent hand from the remaining deck.'''
    dead = {str(c) for c in hero}
    for c in board:
        dead.add(str(c))
    remaining = [c for c in eval7.Deck().cards if str(c) not in dead]
    board_rem = 5 - len(board)
    wins = ties = 0
    for _ in range(samples):
        random.shuffle(remaining)
        opp = remaining[:2]
        run = remaining[2:2 + board_rem]
        full_board = board + run if board_rem else board
        s1 = eval7.evaluate(hero + full_board)
        s2 = eval7.evaluate(opp + full_board)
        if s1 > s2:
            wins += 1
        elif s1 == s2:
            ties += 1
    return (wins + 0.5 * ties) / samples


def _generate_all_combos():
    '''Return all 1326 two-card combos as tuples of card strings.'''
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    cards = [r + s for r in ranks for s in suits]
    combos = []
    for i in range(52):
        for j in range(i + 1, 52):
            combos.append((cards[i], cards[j]))
    return combos


def _build_range_combo_lists(preflop_eq_table):
    '''Return dict {range_pct: [(card1, card2), ...]} of top ranked combos.'''
    all_combos = _generate_all_combos()

    def combo_eq(combo):
        k = canonical_hand_key(list(combo))
        return preflop_eq_table.get(k, 0.5)

    ranked = sorted(all_combos, key=lambda c: -combo_eq(c))
    out = {}
    for pct in (0.05, 0.10, 0.20, 0.40, 1.00):
        n = max(1, int(round(len(ranked) * pct)))
        out[pct] = ranked[:n]
    return out


def mc_equity_vs_range(hero, board, range_combos, samples):
    '''Monte Carlo equity with opponent sampled from given range combos.'''
    hero_strs = {str(c) for c in hero}
    dead_board = {str(c) for c in board}
    dead = hero_strs | dead_board
    valid = [(a, b) for (a, b) in range_combos if a not in dead and b not in dead]
    if not valid:
        return fast_equity(hero, board, samples)

    all_52 = eval7.Deck().cards
    remaining_full = [c for c in all_52 if str(c) not in dead]
    board_rem = 5 - len(board)
    wins = ties = 0
    for _ in range(samples):
        a, b = random.choice(valid)
        opp = [eval7.Card(a), eval7.Card(b)]
        opp_strs = {a, b}
        run_pool = [c for c in remaining_full if str(c) not in opp_strs]
        random.shuffle(run_pool)
        run = run_pool[:board_rem] if board_rem else []
        full_board = board + run if board_rem else board
        s1 = eval7.evaluate(hero + full_board)
        s2 = eval7.evaluate(opp + full_board)
        if s1 > s2:
            wins += 1
        elif s1 == s2:
            ties += 1
    return (wins + 0.5 * ties) / samples


def parse_cards(strs):
    return [eval7.Card(s) for s in strs]


def classify_preflop(cards_str):
    '''Return one of: premium | strong | medium | openable | trash.'''
    r1 = cards_str[0][0]
    r2 = cards_str[1][0]
    s1 = cards_str[0][1]
    s2 = cards_str[1][1]
    v1 = ranks_currV[r1]
    v2 = ranks_currV[r2]
    if v1 < v2:
        v1, v2 = v2, v1
    suited = s1 == s2
    pair = v1 == v2
    if pair and v1 >= 11:
        return 'premium'
    if v1 == 14 and v2 == 13:
        return 'premium'
    if v1 == 14 and v2 == 12 and suited:
        return 'premium'
    if pair and v1 >= 9:
        return 'strong'
    if v1 == 14 and v2 == 12:
        return 'strong'
    if v1 == 14 and v2 == 11 and suited:
        return 'strong'
    if v1 == 13 and v2 == 12 and suited:
        return 'strong'
    if pair:
        return 'medium'
    if v1 == 14:
        return 'medium'
    if v1 == 13 and v2 >= 9:
        return 'medium'
    if v1 == 13 and suited:
        return 'medium'
    if v1 == 12 and v2 >= 9:
        return 'medium'
    if v1 == 12 and suited:
        return 'medium'
    if v1 == 11 and v2 >= 9:
        return 'medium'
    if v1 == 11 and suited:
        return 'medium'
    if v1 == 10 and v2 >= 8:
        return 'medium'
    if suited and (v1 - v2) <= 2 and v1 >= 6:
        return 'medium'
    if v1 >= 10:
        return 'openable'
    if suited and (v1 - v2) <= 3:
        return 'openable'
    return 'trash'


class OpponentModel:
    '''Track simple aggregate stats on the opponent for exploit tilts.'''

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
        self.rounds = 0
        self.showdowns_won_at = []
        self.last_action_aggressive = False
        self.bet_fold_freq_num = 0
        self.bet_fold_freq_den = 0

    def prior_fold_rate(self):
        n = self.preflop_folds + self.preflop_calls + self.preflop_raises
        return (self.preflop_folds + 3) / (n + 6)

    def prior_preflop_raise(self):
        n = self.preflop_actions
        return (self.preflop_raises + 1) / (n + 4)

    def prior_postflop_bet(self):
        n = self.postflop_actions
        return (self.postflop_bets + self.postflop_raises + 2) / (n + 6)

    def prior_postflop_fold_to_bet(self):
        n = self.bet_fold_freq_den
        return (self.bet_fold_freq_num + 2) / (n + 5)


@dataclass(frozen=True)
class DecisionProfile:
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


class Player(Bot):
    def __init__(self):
        self.ranks_currV = ranks_currV
        self.model = load_model()
        random.seed(self.model['seed'])
        try:
            self.preflop_equity = _precompute_preflop_equity(samples_per_hand=self.model['preflop_samples'])
        except Exception:
            self.preflop_equity = {}
        try:
            self.range_combos = _build_range_combo_lists(self.preflop_equity)
        except Exception:
            self.range_combos = {1.00: _generate_all_combos()}
        self.opp = OpponentModel()
        self.round_history = []
        self.prev_opp_pip = 0
        self.prev_my_pip = 0
        self.last_street_seen = -1
        self.my_last_action = None
        self.bankroll_running = 0

    def time_samples(self, game_clock, street):
        scale = self.model['mc_scale']
        if game_clock < 2.5:
            return 0 if scale >= 1.0 else int(40 * scale)
        if game_clock < 6.0:
            return int(120 * scale)
        if game_clock < 15.0:
            return int(240 * scale)
        if game_clock < 30.0:
            return int(380 * scale)
        return int(520 * scale)

    def preflop_lookup(self, cards_str):
        key = canonical_hand_key(cards_str)
        return self.preflop_equity.get(key, 0.5)

    def estimate_opp_range_pct(self, opp_contribution, facing_bet, street):
        committed = opp_contribution / STARTING_STACK
        if not facing_bet:
            return 1.00
        elif committed >= 0.80:
            base = 0.05
        elif committed >= 0.40:
            base = 0.10
        elif committed >= 0.15:
            base = 0.20
        elif committed >= 0.06:
            base = 0.40
        else:
            base = 1.00

        if facing_bet:
            aggression = 0.45 * self.opp.prior_preflop_raise() + 0.55 * self.opp.prior_postflop_bet()
            if aggression >= 0.56:
                base *= 1.45
            elif aggression <= 0.38:
                base *= 0.70

            fold_rate = self.opp.prior_fold_rate()
            if fold_rate >= 0.55:
                base *= 0.85
            elif fold_rate <= 0.35:
                base *= 1.12

        base = max(0.05, min(1.00, base))
        buckets = (0.05, 0.10, 0.20, 0.40, 1.00)
        return min(buckets, key=lambda b: abs(b - base))

    def mc_equity(self, hero_str, board_str, samples, opp_contribution=0, facing_bet=False):
        hero = parse_cards(hero_str)
        board = parse_cards(board_str)
        if samples <= 0:
            if not board:
                return self.preflop_lookup(hero_str)
            return self._rough_made_strength(hero_str, board_str)
        range_pct = self.estimate_opp_range_pct(opp_contribution, facing_bet, len(board_str))
        combos = self.range_combos.get(range_pct) or self.range_combos.get(1.00)
        if combos is None:
            return fast_equity(hero, board, samples)
        return mc_equity_vs_range(hero, board, combos, samples)

    def _rough_made_strength(self, hero_str, board_str):
        hero_ranks = [c[0] for c in hero_str]
        board_ranks = [c[0] for c in board_str]
        if hero_ranks[0] == hero_ranks[1]:
            v = ranks_currV[hero_ranks[0]]
            return min(0.85, 0.55 + v * 0.015)
        top_board = max(ranks_currV[r] for r in board_ranks) if board_ranks else 0
        have_tp = any(ranks_currV[h] == top_board for h in hero_ranks)
        have_board_pair = len(set(board_ranks)) < len(board_ranks)
        if have_tp:
            return 0.6
        if have_board_pair:
            return 0.45
        high = max(ranks_currV[h] for h in hero_ranks)
        return 0.30 + 0.01 * high

    def bounty_expected_multiplier(self, my_bounty, hero_str, board_str, street):
        visible_hole_ranks = {c[0] for c in hero_str}
        visible_board_ranks = {c[0] for c in board_str}
        remaining = 5 - len(board_str)
        if my_bounty in visible_hole_ranks or my_bounty in visible_board_ranks:
            return BOUNTY_RATIO, float(bit_check)
        p_hit = bounty_hit_prob_future(my_bounty, hero_str, board_str, remaining)
        mult = 1.0 + (BOUNTY_RATIO - 1.0) * p_hit
        add = bit_check * p_hit
        return mult, add

    def opp_bounty_expected_multiplier(self, hero_str, board_str, street):
        visible_hole_ranks = {c[0] for c in hero_str}
        visible_board_ranks = {c[0] for c in board_str}
        remaining = 5 - len(board_str)
        total_p = 0.0
        for rank in target_mapping:
            if rank in visible_board_ranks:
                p_hit = 1.0
            else:
                cards_removed = 2 + len(board_str)
                rank_left = 4 - (1 if rank in visible_hole_ranks else 0)
                hidden = 52 - cards_removed
                opp_hole = 2
                future_board = remaining
                drawn = opp_hole + future_board
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
        mult = 1.0 + (BOUNTY_RATIO - 1.0) * avg_p_hit
        add = bit_check * avg_p_hit
        return mult, add

    def marginal_payoffs(self, pot, my_contrib, continue_cost, my_bm, my_bb, opp_bm, opp_bb):
        opp_contrib = pot - my_contrib
        win_marginal = opp_contrib * my_bm + my_bb + my_contrib
        lose_marginal = (my_contrib + continue_cost) * opp_bm + opp_bb - my_contrib
        return win_marginal, lose_marginal

    def required_equity(self, continue_cost, pot, opp_bm, opp_bb, my_bm, my_bb, my_contrib):
        win_marginal, lose_marginal = self.marginal_payoffs(
            pot, my_contrib, continue_cost, my_bm, my_bb, opp_bm, opp_bb
        )
        denom = win_marginal + lose_marginal
        if denom <= 0:
            return 1.0
        return max(0.0, lose_marginal / denom)

    def pick_raise_size(self, equity, pot, my_pip, opp_pip, my_stack, opp_stack, min_raise, max_raise, street, aggression_shift):
        if equity > 0.85:
            frac = 1.1
        elif equity > 0.72:
            frac = 0.8
        elif equity > 0.60:
            frac = 0.6
        else:
            frac = 0.5
        frac += aggression_shift
        frac = max(0.35, min(1.3, frac))
        target_total_bet = opp_pip + int(max(pot, 4) * frac)
        target_total_bet = max(target_total_bet, min_raise)
        target_total_bet = min(target_total_bet, max_raise)
        return target_total_bet

    def estimate_opponent_tightness(self, my_contribution, opp_contribution, continue_cost, street, facing_bet):
        if not facing_bet:
            return 0.01
        committed = opp_contribution / STARTING_STACK
        if street == 0:
            if committed >= 0.85:
                return 0.22
            if committed >= 0.40:
                return 0.14
            if committed >= 0.15:
                return 0.08
            if committed >= 0.06:
                return 0.05
            return 0.02
        else:
            if committed >= 0.80:
                return 0.18
            if committed >= 0.40:
                return 0.12
            if committed >= 0.20:
                return 0.08
            if committed >= 0.08:
                return 0.04
            return 0.02

    def match_mode(self, game_state):
        horizon = 500 if game_state.round_num <= 500 else 1000
        rounds_left = max(0, horizon - game_state.round_num)
        lead = game_state.bankroll
        if lead >= max(140, int(rounds_left * 0.36)):
            return 'protect'
        if lead <= -max(200, int(rounds_left * 0.42)):
            return 'chase'
        return 'neutral'

    def build_decision_profile(self, game_state, street):
        pre_raise = self.opp.prior_preflop_raise()
        if self.opp.preflop_actions < 12:
            passive_weight = 0.35
        else:
            if pre_raise <= 0.34:
                passive_weight = 0.95
            elif pre_raise >= 0.50:
                passive_weight = 0.0
            else:
                passive_weight = max(0.0, min(0.95, (0.50 - pre_raise) / 0.16))

        def blend(def_key, attack_key):
            base = self.model[def_key]
            attack = ATTACK_PROFILE[attack_key]
            return base + (attack - base) * passive_weight

        profile = DecisionProfile(
            preflop_call_margin=blend('preflop_call_margin', 'preflop_call_margin'),
            postflop_call_margin=blend('postflop_call_margin', 'postflop_call_margin'),
            value_raise_eq=blend('postflop_value_raise_eq', 'postflop_value_raise_eq'),
            semi_raise_eq_low=blend('semi_raise_eq_low', 'semi_raise_eq_low'),
            semi_raise_eq_high=blend('semi_raise_eq_high', 'semi_raise_eq_high'),
            semi_raise_fold_prior=blend('semi_raise_fold_prior', 'semi_raise_fold_prior'),
            river_probe_raise_prob=blend('river_probe_raise_prob', 'river_probe_raise_prob'),
            low_equity_bluff_prob=blend('low_equity_bluff_prob', 'low_equity_bluff_prob'),
            aggro_up_threshold=blend('aggro_up_threshold', 'aggro_up_threshold'),
            aggro_down_threshold=blend('aggro_down_threshold', 'aggro_down_threshold'),
            aggro_step=blend('aggro_step', 'aggro_step'),
        )

        mode = self.match_mode(game_state)
        if mode == 'protect':
            profile = DecisionProfile(
                preflop_call_margin=profile.preflop_call_margin + 0.012,
                postflop_call_margin=profile.postflop_call_margin + 0.016,
                value_raise_eq=min(0.92, profile.value_raise_eq + 0.014),
                semi_raise_eq_low=min(0.85, profile.semi_raise_eq_low + 0.014),
                semi_raise_eq_high=min(0.92, profile.semi_raise_eq_high + 0.01),
                semi_raise_fold_prior=min(0.90, profile.semi_raise_fold_prior + 0.02),
                river_probe_raise_prob=max(0.0, profile.river_probe_raise_prob * 0.80),
                low_equity_bluff_prob=max(0.0, profile.low_equity_bluff_prob * 0.70),
                aggro_up_threshold=profile.aggro_up_threshold + 0.02,
                aggro_down_threshold=min(0.65, profile.aggro_down_threshold + 0.02),
                aggro_step=profile.aggro_step * 0.90,
            )
        elif mode == 'chase':
            profile = DecisionProfile(
                preflop_call_margin=profile.preflop_call_margin - 0.003,
                postflop_call_margin=profile.postflop_call_margin - 0.003,
                value_raise_eq=max(0.55, profile.value_raise_eq - 0.003),
                semi_raise_eq_low=max(0.30, profile.semi_raise_eq_low - 0.006),
                semi_raise_eq_high=min(0.92, profile.semi_raise_eq_high + 0.006),
                semi_raise_fold_prior=max(0.25, profile.semi_raise_fold_prior - 0.006),
                river_probe_raise_prob=min(0.95, profile.river_probe_raise_prob * 1.02),
                low_equity_bluff_prob=min(0.50, profile.low_equity_bluff_prob * 1.03),
                aggro_up_threshold=max(0.30, profile.aggro_up_threshold - 0.01),
                aggro_down_threshold=max(0.08, profile.aggro_down_threshold - 0.01),
                aggro_step=min(0.30, profile.aggro_step * 1.02),
            )

        if street == 5:
            profile = DecisionProfile(
                preflop_call_margin=profile.preflop_call_margin,
                postflop_call_margin=profile.postflop_call_margin,
                value_raise_eq=max(0.55, profile.value_raise_eq - 0.01),
                semi_raise_eq_low=max(0.30, profile.semi_raise_eq_low - 0.01),
                semi_raise_eq_high=profile.semi_raise_eq_high,
                semi_raise_fold_prior=profile.semi_raise_fold_prior,
                river_probe_raise_prob=profile.river_probe_raise_prob,
                low_equity_bluff_prob=profile.low_equity_bluff_prob,
                aggro_up_threshold=profile.aggro_up_threshold,
                aggro_down_threshold=profile.aggro_down_threshold,
                aggro_step=profile.aggro_step,
            )

        return profile

    def handle_new_round(self, game_state, round_state, active):
        self.my_last_action = None
        self.last_street_seen = 0
        self.prev_my_pip = 0
        self.prev_opp_pip = 0
        self.opp.rounds += 1

    def handle_round_over(self, game_state, terminal_state, active):
        my_delta = terminal_state.deltas[active]
        self.bankroll_running += my_delta

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:street] if street > 0 else []
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]
        continue_cost = opp_pip - my_pip
        my_bounty = round_state.bounties[active]
        my_contribution = STARTING_STACK - my_stack
        opp_contribution = STARTING_STACK - opp_stack
        pot = my_contribution + opp_contribution
        game_clock = game_state.game_clock

        self._track_opponent(round_state, active, street)
        facing_bet = continue_cost > 0
        min_raise_total = max_raise_total = 0
        if RaiseAction in legal_actions:
            min_raise_total, max_raise_total = round_state.raise_bounds()

        if game_clock < 1.2:
            if CheckAction in legal_actions:
                return CheckAction()
            if continue_cost <= 2 and CallAction in legal_actions:
                return CallAction()
            if FoldAction in legal_actions:
                return FoldAction()
            return CallAction()

        samples = self.time_samples(game_clock, street)
        if street == 0:
            base_equity = self.preflop_lookup(my_cards)
        else:
            base_equity = self.mc_equity(
                my_cards, board_cards, samples,
                opp_contribution=opp_contribution,
                facing_bet=facing_bet,
            )

        my_bm, my_bb = self.bounty_expected_multiplier(my_bounty, my_cards, board_cards, street)
        opp_bm, opp_bb = self.opp_bounty_expected_multiplier(my_cards, board_cards, street)

        if street == 0:
            haircut = self.estimate_opponent_tightness(
                my_contribution, opp_contribution, continue_cost, street, facing_bet
            )
            haircut *= self.model['haircut_scale']
        else:
            haircut = 0.0

        profile = self.build_decision_profile(game_state, street)
        fold_prior = self.opp.prior_postflop_fold_to_bet()
        aggro_shift = 0.0
        if fold_prior > profile.aggro_up_threshold:
            aggro_shift += profile.aggro_step
        elif fold_prior < profile.aggro_down_threshold:
            aggro_shift -= profile.aggro_step

        bounty_bump = self.model['bounty_bump_scale'] * (0.02 * (my_bm - 1.0) + (my_bb / 450.0))

        if street == 0:
            hero_ranks = [c[0] for c in my_cards]
            has_bounty_rank = my_bounty in hero_ranks
            eq_raw = base_equity
            eq_vs_opp = max(0.05, eq_raw - haircut)
            eq_eff = eq_vs_opp + bounty_bump
            klass = classify_preflop(my_cards)
            opp_committed = opp_contribution / STARTING_STACK
            req = self.required_equity(continue_cost, pot, opp_bm, opp_bb, my_bm, my_bb, my_contribution)

            if not facing_bet:
                if RaiseAction in legal_actions:
                    if klass == 'premium':
                        target = opp_pip + max(6, int(pot * 1.6))
                    elif klass == 'strong':
                        target = opp_pip + max(5, int(pot * 1.2))
                    elif klass in ('medium', 'openable'):
                        target = opp_pip + max(4, int(pot * 0.9))
                    elif has_bounty_rank:
                        target = opp_pip + max(4, int(pot * 0.8))
                    else:
                        target = None
                    if target is not None:
                        return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                if CheckAction in legal_actions:
                    return CheckAction()
                if CallAction in legal_actions:
                    return CallAction()

            if opp_committed >= 0.65 or continue_cost >= my_stack * 0.55:
                if klass == 'premium' and CallAction in legal_actions:
                    return CallAction()
                if FoldAction in legal_actions:
                    return FoldAction()
                if CheckAction in legal_actions:
                    return CheckAction()
                return CallAction()

            if opp_pip >= 25 or opp_committed >= 0.12:
                if RaiseAction in legal_actions and klass == 'premium':
                    _, max_rr = round_state.raise_bounds()
                    jam_target = min(opp_pip + int((pot + continue_cost) * 2.2), max_rr)
                    return RaiseAction(max(jam_target, min_raise_total))
                if CallAction in legal_actions and klass in ('premium', 'strong'):
                    return CallAction()
                if CallAction in legal_actions and klass == 'medium' and continue_cost <= pot * 0.55:
                    return CallAction()
                if FoldAction in legal_actions:
                    return FoldAction()
                if CallAction in legal_actions:
                    return CallAction()
                return CheckAction() if CheckAction in legal_actions else CallAction()

            if opp_pip >= 5:
                if RaiseAction in legal_actions and klass == 'premium':
                    target = opp_pip + max(min_raise_total - opp_pip, int(continue_cost * 3.0))
                    return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                if RaiseAction in legal_actions and klass == 'strong':
                    target = opp_pip + max(min_raise_total - opp_pip, int(continue_cost * 2.6))
                    return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                if CallAction in legal_actions and klass in ('premium', 'strong', 'medium'):
                    return CallAction()
                if CallAction in legal_actions and has_bounty_rank and klass == 'openable' and continue_cost <= 8:
                    return CallAction()
                if FoldAction in legal_actions:
                    return FoldAction()
                if CheckAction in legal_actions:
                    return CheckAction()
                return CallAction()

            if continue_cost <= 2:
                if RaiseAction in legal_actions and klass == 'premium':
                    target = opp_pip + max(min_raise_total - opp_pip, 6)
                    return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                if CallAction in legal_actions and klass != 'trash':
                    return CallAction()
                if CallAction in legal_actions and has_bounty_rank:
                    return CallAction()
                if FoldAction in legal_actions:
                    return FoldAction()
                return CallAction() if CallAction in legal_actions else CheckAction()

            if CallAction in legal_actions and eq_eff >= req + profile.preflop_call_margin and klass != 'trash':
                return CallAction()
            if FoldAction in legal_actions:
                return FoldAction()
            if CheckAction in legal_actions:
                return CheckAction()
            return CallAction() if CallAction in legal_actions else CheckAction()

        eq_raw = base_equity
        eq_vs_opp = max(0.02, eq_raw - haircut)
        eq_eff = eq_vs_opp + bounty_bump
        req = self.required_equity(continue_cost, pot, opp_bm, opp_bb, my_bm, my_bb, my_contribution)
        opp_committed = opp_contribution / STARTING_STACK

        if not facing_bet:
            if RaiseAction in legal_actions and pot >= 4:
                if eq_raw >= 0.68:
                    target = self.pick_raise_size(eq_raw, pot, my_pip, opp_pip, my_stack, opp_stack, min_raise_total, max_raise_total, street, aggro_shift)
                    return RaiseAction(target)
                if eq_raw >= 0.52:
                    frac = 0.55 + aggro_shift
                    frac = max(0.4, min(0.8, frac))
                    target = opp_pip + max(min_raise_total - opp_pip, int(max(pot, 4) * frac))
                    return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                if eq_raw >= 0.38 and street == 3:
                    if random.random() < profile.river_probe_raise_prob:
                        frac = 0.45
                        target = opp_pip + max(min_raise_total - opp_pip, int(max(pot, 4) * frac))
                        return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                if eq_raw < 0.30 and pot >= 10 and random.random() < profile.low_equity_bluff_prob:
                    target = opp_pip + max(min_raise_total - opp_pip, int(max(pot, 4) * 0.45))
                    return RaiseAction(min(max(target, min_raise_total), max_raise_total))
            if CheckAction in legal_actions:
                return CheckAction()

        if RaiseAction in legal_actions and eq_eff >= profile.value_raise_eq and opp_committed < 0.55:
            target = self.pick_raise_size(eq_raw, pot + continue_cost, my_pip, opp_pip, my_stack, opp_stack, min_raise_total, max_raise_total, street, aggro_shift)
            return RaiseAction(target)

        if (RaiseAction in legal_actions and profile.semi_raise_eq_low <= eq_eff < profile.semi_raise_eq_high and fold_prior > profile.semi_raise_fold_prior and continue_cost <= pot * 0.6 and opp_committed < 0.35):
            target = opp_pip + max(min_raise_total - opp_pip, int((pot + continue_cost) * 0.75))
            return RaiseAction(min(max(target, min_raise_total), max_raise_total))

        pot_odds = continue_cost / max(1, (pot + continue_cost))
        has_pair_or_better = self._has_pair_or_better(my_cards, board_cards)
        big_bet = continue_cost >= pot * 0.55
        if CallAction in legal_actions:
            if eq_eff >= req + profile.postflop_call_margin:
                if big_bet and not has_pair_or_better and eq_raw < 0.58:
                    pass
                else:
                    return CallAction()
            if continue_cost <= max(3, pot * 0.15) and eq_raw >= 0.32:
                return CallAction()
            if street == 5 and fold_prior < 0.25 and eq_eff >= pot_odds and has_pair_or_better:
                return CallAction()

        if FoldAction in legal_actions:
            return FoldAction()
        if CheckAction in legal_actions:
            return CheckAction()
        return CallAction()

    def _has_pair_or_better(self, hero_str, board_str):
        if not board_str:
            return hero_str[0][0] == hero_str[1][0]
        hr = [c[0] for c in hero_str]
        br = [c[0] for c in board_str]
        if hr[0] == hr[1]:
            return True
        if hr[0] in br or hr[1] in br:
            return True
        hs = [c[1] for c in hero_str]
        bs = [c[1] for c in board_str]
        for suit in set(hs):
            if hs.count(suit) + bs.count(suit) >= 4:
                return True
        try:
            ranks = sorted(set(ranks_currV[r] for r in hr + br))
            for start in range(2, 11):
                window = set(range(start, start + 5))
                if len(window & set(ranks)) >= 4:
                    return True
        except Exception:
            pass
        return False

    def _track_opponent(self, round_state, active, street):
        opp_pip = round_state.pips[1 - active]
        my_pip = round_state.pips[active]
        if street != self.last_street_seen:
            self.last_street_seen = street
            self.prev_opp_pip = opp_pip
            self.prev_my_pip = my_pip
            return
        if opp_pip > self.prev_opp_pip:
            if street == 0:
                self.opp.preflop_actions += 1
                self.opp.preflop_raises += 1
            else:
                self.opp.postflop_actions += 1
                if self.prev_my_pip > self.prev_opp_pip:
                    self.opp.postflop_raises += 1
                else:
                    self.opp.postflop_bets += 1
            self.opp.last_action_aggressive = True
        self.prev_opp_pip = opp_pip
        self.prev_my_pip = my_pip


if __name__ == '__main__':
    run_bot(Player(), parse_args())
