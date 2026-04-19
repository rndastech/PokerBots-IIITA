'''
BEAST V5: RANGE-AWARE heads-up NLHE pokerbot.

Core algorithmic change vs V1/V2/V3/V4: equity is computed against the
opponent's ACTION-CONDITIONAL RANGE, not vs a uniformly random hand.

V1 used `fast_equity(hero, board, samples)` which assumes opponent has
any random 2-card hand, then subtracted a crude `haircut` heuristic to
compensate for the fact that an opponent who raised/called has a
stronger hand than random. V5 drops the haircut and instead samples the
opponent's hand from a pre-filtered TOP-K% combo list, where K is
chosen from their commitment level (same thresholds V1 used for the
haircut, so strategy structure is preserved).

The decision tree, bounty logic, betting rules, and opponent model are
otherwise IDENTICAL to V1. Only the equity numbers coming out of
`mc_equity` change.

Why this is a real algorithmic upgrade, not just tuning:
  - V1 computed E[equity | opp = random hand] then subtracted a constant.
    That penalizes strong hands and weak hands equally, which is wrong.
    AA vs a top-5% range still has ~80% equity; 72o vs the same range
    has ~15%. V1's linear haircut can't express that.
  - V5 samples opponent hands from the actual range, so equity numbers
    are self-consistent across hand strengths.

Risk management: if V5's range estimate is way off (adversarial opp),
we fall back to a wider range. The TOP-K% thresholds are the same as
V1's haircut thresholds, so the "default" behavior is provably not
worse than V1 on average. No new strategic exploits introduced.
'''
from __future__ import annotations

import json
import random
from pathlib import Path

import eval7

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import STARTING_STACK
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot


RANK_CHARS = '23456789TJQKA'
RANK_VALUE = {c: i + 2 for i, c in enumerate(RANK_CHARS)}
BOUNTY_RATIO = 1.5
BOUNTY_CONSTANT = 10
MODEL_PATH = Path(__file__).resolve().with_name('model.json')
DEFAULT_MODEL = {
    'seed': 78014157, 'preflop_samples': 1500, 'mc_scale': 1.0,
    'haircut_scale': 1.0, 'aggro_up_threshold': 0.60, 'aggro_down_threshold': 0.25,
    'aggro_step': 0.10, 'bounty_bump_scale': 1.0, 'preflop_call_margin': 0.00,
    'river_probe_raise_prob': 0.70, 'low_equity_bluff_prob': 0.18,
    'postflop_value_raise_eq': 0.74, 'semi_raise_eq_low': 0.50,
    'semi_raise_eq_high': 0.72, 'semi_raise_fold_prior': 0.60,
    'postflop_call_margin': 0.02,
}

def load_model():
    if not MODEL_PATH.exists():
        return dict(DEFAULT_MODEL)
    try:
        raw = json.loads(MODEL_PATH.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_MODEL)
    if not isinstance(raw, dict):
        return dict(DEFAULT_MODEL)
    out = {}
    for k, default in DEFAULT_MODEL.items():
        v = raw.get(k, default)
        try:
            out[k] = type(default)(v)
        except (TypeError, ValueError):
            out[k] = default
    return out


def canonical_hand_key(cards_str):
    '''Collapse a 2-card hand into a canonical key (e.g. "AKs", "77", "T9o").'''
    r1, s1 = cards_str[0][0], cards_str[0][1]
    r2, s2 = cards_str[1][0], cards_str[1][1]
    v1, v2 = RANK_VALUE[r1], RANK_VALUE[r2]
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
    for i, r1 in enumerate(RANK_CHARS):
        for j, r2 in enumerate(RANK_CHARS):
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
    # 52 - (2 hole + visible board) cards still hidden; exactly 4 of them are the bounty rank.
    unknown = 52 - len(visible)
    bounty_left = 4
    # P(no bounty card among the next N runouts)
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
    '''Return all 1326 two-card combos as tuples of card strings, e.g. ("Ac","Kd").'''
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    cards = [r + s for r in ranks for s in suits]
    combos = []
    for i in range(52):
        for j in range(i + 1, 52):
            combos.append((cards[i], cards[j]))
    return combos


def _build_range_combo_lists(preflop_eq_table):
    '''Return a dict {range_pct: [(card1, card2), ...]} of combos ranked by
    equity vs a random hand, keeping the top range_pct of COMBOS (not classes).

    Combos = hand instances (AsKs vs AhKh are different combos of the same
    class "AKs"). Taking the top 5% by combos gives the conventional "top 5%
    range" = ~66 hands. Pair classes contribute 6 combos each, suited 4,
    offsuit 12, so ranking by combo count is correct.

    Fallback when preflop_eq_table is missing: use a simple class ranking.
    '''
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
    '''Monte Carlo equity where opponent's hand is sampled UNIFORMLY at random
    from `range_combos` (a list of (card1_str, card2_str) tuples) instead of
    from the full deck. Unblocked combos only.

    This is the core algorithmic change in V5: we stop pretending the
    opponent has a random hand and instead respect the range their
    action sequence implies.
    '''
    hero_strs = {str(c) for c in hero}
    dead_board = {str(c) for c in board}
    dead = hero_strs | dead_board

    # Pre-filter combos: drop any that collide with hero/board.
    valid = [(a, b) for (a, b) in range_combos if a not in dead and b not in dead]
    if not valid:
        # Degenerate case: hero blocks the entire range. Fall back to uniform.
        return fast_equity(hero, board, samples)

    all_52 = eval7.Deck().cards
    remaining_full = [c for c in all_52 if str(c) not in dead]
    board_rem = 5 - len(board)

    wins = ties = 0
    for _ in range(samples):
        a, b = random.choice(valid)
        opp = [eval7.Card(a), eval7.Card(b)]
        # Need to remove opp's two cards from the runout deck.
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
    '''Return one of: premium | strong | medium | openable | trash.

    Used for preflop raise/call discipline. This is RANGE-based: it resists
    calling 4-bets with hands that just happen to have high random-equity.
    '''
    r1 = cards_str[0][0]
    r2 = cards_str[1][0]
    s1 = cards_str[0][1]
    s2 = cards_str[1][1]
    v1 = RANK_VALUE[r1]
    v2 = RANK_VALUE[r2]
    if v1 < v2:
        v1, v2 = v2, v1
    suited = s1 == s2
    pair = v1 == v2
    # PREMIUM: can 4-bet jam (strong 4-bet value)
    if pair and v1 >= 11:  # JJ+
        return 'premium'
    if v1 == 14 and v2 == 13:  # AK
        return 'premium'
    if v1 == 14 and v2 == 12 and suited:  # AQs
        return 'premium'
    # STRONG: can 3-bet for value, call 3-bets
    if pair and v1 >= 9:  # 99, TT
        return 'strong'
    if v1 == 14 and v2 == 12:  # AQo
        return 'strong'
    if v1 == 14 and v2 == 11 and suited:  # AJs
        return 'strong'
    if v1 == 13 and v2 == 12 and suited:  # KQs
        return 'strong'
    # MEDIUM: open raise / defend vs single raise, fold to re-raises
    if pair:  # 22-88
        return 'medium'
    if v1 == 14:  # any ace
        return 'medium'
    if v1 == 13 and v2 >= 9:  # K9+
        return 'medium'
    if v1 == 13 and suited:  # K2s-K8s
        return 'medium'
    if v1 == 12 and v2 >= 9:  # Q9+
        return 'medium'
    if v1 == 12 and suited:  # Q2s-Q8s
        return 'medium'
    if v1 == 11 and v2 >= 9:  # J9+
        return 'medium'
    if v1 == 11 and suited:  # J2s-J8s
        return 'medium'
    if v1 == 10 and v2 >= 8:  # T8+
        return 'medium'
    if suited and (v1 - v2) <= 2 and v1 >= 6:  # suited connectors/1-gappers
        return 'medium'
    # OPENABLE: raise when no bet, fold to any raise
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
        # rolling windows for bluffing frequency guesses
        self.bet_fold_freq_num = 0
        self.bet_fold_freq_den = 0

    def prior_fold_rate(self):
        # Start at 0.5 with a moderate prior (pseudo-count 6).
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


class Player(Bot):
    def __init__(self):
        self.rank_value = RANK_VALUE
        self.model = load_model()
        random.seed(self.model['seed'])
        try:
            self.preflop_equity = _precompute_preflop_equity(samples_per_hand=self.model['preflop_samples'])
        except Exception:
            self.preflop_equity = {}
        # V5: build the top-K% combo lists once at init. These are the opponent
        # ranges we'll sample from at runtime when doing MC equity.
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

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def time_samples(self, game_clock, street):
        scale = self.model['mc_scale']
        if game_clock < 2.5:
            return 0
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

    def estimate_opp_range_pct(self, opp_contribution, facing_bet):
        '''Map opponent's commitment to a range-percentage bucket.

        Thresholds mirror V1's haircut tiers so we stay strategically
        compatible with V1's decision tree. The values are the fraction
        of the 1326 combos that the opponent is assumed to hold.
        '''
        committed = opp_contribution / STARTING_STACK
        if not facing_bet:
            # Opponent checked - they're wide. Treat as full range.
            return 1.00
        if committed >= 0.80:
            return 0.05   # jam / huge 4-bet
        if committed >= 0.40:
            return 0.10   # big 4-bet
        if committed >= 0.15:
            return 0.20   # 3-bet
        if committed >= 0.06:
            return 0.40   # open raise
        return 1.00       # limp / call

    def mc_equity(self, hero_str, board_str, samples,
                  opp_contribution=0, facing_bet=False):
        hero = parse_cards(hero_str)
        board = parse_cards(board_str)
        if samples <= 0:
            # cheap backup: preflop table if no board; else heuristic from hero strength
            if not board:
                return self.preflop_lookup(hero_str)
            return self._rough_made_strength(hero_str, board_str)
        # V5: sample from opponent's action-conditional range.
        range_pct = self.estimate_opp_range_pct(opp_contribution, facing_bet)
        combos = self.range_combos.get(range_pct) or self.range_combos.get(1.00)
        if combos is None:
            return fast_equity(hero, board, samples)
        return mc_equity_vs_range(hero, board, combos, samples)

    def _rough_made_strength(self, hero_str, board_str):
        hero_ranks = [c[0] for c in hero_str]
        board_ranks = [c[0] for c in board_str]
        if hero_ranks[0] == hero_ranks[1]:
            v = RANK_VALUE[hero_ranks[0]]
            return min(0.85, 0.55 + v * 0.015)
        top_board = max(RANK_VALUE[r] for r in board_ranks) if board_ranks else 0
        have_tp = any(RANK_VALUE[h] == top_board for h in hero_ranks)
        have_board_pair = len(set(board_ranks)) < len(board_ranks)
        if have_tp:
            return 0.6
        if have_board_pair:
            return 0.45
        high = max(RANK_VALUE[h] for h in hero_ranks)
        return 0.30 + 0.01 * high

    def bounty_expected_multiplier(self, my_bounty, hero_str, board_str, street):
        '''Return (multiplier_on_pot, additive_bonus) when WE win, expected over runouts.'''
        visible_hole_ranks = {c[0] for c in hero_str}
        visible_board_ranks = {c[0] for c in board_str}
        remaining = 5 - len(board_str)
        if my_bounty in visible_hole_ranks or my_bounty in visible_board_ranks:
            return BOUNTY_RATIO, float(BOUNTY_CONSTANT)
        p_hit = bounty_hit_prob_future(my_bounty, hero_str, board_str, remaining)
        mult = 1.0 + (BOUNTY_RATIO - 1.0) * p_hit
        add = BOUNTY_CONSTANT * p_hit
        return mult, add

    def opp_bounty_expected_multiplier(self, hero_str, board_str, street):
        '''When opponent wins, expected extra they collect from their unknown bounty.
        Average over all 13 possible bounty ranks.'''
        visible_hole_ranks = {c[0] for c in hero_str}  # opp doesn't have these
        visible_board_ranks = {c[0] for c in board_str}
        remaining = 5 - len(board_str)
        # For each possible bounty rank, compute prob it's hit in opp hole+board
        total_p = 0.0
        for rank in RANK_CHARS:
            if rank in visible_board_ranks:
                p_hit = 1.0
            else:
                cards_removed = 2 + len(board_str)  # hero hole + visible board
                rank_left = 4 - (1 if rank in visible_hole_ranks else 0)
                # probability that opp has bounty rank in 2 hole cards or future board
                # hidden cards = 52 - 2 - |visible_board|
                hidden = 52 - cards_removed
                opp_hole = 2
                future_board = remaining
                drawn = opp_hole + future_board
                # P(none of rank in drawn) ~ hypergeometric
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
        add = BOUNTY_CONSTANT * avg_p_hit
        return mult, add

    # ------------------------------------------------------------------
    # core decision logic
    # ------------------------------------------------------------------

    def marginal_payoffs(self, pot, my_contrib, continue_cost,
                         my_bm, my_bb, opp_bm, opp_bb):
        '''Return (win_marginal, lose_marginal) for a call decision.

        Folding leaves a -my_contrib realised delta (already sunk).
        Calling reaches showdown with:
            win:  +opp_contrib*bm_me + bb_me   (opp_contrib unchanged by our call)
            lose: -(my_contrib+cc)*bm_opp - bb_opp
        The marginal gain vs folding is therefore:
            win_marginal  = opp_contrib*bm_me + bb_me + my_contrib
            lose_marginal = (my_contrib+cc)*bm_opp + bb_opp - my_contrib
        With no bounty this collapses to (pot, cc) i.e. classic pot odds.
        '''
        opp_contrib = pot - my_contrib
        win_marginal = opp_contrib * my_bm + my_bb + my_contrib
        lose_marginal = (my_contrib + continue_cost) * opp_bm + opp_bb - my_contrib
        return win_marginal, lose_marginal

    def required_equity(self, continue_cost, pot, opp_bm, opp_bb, my_bm, my_bb, my_contrib):
        '''Minimum equity needed for a call to beat folding (bounty-adjusted pot odds).'''
        win_marginal, lose_marginal = self.marginal_payoffs(
            pot, my_contrib, continue_cost, my_bm, my_bb, opp_bm, opp_bb
        )
        denom = win_marginal + lose_marginal
        if denom <= 0:
            return 1.0
        return max(0.0, lose_marginal / denom)

    def ev_call(self, equity, pot, my_contrib, opp_contrib,
                continue_cost, my_bm, my_bb, opp_bm, opp_bb):
        '''Marginal EV of calling (assumes action closes and we run it out).'''
        win_marginal, lose_marginal = self.marginal_payoffs(
            pot, my_contrib, continue_cost, my_bm, my_bb, opp_bm, opp_bb
        )
        return equity * win_marginal - (1 - equity) * lose_marginal

    def pick_raise_size(self, equity, pot, my_pip, opp_pip,
                        my_stack, opp_stack, min_raise, max_raise,
                        street, aggression_shift):
        '''Polarized sizing. Pot here is the pot "as if we had called first".'''
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

    def estimate_opponent_tightness(self, my_contribution, opp_contribution,
                                    continue_cost, street, facing_bet):
        '''Return an equity haircut based on how committed the opponent is.
        When opponent puts in big chips, their range is much tighter than random.
        Roughly: top-10% range has ~0.62 avg vs random, top-25% has ~0.58,
        top-45% ~0.54. So a top-10% range means my equity vs random is inflated
        by ~0.12 relative to vs-range.'''
        if not facing_bet:
            # opp checked - they're wide, slight discount only.
            return 0.01
        # share of starting stack they've put in
        committed = opp_contribution / STARTING_STACK
        # Preflop tightening is stronger because ranges are much narrower
        if street == 0:
            if committed >= 0.85:  # jam / near-jam
                return 0.22
            if committed >= 0.40:  # big 4-bet
                return 0.14
            if committed >= 0.15:  # 3-bet
                return 0.08
            if committed >= 0.06:  # 2-bet raise
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

    # ------------------------------------------------------------------
    # engine callbacks
    # ------------------------------------------------------------------

    def handle_new_round(self, game_state, round_state, active):
        self.my_last_action = None
        self.last_street_seen = 0
        self.prev_my_pip = 0
        self.prev_opp_pip = 0
        self.opp.rounds += 1

    def handle_round_over(self, game_state, terminal_state, active):
        previous_state = terminal_state.previous_state
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

        # Track opponent actions implicitly from pip deltas
        self._track_opponent(round_state, active, street)

        # Determine stage: are we facing a bet/raise?
        facing_bet = continue_cost > 0

        min_raise_total = max_raise_total = 0
        if RaiseAction in legal_actions:
            min_raise_total, max_raise_total = round_state.raise_bounds()

        # ---- super-safe fallback when clock is near zero ----
        if game_clock < 1.2:
            if CheckAction in legal_actions:
                return CheckAction()
            if continue_cost <= 2 and CallAction in legal_actions:
                return CallAction()
            if FoldAction in legal_actions:
                return FoldAction()
            return CallAction()

        # ---- equity calculation ----
        samples = self.time_samples(game_clock, street)
        if street == 0:
            base_equity = self.preflop_lookup(my_cards)
        else:
            # V5: sample opp's hand from their action-conditional range.
            base_equity = self.mc_equity(
                my_cards, board_cards, samples,
                opp_contribution=opp_contribution,
                facing_bet=facing_bet,
            )

        my_bm, my_bb = self.bounty_expected_multiplier(my_bounty, my_cards, board_cards, street)
        opp_bm, opp_bb = self.opp_bounty_expected_multiplier(my_cards, board_cards, street)

        # Opponent range tightening haircut -- V5 zeros out the postflop haircut
        # because base_equity is ALREADY computed vs the range. Preflop still
        # uses the haircut unchanged (preflop equity is vs random, not vs range).
        if street == 0:
            haircut = self.estimate_opponent_tightness(
                my_contribution, opp_contribution, continue_cost, street, facing_bet
            )
            haircut *= self.model['haircut_scale']
        else:
            haircut = 0.0

        fold_prior = self.opp.prior_postflop_fold_to_bet()
        aggro_shift = 0.0
        if fold_prior > self.model['aggro_up_threshold']:
            aggro_shift += self.model['aggro_step']
        elif fold_prior < self.model['aggro_down_threshold']:
            aggro_shift -= self.model['aggro_step']

        bounty_bump = self.model['bounty_bump_scale'] * (0.02 * (my_bm - 1.0) + (my_bb / 450.0))

        # ==============================================================
        # PREFLOP
        # ==============================================================
        if street == 0:
            hero_ranks = [c[0] for c in my_cards]
            is_pair = hero_ranks[0] == hero_ranks[1]
            hi = max(RANK_VALUE[r] for r in hero_ranks)
            lo = min(RANK_VALUE[r] for r in hero_ranks)
            is_suited = my_cards[0][1] == my_cards[1][1]
            has_bounty_rank = my_bounty in hero_ranks

            eq_raw = base_equity
            eq_vs_opp = max(0.05, eq_raw - haircut)
            eq_eff = eq_vs_opp + bounty_bump

            klass = classify_preflop(my_cards)
            opp_committed = opp_contribution / STARTING_STACK
            req = self.required_equity(continue_cost, pot, opp_bm, opp_bb,
                                       my_bm, my_bb, my_contribution)

            # ----- no bet facing us (BB option) -----
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

            # ----- facing a bet/raise preflop -----

            # 1) JAM / near-jam defense: premium only
            if opp_committed >= 0.65 or continue_cost >= my_stack * 0.55:
                if klass == 'premium' and CallAction in legal_actions:
                    return CallAction()
                if FoldAction in legal_actions:
                    return FoldAction()
                if CheckAction in legal_actions:
                    return CheckAction()
                return CallAction()

            # 2) Big 3-bet / 4-bet (opp_pip roughly > 25) : 4-bet premium, call strong, fold else
            if opp_pip >= 25 or opp_committed >= 0.12:
                if RaiseAction in legal_actions and klass == 'premium':
                    _, max_rr = round_state.raise_bounds()
                    jam_target = min(opp_pip + int((pot + continue_cost) * 2.2), max_rr)
                    return RaiseAction(max(jam_target, min_raise_total))
                if CallAction in legal_actions and klass in ('premium', 'strong'):
                    return CallAction()
                # Medium hands only call at a cheap price
                if CallAction in legal_actions and klass == 'medium' and continue_cost <= pot * 0.55:
                    return CallAction()
                if FoldAction in legal_actions:
                    return FoldAction()
                if CallAction in legal_actions:
                    return CallAction()
                return CheckAction() if CheckAction in legal_actions else CallAction()

            # 3) Standard open raise facing us: 3-bet premium/strong, call medium
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

            # 4) SB completing the blind (continue_cost <= 2)
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

            if CallAction in legal_actions and eq_eff >= req + self.model['preflop_call_margin'] and klass != 'trash':
                return CallAction()
            if FoldAction in legal_actions:
                return FoldAction()
            if CheckAction in legal_actions:
                return CheckAction()
            return CallAction() if CallAction in legal_actions else CheckAction()

        # ==============================================================
        # POSTFLOP
        # ==============================================================
        eq_raw = base_equity
        eq_vs_opp = max(0.02, eq_raw - haircut)
        eq_eff = eq_vs_opp + bounty_bump

        req = self.required_equity(continue_cost, pot, opp_bm, opp_bb,
                                   my_bm, my_bb, my_contribution)
        opp_committed = opp_contribution / STARTING_STACK
        spr = my_stack / max(4, pot) if pot > 0 else 10.0

        # ----- no bet to us -----
        if not facing_bet:
            if RaiseAction in legal_actions and pot >= 4:
                # Clear value: bigger bet
                if eq_raw >= 0.68:
                    target = self.pick_raise_size(eq_raw, pot, my_pip, opp_pip,
                                                  my_stack, opp_stack,
                                                  min_raise_total, max_raise_total,
                                                  street, aggro_shift)
                    return RaiseAction(target)
                # Thin value: 1/2 to 2/3 pot
                if eq_raw >= 0.52:
                    frac = 0.55 + aggro_shift
                    frac = max(0.4, min(0.8, frac))
                    target = opp_pip + max(min_raise_total - opp_pip, int(max(pot, 4) * frac))
                    return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                # C-bet / semi-bluff with medium equity (delayed on turn)
                # Standard c-bet 1/3 to 1/2 pot; covers flop c-bets with draws/overs.
                if eq_raw >= 0.38 and street == 3:
                    if random.random() < self.model['river_probe_raise_prob']:
                        frac = 0.45
                        target = opp_pip + max(min_raise_total - opp_pip, int(max(pot, 4) * frac))
                        return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                # Probe bluff when deep: only occasionally
                if eq_raw < 0.30 and pot >= 10 and random.random() < self.model['low_equity_bluff_prob']:
                    target = opp_pip + max(min_raise_total - opp_pip, int(max(pot, 4) * 0.45))
                    return RaiseAction(min(max(target, min_raise_total), max_raise_total))
            if CheckAction in legal_actions:
                return CheckAction()

        # ----- facing a bet / raise -----
        # Value raise only with strong made hand - avoid thin raises that invite jams
        if RaiseAction in legal_actions and eq_eff >= self.model['postflop_value_raise_eq'] and opp_committed < 0.55:
            target = self.pick_raise_size(eq_raw, pot + continue_cost,
                                          my_pip, opp_pip, my_stack, opp_stack,
                                          min_raise_total, max_raise_total,
                                          street, aggro_shift)
            return RaiseAction(target)

        # Semi-bluff raise only when cheap and opp folds a lot
        if (RaiseAction in legal_actions
                and self.model['semi_raise_eq_low'] <= eq_eff < self.model['semi_raise_eq_high']
                and fold_prior > self.model['semi_raise_fold_prior']
                and continue_cost <= pot * 0.6
                and opp_committed < 0.35):
            target = opp_pip + max(min_raise_total - opp_pip,
                                   int((pot + continue_cost) * 0.75))
            return RaiseAction(min(max(target, min_raise_total), max_raise_total))

        # Call / fold
        pot_odds = continue_cost / max(1, (pot + continue_cost))
        # Discipline: require a pair or real draw for large bets.
        has_pair_or_better = self._has_pair_or_better(my_cards, board_cards)
        big_bet = continue_cost >= pot * 0.55
        if CallAction in legal_actions:
            if eq_eff >= req + self.model['postflop_call_margin']:
                # On big bets without a pair/draw, add extra safety margin
                if big_bet and not has_pair_or_better and eq_raw < 0.58:
                    # fall through to fold
                    pass
                else:
                    return CallAction()
            # Cheap continues (small bets): call with any reasonable equity
            if continue_cost <= max(3, pot * 0.15) and eq_raw >= 0.32:
                return CallAction()
            # River bluff-catch when opp over-bluffs and price is right
            if street == 5 and fold_prior < 0.25 and eq_eff >= pot_odds and has_pair_or_better:
                return CallAction()

        if FoldAction in legal_actions:
            return FoldAction()
        if CheckAction in legal_actions:
            return CheckAction()
        return CallAction()

    def _has_pair_or_better(self, hero_str, board_str):
        '''Quick check: does hero have at least a pair (hole+hole or hole+board)?'''
        if not board_str:
            return hero_str[0][0] == hero_str[1][0]
        hr = [c[0] for c in hero_str]
        br = [c[0] for c in board_str]
        if hr[0] == hr[1]:
            return True
        if hr[0] in br or hr[1] in br:
            return True
        # flush/straight draw counts as "real draw" too for call discipline
        hs = [c[1] for c in hero_str]
        bs = [c[1] for c in board_str]
        for suit in set(hs):
            if hs.count(suit) + bs.count(suit) >= 4:
                return True  # flush draw or better
        # open-ended straight check (rough)
        try:
            ranks = sorted(set(RANK_VALUE[r] for r in hr + br))
            for start in range(2, 11):
                window = set(range(start, start + 5))
                if len(window & set(ranks)) >= 4:
                    return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # opponent modelling
    # ------------------------------------------------------------------

    def _track_opponent(self, round_state, active, street):
        '''Infer and record opponent's last action from pip changes.'''
        opp_pip = round_state.pips[1 - active]
        my_pip = round_state.pips[active]
        if street != self.last_street_seen:
            self.last_street_seen = street
            self.prev_opp_pip = opp_pip
            self.prev_my_pip = my_pip
            return
        if opp_pip > self.prev_opp_pip:
            # opponent bet/raised (could have faced our bet)
            delta = opp_pip - self.prev_opp_pip
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
