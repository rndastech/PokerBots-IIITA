'''
BEAST v6.0 — Bounty-Aware Exploit Superbot
============================================
Upgrades over v4:
  U1  Nash Push/Fold lookup table (rounds <= 15BB — mathematically unexploitable)
  U2  Early-game blitz window (rounds 1-25): hyper-bluff since all profiling bots
      are passive/balanced until they have enough sample data
  U3  Anti-regime-shift camouflage: rotates strategy style every ~20 hands so
      windowed opponent models (BackPropagators) never lock in our profile
  U4  4-bet shove range expansion: TT, AQ, KQs added — opponents with tight
      all-in call tiers (IEC2025002, IIT2023504) fold these hands
  U5  Polarized-only sizing (35% / 65%) — avoids small_bet_ratio > 0.4 which
      triggers "exploit_bot" counter-mode in IIT2023504
  U6  Opponent "unknown guard" exploit: fire bluffs in rounds 1-30 because
      IEC2025002 disables its bluffs until they have 30 hands of data
  U7  Regime-shift evasion: tighten for 3-4 hands every 20 hands
  U8  Improved opponent stat tracking with fold-to-3bet and VPIP windows
  U9  Range-weighted MC equity — filters opp hole cards by preflop range percentile
      giving far more accurate equity estimates vs NITs and TAGs
  U10 MDF (Minimum Defense Frequency) guard — never fold-too-much to small bets
  U11 Turn-specific tightening — eliminates the -4253 turn EV leak by requiring
      stronger equity before betting/calling on turn
  U12 Dynamic bluff frequency using measured opp fold rate instead of fixed 28%
  U13 Position awareness — tighter OOP, wider IP on all streets
  U14 River blocker-aware hero calls — hold pair/draw when opp over-bets river
'''
from __future__ import annotations

import random
import time

import eval7

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import STARTING_STACK
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

try:
    from skeleton.states import NUM_ROUNDS
except Exception:
    NUM_ROUNDS = 1000

BIG_BLIND = 2
RANK_CHARS = '23456789TJQKA'
RANK_VALUE = {c: i + 2 for i, c in enumerate(RANK_CHARS)}
BOUNTY_RATIO = 1.5
BOUNTY_CONSTANT = 10

# ─── U1: Nash Push/Fold Lookup Table ──────────────────────────────────────────
# (eff_bb_cutoff, push_threshold, call_threshold)
# push/call thresholds are preflop equity vs random hand
_NASH = [
    (15, 0.32, 0.50),
    (12, 0.25, 0.46),
    (10, 0.20, 0.42),
    ( 8, 0.12, 0.38),
    ( 6, 0.04, 0.33),
    ( 4, 0.25, 0.20),   
    ( 2, 0.34, 0.30),   # require ~33%+ equity vs APEX 100% callers
]

def nash_thresholds(eff_bb):
    for cutoff, push_t, call_t in _NASH:
        if eff_bb >= cutoff:
            return push_t, call_t
    return 0.0, 0.20


# ─── Hand Utilities ───────────────────────────────────────────────────────────
def canonical_hand_key(cards_str):
    r1, s1 = cards_str[0][0], cards_str[0][1]
    r2, s2 = cards_str[1][0], cards_str[1][1]
    v1, v2 = RANK_VALUE[r1], RANK_VALUE[r2]
    if v1 < v2:
        r1, r2, s1, s2, v1, v2 = r2, r1, s2, s1, v2, v1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ('s' if s1 == s2 else 'o')


def build_hand_from_key(key):
    r1 = key[0]; r2 = key[1]
    if len(key) == 2:
        return [eval7.Card(r1 + 's'), eval7.Card(r2 + 'h')]
    if key[2] == 's':
        return [eval7.Card(r1 + 's'), eval7.Card(r2 + 's')]
    return [eval7.Card(r1 + 's'), eval7.Card(r2 + 'h')]


def _precompute_preflop_equity(samples_per_hand=600):
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


def fast_equity(hero, board, samples, opp_range_min=0.0, pf_table=None):
    '''U9: Range-weighted MC — filters opp hands by preflop equity threshold.
    opp_range_min=0.0 → random (vs unknown); 0.22 → TAG; 0.52 → NIT.
    '''
    dead = {str(c) for c in hero}
    for c in board:
        dead.add(str(c))
    remaining = [c for c in eval7.Deck().cards if str(c) not in dead]
    board_rem = 5 - len(board)
    wins = ties = total = 0
    t0 = time.monotonic()
    attempts = 0
    limit = max(samples * 12, 1500)
    while total < samples and attempts < limit:
        # clock-guard: abort MC if we've spent >45ms and have enough samples
        if attempts > 0 and attempts % 50 == 0:
            if (time.monotonic() - t0) * 1000 > 45 and total >= 80:
                break
        attempts += 1
        random.shuffle(remaining)
        opp = remaining[:2]
        # U9: range filter — skip opp hands outside their range
        if opp_range_min > 0.02 and pf_table is not None:
            try:
                r1, s1 = opp[0].rank, opp[0].suit
                r2, s2 = opp[1].rank, opp[1].suit
                hi, lo = max(r1, r2), min(r1, r2)
                skey = 's' if s1 == s2 else 'o'
                rchars = '23456789TJQKA'
                key = (rchars[hi - 2] + rchars[lo - 2] + ('' if hi == lo else skey))
                pf_eq = pf_table.get(key, 0.5)
                if pf_eq < opp_range_min and random.random() > 0.15:
                    continue   # reject out-of-range hand (allow 15% to prevent bias)
            except Exception:
                pass
        run = remaining[2:2 + board_rem]
        full_board = board + run if board_rem else board
        s1 = eval7.evaluate(hero + full_board)
        s2 = eval7.evaluate(opp + full_board)
        if s1 > s2:
            wins += 1
        elif s1 == s2:
            ties += 1
        total += 1
    return (wins + 0.5 * ties) / max(1, total)


def parse_cards(strs):
    return [eval7.Card(s) for s in strs]


# ─── U4: Extended Preflop Classification ─────────────────────────────────────
def classify_preflop(cards_str):
    '''Returns: premium | strong | medium | openable | trash'''
    r1, s1 = cards_str[0][0], cards_str[0][1]
    r2, s2 = cards_str[1][0], cards_str[1][1]
    v1, v2 = RANK_VALUE[r1], RANK_VALUE[r2]
    if v1 < v2:
        v1, v2, r1, r2, s1, s2 = v2, v1, r2, r1, s2, s1
    suited = s1 == s2
    pair = v1 == v2
    # PREMIUM: 4-bet jam candidates
    if pair and v1 >= 11:                                   # JJ+
        return 'premium'
    if v1 == 14 and v2 == 13:                               # AK
        return 'premium'
    if v1 == 14 and v2 == 12 and suited:                   # AQs
        return 'premium'
    # U4: expanded 4-bet shove range — tight bots fold these
    if pair and v1 == 10:                                   # TT (new)
        return 'premium'
    if v1 == 14 and v2 == 12:                               # AQo (now premium)
        return 'premium'
    if v1 == 13 and v2 == 12 and suited:                   # KQs (new)
        return 'premium'
    # STRONG
    if pair and v1 >= 9:
        return 'strong'
    if v1 == 14 and v2 == 11 and suited:
        return 'strong'
    if v1 == 14 and v2 >= 10:
        return 'strong'
    if v1 == 13 and v2 >= 11:
        return 'strong'
    if v1 == 13 and v2 >= 10 and suited:
        return 'strong'
    # MEDIUM
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
    # OPENABLE
    if v1 >= 10:
        return 'openable'
    if suited and (v1 - v2) <= 3:
        return 'openable'
    return 'trash'


# ─── Opponent Model ───────────────────────────────────────────────────────────
class OpponentModel:
    '''
    Windowed opponent model for regime-shift detection (U7).
    Tracks per-hand records in a sliding window of 50 hands.
    '''
    def __init__(self):
        self.rounds = 0
        self.preflop_actions = 0
        self.preflop_raises = 0
        self.preflop_calls = 0
        self.preflop_folds = 0
        self.postflop_actions = 0
        self.postflop_bets = 0
        self.postflop_raises = 0
        self.postflop_calls = 0
        self.postflop_checks = 0
        self.postflop_folds = 0
        self.vpip_rounds = 0
        self.showdowns_won_at = []
        # Fold-to-bet tracking
        self.bet_fold_freq_num = 0
        self.bet_fold_freq_den = 0
        self.last_action_aggressive = False
        # Windowed history: list of hand dicts, max 100
        self._history = []
        self._cur = {}
        self.is_baseline = None
        self._pf_raise_counts = 0   # total PF raises ever
        self._hands_seen = 0

    def new_hand(self):
        if self._cur:
            self._history.append(self._cur)
            if len(self._history) > 100:
                self._history.pop(0)
        self._cur = {'vpip': False, 'pfr': False, 'pb': 0, 'pcl': 0, 'pf': 0}
        self._hands_seen += 1

    def record_pf_raise(self):
        self._pf_raise_counts += 1
        self._cur['pfr'] = True
        self._cur['vpip'] = True

    def record_pf_call(self):
        self._cur['vpip'] = True

    def record_pf_fold(self):
        pass

    def record_postflop_bet(self):
        self._cur['pb'] = self._cur.get('pb', 0) + 1

    def record_postflop_call(self):
        self._cur['pcl'] = self._cur.get('pcl', 0) + 1

    def record_postflop_fold(self):
        self._cur['pf'] = self._cur.get('pf', 0) + 1

    def _window_stats(self, window=50):
        recs = self._history[-window:]
        n = len(recs)
        if n == 0:
            return {'vpip': 0.5, 'pfr': 0.30, 'af': 1.0, 'n': 0}
        vpip = sum(1 for r in recs if r.get('vpip')) / n
        pfr = sum(1 for r in recs if r.get('pfr')) / n
        tb = sum(r.get('pb', 0) for r in recs)
        tc = sum(r.get('pcl', 0) for r in recs)
        af = tb / max(1, tc)
        return {'vpip': vpip, 'pfr': pfr, 'af': af, 'n': n}

    # U7: detect if opponent changed strategy (regime shift)
    def detect_regime_shift(self):
        if len(self._history) < 25:
            return False
        rec12 = self._window_stats(12)
        rec_all = self._window_stats()
        return (abs(rec12['vpip'] - rec_all['vpip']) > 0.20 or
                abs(rec12['pfr'] - rec_all['pfr']) > 0.15 or
                abs(rec12['af'] - rec_all['af']) > 0.20)

    # Classification properties using blended windows
    def _stats(self):
        w50 = self._window_stats(50)
        w_all = self._window_stats()
        n = w50['n']
        if n < 8:
            return w_all
        # regime shift → down-weight history
        if self.detect_regime_shift():
            w12 = self._window_stats(12)
            return {k: 0.70 * w12.get(k, 0.5) + 0.30 * w50.get(k, 0.5)
                    for k in ('vpip', 'pfr', 'af')}
        return {k: 0.50 * w50.get(k, v) + 0.50 * w_all.get(k, v)
                for k, v in {'vpip': 0.5, 'pfr': 0.3, 'af': 1.0}.items()}

    @property
    def vpip(self):       return self._stats().get('vpip', 0.50)
    @property
    def pfr(self):        return self._stats().get('pfr', 0.30)
    @property
    def af(self):         return self._stats().get('af', 1.0)

    @property
    def is_nit(self):     return self.vpip < 0.38 and self.pfr < 0.28
    @property
    def is_station(self): return self.vpip > 0.72 and self.pfr < 0.20
    @property
    def is_maniac(self):  return self.pfr > 0.68 or (self.af > 4.5 and self.pfr > 0.58)
    @property
    def is_lag(self):     return self.vpip >= 0.70 and self.pfr >= 0.48
    @property
    def is_tag(self):     return 0.38 <= self.vpip < 0.72 and 0.28 <= self.pfr < 0.62

    def profile(self):
        if self._hands_seen < 8:    return 'UNKNOWN'
        if self.is_maniac:          return 'MANIAC'
        if self.is_nit:             return 'NIT'
        if self.is_station:         return 'STATION'
        if self.is_lag:             return 'LAG'
        if self.is_tag:             return 'TAG'
        return 'UNKNOWN'

    def check_baseline(self):
        # Baseline check is now handled perfectly inside _track_opponent via bet sizing fingerprints
        pass

    def prior_fold_rate(self):
        n = self.preflop_folds + self.preflop_calls + self.preflop_raises
        return (self.preflop_folds + 3) / (n + 6)

    def prior_postflop_bet(self):
        n = self.postflop_actions
        return (self.postflop_bets + self.postflop_raises + 2) / (n + 6)

    def prior_postflop_fold_to_bet(self):
        n = self.bet_fold_freq_den
        return (self.bet_fold_freq_num + 2) / (n + 5)

    def _range_min(self):
        '''U9: Return the minimum preflop equity threshold for opponent range.
        Used to filter Monte Carlo samples — only simulate hands they would play.
        Returns 0.0 (random) until we have enough data.
        '''
        if self._hands_seen < 12:
            return 0.0   # not enough data — assume random
        if self.is_nit:     return 0.50   # NIT plays top ~30% → eq ≥ 0.50
        if self.is_tag:     return 0.22   # TAG plays ~50% → eq ≥ 0.22
        if self.is_station: return 0.05   # Station plays almost any two
        if self.is_maniac:  return 0.00   # Maniac = random
        if self.is_lag:     return 0.12   # LAG plays ~70%
        return 0.0


# ─── Main Player ──────────────────────────────────────────────────────────────
class Player(Bot):
    def __init__(self):
        random.seed(0xBEA57ACE)
        try:
            self.preflop_equity = _precompute_preflop_equity(samples_per_hand=600)
        except Exception:
            self.preflop_equity = {}
        self.opp = OpponentModel()
        self.round_count = 0          # total rounds played
        self.bankroll_running = 0
        self.opp_folds = 0
        self.bluff_opportunities = 0
        # Per-hand state
        self.prev_opp_pip = 0
        self.prev_my_pip = 0
        self.last_street_seen = -1
        self.my_last_action = None
        # U3: anti-regime camouflage style counter
        self._style_phase = 0         # 0=aggressive, 1=tight(camouflage)
        self._style_hand_counter = 0

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def time_samples(self, game_clock, street):
        if game_clock < 2.5:  return 0
        if game_clock < 6.0:  return 100
        if game_clock < 15.0: return 240
        if game_clock < 30.0: return 400
        return 550

    def preflop_lookup(self, cards_str):
        key = canonical_hand_key(cards_str)
        return self.preflop_equity.get(key, 0.5)

    def mc_equity(self, hero_str, board_str, samples, opp_range_min=0.0):
        hero = parse_cards(hero_str)
        board = parse_cards(board_str)
        if samples <= 0:
            if not board:
                return self.preflop_lookup(hero_str)
            return self._rough_made_strength(hero_str, board_str)
        # U9: pass range filter and preflop table to fast_equity
        return fast_equity(hero, board, samples,
                           opp_range_min=opp_range_min,
                           pf_table=self.preflop_equity)

    def _rough_made_strength(self, hero_str, board_str):
        hero_ranks = [c[0] for c in hero_str]
        board_ranks = [c[0] for c in board_str]
        if hero_ranks[0] == hero_ranks[1]:
            v = RANK_VALUE[hero_ranks[0]]
            return min(0.85, 0.55 + v * 0.015)
        top_board = max(RANK_VALUE[r] for r in board_ranks) if board_ranks else 0
        have_tp = any(RANK_VALUE[h] == top_board for h in hero_ranks)
        if have_tp:
            return 0.62
        high = max(RANK_VALUE[h] for h in hero_ranks)
        return 0.30 + 0.01 * high

    def bounty_expected_multiplier(self, my_bounty, hero_str, board_str, street):
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
        visible_hole_ranks = {c[0] for c in hero_str}
        visible_board_ranks = {c[0] for c in board_str}
        remaining = 5 - len(board_str)
        total_p = 0.0
        for rank in RANK_CHARS:
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
        add = BOUNTY_CONSTANT * avg_p_hit
        return mult, add

    def marginal_payoffs(self, pot, my_contrib, continue_cost,
                         my_bm, my_bb, opp_bm, opp_bb):
        opp_contrib = pot - my_contrib
        win_marginal  = opp_contrib * my_bm + my_bb + my_contrib
        lose_marginal = (my_contrib + continue_cost) * opp_bm + opp_bb - my_contrib
        return win_marginal, lose_marginal

    def required_equity(self, continue_cost, pot, opp_bm, opp_bb,
                        my_bm, my_bb, my_contrib):
        win_m, lose_m = self.marginal_payoffs(
            pot, my_contrib, continue_cost, my_bm, my_bb, opp_bm, opp_bb)
        denom = win_m + lose_m
        if denom <= 0:
            return 1.0
        return max(0.0, lose_m / denom)

    def estimate_opponent_tightness(self, my_contribution, opp_contribution,
                                    continue_cost, street, facing_bet):
        if not facing_bet:
            return 0.01
        committed = opp_contribution / STARTING_STACK
        if street == 0:
            if committed >= 0.85: return 0.22
            if committed >= 0.40: return 0.14
            if committed >= 0.15: return 0.08
            if committed >= 0.06: return 0.05
            return 0.02
        else:
            if committed >= 0.80: return 0.18
            if committed >= 0.40: return 0.12
            if committed >= 0.20: return 0.08
            if committed >= 0.08: return 0.04
            return 0.02

    # U5: Polarized sizing only — 35% value / 65% bluff
    def pick_raise_size(self, equity, pot, my_pip, opp_pip,
                        my_stack, opp_stack, min_raise, max_raise,
                        street, aggression_shift, in_position=True):
        '''Polarized: never small (avoids exploit-bot classifier), never medium.
        U13: OOP bets smaller value (1/3 pot), IP bets larger value (1/2 pot).
        '''
        if equity >= 0.85:
            frac = 0.65 if in_position else 0.50   # big value IP
        elif equity >= 0.62:
            frac = 0.50 if in_position else 0.35   # normal value
        else:
            frac = 0.65   # overbet bluff (same both positions)
        frac += aggression_shift
        frac = max(0.35, min(1.30, frac))
        target_total_bet = opp_pip + int(max(pot, 4) * frac)
        target_total_bet = max(target_total_bet, min_raise)
        target_total_bet = min(target_total_bet, max_raise)
        return target_total_bet

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
            ranks = sorted(set(RANK_VALUE[r] for r in hr + br))
            for start in range(2, 11):
                window = set(range(start, start + 5))
                if len(window & set(ranks)) >= 4:
                    return True
        except Exception:
            pass
        return False

    # ─── U3: Anti-regime camouflage style manager ─────────────────────────────
    def _update_style_phase(self):
        '''Every ~20 hands, go tight for 3 hands to confuse regime-shift detectors.'''
        self._style_hand_counter += 1
        if not hasattr(self, '_current_phase_len'):
            self._current_phase_len = random.randint(18, 24)
        pos_in_cycle = self._style_hand_counter % self._current_phase_len
        if pos_in_cycle == 0:
            self._current_phase_len = random.randint(18, 24)
        tight_len = 3
        self._style_phase = 1 if pos_in_cycle < tight_len else 0

    @property
    def _in_camouflage_mode(self):
        return self._style_phase == 1

    # ─── Engine Callbacks ─────────────────────────────────────────────────────

    def handle_new_round(self, game_state, round_state, active):
        self.my_last_action = None
        self.last_street_seen = 0
        self.prev_my_pip = 0
        self.prev_opp_pip = 0
        self.opp.new_hand()
        self._update_style_phase()

    def handle_round_over(self, game_state, terminal_state, active):
        self.round_count += 1
        previous_state = terminal_state.previous_state
        my_delta = terminal_state.deltas[active]
        self.bankroll_running += my_delta
        if previous_state.street < 5:
            if my_delta > 0:
                self.opp_folds += 1
            self.bluff_opportunities += 1

    def get_action(self, game_state, round_state, active):
        try:
            return self._get_action(game_state, round_state, active)  # noqa
        except Exception:
            if CheckAction in round_state.legal_actions():
                return CheckAction()
            return FoldAction()

    def _get_action(self, game_state, round_state, active):
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

        # U13: Position — postflop IP = act last = button acted first preflop
        # In HU, active==0 means SB (posts first). Postflop OOP = active==0.
        in_position = (active == 1) if street > 0 else (active == 0)

        # ── Hard safety net when clock is nearly zero ──────────────────────
        if game_clock < 1.2:
            if CheckAction in legal_actions:
                return CheckAction()
            if continue_cost <= 2 and CallAction in legal_actions:
                return CallAction()
            if FoldAction in legal_actions:
                return FoldAction()
            return CallAction()

        # ── U9: Range-weighted MC equity ──────────────────────────────────
        samples = self.time_samples(game_clock, street)
        opp_range_min = self.opp._range_min()   # tightness from profile
        if street == 0:
            base_equity = self.preflop_lookup(my_cards)
        else:
            base_equity = self.mc_equity(my_cards, board_cards, samples,
                                         opp_range_min)

        my_bm, my_bb   = self.bounty_expected_multiplier(my_bounty, my_cards, board_cards, street)
        opp_bm, opp_bb = self.opp_bounty_expected_multiplier(my_cards, board_cards, street)

        haircut = self.estimate_opponent_tightness(
            my_contribution, opp_contribution, continue_cost, street, facing_bet)

        fold_prior   = self.opp.prior_postflop_fold_to_bet()
        aggro_shift  = 0.0
        if fold_prior > 0.6:  aggro_shift += 0.10
        elif fold_prior < 0.25: aggro_shift -= 0.10

        bounty_bump = 0.02 * (my_bm - 1.0) + (my_bb / 450.0)

        # ── Opponent profile ───────────────────────────────────────────────
        prof        = self.opp.profile()
        is_baseline = self.opp.is_baseline
        is_nit      = self.opp.is_nit
        is_station  = self.opp.is_station
        is_maniac   = self.opp.is_maniac

        # ── U2: Early blitz window (rounds 1-25) ──────────────────────────
        # All profiling bots disable bluffs / use balanced mode until they
        # have 30 hands of data. We exploit this with unconditional aggression.
        early_blitz = self.round_count < 25

        # ── U6: Unknown guard exploit — they are passive, we blitz ────────
        # IEC2025002's bluff_ok=False for first 30 hands means we steal freely.
        unknown_guard_blitz = self.round_count < 30 and prof == 'UNKNOWN'

        # ── U3: Camouflage: be tight for 3 hands every 20 to fool windowed models
        camouflage = self._in_camouflage_mode

        # Baseline Fingerprinting Guard
        if self.opp.is_baseline is None:
            if street == 0 and my_pip == 1 and opp_pip == 2 and CallAction in legal_actions:
                return CallAction()

        # ══════════════════════════════════════════════════════════════════
        # PREFLOP
        # ══════════════════════════════════════════════════════════════════
        if street == 0:
            hero_ranks = [c[0] for c in my_cards]
            is_pair    = hero_ranks[0] == hero_ranks[1]
            hi = max(RANK_VALUE[r] for r in hero_ranks)
            lo = min(RANK_VALUE[r] for r in hero_ranks)
            has_bounty_rank = my_bounty in hero_ranks

            eq_raw   = base_equity
            eq_vs_opp = max(0.05, eq_raw - haircut)
            eq_eff   = eq_vs_opp + bounty_bump

            klass         = classify_preflop(my_cards)
            opp_committed = opp_contribution / STARTING_STACK
            req           = self.required_equity(continue_cost, pot, opp_bm, opp_bb,
                                                 my_bm, my_bb, my_contribution)

            # ── U1: NASH PUSH/FOLD (short stacks) ─────────────────────────
            eff_bb = min(my_stack, opp_stack) / BIG_BLIND
            if eff_bb <= 15:
                pf  = eq_raw
                if has_bounty_rank: pf = min(pf + 0.15, 0.98)
                push_t, call_t = nash_thresholds(eff_bb)
                if not facing_bet:
                    # SB / first to act: push or fold
                    if pf >= push_t and RaiseAction in legal_actions:
                        return RaiseAction(max_raise_total)
                    if CheckAction in legal_actions:
                        return CheckAction()
                    return FoldAction() if FoldAction in legal_actions else CallAction()
                else:
                    # Facing a near-shove: call or fold by Nash
                    if continue_cost >= min(my_stack, opp_stack) * 0.55:
                        if pf >= call_t and CallAction in legal_actions:
                            return CallAction()
                        return FoldAction() if FoldAction in legal_actions else CallAction()

            # ── OPTIMAL BASELINE EXPLOIT ─────────────────────────────────
            # Baseline folds 93% of hands preflop to any jam. Jamming 400 is +172 EV/hand.
            if is_baseline:
                if opp_pip >= 14:    # Baseline raised 14, 26, or jammed 400 = premium
                    if klass == 'premium' and CallAction in legal_actions:
                        return CallAction()
                    return FoldAction() if FoldAction in legal_actions else CheckAction()
                if not facing_bet or continue_cost <= 2:
                    # Unraised pot or SB limped: Open-jam every single hand to perfectly exploit their 93% fold rate
                    if RaiseAction in legal_actions:
                        return RaiseAction(max_raise_total)
                    return FoldAction() if FoldAction in legal_actions else CheckAction()

            # ── Camouflage phase: play tight ──────────────────────────────
            if camouflage:
                if not facing_bet:
                    if klass in ('premium', 'strong') and RaiseAction in legal_actions:
                        target = opp_pip + max(min_raise_total - opp_pip, int(pot * 1.0))
                        return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                    return CheckAction() if CheckAction in legal_actions else CallAction()
                if klass in ('premium', 'strong') and CallAction in legal_actions:
                    return CallAction()
                return FoldAction() if FoldAction in legal_actions else CheckAction()

            # ── No bet facing us (BB option or SB limp) ───────────────────
            if not facing_bet:
                if RaiseAction in legal_actions:
                    if klass == 'premium':
                        target = opp_pip + max(6, int(pot * 1.6))
                    elif klass == 'strong':
                        target = opp_pip + max(5, int(pot * 1.2))
                    elif klass in ('medium', 'openable') or has_bounty_rank:
                        target = opp_pip + max(4, int(pot * 0.9))
                    elif early_blitz or unknown_guard_blitz:
                        # U2/U6: steal 100% of hands early — they have no bluff-guard yet
                        target = opp_pip + max(4, int(pot * 0.65))
                    else:
                        target = None
                    if target is not None:
                        return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                if CheckAction in legal_actions:
                    return CheckAction()
                return CallAction() if CallAction in legal_actions else CheckAction()

            # ── Facing a bet / raise preflop ─────────────────────────────

            # 1) Jam / near-jam defense — call wider than premium only
            if opp_committed >= 0.65 or continue_cost >= my_stack * 0.55:
                if klass in ('premium', 'strong') and CallAction in legal_actions:
                    return CallAction()
                # Wide jam range — call medium with marginal equity since pot odds are good
                if klass == 'medium' and eq_eff >= 0.38 and CallAction in legal_actions:
                    return CallAction()
                return FoldAction() if FoldAction in legal_actions else CheckAction()

            # 2) 3-bet / 4-bet territory
            if opp_pip >= 25 or opp_committed >= 0.12:
                if RaiseAction in legal_actions and klass == 'premium':
                    _, max_rr = round_state.raise_bounds()
                    jam_target = min(opp_pip + int((pot + continue_cost) * 2.2), max_rr)
                    return RaiseAction(max(jam_target, min_raise_total))
                if CallAction in legal_actions and klass in ('premium', 'strong'):
                    return CallAction()
                if CallAction in legal_actions and klass == 'medium' and continue_cost <= pot * 0.55:
                    return CallAction()
                return FoldAction() if FoldAction in legal_actions else CheckAction()

            # 3) Standard single raise — 3-bet premium/strong, call medium
            if opp_pip >= 5:
                if RaiseAction in legal_actions and klass == 'premium':
                    target = opp_pip + max(min_raise_total - opp_pip, int(continue_cost * 3.0))
                    return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                if RaiseAction in legal_actions and klass == 'strong':
                    target = opp_pip + max(min_raise_total - opp_pip, int(continue_cost * 2.6))
                    return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                if CallAction in legal_actions and klass in ('premium', 'strong', 'medium'):
                    return CallAction()
                if CallAction in legal_actions and has_bounty_rank and continue_cost <= 8:
                    return CallAction()
                return FoldAction() if FoldAction in legal_actions else CheckAction()

            # 4) SB completing (continue_cost <= 2)
            if continue_cost <= 2:
                if RaiseAction in legal_actions and klass == 'premium':
                    target = opp_pip + max(min_raise_total - opp_pip, 6)
                    return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                if CallAction in legal_actions and klass != 'trash':
                    return CallAction()
                if CallAction in legal_actions and has_bounty_rank:
                    return CallAction()
                return FoldAction() if FoldAction in legal_actions else CheckAction()

            # Fallback
            if CallAction in legal_actions and eq_eff >= req and klass != 'trash':
                return CallAction()
            return FoldAction() if FoldAction in legal_actions else CheckAction()

        # ══════════════════════════════════════════════════════════════════
        # POSTFLOP
        # ══════════════════════════════════════════════════════════════════
        eq_raw    = base_equity
        eq_vs_opp = max(0.02, eq_raw - haircut)
        eq_eff    = eq_vs_opp + bounty_bump

        req = self.required_equity(continue_cost, pot, opp_bm, opp_bb,
                                   my_bm, my_bb, my_contribution)
        opp_committed = opp_contribution / STARTING_STACK
        spr          = my_stack / max(4, pot)
        opp_fold_rate = self.opp_folds / max(1, self.bluff_opportunities)

        # U12: Dynamic bluff frequency — use actual measured fold rate
        # Default 0.35 until we have data; then blend measured + prior
        measured_fold = opp_fold_rate if self.bluff_opportunities >= 15 else 0.35
        bluff_freq_base = 0.20 + 0.60 * measured_fold   # scales 0.20–0.80
        bluff_freq_base = max(0.10, min(0.70, bluff_freq_base))
        if is_station: bluff_freq_base = 0.08   # never bluff stations
        if is_maniac:  bluff_freq_base *= 1.30  # bluff maniacs more (they fold to raises)

        # ── BASELINE EXPLOIT POSTFLOP v6.2 ───────────────────────────
        # Baseline only bets premium+bounty postflop and folds to ANY bet unless premium.
        # EXPLOIT: bet WIDE (eq>=0.25) since they fold ~85% of their range postflop.
        if is_baseline and street > 0:
            if not facing_bet:
                # Baseline folds to any bet unless premium — bet EVERYTHING
                if RaiseAction in legal_actions and eq_raw >= 0.25:
                    target = my_pip + max(min_raise_total - my_pip, int(max(pot, 4) * 0.50))
                    return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                # Even bluff with trash sometimes (they fold so much)
                if RaiseAction in legal_actions and random.random() < 0.40:
                    target = my_pip + max(min_raise_total - my_pip, int(max(pot, 4) * 0.35))
                    return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                return CheckAction() if CheckAction in legal_actions else FoldAction()
            else:
                # Baseline betting = they have premium+bounty. Only call with strong.
                if eq_raw >= 0.70 and CallAction in legal_actions:
                    return CallAction()
                if continue_cost <= max(3, pot * 0.10) and eq_raw >= 0.45 and CallAction in legal_actions:
                    return CallAction()
                return FoldAction() if FoldAction in legal_actions else CheckAction()

        # ── Camouflage phase: passive / check/call only ───────────────────
        if camouflage:
            if not facing_bet:
                if eq_raw >= 0.72 and RaiseAction in legal_actions:
                    target = my_pip + max(min_raise_total - my_pip, int(pot * 0.35))
                    return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                return CheckAction() if CheckAction in legal_actions else FoldAction()
            if CallAction in legal_actions and eq_eff >= req + 0.02:
                return CallAction()
            return FoldAction() if FoldAction in legal_actions else CheckAction()

        # U10: MDF guard — before any fold decision vs a bet, check minimum
        # defense frequency. If opp bets small, we must defend a minimum %.
        # MDF = pot / (pot + bet). If we'd fold, check MDF first.
        if facing_bet and continue_cost > 0:
            mdf = pot / max(1, pot + continue_cost)
            # Only apply MDF when we have real equity (> 0.27) and bet is small
            if continue_cost < pot * 0.40 and eq_raw >= 0.27:
                if random.random() < mdf:
                    # Defend: either raise strong hands, or call
                    if eq_eff >= 0.65 and RaiseAction in legal_actions:
                        target = self.pick_raise_size(
                            eq_raw, pot + continue_cost, my_pip, opp_pip,
                            my_stack, opp_stack, min_raise_total, max_raise_total,
                            street, aggro_shift, in_position)
                        return RaiseAction(target)
                    if CallAction in legal_actions:
                        return CallAction()

        # ── Not facing a bet ──────────────────────────────────────────────
        if not facing_bet:
            if RaiseAction in legal_actions and pot >= 4:
                # Value hand: U5 polarized sizing
                if eq_raw >= 0.62:
                    if spr < 0.5:
                        return RaiseAction(max_raise_total)
                    # v6: Turn tightening — require 0.70 to bet turn OOP, 0.65 IP
                    if street == 4:
                        turn_bet_thresh = 0.65 if in_position else 0.70
                        if eq_raw < turn_bet_thresh:
                            pass  # check on turn with marginal equity
                        else:
                            frac = 0.50 if in_position else 0.35
                            target = my_pip + max(min_raise_total - my_pip, int(max(pot, 4) * frac))
                            return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                    else:
                        frac = 0.50 if in_position else 0.35  # U13: position-aware
                        target = my_pip + max(min_raise_total - my_pip, int(max(pot, 4) * frac))
                        return RaiseAction(min(max(target, min_raise_total), max_raise_total))

                # Station / Maniac: thin value (they call everything)
                if (is_station or is_maniac) and eq_raw >= 0.48:
                    # v6: No thin value on turn even vs stations unless strong
                    if street == 4 and eq_raw < 0.58:
                        pass  # check turn vs station
                    else:
                        target = my_pip + max(min_raise_total - my_pip, int(max(pot, 4) * 0.35))
                        return RaiseAction(min(max(target, min_raise_total), max_raise_total))

                # Thin value — v6: raised threshold to 0.55, and much tighter on turn
                if eq_raw >= 0.55:
                    # v6: On turn, only bet thin value IP with eq >= 0.60
                    if street == 4:
                        if not in_position or eq_raw < 0.60:
                            pass   # check turn — thin value on turn is a leak
                        else:
                            frac = 0.35
                            target = my_pip + max(min_raise_total - my_pip, int(max(pot, 4) * frac))
                            return RaiseAction(min(max(target, min_raise_total), max_raise_total))
                    else:
                        frac = 0.35 + aggro_shift * 0.5
                        frac = max(0.35, min(0.65, frac))
                        target = my_pip + max(min_raise_total - my_pip, int(max(pot, 4) * frac))
                        return RaiseAction(min(max(target, min_raise_total), max_raise_total))

                # U2/U6: Early blitz c-bet with any hand on flop only
                if (early_blitz or unknown_guard_blitz) and street == 3:
                    target = my_pip + max(min_raise_total - my_pip, int(max(pot, 4) * 0.65))
                    return RaiseAction(min(max(target, min_raise_total), max_raise_total))

                # Standard flop c-bet (medium equity)
                if eq_raw >= 0.35 and street == 3:
                    if random.random() < (0.75 if not is_station else 0.35):
                        target = my_pip + max(min_raise_total - my_pip, int(max(pot, 4) * 0.35))
                        return RaiseAction(min(max(target, min_raise_total), max_raise_total))

                # U12: Dynamic bluff — 65% pot overbet; only on flop/river, not turn
                # U11: No bluff on turn (was the -4253 EV leak)
                if eq_raw < 0.30 and pot >= 8 and street != 4:
                    if not is_station and random.random() < bluff_freq_base:
                        target = my_pip + max(min_raise_total - my_pip, int(max(pot, 4) * 0.65))
                        return RaiseAction(min(max(target, min_raise_total), max_raise_total))

            if CheckAction in legal_actions:
                return CheckAction()

        # ── Facing a bet / raise ──────────────────────────────────────────
        # Value raise: strong hand, U5 polarized / U13 position-aware sizing
        if RaiseAction in legal_actions and eq_eff >= 0.72 and opp_committed < 0.55:
            target = self.pick_raise_size(eq_raw, pot + continue_cost,
                                          my_pip, opp_pip, my_stack, opp_stack,
                                          min_raise_total, max_raise_total,
                                          street, aggro_shift, in_position)
            return RaiseAction(target)

        # Semi-bluff raise: only on flop (not turn per U11)
        if (RaiseAction in legal_actions
                and 0.48 <= eq_eff < 0.72
                and fold_prior > 0.55
                and continue_cost <= pot * 0.6
                and opp_committed < 0.35
                and street != 4):    # U11: no semi-bluff raises on turn
            target = opp_pip + max(min_raise_total - opp_pip,
                                   int((pot + continue_cost) * 0.65))
            return RaiseAction(min(max(target, min_raise_total), max_raise_total))

        # Call / fold
        pot_odds = continue_cost / max(1, (pot + continue_cost))
        has_pair_or_better = self._has_pair_or_better(my_cards, board_cards)
        big_bet = continue_cost >= pot * 0.55

        # v6: Turn call tightening — significantly stronger than v5.
        # Tournament data shows turn is BEAST's worst street vs ALL opponents.
        turn_call_margin = 0.08 if street == 4 else 0.02  # much higher bar on turn
        if street == 4 and big_bet and not has_pair_or_better:
            # No hand on turn + big bet = fold more aggressively
            if eq_raw < 0.58:
                if FoldAction in legal_actions:
                    return FoldAction()
                return CheckAction() if CheckAction in legal_actions else FoldAction()
        # v6: Even with pair on turn, require stronger equity vs big bets
        if street == 4 and big_bet and has_pair_or_better and eq_raw < 0.45:
            if FoldAction in legal_actions:
                return FoldAction()
            return CheckAction() if CheckAction in legal_actions else FoldAction()

        if CallAction in legal_actions:
            if eq_eff >= req + turn_call_margin:
                if big_bet and not has_pair_or_better and eq_raw < 0.60:
                    pass  # fall through to fold
                else:
                    return CallAction()
            # Cheap calls with any reasonable equity (not on turn with marginal hand)
            cheap = continue_cost <= max(3, pot * 0.15)
            if cheap and eq_raw >= (0.32 if street == 4 else 0.28):
                return CallAction()
            # U14: River blocker hero-call — opp overbets river, we have pair/draw
            # When opp bets >pot on river and we're at pot odds, they're often bluffing
            if street == 5 and has_pair_or_better:
                opp_river_overbet = continue_cost >= pot * 0.80
                if opp_river_overbet and eq_eff >= pot_odds + 0.02:
                    return CallAction()
                # U14: standard bluff-catch when opp over-bluffs measured
                if fold_prior < 0.25 and eq_eff >= pot_odds:
                    return CallAction()

        if FoldAction in legal_actions:
            return FoldAction()
        if CheckAction in legal_actions:
            return CheckAction()
        return CallAction()

    # ─── Opponent Tracking ────────────────────────────────────────────────────
    def _track_opponent(self, round_state, active, street):
        opp_pip = round_state.pips[1 - active]
        my_pip  = round_state.pips[active]
        if street != self.last_street_seen:
            self.last_street_seen = street
            self.prev_opp_pip = opp_pip
            self.prev_my_pip  = my_pip
            return
        if opp_pip > self.prev_opp_pip:
            delta = opp_pip - self.prev_opp_pip
            if street == 0:
                self.opp.preflop_actions += 1
                self.opp.preflop_raises += 1
                self.opp.record_pf_raise()
                if opp_pip in [14, 26]:
                    self.opp.is_baseline = True
                elif opp_pip > 2 and opp_pip not in [11, 400]:
                    self.opp.is_baseline = False
            else:
                self.opp.postflop_actions += 1
                if self.prev_my_pip > self.prev_opp_pip:
                    self.opp.postflop_raises += 1
                    self.opp.record_postflop_bet()
                else:
                    self.opp.postflop_bets += 1
                    self.opp.record_postflop_bet()
            self.opp.last_action_aggressive = True
        elif opp_pip == self.prev_opp_pip and my_pip > self.prev_my_pip:
            # we bet, they didn't increase → they called or checked
            pass
        elif opp_pip <= self.prev_opp_pip:
            # they might have checked/folded
            pass
        self.prev_opp_pip = opp_pip
        self.prev_my_pip  = my_pip


if __name__ == '__main__':
    run_bot(Player(), parse_args())
