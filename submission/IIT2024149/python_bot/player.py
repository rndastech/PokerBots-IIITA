'''
=============================================================
  FAMILY-KILLER BOUNTY HOLD'EM POKER BOT v5.0 -- newplayer5
=============================================================

Improvements over v2.0:
  L14 Information-set mixer for strategic unpredictability
  L15 Confidence-aware soft opponent profiling
  L16 Anti-exploit safety guards and leak caps
  L17 Profile-poisoning defense (windowed stats + regime shift)
  +   Bayesian bounty inference (posterior over ranks)
  +   Range-aware Monte Carlo equity
  +   Discrete polarized bet sizing menu
  +   Blocker-aware river decisions
  +   Bankroll-trend adaptation

All v2.0 fixes preserved:
  - No catastrophic anti-nit override
  - Equity-gated stealing (not 100%)
  - No equity corruption from Bayesian range discount
  - Strong+ hands never fold to standard river bets
  - Opponent adjustments are soft overlays, not overrides
  - Minimum 80 MC iterations for stability
  - Drawing hands reclassified on river
'''

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import math

# ---------- eval7 ----------
try:
    import eval7
    EVAL7_AVAILABLE = True
except ImportError:
    EVAL7_AVAILABLE = False

if EVAL7_AVAILABLE:
    EVAL7_CARD_CACHE = {
        rank + suit: eval7.Card(rank + suit)
        for rank in '23456789TJQKA'
        for suit in 'cdhs'
    }
else:
    EVAL7_CARD_CACHE = {}

# ---------- Constants ----------
BOUNTY_RATIO = 1.5
BOUNTY_CONSTANT = 10
ROUNDS_PER_BOUNTY = 25

RANK_VALUE = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
}
VALUE_RANK = {v: k for k, v in RANK_VALUE.items()}
ALL_RANKS = list(RANK_VALUE.keys())

# Range tightness thresholds (preflop equity cutoff for "would play")
RANGE_THRESHOLDS = {
    'nit': 0.58, 'tag': 0.50, 'default': 0.42,
    'lag': 0.38, 'maniac': 0.34, 'station': 0.38,
}


# =====================================================================
#  L1 -- PRE-COMPUTED PREFLOP EQUITY TABLE
# =====================================================================
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


def cached_eval7_cards(card_strs):
    '''Convert string cards to cached eval7 card objects.'''
    if not EVAL7_AVAILABLE:
        return []
    return [EVAL7_CARD_CACHE[str(card)] for card in card_strs]


# =====================================================================
#  HELPER FUNCTIONS
# =====================================================================

def minimum_defense_frequency(pot, bet_size):
    if bet_size <= 0 or pot <= 0:
        return 1.0
    return pot / (pot + bet_size)


def analyze_board_texture(board_cards):
    if not board_cards or len(board_cards) < 3:
        return 'neutral'
    suits = [c[1] for c in board_cards if len(c) > 1]
    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    max_suit = max(suit_counts.values()) if suit_counts else 0
    ranks = sorted([RANK_VALUE.get(c[0], 0) for c in board_cards])
    max_con = 1
    cur = 1
    for i in range(1, len(ranks)):
        if ranks[i] == ranks[i-1] + 1:
            cur += 1
            max_con = max(max_con, cur)
        elif ranks[i] != ranks[i-1]:
            cur = 1
    if max_suit >= 3 or max_con >= 3:
        return 'wet'
    elif max_suit <= 1 and max_con <= 1:
        return 'dry'
    return 'neutral'


def board_is_paired(board_cards):
    if not board_cards:
        return False
    ranks = [c[0] for c in board_cards]
    return len(set(ranks)) < len(ranks)


def board_pair_multiplicity(board_cards):
    if not board_cards:
        return 1
    counts = {}
    for card in board_cards:
        rank = card[0]
        counts[rank] = counts.get(rank, 0) + 1
    return max(counts.values(), default=1)


def board_danger_score(board_cards):
    '''Compact danger score for paired, monotone, and connected boards.'''
    if not board_cards:
        return 0.0

    score = 0.0
    suits = [c[1] for c in board_cards if len(c) > 1]
    suit_counts = {}
    for suit in suits:
        suit_counts[suit] = suit_counts.get(suit, 0) + 1
    max_suit = max(suit_counts.values(), default=0)
    if max_suit >= 4:
        score += 3.0
    elif max_suit == 3:
        score += 1.6

    ranks = sorted(set(RANK_VALUE.get(c[0], 0) for c in board_cards))
    if 14 in ranks:
        ranks = sorted(set(ranks + [1]))
    max_run = 1
    cur = 1
    for i in range(1, len(ranks)):
        if ranks[i] == ranks[i - 1] + 1:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    if max_run >= 4:
        score += 2.6
    elif max_run == 3:
        score += 1.2

    pair_mult = board_pair_multiplicity(board_cards)
    if pair_mult >= 3:
        score += 3.0
    elif board_is_paired(board_cards):
        score += 1.7

    if len(board_cards) >= 4 and max_suit >= 3 and max_run >= 3:
        score += 1.0
    return score


def bounty_pot_odds(cost, pot, bounty_active):
    if cost <= 0:
        return 0.0
    eff_pot = pot * BOUNTY_RATIO + BOUNTY_CONSTANT if bounty_active else pot
    return cost / (eff_pot + cost)


def canonical_card_class(cards):
    r1 = RANK_VALUE.get(cards[0][0], 5)
    r2 = RANK_VALUE.get(cards[1][0], 5)
    high, low = max(r1, r2), min(r1, r2)
    is_pair = r1 == r2
    is_suited = (len(cards[0]) > 1 and len(cards[1]) > 1
                 and cards[0][1] == cards[1][1])
    return high * 15 + low + (100 if is_pair else 0) + (50 if is_suited else 0)


def pot_bucket(pot):
    ratio = pot / max(1, STARTING_STACK)
    if ratio < 0.10:
        return 0
    elif ratio < 0.25:
        return 1
    elif ratio < 0.50:
        return 2
    elif ratio < 0.75:
        return 3
    return 4


def preflop_allin_call_tier(cards):
    '''Tight call tiers for full-stack preflop commitments.'''
    r1 = RANK_VALUE.get(cards[0][0], 5)
    r2 = RANK_VALUE.get(cards[1][0], 5)
    high, low = max(r1, r2), min(r1, r2)
    is_pair = r1 == r2
    is_suited = (len(cards[0]) > 1 and len(cards[1]) > 1
                 and cards[0][1] == cards[1][1])

    if is_pair and high >= 13:
        return 3  # AA, KK
    if is_pair and high == 12:
        return 2  # QQ
    if high == 14 and low == 13:
        return 2  # AK
    if is_pair and high == 11:
        return 1  # JJ
    if is_suited and high == 14 and low == 12:
        return 1  # AQs
    return 0


def adaptive_jam_threshold(opp_jam_rate, bounty_active, bankroll, phase,
                           family_mode=False):
    '''Continuous threshold for true near-all-in preflop calls.'''
    if opp_jam_rate < 0.04:
        base = 0.76
    elif opp_jam_rate < 0.08:
        base = 0.71
    elif opp_jam_rate < 0.15:
        base = 0.64
    elif opp_jam_rate < 0.25:
        base = 0.59
    else:
        base = 0.55

    if bounty_active:
        base -= 0.03
    if family_mode:
        base -= 0.02
    if phase == 'late' and bankroll > 200:
        base += 0.02
    elif phase == 'late' and bankroll < -200:
        base -= 0.02
    return max(0.50, min(0.82, base))


def max_reraise_by_tier(tier):
    '''Cap re-raises only once the street becomes an escalation war.'''
    return {
        'monster': 99,
        'very_strong': 2,
        'strong': 1,
        'medium': 0,
        'overcards': 0,
        'drawing': 1,
        'marginal': 0,
        'weak': 0,
    }.get(tier, 0)


# =====================================================================
#  L14 -- INFORMATION-SET MIXER
# =====================================================================

class ISMixer:
    '''Deterministic-seeded mixer for non-patterned mixed strategies.'''

    def __init__(self):
        self._seed = 0

    def new_round(self, round_num, cards):
        h = 0
        for c in cards:
            h = h * 31 + ord(c[0]) * 7 + (ord(c[1]) if len(c) > 1 else 0)
        self._seed = round_num * 97 + h

    def mix(self, street, pos, cc, bb, pb, action_id):
        '''Returns float in [0, 1) for mixed strategy decisions.'''
        s = (self._seed * 1000003 + street * 7919 + pos * 104729 +
             cc * 1299709 + bb * 15485863 + pb * 32452843 +
             action_id * 49979687)
        s = ((s ^ (s >> 16)) * 0x45d9f3b) & 0xFFFFFFFF
        s = ((s ^ (s >> 16)) * 0x45d9f3b) & 0xFFFFFFFF
        s = s ^ (s >> 16)
        return (s % 10000) / 10000.0


# =====================================================================
#  L3 -- HAND CLASSIFICATION
# =====================================================================

HAND_RANK_NAMES = {
    0: 'high_card', 1: 'pair', 2: 'two_pair', 3: 'trips',
    4: 'straight', 5: 'flush', 6: 'full_house',
    7: 'quads', 8: 'straight_flush',
}


def normalize_hand_category(hand_type):
    if isinstance(hand_type, str):
        key = hand_type.strip().lower().replace(' ', '_')
        aliases = {
            'pair': 'pair',
            'two_pair': 'two_pair',
            'three_of_a_kind': 'trips',
            'trips': 'trips',
            'straight': 'straight',
            'flush': 'flush',
            'full_house': 'full_house',
            'four_of_a_kind': 'quads',
            'quads': 'quads',
            'straight_flush': 'straight_flush',
            'high_card': 'high_card',
        }
        return aliases.get(key, 'unknown')
    return HAND_RANK_NAMES.get(hand_type, 'unknown')


def classify_hand_strength(my_cards, board_cards):
    if not EVAL7_AVAILABLE or not board_cards:
        return ('unknown', 0, 0.5)
    try:
        my_hand = cached_eval7_cards(my_cards)
        board = cached_eval7_cards(board_cards)
        all_cards = my_hand + board
        if len(all_cards) < 5:
            return ('unknown', 0, 0.5)
        score = eval7.evaluate(all_cards)
        hand_type = eval7.handtype(score)
        category = normalize_hand_category(hand_type)
        deck = eval7.Deck()
        dead = set(my_hand + board)
        live = [c for c in deck.cards if c not in dead]
        beat = 0
        total = 0
        sample = min(150, len(live) * (len(live) - 1) // 2)
        for _ in range(sample):
            random.shuffle(live)
            opp_score = eval7.evaluate(live[:2] + board)
            if score > opp_score:
                beat += 1
            elif score == opp_score:
                beat += 0.5
            total += 1
        pct = beat / total if total > 0 else 0.5
        return (category, score, pct)
    except Exception:
        return ('unknown', 0, 0.5)


def quick_hand_class(my_cards, board_cards):
    if not board_cards:
        return 'unknown'
    board_rank_chars = [c[0] for c in board_cards]
    my_rank_chars = [c[0] for c in my_cards]
    pair_mult = board_pair_multiplicity(board_cards)
    if my_cards[0][0] == my_cards[1][0] and my_cards[0][0] in board_rank_chars:
        return 'monster'
    hits = sum(1 for r in my_rank_chars if r in board_rank_chars)
    if hits >= 2:
        return 'strong' if pair_mult >= 2 else 'very_strong'
    if hits >= 1:
        brs = sorted([RANK_VALUE.get(c[0], 0) for c in board_cards], reverse=True)
        for c in my_cards:
            if c[0] in board_rank_chars:
                kicker = max(
                    (RANK_VALUE.get(x[0], 0) for x in my_cards if x[0] != c[0]),
                    default=0,
                )
                if RANK_VALUE.get(c[0], 0) >= brs[0]:
                    if pair_mult >= 2 or (len(board_cards) >= 4 and kicker <= 10):
                        return 'medium'
                    return 'strong'
                return 'medium'
    if len(board_cards) >= 3:
        all_suits = [c[1] for c in my_cards + board_cards if len(c) > 1]
        sc = {}
        for s in all_suits:
            sc[s] = sc.get(s, 0) + 1
        if max(sc.values(), default=0) >= 5:
            return 'monster'
        if max(sc.values(), default=0) >= 4:
            return 'drawing'
        all_ranks = set(RANK_VALUE.get(c[0], 0) for c in my_cards + board_cards)
        for base in range(1, 11):
            if len(all_ranks & set(range(base, base + 5))) >= 5:
                return 'very_strong'
            if len(all_ranks & set(range(base, base + 5))) >= 4:
                return 'drawing'
    board_max = max((RANK_VALUE.get(c[0], 0) for c in board_cards), default=0)
    oc = sum(1 for c in my_cards if RANK_VALUE.get(c[0], 0) > board_max)
    if oc >= 2:
        return 'overcards' if RANK_VALUE.get(my_cards[0][0], 0) >= 11 else 'weak'
    return 'weak'


# =====================================================================
#  L2 -- RANGE-AWARE MONTE CARLO EQUITY
# =====================================================================

def range_weighted_equity(my_cards, board_cards, opp_tightness='default',
                          iterations=200):
    '''MC equity with opponent range weighting.'''
    if not EVAL7_AVAILABLE or not board_cards:
        return get_preflop_equity(my_cards)
    try:
        my_hand = cached_eval7_cards(my_cards)
        board = cached_eval7_cards(board_cards)
    except Exception:
        return get_preflop_equity(my_cards)
    deck = eval7.Deck()
    dead = set(my_hand + board)
    live = [c for c in deck.cards if c not in dead]
    iterations = max(80, iterations)
    range_thresh = RANGE_THRESHOLDS.get(opp_tightness, 0.42)
    wins = 0.0
    total = 0
    cards_needed = 5 - len(board)
    for _ in range(iterations):
        random.shuffle(live)
        opp = live[:2]
        # Range filter: check if opp would play these cards
        opp_strs = [str(c) for c in opp]
        opp_eq = get_preflop_equity(opp_strs)
        # Accept in-range hands; accept 15% of out-of-range to avoid extreme bias
        if opp_eq < range_thresh and random.random() > 0.15:
            continue
        sim_board = board + live[2:2 + cards_needed]
        my_score = eval7.evaluate(my_hand + sim_board)
        opp_score = eval7.evaluate(opp + sim_board)
        if my_score > opp_score:
            wins += 1.0
        elif my_score == opp_score:
            wins += 0.5
        total += 1
    return wins / total if total > 0 else 0.5


def exact_range_weighted_river_equity(my_cards, board_cards,
                                      opp_tightness='default'):
    '''Exact river equity with the same preflop range weighting used in MC.'''
    if not EVAL7_AVAILABLE or len(board_cards) != 5:
        return range_weighted_equity(my_cards, board_cards, opp_tightness, 200)
    try:
        my_hand = cached_eval7_cards(my_cards)
        board = cached_eval7_cards(board_cards)
    except Exception:
        return 0.5

    deck = eval7.Deck()
    dead = set(my_hand + board)
    live = [c for c in deck.cards if c not in dead]
    my_score = eval7.evaluate(my_hand + board)
    range_thresh = RANGE_THRESHOLDS.get(opp_tightness, 0.42)
    wins = 0.0
    total = 0.0

    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            opp = [live[i], live[j]]
            opp_eq = get_preflop_equity([str(opp[0]), str(opp[1])])
            weight = 1.0 if opp_eq >= range_thresh else 0.15
            opp_score = eval7.evaluate(opp + board)
            if my_score > opp_score:
                wins += weight
            elif my_score == opp_score:
                wins += 0.5 * weight
            total += weight
    return wins / total if total > 0 else 0.5


def quick_postflop_equity(my_cards, board_cards):
    '''Fast heuristic postflop equity without Monte Carlo.'''
    base = get_preflop_equity(my_cards)
    if not board_cards:
        return base
    brc = [c[0] for c in board_cards]
    mrc = [c[0] for c in my_cards]
    if my_cards[0][0] == my_cards[1][0] and my_cards[0][0] in brc:
        return min(0.95, base + 0.35)
    hits = sum(1 for r in mrc if r in brc)
    if hits >= 2:
        return min(0.92, base + 0.25)
    if hits >= 1:
        bmax = max(RANK_VALUE.get(c[0], 0) for c in board_cards)
        for c in my_cards:
            if c[0] in brc:
                if RANK_VALUE.get(c[0], 0) >= bmax:
                    return min(0.85, base + 0.18)
                return min(0.75, base + 0.10)
    if len(board_cards) >= 3:
        asuits = [c[1] for c in my_cards + board_cards if len(c) > 1]
        sc = {}
        for s in asuits:
            sc[s] = sc.get(s, 0) + 1
        ms = max(sc.values(), default=0)
        if ms >= 5:
            return min(0.92, base + 0.30)
        if ms >= 4:
            return min(0.70, base + 0.08)
    bmax = max((RANK_VALUE.get(c[0], 0) for c in board_cards), default=0)
    oc = sum(1 for c in my_cards if RANK_VALUE.get(c[0], 0) > bmax)
    if oc >= 2 and RANK_VALUE.get(my_cards[0][0], 0) >= 11:
        return base
    if oc >= 1 and RANK_VALUE.get(my_cards[0][0], 0) >= 13:
        return max(base - 0.05, 0.30)
    return max(base - 0.12, 0.15)


# =====================================================================
#  L5 -- BAYESIAN BOUNTY TRACKER
# =====================================================================

class BayesianBountyTracker:
    '''Posterior tracker over opponent hidden bounty rank.'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.posterior = {r: 1.0 / 13.0 for r in ALL_RANKS}
        self.total_rounds = 0
        self.total_hits = 0

    def update(self, opp_hit, board_ranks, opp_card_ranks=None):
        self.total_rounds += 1
        if opp_hit:
            self.total_hits += 1
        all_vis = set(board_ranks)
        saw_opp = opp_card_ranks is not None
        if saw_opp:
            all_vis |= set(opp_card_ranks)
        new_p = dict(self.posterior)
        for rank in ALL_RANKS:
            if opp_hit:
                if saw_opp:
                    if rank not in all_vis:
                        new_p[rank] = 0.0
                else:
                    if rank in board_ranks:
                        new_p[rank] *= 1.4
                    else:
                        new_p[rank] *= 0.90
            else:
                if saw_opp:
                    if rank in all_vis:
                        new_p[rank] = 0.0
                else:
                    if rank in board_ranks:
                        new_p[rank] = 0.0
        total = sum(new_p.values())
        if total > 0:
            self.posterior = {r: p / total for r, p in new_p.items()}
        else:
            self.reset()

    def estimate(self):
        best = max(self.posterior, key=self.posterior.get)
        conf = max(0.0, self.posterior[best] - 1.0 / 13.0)
        return best, min(0.90, conf)

    def board_likely_hits_opponent(self, board_cards, threshold=0.35):
        est, conf = self.estimate()
        if conf < threshold:
            return False
        return any(c[0] == est for c in board_cards)


# =====================================================================
#  L4/L17 -- WINDOWED OPPONENT MODEL
# =====================================================================

class WindowedOpponentModel:
    '''Opponent model with per-hand records, windowed stats, and
    regime-shift detection for profile-poisoning defense.'''

    def __init__(self):
        self.history = []
        self._round = {}
        self.hands_played = 0
        self.showdowns = 0
        self.postflop_bluffs_detected = 0
        self.postflop_value_detected = 0
        self.raise_faced = 0
        self.folds_to_raise = 0
        self._reset_round()

    def _reset_round(self):
        self._round = {
            'vpip': False, 'pfr': False, 'preflop_jam': False,
            'pb': 0, 'pc': 0, 'pcl': 0, 'pf': 0,
            'folded': False, 'sizes': [],
            'cbet_opp': False, 'did_cbet': False,
            'small_probe_flop': False,
            'small_probe_turn': False,
            'small_probe_river': False,
            'late_big_bet': False,
            'checked_streets': set(),
        }

    def new_round(self):
        self.hands_played += 1
        self._reset_round()

    def record_action(self, action_str, street, pip_amount, pot_size):
        if action_str == 'raise':
            if street == 0:
                self._round['pfr'] = True
                self._round['vpip'] = True
                if pip_amount >= STARTING_STACK * 0.85:
                    self._round['preflop_jam'] = True
            else:
                self._round['pb'] += 1
            if pot_size > 0 and pip_amount > 0:
                ratio = pip_amount / pot_size
                self._round['sizes'].append(ratio)
                if street == 3 and ratio <= 0.33:
                    self._round['small_probe_flop'] = True
                elif street == 4 and ratio <= 0.33:
                    self._round['small_probe_turn'] = True
                elif street == 5 and ratio <= 0.33:
                    self._round['small_probe_river'] = True
                if street >= 4 and ratio >= 0.65:
                    self._round['late_big_bet'] = True
        elif action_str == 'call':
            if street == 0:
                self._round['vpip'] = True
            else:
                self._round['pcl'] += 1
        elif action_str == 'check':
            if street > 0:
                self._round['pc'] += 1
                self._round['checked_streets'].add(street)
        elif action_str == 'fold':
            self._round['folded'] = True
            if street > 0:
                self._round['pf'] += 1

    def record_cbet_opportunity(self, did_cbet):
        self._round['cbet_opp'] = True
        self._round['did_cbet'] = did_cbet

    def end_round(self, went_to_showdown, delta, opp_bet_postflop):
        rec = dict(self._round)
        checked_streets = rec.get('checked_streets', set())
        rec['checked_street_count'] = len(checked_streets)
        rec['capped_line'] = (
            (len(checked_streets) >= 1 and (
                rec.get('small_probe_flop')
                or rec.get('small_probe_turn')
                or rec.get('small_probe_river')
            ))
            or (len(checked_streets) >= 2 and not rec.get('late_big_bet'))
        )
        rec.pop('checked_streets', None)
        rec['sd'] = went_to_showdown
        if went_to_showdown:
            self.showdowns += 1
            if opp_bet_postflop:
                if delta > 0:
                    self.postflop_bluffs_detected += 1
                    rec['bluff'] = True
                else:
                    self.postflop_value_detected += 1
                    rec['bluff'] = False
        self.history.append(rec)

    def _stats(self, records):
        n = len(records)
        if n == 0:
            return {'vpip': 0.5, 'pfr': 0.25, 'af': 0.5,
                    'pf_fold': 0.3, 'avg_sz': 0.5, 'cbet': 0.5,
                    'sd_rate': 0.3, 'jam': 0.0, 'small_probe_flop': 0.0,
                    'small_probe_turn': 0.0, 'small_probe_river': 0.0,
                    'late_big_bet': 0.0, 'capped_line': 0.0,
                    'checked_street_count': 0.0, 'n': 0}
        vpip = sum(1 for r in records if r.get('vpip')) / n
        pfr = sum(1 for r in records if r.get('pfr')) / n
        tb = sum(r.get('pb', 0) for r in records)
        tc = sum(r.get('pc', 0) for r in records)
        af = tb / max(1, tb + tc)
        tp = sum(r.get('pb', 0) + r.get('pc', 0) + r.get('pcl', 0) + r.get('pf', 0) for r in records)
        tf = sum(r.get('pf', 0) for r in records)
        pff = tf / max(1, tp)
        szs = []
        for r in records:
            szs.extend(r.get('sizes', []))
        avg_sz = sum(szs) / len(szs) if szs else 0.5
        co = sum(1 for r in records if r.get('cbet_opp'))
        cb = sum(1 for r in records if r.get('did_cbet'))
        cbet = cb / max(1, co) if co >= 3 else 0.5
        sd = sum(1 for r in records if r.get('sd')) / n
        jm = sum(1 for r in records if r.get('preflop_jam')) / n
        spf = sum(1 for r in records if r.get('small_probe_flop')) / n
        spt = sum(1 for r in records if r.get('small_probe_turn')) / n
        spr = sum(1 for r in records if r.get('small_probe_river')) / n
        late_big = sum(1 for r in records if r.get('late_big_bet')) / n
        capped = sum(1 for r in records if r.get('capped_line')) / n
        checked = sum(r.get('checked_street_count', 0) for r in records) / n
        return {'vpip': vpip, 'pfr': pfr, 'af': af, 'pf_fold': pff,
                'avg_sz': avg_sz, 'cbet': cbet, 'sd_rate': sd, 'jam': jm,
                'small_probe_flop': spf, 'small_probe_turn': spt,
                'small_probe_river': spr, 'late_big_bet': late_big,
                'capped_line': capped, 'checked_street_count': checked, 'n': n}

    def get_stats(self, window=None):
        recs = self.history if window is None else self.history[-window:]
        return self._stats(recs)

    def get_blended_stats(self):
        if self.detect_regime_shift():
            r12 = self.get_stats(12)
            r50 = self.get_stats(50)
            bl = {}
            for k in r12:
                if k == 'n':
                    bl[k] = r12[k]
                    continue
                bl[k] = 0.70 * r12[k] + 0.20 * r50[k] + 0.10 * 0.5
            bl['regime_shift'] = True
            return bl
        r50 = self.get_stats(50)
        r100 = self.get_stats(100)
        lt = self.get_stats()
        bl = {}
        for k in r50:
            if k == 'n':
                bl[k] = lt[k]
                continue
            bl[k] = 0.50 * r50[k] + 0.30 * r100[k] + 0.20 * lt[k]
        bl['regime_shift'] = False
        return bl

    def detect_regime_shift(self):
        if len(self.history) < 25:
            return False
        rec = self._stats(self.history[-12:])
        lt = self._stats(self.history)
        return (abs(rec['vpip'] - lt['vpip']) > 0.20 or
                abs(rec['af'] - lt['af']) > 0.20 or
                abs(rec['pfr'] - lt['pfr']) > 0.15)

    def soft_classify(self):
        stats = self.get_blended_stats()
        n = stats['n']
        sc = min(1.0, n / 50.0) if n > 0 else 0.0
        if n < 10:
            return {'unknown': 1.0, 'nit': 0.0, 'tag': 0.0, 'lag': 0.0,
                    'maniac': 0.0, 'station': 0.0, 'foldy': 0.0, 'trapper': 0.0}
        v, p, a = stats['vpip'], stats['pfr'], stats['af']
        pf = stats['pf_fold']
        sd = stats['sd_rate']
        s = {}
        s['nit'] = max(0, (0.25 - v) / 0.15) * sc
        s['tag'] = max(0, min((0.45 - v) / 0.15, (p - 0.15) / 0.10, (a - 0.35) / 0.15)) * sc
        s['lag'] = max(0, min((v - 0.45) / 0.15, (p - 0.20) / 0.10, (a - 0.40) / 0.15)) * sc
        s['maniac'] = max(0, min((v - 0.60) / 0.10, (a - 0.50) / 0.10)) * sc
        s['station'] = max(0, min((v - 0.45) / 0.15, (0.40 - a) / 0.10)) * sc
        s['foldy'] = max(0, (pf - 0.40) / 0.15) * sc
        s['trapper'] = max(0, min((0.35 - a) / 0.10, (sd - 0.35) / 0.10)) * sc
        mx = max(s.values()) if s else 0
        s['unknown'] = max(0, 1.0 - mx * 1.5)
        tot = sum(s.values())
        if tot > 0:
            s = {k: vv / tot for k, vv in s.items()}
        return s

    def get_dominant_profile(self):
        pr = self.soft_classify()
        best = max(pr, key=pr.get)
        return best, pr[best]

    def get_confidence(self):
        pr = self.soft_classify()
        return 1.0 - pr.get('unknown', 1.0)

    def get_fold_to_raise_rate(self):
        if self.raise_faced < 5:
            return 0.3
        return self.folds_to_raise / max(1, self.raise_faced)

    def is_honest_bettor(self):
        t = self.postflop_bluffs_detected + self.postflop_value_detected
        if t < 8:
            return False
        return self.postflop_bluffs_detected / t < 0.20

    def get_postflop_bluff_rate(self):
        t = self.postflop_bluffs_detected + self.postflop_value_detected
        if t < 5:
            return 0.35
        return self.postflop_bluffs_detected / t

    def get_bluff_sample_count(self):
        return self.postflop_bluffs_detected + self.postflop_value_detected

    def get_family_score(self):
        '''Detect the conservative exploit-heavy newplayer2 family.'''
        stats = self.get_blended_stats()
        n = stats.get('n', 0)
        if n < 12:
            return 0.0

        score = 0.0
        vpip = stats.get('vpip', 0.5)
        pfr = stats.get('pfr', 0.25)
        af = stats.get('af', 0.5)
        jam = stats.get('jam', 0.0)
        avg_sz = stats.get('avg_sz', 0.5)
        pf_fold = stats.get('pf_fold', 0.3)
        fold_to_raise = self.get_fold_to_raise_rate()
        bluff_samples = self.get_bluff_sample_count()
        bluff_rate = self.get_postflop_bluff_rate()
        if bluff_samples < 5:
            bluff_rate = 0.5 * bluff_rate + 0.5 * 0.35

        if 0.34 <= vpip <= 0.60:
            score += 0.18
        if 0.15 <= pfr <= 0.42:
            score += 0.14
        if af <= 0.58:
            score += 0.10
        if bluff_rate <= 0.28:
            score += 0.18
        if 0.50 <= avg_sz <= 0.95:
            score += 0.12
        if (stats.get('small_probe_flop', 0.0) + stats.get('small_probe_river', 0.0) >= 0.18 or
                stats.get('capped_line', 0.0) >= 0.18):
            score += 0.12
        if max(pf_fold, fold_to_raise) >= 0.35:
            score += 0.09
        if 0.02 <= jam <= 0.16:
            score += 0.07

        if vpip > 0.65 and af > 0.60:
            score -= 0.20
        if jam > 0.25:
            score -= 0.10
        if bluff_rate > 0.40:
            score -= 0.12
        return max(0.0, min(1.0, score))

    def is_newplayer2_family(self, threshold=0.58):
        return self.get_family_score() >= threshold

    def get_pressure_score(self):
        '''Detect balanced pressure bots such as newplayer3 and similar unknowns.'''
        stats = self.get_blended_stats()
        n = stats.get('n', 0)
        if n < 10:
            return 0.0

        score = 0.0
        vpip = stats.get('vpip', 0.5)
        pfr = stats.get('pfr', 0.25)
        af = stats.get('af', 0.5)
        avg_sz = stats.get('avg_sz', 0.5)
        cbet = stats.get('cbet', 0.5)
        sd = stats.get('sd_rate', 0.3)
        jam = stats.get('jam', 0.0)
        capped = stats.get('capped_line', 0.0)
        checked = stats.get('checked_street_count', 0.0)
        fold_to_raise = self.get_fold_to_raise_rate()
        bluff_samples = self.get_bluff_sample_count()
        bluff_rate = self.get_postflop_bluff_rate()
        if bluff_samples < 5:
            bluff_rate = 0.55 * bluff_rate + 0.45 * 0.28

        if 0.34 <= vpip <= 0.62:
            score += 0.15
        if 0.18 <= pfr <= 0.42:
            score += 0.14
        if 0.44 <= af <= 0.78:
            score += 0.18
        if 0.52 <= cbet <= 0.86:
            score += 0.12
        if 0.14 <= bluff_rate <= 0.42:
            score += 0.12
        if fold_to_raise <= 0.34:
            score += 0.10
        if 0.22 <= sd <= 0.48:
            score += 0.08
        if 0.22 <= avg_sz <= 0.78:
            score += 0.06
        if capped <= 0.16 and checked <= 1.25:
            score += 0.08

        if jam > 0.18:
            score -= 0.08
        if af < 0.32:
            score -= 0.12
        if bluff_rate < 0.12 and avg_sz > 0.68:
            score -= 0.10
        return max(0.0, min(1.0, score))

    def is_pressure_bot(self, threshold=0.56):
        return (not self.is_newplayer2_family(0.64) and
                self.get_pressure_score() >= threshold)

    def is_provisionally_honest_bettor(self):
        '''Detect low-bluff large-sizing opponents faster than the full sample rule.'''
        t = self.postflop_bluffs_detected + self.postflop_value_detected
        if t >= 3:
            return self.postflop_bluffs_detected / max(1, t) < 0.20
        stats = self.get_blended_stats()
        if stats.get('n', 0) < 10:
            return False
        return (stats.get('jam', 0.0) >= 0.04 and
                stats.get('avg_sz', 0.5) >= 0.70 and
                stats.get('vpip', 0.5) <= 0.55 and
                stats.get('pfr', 0.25) <= 0.40)

    def is_baseline_style(self):
        '''Large preflop raises, low observed bluffing, and not especially loose.'''
        stats = self.get_blended_stats()
        if stats.get('n', 0) < 10:
            return False
        if stats.get('jam', 0.0) < 0.04:
            return False
        if stats.get('avg_sz', 0.5) < 0.70:
            return False
        if stats.get('vpip', 0.5) > 0.60:
            return False
        if stats.get('pfr', 0.25) > 0.45:
            return False
        return self.is_honest_bettor() or self.is_provisionally_honest_bettor()


# =====================================================================
#  L7 -- DISCRETE POLARIZED SIZING
# =====================================================================

def select_bet_size(pot, equity, street, texture, profiles, bounty_active,
                    is_value, mixer_val, mn, mx, stack_depth=400):
    if pot <= 0:
        return mn
    if is_value:
        if equity >= 0.85:
            sizes = [0.80, 1.00, 1.25]
        elif equity >= 0.72:
            sizes = [0.66, 0.80, 1.00]
        elif equity >= 0.58:
            sizes = [0.50, 0.66]
        else:
            sizes = [0.33, 0.50]
    else:
        sizes = [0.66, 0.80] if street >= 5 else [0.50, 0.66]
    # Opponent adjustments
    st_c = profiles.get('station', 0)
    ni_c = profiles.get('nit', 0)
    fo_c = profiles.get('foldy', 0)
    if st_c > 0.3 and is_value:
        sizes = [min(1.25, s + 0.15) for s in sizes]
    elif ni_c > 0.3 and is_value:
        sizes = [max(0.25, s - 0.15) for s in sizes]
    elif fo_c > 0.3 and not is_value:
        sizes = [max(0.25, s - 0.10) for s in sizes]
    # Texture
    if texture == 'wet' and is_value:
        sizes = [min(1.25, s + 0.10) for s in sizes]
    elif texture == 'dry' and not is_value:
        sizes = [max(0.25, s - 0.08) for s in sizes]
    if bounty_active and is_value:
        sizes = [min(1.25, s + 0.08) for s in sizes]
    # SPR — short stack shove
    spr = stack_depth / max(1, pot)
    if spr < 2 and is_value and equity >= 0.60:
        return mx
    idx = int(mixer_val * len(sizes)) % len(sizes)
    bet = max(mn, int(pot * sizes[idx]))
    return min(bet, mx)


# =====================================================================
#  BLOCKER LOGIC
# =====================================================================

def has_flush_blocker(my_cards, board_cards):
    if not board_cards or len(board_cards) < 3:
        return False
    sc = {}
    for c in board_cards:
        if len(c) > 1:
            sc[c[1]] = sc.get(c[1], 0) + 1
    if not sc:
        return False
    ds = max(sc, key=sc.get)
    if sc[ds] < 3:
        return False
    return any(len(c) > 1 and c[1] == ds and c[0] in ('A', 'K') for c in my_cards)


def has_top_pair_blocker(my_cards, board_cards):
    if not board_cards:
        return False
    bmax = max(RANK_VALUE.get(c[0], 0) for c in board_cards)
    bchar = VALUE_RANK.get(bmax, '')
    return any(c[0] == bchar for c in my_cards)


def late_street_hand_flags(my_cards, board_cards, hcat, pct):
    if not board_cards:
        return {
            'weak_top_pair': False,
            'vulnerable_two_pair': False,
            'vulnerable_trips': False,
            'low_two_pair': False,
            'paired_two_pair': False,
            'fragile_trips': False,
            'thin_value': False,
        }

    board_vals = [RANK_VALUE.get(c[0], 0) for c in board_cards]
    hole_vals = [RANK_VALUE.get(c[0], 0) for c in my_cards]
    board_counts = {}
    for value in board_vals:
        board_counts[value] = board_counts.get(value, 0) + 1

    top_board = max(board_vals, default=0)
    pair_mult = max(board_counts.values(), default=1)
    shared = [v for v in hole_vals if v in board_counts]
    top_hits = [v for v in hole_vals if v == top_board]
    kicker = max((v for v in hole_vals if v != top_board), default=0)

    weak_top_pair = (
        len(top_hits) == 1
        and len(shared) == 1
        and kicker <= 10
        and pct < 0.78
    )

    vulnerable_two_pair = (
        hcat == 'two_pair'
        and pair_mult >= 2
        and pct < 0.86
    )
    low_two_pair = (
        hcat == 'two_pair'
        and len(set(shared)) >= 2
        and max(shared, default=0) < top_board
        and pct < 0.90
    )
    paired_two_pair = (
        hcat == 'two_pair'
        and pair_mult >= 2
        and pct < 0.95
    )

    paired_rank = next((rank for rank, count in board_counts.items() if count >= 2), None)
    trip_kicker = 14
    if paired_rank is not None and paired_rank in hole_vals:
        others = [v for v in hole_vals if v != paired_rank]
        trip_kicker = max(others, default=14)

    vulnerable_trips = (
        hcat == 'trips'
        and pair_mult >= 2
        and trip_kicker <= 11
        and pct < 0.92
    )
    fragile_trips = (
        hcat == 'trips'
        and pair_mult >= 2
        and trip_kicker <= 12
        and pct < 0.97
    )

    thin_value = pct < 0.72 and (
        pair_mult >= 2 or analyze_board_texture(board_cards) == 'wet'
    )

    return {
        'weak_top_pair': weak_top_pair,
        'vulnerable_two_pair': vulnerable_two_pair,
        'vulnerable_trips': vulnerable_trips,
        'low_two_pair': low_two_pair,
        'paired_two_pair': paired_two_pair,
        'fragile_trips': fragile_trips,
        'thin_value': thin_value,
    }


# =====================================================================
#  MAIN PLAYER CLASS
# =====================================================================

class Player(Bot):
    '''Family-first exploit bot derived from newplayer3.'''

    def __init__(self):
        self.opp = WindowedOpponentModel()
        self.bounty_tracker = BayesianBountyTracker()
        self.mixer = ISMixer()
        self.round_num = 0
        self.my_active = 0
        self.did_raise_preflop = False
        self.committed_to_draw = False
        self.slow_play = False
        self._eq_cache = {}
        self._str_cache = {}
        self._opp_raised_pf = False
        self._last_opp_bet_st = -1
        self._opp_chk_sts = set()
        self._last_opp_key = None
        self._we_raised_pf = False
        self._opp_bet_pf = False
        self._is_sb = False
        self._bk_hist = []
        self._our_raise_count_street = {}
        self._opp_raise_count_street = {}
        self._family_mode = False
        self._pressure_mode = False

    def handle_new_round(self, game_state, round_state, active):
        self.my_active = active
        self.round_num = game_state.round_num
        self.did_raise_preflop = False
        self.committed_to_draw = False
        self.slow_play = False
        self._eq_cache = {}
        self._str_cache = {}
        self._opp_raised_pf = False
        self._last_opp_bet_st = -1
        self._opp_chk_sts = set()
        self._last_opp_key = None
        self._we_raised_pf = False
        self._opp_bet_pf = False
        self._is_sb = (active == 0)
        self._our_raise_count_street = {}
        self._opp_raise_count_street = {}
        self._family_mode = False
        self._pressure_mode = False
        if (self.round_num - 1) % ROUNDS_PER_BOUNTY == 0:
            self.bounty_tracker.reset()
        self.opp.new_round()
        self.mixer.new_round(self.round_num, round_state.hands[active])

    def handle_round_over(self, game_state, terminal_state, active):
        prev = terminal_state.previous_state
        bhits = terminal_state.bounty_hits
        delta = terminal_state.deltas[active]
        self._bk_hist.append(delta)
        went_sd = False
        opp_cr = None
        board_ranks = set()
        if isinstance(prev, RoundState):
            oc = prev.hands[1 - active]
            went_sd = oc and len(oc) >= 2 and len(oc[0]) >= 2
            if went_sd:
                opp_cr = set(c[0] for c in oc if c)
            bd = prev.deck
            if bd:
                board_ranks = set(c[0] for c in bd if c and len(c) >= 1)
            if self.did_raise_preflop or self._we_raised_pf:
                self.opp.raise_faced += 1
                if not went_sd and delta > 0:
                    self.opp.folds_to_raise += 1
                    self.opp.record_action('fold', prev.street, 0, 0)
            if self._opp_raised_pf and prev.street >= 3:
                self.opp.record_cbet_opportunity(self._last_opp_bet_st == 3)
            self.opp.end_round(went_sd, delta, self._opp_bet_pf)
        if bhits is not None:
            opp_hit = bhits[1 - active]
            self.bounty_tracker.update(opp_hit, board_ranks, opp_cr)

    # --- caching ---
    def _ckey(self, cards):
        return tuple(str(c) for c in cards if c)

    def _ceq(self, cards, board, iters, rng='default'):
        k = (self._ckey(cards), self._ckey(board), iters, rng)
        if k not in self._eq_cache:
            if len(board) == 5:
                self._eq_cache[k] = exact_range_weighted_river_equity(
                    cards, board, opp_tightness=rng)
            else:
                self._eq_cache[k] = range_weighted_equity(
                    cards, board, opp_tightness=rng, iterations=iters)
        return self._eq_cache[k]

    def _cstr(self, cards, board):
        k = (self._ckey(cards), self._ckey(board))
        if k not in self._str_cache:
            self._str_cache[k] = classify_hand_strength(cards, board)
        return self._str_cache[k]

    def _bk_trend(self, w=50):
        if len(self._bk_hist) < 10:
            return 0
        return sum(self._bk_hist[-min(w, len(self._bk_hist)):])

    def _opp_range(self):
        prof, conf = self.opp.get_dominant_profile()
        if conf < 0.3:
            return 'default'
        return prof if prof in RANGE_THRESHOLDS else 'default'

    # =================================================================
    #  CORE DECISION ENGINE
    # =================================================================

    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        street = round_state.street
        cards = round_state.hands[active]
        board = round_state.deck[:street]
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]
        cost = opp_pip - my_pip
        pot = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)
        clock = game_state.game_clock
        bankroll = game_state.bankroll
        my_bounty = round_state.bounties[active]
        is_ip = self._is_sb
        is_bb = not self._is_sb

        # L13: Clock safety
        if clock < 1.5:
            return self._emergency(legal, cards, cost)
        if clock < 3.0:
            return self._clock_safe(legal, cards, board, cost, pot)

        # Opponent action tracking
        okey = (opp_pip, street, round_state.button)
        if okey != self._last_opp_key:
            if street == 0:
                if opp_pip > BIG_BLIND and not self._opp_raised_pf:
                    self.opp.record_action('raise', 0, opp_pip, pot)
                    self._opp_raised_pf = True
                elif cost == 0 and opp_pip <= BIG_BLIND and is_bb:
                    self.opp.record_action('call', 0, opp_pip, pot)
            else:
                if cost > 0 and self._last_opp_bet_st != street:
                    self.opp.record_action('raise', street, opp_pip, pot)
                    self._last_opp_bet_st = street
                    self._opp_bet_pf = True
                    self._opp_raise_count_street[street] = (
                        self._opp_raise_count_street.get(street, 0) + 1
                    )
                elif cost == 0 and street not in self._opp_chk_sts:
                    self.opp.record_action('check', street, 0, pot)
                    self._opp_chk_sts.add(street)
            self._last_opp_key = okey

        # Profiling
        profiles = self.opp.soft_classify()
        opp_dom, opp_conf = self.opp.get_dominant_profile()
        regime = self.opp.detect_regime_shift()
        model_conf = self.opp.get_confidence()
        provisional_honest = self.opp.is_provisionally_honest_bettor()
        baseline_safe = self.opp.is_baseline_style()
        family_score = self.opp.get_family_score()
        family_mode = self.opp.is_newplayer2_family()
        pressure_score = self.opp.get_pressure_score()
        pressure_mode = (not family_mode) and self.opp.is_pressure_bot()
        self._family_mode = family_mode
        self._pressure_mode = pressure_mode
        cc = canonical_card_class(cards)
        pb = pot_bucket(pot)
        pi = 1 if is_ip else 0

        # Equity
        if street == 0:
            equity = get_preflop_equity(cards)
        else:
            big = cost > 0.12 * STARTING_STACK or pot > 0.25 * STARTING_STACK
            medium = cost > 0.06 * STARTING_STACK or pot > 0.14 * STARTING_STACK
            if EVAL7_AVAILABLE and len(board) == 5 and clock > 4.0:
                equity = self._ceq(cards, board, 0, self._opp_range())
            elif big and EVAL7_AVAILABLE and clock > 5.0:
                it = 320
                if clock < 8.0:
                    it = 110
                elif clock < 15.0:
                    it = 170
                elif clock < 25.0:
                    it = 240
                equity = self._ceq(cards, board, it, self._opp_range())
            elif medium and EVAL7_AVAILABLE and clock > 7.0:
                it = 120 if clock < 12.0 else 170
                equity = self._ceq(cards, board, it, self._opp_range())
            else:
                equity = quick_postflop_equity(cards, board)

        # Hand strength
        if street > 0 and EVAL7_AVAILABLE and clock > 8.0:
            hcat, hscore, pct = self._cstr(cards, board)
        else:
            hcat = 'unknown'
            pct = equity
            if street > 0:
                hcat = quick_hand_class(cards, board)
                if hcat == 'monster':
                    pct = max(pct, 0.90)
                elif hcat == 'very_strong':
                    pct = max(pct, 0.80)
                elif hcat == 'strong':
                    pct = max(pct, 0.65)
                elif hcat == 'overcards':
                    pct = max(pct, 0.42)

        # Bounty
        has_b = cards[0][0] == my_bounty or cards[1][0] == my_bounty
        b_board = any(c[0] == my_bounty for c in board) if board else False
        b_active = has_b or b_board
        opp_b_danger = self.bounty_tracker.board_likely_hits_opponent(board) if board else False

        pot_od = bounty_pot_odds(cost, pot, b_active)
        b_boost = 0.0
        if cost == 0 and b_active:
            b_boost = min(0.15, BOUNTY_CONSTANT / (pot + 20))
        if opp_b_danger:
            b_boost -= 0.03
        eff_eq = min(0.96, max(0.05, equity + b_boost))

        # Calling threshold
        c_bonus = 0.0
        if street > 0 and cost > 0:
            pi_ratio = pot / STARTING_STACK
            if pi_ratio > 0.50:
                if opp_dom in ('maniac', 'lag'):
                    c_bonus = 0.0
                elif opp_dom in ('nit', 'tag') and opp_conf > 0.2:
                    c_bonus = min(0.08, (pi_ratio - 0.50) * 0.15)
                else:
                    c_bonus = min(0.05, (pi_ratio - 0.50) * 0.10)

        mdf = minimum_defense_frequency(pot, cost) if cost > 0 else 1.0
        tex = analyze_board_texture(board) if street > 0 else 'neutral'

        mn = mx = 0
        if RaiseAction in legal:
            mn, mx = round_state.raise_bounds()

        phase = 'early' if self.round_num < 200 else ('mid' if self.round_num < 500 else 'late')
        bk_tr = self._bk_trend(50)
        losing = bk_tr < -200
        winning = bankroll > 1500
        protect_lead = phase == 'late' and bankroll > 80
        qualification_mode = ((baseline_safe and not family_mode) or
                              (provisional_honest and protect_lead))

        base_w = 0.0
        if regime:
            base_w = 0.5
        elif model_conf < 0.3:
            base_w = 0.3

        if street == 0:
            act = self._preflop(legal, cards, my_bounty, eff_eq, pot_od, cost,
                                pot, my_pip, opp_pip, my_stack, opp_stack,
                                bankroll, b_active, profiles, opp_dom, opp_conf,
                                is_bb, mn, mx, phase, cc, pb, pi, losing,
                                qualification_mode, family_mode, family_score,
                                pressure_mode, pressure_score)
            if isinstance(act, RaiseAction):
                self.did_raise_preflop = True
                self._our_raise_count_street[street] = (
                    self._our_raise_count_street.get(street, 0) + 1
                )
            return act
        else:
            bd_score = board_danger_score(board)
            act = self._postflop(legal, cards, board, my_bounty, eff_eq, pct,
                                 pot_od, cost, pot, my_pip, opp_pip, my_stack,
                                 opp_stack, bankroll, b_active, opp_b_danger,
                                 profiles, opp_dom, opp_conf, is_bb, mn, mx,
                                 hcat, street, phase, clock, mdf, tex, is_ip,
                                 c_bonus, cc, pb, pi, base_w, losing, winning,
                                 model_conf, qualification_mode,
                                 provisional_honest, bd_score, family_mode,
                                 family_score, pressure_mode, pressure_score)
            if isinstance(act, RaiseAction):
                self._we_raised_pf = True
                self._our_raise_count_street[street] = (
                    self._our_raise_count_street.get(street, 0) + 1
                )
            return act

    # =================================================================
    #  PREFLOP
    # =================================================================

    def _preflop(self, legal, cards, bounty, eq, pot_od, cost, pot,
                 my_pip, opp_pip, my_stack, opp_stack, bankroll,
                 b_active, pr, opp_dom, opp_conf, is_bb, mn, mx,
                 phase, cc, pb, pi, losing, qualification_mode,
                 family_mode, family_score, pressure_mode, pressure_score):
        high = max(RANK_VALUE.get(cards[0][0], 5), RANK_VALUE.get(cards[1][0], 5))
        low = min(RANK_VALUE.get(cards[0][0], 5), RANK_VALUE.get(cards[1][0], 5))
        is_pair = cards[0][0] == cards[1][0]
        nit_c = pr.get('nit', 0)
        st_c = pr.get('station', 0)
        man_c = pr.get('maniac', 0)
        lag_c = pr.get('lag', 0)
        fol_c = pr.get('foldy', 0)
        jam_tier = preflop_allin_call_tier(cards)

        premium = eq >= 0.66
        strong = eq >= 0.56
        medium = eq >= (0.44 if b_active else 0.48)
        playable = eq >= (0.38 if b_active else 0.42)

        # --- OPEN ---
        if cost == 0:
            # Steal vs nits
            if nit_c > 0.25 and not is_bb and not premium and not strong:
                m = self.mixer.mix(0, pi, cc, 0, pb, 1)
                freq = 0.60 + nit_c * 0.20
                if eq >= 0.35 and RaiseAction in legal and m < freq:
                    return RaiseAction(mn)
            if family_mode and not is_bb and not premium and RaiseAction in legal:
                m = self.mixer.mix(0, pi, cc, 0, pb, 11)
                fam_freq = 0.28 + 0.22 * family_score
                if eq >= 0.33 and m < fam_freq:
                    return RaiseAction(min(max(mn, my_pip + 6), mx))
            if premium:
                if RaiseAction in legal:
                    if st_c > 0.3:
                        t = max(mn, my_pip + 30)
                    elif nit_c > 0.3:
                        t = max(mn, my_pip + 14)
                    elif man_c > 0.3:
                        t = max(mn, my_pip + 18)
                    else:
                        t = max(mn, my_pip + 22)
                    return RaiseAction(min(t, mx))
                return CheckAction()
            if strong:
                if RaiseAction in legal:
                    t = max(mn, my_pip + 16)
                    if man_c > 0.3:
                        t = max(mn, my_pip + 12)
                    elif st_c > 0.3:
                        t = max(mn, my_pip + 20)
                    return RaiseAction(min(t, mx))
                return CheckAction()
            if medium:
                if RaiseAction in legal:
                    m = self.mixer.mix(0, pi, cc, 0, pb, 2)
                    freq = 0.55 + (0.10 * family_score if family_mode and not is_bb else 0.0)
                    if is_bb or m < min(0.80, freq):
                        return RaiseAction(min(max(mn, my_pip + 8), mx))
                return CheckAction()
            if playable:
                m = self.mixer.mix(0, pi, cc, 0, pb, 3)
                if (nit_c > 0.2 or fol_c > 0.2) and RaiseAction in legal and m < 0.35:
                    return RaiseAction(mn)
                if b_active and st_c < 0.3 and RaiseAction in legal and m < 0.40:
                    return RaiseAction(mn)
                if family_mode and not is_bb and RaiseAction in legal and m < (0.20 + 0.12 * family_score):
                    return RaiseAction(min(max(mn, my_pip + 6), mx))
                return CheckAction()
            # Trash: SB steal or check
            if not is_bb:
                m = self.mixer.mix(0, pi, cc, 0, pb, 4)
                steal_freq = 0.25 + (0.10 * family_score if family_mode else 0.0)
                if eq >= 0.36 and RaiseAction in legal and m < steal_freq:
                    return RaiseAction(min(max(mn, my_pip + 6), mx))
            return CheckAction()

        # --- FACING A RAISE ---
        if cost > 0:
            # Handle true near-all-in pressure with a continuous threshold.
            if cost >= STARTING_STACK * 0.80:
                jam_rate = 0.05
                if self.opp.hands_played >= 10:
                    jam_rate = self.opp.get_blended_stats().get('jam', 0.05)
                jam_th = adaptive_jam_threshold(
                    jam_rate, b_active, bankroll, phase, family_mode)
                if (eq >= jam_th or
                        (self.opp.hands_played < 10 and jam_tier >= 2) or
                        (family_mode and jam_tier >= 1 and eq >= jam_th - 0.03)):
                    if CallAction in legal:
                        return CallAction()
                if FoldAction in legal:
                    return FoldAction()
                return CallAction() if CallAction in legal else CheckAction()

            if premium:
                if RaiseAction in legal and my_stack > cost * 2:
                    t = max(mn, opp_pip * 3)
                    if is_pair and high >= 13:
                        t = max(mn, opp_pip * 3 + 10)
                    return RaiseAction(min(max(t, mn), mx))
                if cost >= STARTING_STACK * 0.85:
                    if eq >= 0.60 and CallAction in legal:
                        return CallAction()
                    if FoldAction in legal:
                        return FoldAction()
                if CallAction in legal:
                    return CallAction()

            if cost >= 20 and jam_tier == 0 and not family_mode:
                if FoldAction in legal:
                    return FoldAction()
                return CallAction()

            if strong:
                if cost <= 15:
                    if RaiseAction in legal and my_stack > cost * 3:
                        if man_c > 0.2 or lag_c > 0.2:
                            t = max(mn, opp_pip * 3)
                            return RaiseAction(min(max(t, mn), mx))
                        if CallAction in legal:
                            return CallAction()
                    if CallAction in legal:
                        return CallAction()
                elif cost <= 40:
                    if CallAction in legal:
                        return CallAction()
                if FoldAction in legal:
                    return FoldAction()
                return CallAction()

            if medium:
                th = 18 if b_active else 12
                if man_c > 0.2 or lag_c > 0.2:
                    th += 8
                if losing:
                    th = max(8, th - 4)
                if family_mode and cost <= 16:
                    th += 4
                if pressure_mode and cost <= 18:
                    th += 3 + (2 if not is_bb else 0)
                if cost <= th:
                    if CallAction in legal:
                        return CallAction()
                if FoldAction in legal:
                    return FoldAction()
                return CallAction()

            if playable:
                th = 8 if b_active else 5
                if is_pair:
                    th = 15
                if family_mode and cost <= 10:
                    th += 2
                if pressure_mode and (high >= 11 or low >= 9 or cards[0][1] == cards[1][1]):
                    th += 2
                if cost <= th:
                    if CallAction in legal:
                        return CallAction()
                if FoldAction in legal:
                    return FoldAction()
                return CallAction()

            # SB trash
            if cost <= 1 and not is_bb:
                if eq >= 0.36 and RaiseAction in legal:
                    return RaiseAction(min(max(mn, my_pip + 6), mx))
                if FoldAction in legal:
                    return FoldAction()
                return CallAction()
            if cost <= 1 and CallAction in legal:
                return CallAction()
            if FoldAction in legal:
                return FoldAction()
            return CallAction()

        return CheckAction() if CheckAction in legal else CallAction()

    # =================================================================
    #  POSTFLOP
    # =================================================================

    def _postflop(self, legal, cards, board, bounty, eq, pct, pot_od,
                  cost, pot, my_pip, opp_pip, my_stack, opp_stack,
                  bankroll, b_active, opp_bd, pr, opp_dom, opp_conf,
                  is_bb, mn, mx, hcat, street, phase, clock, mdf,
                  tex, is_ip, c_bonus, cc, pb, pi, base_w, losing,
                  winning, model_conf, qualification_mode,
                  provisional_honest, bd_score, family_mode,
                  family_score, pressure_mode, pressure_score):
        nit_c = pr.get('nit', 0)
        man_c = pr.get('maniac', 0)
        lag_c = pr.get('lag', 0)
        st_c = pr.get('station', 0)
        fol_c = pr.get('foldy', 0)
        is_nit = nit_c > 0.25 or (opp_dom == 'nit' and opp_conf > 0.2)
        is_agg = man_c > 0.2 or lag_c > 0.2
        is_sta = st_c > 0.25 or (opp_dom == 'station' and opp_conf > 0.2)
        is_fol = fol_c > 0.2 or (self.opp.raise_faced >= 5 and self.opp.get_fold_to_raise_rate() > 0.55)

        # Tier
        if hcat in ('monster', 'straight_flush', 'quads', 'full_house', 'trips'):
            tier = 'monster'
        elif hcat in ('very_strong', 'flush', 'straight', 'two_pair'):
            tier = 'very_strong'
        elif hcat == 'strong' or pct >= 0.72:
            tier = 'strong'
        elif hcat == 'medium' or pct >= 0.55:
            tier = 'medium'
        elif hcat == 'overcards':
            tier = 'overcards'
        elif hcat == 'drawing':
            tier = 'drawing'
        elif pct >= 0.45:
            tier = 'marginal'
        else:
            tier = 'weak'
        if pct >= 0.88 and tier != 'monster':
            tier = 'monster'
        elif pct >= 0.78 and tier in ('medium', 'overcards', 'drawing', 'marginal', 'weak'):
            tier = 'very_strong'

        # Never let one-pair hands masquerade as monsters just because their
        # hot-board percentile runs high against random ranges.
        if hcat in ('pair', 'strong') and tier in ('monster', 'very_strong'):
            tier = 'strong'

        # L12: Draw commitment
        if tier == 'drawing' and street == 3:
            outs = 9 if 'flush' in str(hcat) else 8
            if min(0.60, outs * 0.042) > 0.30:
                self.committed_to_draw = True

        # L8: Slow-play
        if tier == 'monster' and not self.slow_play:
            m = self.mixer.mix(street, pi, cc, 0, pb, 10)
            if street == 5:
                if is_agg and m < 0.40:
                    self.slow_play = True
                elif m < 0.12:
                    self.slow_play = True
            elif street >= 4 and tex == 'dry':
                if is_agg and m < 0.30:
                    self.slow_play = True

        adj_po = pot_od + c_bonus
        mdf_def = pct >= (1.0 - mdf) if cost > 0 else True
        fb = has_flush_blocker(cards, board) if street >= 3 else False
        tb = has_top_pair_blocker(cards, board) if street >= 3 else False
        late_flags = late_street_hand_flags(cards, board, hcat, pct)
        paired_board = board_is_paired(board)
        post_raise_pot_gate = STARTING_STACK * (0.30 if pressure_mode else 0.45)
        post_raise_trap = (
            street >= 4 and
            (hcat in ('pair', 'strong') or
             (pressure_mode and hcat in ('two_pair', 'trips') and
              (paired_board or bd_score >= 1.2))) and
            pot >= post_raise_pot_gate and
            self._opp_raise_count_street.get(street - 1, 0) > 0
        )

        if pressure_mode and paired_board:
            if hcat in ('pair', 'strong') and tier == 'strong':
                tier = 'medium'
            elif hcat == 'two_pair' and late_flags['paired_two_pair']:
                if tier == 'monster':
                    tier = 'very_strong'
                elif tier == 'very_strong':
                    tier = 'strong'
            elif hcat == 'trips' and late_flags['fragile_trips'] and tier == 'monster':
                tier = 'very_strong'

        if street >= 4:
            if late_flags['weak_top_pair']:
                if tier == 'strong':
                    tier = 'medium'
                elif tier == 'medium':
                    tier = 'marginal'
            if late_flags['vulnerable_two_pair']:
                if tier == 'very_strong':
                    tier = 'strong'
                elif tier == 'strong':
                    tier = 'medium'
            if late_flags['vulnerable_trips'] and tier == 'monster':
                tier = 'very_strong'
            if street == 5 and late_flags['thin_value']:
                if tier == 'strong' and pct < 0.70:
                    tier = 'medium'
                elif tier == 'medium':
                    tier = 'marginal'
            if hcat in ('pair', 'strong') and bd_score >= 3.0 and tier == 'strong':
                tier = 'medium'

        can_reraise = (
            self._our_raise_count_street.get(street, 0) <
            max_reraise_by_tier(tier)
        )
        if self._our_raise_count_street.get(street, 0) >= 1:
            if late_flags['vulnerable_trips'] or late_flags['low_two_pair']:
                can_reraise = False
            if late_flags['paired_two_pair'] or late_flags['fragile_trips']:
                can_reraise = False
        if pressure_mode and paired_board and self._our_raise_count_street.get(street, 0) >= 1:
            if hcat in ('pair', 'strong', 'two_pair', 'trips'):
                can_reraise = False
        if (pressure_mode and street == 5 and bd_score >= 2.2 and
                self._our_raise_count_street.get(street, 0) >= 1 and
                tier in ('strong', 'very_strong')):
            can_reraise = False

        if cost == 0:
            return self._pf_nobet(
                legal, cards, board, eq, pct, pot, my_stack, b_active,
                mn, mx, street, tier, tex, is_ip, is_nit, is_agg,
                is_sta, is_fol, cc, pb, pi, pr, model_conf, base_w,
                losing, fb, tb, qualification_mode, late_flags,
                family_mode, family_score, pressure_mode, pressure_score,
                post_raise_trap, paired_board)
        else:
            return self._pf_bet(
                legal, cards, board, eq, pct, pot_od, adj_po, cost, pot,
                my_pip, opp_pip, my_stack, opp_stack, bankroll, b_active,
                opp_bd, mn, mx, street, tier, tex, is_ip, mdf, mdf_def,
                is_nit, is_agg, is_sta, is_fol, cc, pb, pi, pr, hcat,
                model_conf, base_w, losing, winning, phase, fb, tb,
                qualification_mode, provisional_honest, late_flags,
                bd_score, can_reraise, family_mode, family_score,
                pressure_mode, pressure_score, paired_board)

    # ----- No Bet -----

    def _pf_nobet(self, legal, cards, board, eq, pct, pot, stk, b_act,
                  mn, mx, street, tier, tex, is_ip, is_nit, is_agg,
                  is_sta, is_fol, cc, pb, pi, pr, mc, bw, losing, fb, tb,
                  qualification_mode, late_flags, family_mode, family_score,
                  pressure_mode, pressure_score, post_raise_trap,
                  paired_board):
        mb = self.mixer.mix(street, pi, cc, hash(tex) % 5, pb, 20)
        mbl = self.mixer.mix(street, pi, cc, hash(tex) % 5, pb, 21)
        ms = self.mixer.mix(street, pi, cc, hash(tex) % 5, pb, 22)
        capped_line_pressure = (
            family_mode or
            self.opp.get_blended_stats().get('capped_line', 0.0) >= 0.18
        )
        qual_bluff_mult = 0.85 if qualification_mode and family_mode else (
            0.55 if qualification_mode else 1.0
        )

        def bet(eq_s=None):
            if RaiseAction not in legal:
                return CheckAction()
            e = eq_s if eq_s is not None else eq
            b = select_bet_size(pot, e, street, tex, pr, b_act,
                                is_value=(e >= 0.55), mixer_val=ms,
                                mn=mn, mx=mx, stack_depth=stk)
            return RaiseAction(b)

        # Monster
        if tier == 'monster':
            if self.slow_play and street <= 4:
                self.slow_play = False
                return CheckAction()
            return bet()

        # Very strong
        if tier == 'very_strong':
            if (pressure_mode and street >= 4 and paired_board and
                    (late_flags['paired_two_pair'] or late_flags['fragile_trips'])):
                return CheckAction()
            if b_act and RaiseAction in legal:
                frac = 0.45 if qualification_mode and street == 5 else 0.70
                b = max(mn, int(pot * frac))
                return RaiseAction(min(b, mx))
            if qualification_mode and street == 5 and tex == 'wet':
                return CheckAction()
            return bet()

        # Strong
        if tier == 'strong':
            if post_raise_trap:
                return CheckAction()
            if pressure_mode and paired_board and street >= 4:
                return CheckAction()
            if street == 5 and late_flags['thin_value']:
                return CheckAction()
            if RaiseAction in legal and stk > 20:
                if is_agg and is_ip and mb < 0.35:
                    return CheckAction()
                return bet()
            return CheckAction()

        # Medium
        if tier == 'medium':
            if post_raise_trap:
                return CheckAction()
            if street >= 4 and (
                late_flags['weak_top_pair']
                or late_flags['vulnerable_two_pair']
                or late_flags['thin_value']
            ):
                return CheckAction()
            if street == 3 and eq > 0.55 and RaiseAction in legal:
                if is_nit:
                    b = max(mn, int(pot * 0.33))
                    return RaiseAction(min(b, mx))
                return bet()
            if street == 4 and capped_line_pressure and eq > 0.47 and RaiseAction in legal:
                return bet(eq_s=max(eq, 0.46))
            if street == 5 and eq > 0.58 and pct >= 0.60 and RaiseAction in legal:
                sz = 0.55 if is_sta else 0.40
                b = max(mn, int(pot * sz))
                return RaiseAction(min(b, mx))
            if is_agg and eq > 0.55 and mb < 0.25:
                return CheckAction()
            return CheckAction()

        # Overcards
        if tier == 'overcards':
            if street <= 4 and RaiseAction in legal:
                obp = 0.18
                if is_nit:
                    obp = 0.08
                if is_sta:
                    obp = 0.05
                elif is_fol:
                    obp = min(0.35, obp + 0.12)
                if capped_line_pressure:
                    obp = min(0.40, obp + 0.10 + 0.08 * family_score)
                obp *= qual_bluff_mult
                if mbl < obp:
                    return bet(eq_s=0.35)
            return CheckAction()

        # Drawing
        if tier == 'drawing':
            if street <= 4 and RaiseAction in legal:
                sbp = 0.35 + max(0.0, min(1.0, (eq - 0.20) / 0.25)) * 0.40
                if is_nit:
                    sbp *= 0.30
                if is_sta:
                    sbp *= 0.40
                sbp *= (1.0 - bw * 0.5)
                if self.committed_to_draw or mbl < sbp:
                    return bet(eq_s=0.40)
            return CheckAction()

        # Marginal
        if tier == 'marginal':
            ow = street in self._opp_chk_sts
            ovw = len(self._opp_chk_sts) >= 2
            bp = 0.12 + max(0.0, min(1.0, (eq - 0.35) / 0.10)) * 0.12
            if is_nit:
                bp = 0.06
            if is_sta:
                bp = 0.02
            elif is_fol:
                bp = min(0.30, bp + 0.10)
            if ovw:
                bp = min(0.40, bp + 0.12)
            if mc < 0.3:
                bp = min(bp, 0.15)
            if losing:
                bp *= 0.5
            if capped_line_pressure:
                bp = min(0.42, bp + 0.10 + 0.08 * family_score)
            bp *= qual_bluff_mult
            bp *= (1.0 - bw * 0.4)
            if (ow or is_fol) and mbl < bp and RaiseAction in legal:
                return bet(eq_s=0.30)
            return CheckAction()

        # Weak
        oc = street in self._opp_chk_sts
        omc = len(self._opp_chk_sts) >= 2
        bp = 0.06
        if is_sta:
            bp = 0.01
        elif is_fol:
            bp = 0.12
        if omc:
            bp = min(0.25, bp + 0.10)
        if street == 5 and (fb or tb):
            bp = min(0.30, bp + 0.08)
        if mc < 0.3:
            bp = min(bp, 0.10)
        if losing:
            bp *= 0.4
        if capped_line_pressure:
            bp = min(0.28, bp + 0.06 + 0.08 * family_score)
        bp *= qual_bluff_mult
        bp *= (1.0 - bw * 0.4)
        if oc and mbl < bp and RaiseAction in legal:
            return bet(eq_s=0.25)
        return CheckAction()

    # ----- Facing Bet -----

    def _pf_bet(self, legal, cards, board, eq, pct, pot_od, adj_po,
                cost, pot, my_pip, opp_pip, my_stack, opp_stack, bankroll,
                b_act, opp_bd, mn, mx, street, tier, tex, is_ip, mdf,
                mdf_def, is_nit, is_agg, is_sta, is_fol, cc, pb, pi,
                pr, hcat, mc, bw, losing, winning, phase, fb, tb,
                qualification_mode, provisional_honest, late_flags,
                bd_score, can_reraise, family_mode, family_score,
                pressure_mode, pressure_score, paired_board):
        mr = self.mixer.mix(street, pi, cc, hash(tex) % 5, pb, 30)
        mcl = self.mixer.mix(street, pi, cc, hash(tex) % 5, pb, 31)
        ms = self.mixer.mix(street, pi, cc, hash(tex) % 5, pb, 32)
        cost_ratio = cost / max(1, pot)
        large_pressure = street >= 4 and cost > max(18, pot * 0.40)
        huge_pressure = street >= 4 and cost > max(40, pot * 0.75)
        small_probe_defense = street <= 4 and cost_ratio <= 0.30

        # Small-bet defense and family punishment: do not let one-third-pot
        # probes print money, especially once the opponent looks like newplayer2.
        if small_probe_defense:
            if pressure_mode and paired_board:
                if tier == 'monster' and hcat == 'trips' and late_flags['fragile_trips']:
                    if CallAction in legal:
                        return CallAction()
                if tier in ('very_strong', 'strong'):
                    if CallAction in legal:
                        return CallAction()
            if tier == 'monster':
                if RaiseAction in legal and can_reraise:
                    sz = 1.0 if is_sta else 0.70
                    t = max(mn, opp_pip + int(pot * sz))
                    return RaiseAction(min(t, mx))
                if CallAction in legal:
                    return CallAction()
            if tier in ('very_strong', 'strong') and RaiseAction in legal and can_reraise:
                if cost_ratio < 0.18 or family_mode:
                    t = max(mn, opp_pip + int(pot * (0.55 if family_mode else 0.50)))
                    return RaiseAction(min(t, mx))
            if tier in ('medium', 'drawing', 'overcards'):
                if CallAction in legal:
                    return CallAction()
            if tier == 'marginal' and (family_mode or is_agg) and eq > adj_po * 0.80:
                if CallAction in legal:
                    return CallAction()

        # Monster
        if tier == 'monster':
            if pressure_mode and paired_board and hcat == 'trips' and late_flags['fragile_trips']:
                if CallAction in legal:
                    return CallAction()
            if RaiseAction in legal and can_reraise:
                sz = 1.0 if is_sta else 0.70
                t = max(mn, opp_pip + int(pot * sz))
                return RaiseAction(min(t, mx))
            if CallAction in legal:
                return CallAction()

        # Very strong
        if tier == 'very_strong':
            if paired_board and (late_flags['paired_two_pair'] or late_flags['fragile_trips']):
                if CallAction in legal:
                    return CallAction()
            if qualification_mode and provisional_honest and street >= 4 and not family_mode:
                if CallAction in legal:
                    return CallAction()
            if (pressure_mode and (
                    (paired_board and (late_flags['paired_two_pair'] or late_flags['fragile_trips'])) or
                    (street == 5 and self._our_raise_count_street.get(street, 0) >= 1 and bd_score >= 2.2))):
                if CallAction in legal:
                    return CallAction()
            if cost < pot * 0.7 and RaiseAction in legal and can_reraise:
                t = max(mn, opp_pip + int(pot * 0.65))
                return RaiseAction(min(t, mx))
            if CallAction in legal:
                return CallAction()

        # Strong — PRESERVED: never fold to standard river bets
        if tier == 'strong':
            if street >= 4:
                safe_cost = pot * 0.45
                if late_flags['weak_top_pair']:
                    safe_cost = pot * 0.28
                elif late_flags['vulnerable_two_pair'] or late_flags['thin_value']:
                    safe_cost = pot * 0.34
                if is_nit:
                    safe_cost *= 0.85
                need_pct = max(adj_po + 0.08, 0.58)
                if (bd_score >= 4.5 and cost_ratio >= 0.70 and
                        self.opp.get_bluff_sample_count() >= 5 and
                        self.opp.get_postflop_bluff_rate() < 0.28 and
                        pct < 0.80):
                    if FoldAction in legal:
                        return FoldAction()
                if large_pressure and pct < need_pct and cost > safe_cost:
                    if FoldAction in legal:
                        return FoldAction()
                if huge_pressure and pct < max(adj_po + 0.12, 0.72):
                    if FoldAction in legal:
                        return FoldAction()
                if (pressure_mode and hcat in ('pair', 'strong') and
                        self._opp_raise_count_street.get(street - 1, 0) > 0 and
                        cost_ratio >= 0.78 and bd_score >= 1.2 and pct < 0.84):
                    if FoldAction in legal:
                        return FoldAction()
            if eq > adj_po or mdf_def:
                if CallAction in legal:
                    return CallAction()
            if cost <= pot * 0.75:
                if CallAction in legal:
                    return CallAction()
            if street == 5:
                if cost <= pot * 1.5:
                    if CallAction in legal:
                        return CallAction()
            if is_nit and street >= 4 and cost > pot * 0.80:
                if FoldAction in legal:
                    return FoldAction()
            if CallAction in legal:
                return CallAction()
            return FoldAction() if FoldAction in legal else CallAction()

        # Medium
        if tier == 'medium':
            if street >= 4:
                safe_cost = pot * 0.26
                if is_agg:
                    safe_cost = pot * 0.32
                if late_flags['weak_top_pair'] or late_flags['vulnerable_two_pair']:
                    safe_cost = min(safe_cost, pot * 0.18)
                elif late_flags['thin_value']:
                    safe_cost = min(safe_cost, pot * 0.22)
                need_pct = max(adj_po + 0.08, 0.54)
                if (cost > safe_cost or
                        (large_pressure and pct < need_pct) or
                        (bd_score >= 3.5 and self.opp.is_honest_bettor() and cost_ratio >= 0.45)):
                    if FoldAction in legal:
                        return FoldAction()
            if qualification_mode and not family_mode:
                if street == 3:
                    if cost <= pot * 0.25 and eq > adj_po and CallAction in legal:
                        return CallAction()
                    if FoldAction in legal:
                        return FoldAction()
                if street >= 4:
                    if provisional_honest:
                        if eq >= 0.62 and cost <= pot * 0.22 and CallAction in legal:
                            return CallAction()
                        if FoldAction in legal:
                            return FoldAction()
                    if cost <= pot * 0.16 and eq > adj_po and CallAction in legal:
                        return CallAction()
                    if FoldAction in legal:
                        return FoldAction()
            if qualification_mode and family_mode and street >= 4:
                if cost <= pot * 0.24 and eq > adj_po * 0.90 and CallAction in legal:
                    return CallAction()
            if is_nit and street >= 4:
                if cost <= pot * 0.22 and eq > adj_po:
                    if CallAction in legal:
                        return CallAction()
                if FoldAction in legal:
                    return FoldAction()
                return CallAction()
            if winning and phase == 'late' and cost > pot * 0.50:
                if FoldAction in legal:
                    return FoldAction()
            od, _ = self.opp.get_dominant_profile()
            if street >= 4 and od in ('unknown', 'tag'):
                tl = pot * 0.20 if od == 'unknown' else pot * 0.25
                if cost > tl:
                    if self.opp.is_honest_bettor():
                        if FoldAction in legal:
                            return FoldAction()
                    if eq < 0.55:
                        if FoldAction in legal:
                            return FoldAction()
            if self.opp.is_honest_bettor() and street >= 4:
                if eq > 0.58 and cost <= pot * 0.40:
                    if CallAction in legal:
                        return CallAction()
                if FoldAction in legal:
                    return FoldAction()
                return CallAction()
            if mdf_def and eq > adj_po * 0.80:
                if CallAction in legal:
                    return CallAction()
            stats = self.opp.get_blended_stats()
            cb_rate = stats.get('cbet', 0.5)
            if street == 3 and cb_rate > 0.70:
                if cost <= pot * 0.60 and CallAction in legal:
                    return CallAction()
            cl = pot * 0.35
            if b_act:
                cl = pot * 0.55
            if is_agg:
                cl = pot * 0.50
            elif od == 'unknown' and self.opp.hands_played < 20:
                cl = pot * 0.30
            if is_sta:
                cl = max(cl, pot * 0.45)
            if losing:
                cl *= 0.75
            if cost <= cl and eq > adj_po * 0.85:
                if CallAction in legal:
                    return CallAction()
            if street == 5 and is_agg and eq > 0.42:
                if CallAction in legal:
                    return CallAction()
            if FoldAction in legal:
                return FoldAction()
            return CallAction()

        # Overcards
        if tier == 'overcards':
            if street <= 4 and cost_ratio <= 0.24 and (
                    family_mode or is_agg or
                    self.opp.get_blended_stats().get('small_probe_flop', 0.0) > 0.10 or
                    self.opp.get_blended_stats().get('small_probe_turn', 0.0) > 0.10):
                if CallAction in legal:
                    return CallAction()
            if street == 3 and cost_ratio <= 0.18 and eq > adj_po * 0.78:
                if CallAction in legal:
                    return CallAction()
            if FoldAction in legal:
                return FoldAction()
            return CallAction()

        # Drawing
        if tier == 'drawing':
            if street <= 4:
                ip = pot + min(opp_stack, pot * 0.8)
                if b_act:
                    ip = ip * BOUNTY_RATIO + BOUNTY_CONSTANT
                io = cost / (ip + cost) if ip > 0 else pot_od
                if qualification_mode and provisional_honest and street >= 4:
                    if eq > io and cost <= pot * 0.25 and CallAction in legal:
                        return CallAction()
                    if FoldAction in legal:
                        return FoldAction()
                if eq > io or (mdf_def and cost <= pot * 0.45):
                    if CallAction in legal:
                        return CallAction()
                if (not is_nit and RaiseAction in legal and can_reraise and
                        mr < 0.20 and cost < pot * 0.3):
                    return RaiseAction(mn)
            if street == 5:
                if pct >= 0.85:
                    if RaiseAction in legal and can_reraise:
                        t = max(mn, opp_pip + int(pot * 0.70))
                        return RaiseAction(min(t, mx))
                    if CallAction in legal:
                        return CallAction()
                elif pct >= 0.65:
                    if CallAction in legal:
                        return CallAction()
            if FoldAction in legal:
                return FoldAction()
            return CallAction()

        # Marginal
        if tier == 'marginal':
            if street >= 4 and (
                cost > max(4, pot * 0.18)
                or late_flags['weak_top_pair']
                or late_flags['thin_value']
            ):
                if FoldAction in legal:
                    return FoldAction()
            if qualification_mode and not family_mode:
                if street == 3 and cost <= max(4, pot * 0.12) and eq > adj_po and CallAction in legal:
                    return CallAction()
                if FoldAction in legal:
                    return FoldAction()
                return CallAction()
            if family_mode and street <= 4 and cost <= max(4, pot * 0.16) and eq > adj_po * 0.82:
                if CallAction in legal:
                    return CallAction()
            if is_nit:
                if FoldAction in legal:
                    return FoldAction()
                return CallAction()
            if mdf_def and cost <= pot * 0.40:
                if CallAction in legal:
                    return CallAction()
            if cost <= max(4, pot * 0.20):
                if CallAction in legal:
                    return CallAction()
            if is_agg and cost <= pot * 0.45 and eq > 0.35:
                if CallAction in legal:
                    return CallAction()
            if losing:
                if FoldAction in legal:
                    return FoldAction()
            if FoldAction in legal:
                return FoldAction()
            return CallAction()

        # Weak
        if qualification_mode and not family_mode:
            if FoldAction in legal:
                return FoldAction()
            return CallAction()
        if family_mode and street <= 4 and cost <= max(3, pot * 0.12) and eq > adj_po * 0.72:
            if CallAction in legal:
                return CallAction()
        if street == 5 and is_agg and eq > 0.38 and cost <= pot * 0.4:
            hcb = 0.15 if mc < 0.5 else 0.25
            if mcl < hcb:
                if CallAction in legal:
                    return CallAction()
        if street == 5 and (fb or tb) and cost <= pot * 0.35:
            if mcl < 0.12 and CallAction in legal:
                return CallAction()
        if mdf_def and cost <= max(3, pot * 0.15):
            if CallAction in legal:
                return CallAction()
        if FoldAction in legal:
            return FoldAction()
        return CallAction()

    # =================================================================
    #  CLOCK SAFETY
    # =================================================================

    def _clock_safe(self, legal, cards, board, cost, pot):
        eq = quick_postflop_equity(cards, board) if board else get_preflop_equity(cards)
        po = cost / (pot + cost) if cost > 0 and pot > 0 else 0
        if cost == 0:
            return CheckAction() if CheckAction in legal else CallAction()
        if eq > po + 0.05:
            if CallAction in legal:
                return CallAction()
        high = max(RANK_VALUE.get(cards[0][0], 5), RANK_VALUE.get(cards[1][0], 5))
        if cards[0][0] == cards[1][0] or high >= 12:
            if CallAction in legal:
                return CallAction()
        if cost <= 3 and CallAction in legal:
            return CallAction()
        if FoldAction in legal:
            return FoldAction()
        return CheckAction() if CheckAction in legal else CallAction()

    def _emergency(self, legal, cards, cost):
        if cost == 0:
            return CheckAction() if CheckAction in legal else CallAction()
        high = max(RANK_VALUE.get(cards[0][0], 5), RANK_VALUE.get(cards[1][0], 5))
        if cards[0][0] == cards[1][0] or high >= 12:
            return CallAction() if CallAction in legal else CheckAction()
        if cost <= 3:
            return CallAction() if CallAction in legal else CheckAction()
        if FoldAction in legal:
            return FoldAction()
        return CheckAction() if CheckAction in legal else CallAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
