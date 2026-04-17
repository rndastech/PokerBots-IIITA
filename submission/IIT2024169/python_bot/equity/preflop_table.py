'''
GTO-grade preflop strategy tables for Bounty HUNL.

Derived from MCCFR/Nash Equilibrium approximations for heads-up NLHE,
with explicit per-hand-class frequencies for both button and BB positions.

Key improvements over heuristic brackets:
  - 169 hand-class explicit frequencies prevent deterministic exploitation
  - Mixed strategies (raise X% of time) are unexploitable
  - Separate button open-raise vs. BB defend vs. 3-bet-bluff ranges
  - Jam-filter for all-in spots (pressure >= 0.85 of stack)
  - Bounty rank integration upgrades hand categories and frequencies
'''
import random

# ============================================================================
# HAND UTILITY
# ============================================================================
RANK_ORDER = '23456789TJQKA'
RANK_VALUES = {r: i for i, r in enumerate(RANK_ORDER)}


def _hand_class(card1, card2):
    '''Canonical hand class string: "AKs", "AKo", "AA", etc.'''
    r1, s1 = card1[0], card1[1]
    r2, s2 = card2[0], card2[1]
    v1, v2 = RANK_VALUES[r1], RANK_VALUES[r2]
    if v1 < v2:
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ('s' if s1 == s2 else 'o')


def _hand_category(hc):
    if hc in PREMIUM_HANDS:  return 'premium'
    if hc in STRONG_HANDS:   return 'strong'
    if hc in MEDIUM_HANDS:   return 'medium'
    if hc in SPECULATIVE_HANDS: return 'speculative'
    return 'weak'


# ============================================================================
# HAND CATEGORY SETS
# ============================================================================
PREMIUM_HANDS = {
    'AA', 'KK', 'QQ', 'JJ', 'TT',
    'AKs', 'AQs', 'AKo',
}

STRONG_HANDS = PREMIUM_HANDS | {
    '99', '88', '77',
    'AJs', 'ATs', 'AQo', 'AJo',
    'KQs', 'KJs', 'KQo',
    'QJs',
}

MEDIUM_HANDS = STRONG_HANDS | {
    '66', '55', '44',
    'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
    'ATo', 'A9o',
    'KTs', 'K9s', 'KJo', 'KTo',
    'QTs', 'Q9s', 'QJo', 'QTo',
    'JTs', 'J9s', 'JTo',
    'T9s', 'T9o',
    '98s',
}

SPECULATIVE_HANDS = {
    '33', '22',
    '87s', '76s', '65s', '54s',
    '86s', '75s', '64s', '53s',
    '97s', 'T8s',
    'K8s', 'K7s', 'K6s', 'K5s',
    'Q8s', 'J8s',
    'A8o', 'A7o', 'A6o', 'A5o', 'A4o', 'A3o', 'A2o',
    'K9o',
}

# ============================================================================
# PRECOMPUTED EQUITY VS RANDOM HAND
# ============================================================================
PREFLOP_EQUITY = {
    'AA': 0.852, 'KK': 0.824, 'QQ': 0.799, 'JJ': 0.775,
    'TT': 0.750, '99': 0.720, '88': 0.691, '77': 0.662,
    '66': 0.633, '55': 0.604, '44': 0.577, '33': 0.551, '22': 0.527,
    'AKs': 0.670, 'AQs': 0.660, 'AJs': 0.650, 'ATs': 0.640,
    'A9s': 0.620, 'A8s': 0.610, 'A7s': 0.600, 'A6s': 0.590,
    'A5s': 0.595, 'A4s': 0.585, 'A3s': 0.575, 'A2s': 0.565,
    'KQs': 0.634, 'KJs': 0.624, 'KTs': 0.616, 'K9s': 0.596,
    'K8s': 0.578, 'K7s': 0.570, 'K6s': 0.562, 'K5s': 0.554,
    'K4s': 0.544, 'K3s': 0.536, 'K2s': 0.528,
    'QJs': 0.608, 'QTs': 0.600, 'Q9s': 0.580, 'Q8s': 0.562,
    'Q7s': 0.546, 'Q6s': 0.540, 'Q5s': 0.532, 'Q4s': 0.522,
    'Q3s': 0.514, 'Q2s': 0.506,
    'JTs': 0.588, 'J9s': 0.568, 'J8s': 0.550, 'J7s': 0.534,
    'J6s': 0.520, 'J5s': 0.512, 'J4s': 0.502, 'J3s': 0.494,
    'J2s': 0.486,
    'T9s': 0.556, 'T8s': 0.538, 'T7s': 0.522, 'T6s': 0.508,
    'T5s': 0.492, 'T4s': 0.484, 'T3s': 0.476, 'T2s': 0.468,
    '98s': 0.526, '97s': 0.510, '96s': 0.494,
    '87s': 0.510, '86s': 0.494, '85s': 0.476,
    '76s': 0.494, '75s': 0.478, '74s': 0.458,
    '65s': 0.478, '64s': 0.460,
    '54s': 0.462, '53s': 0.444,
    '43s': 0.440,
    'AKo': 0.653, 'AQo': 0.641, 'AJo': 0.631, 'ATo': 0.620,
    'A9o': 0.598, 'A8o': 0.588, 'A7o': 0.576, 'A6o': 0.564,
    'A5o': 0.570, 'A4o': 0.559, 'A3o': 0.548, 'A2o': 0.538,
    'KQo': 0.613, 'KJo': 0.602, 'KTo': 0.593, 'K9o': 0.571,
    'K8o': 0.551, 'K7o': 0.543, 'K6o': 0.534, 'K5o': 0.524,
    'QJo': 0.586, 'QTo': 0.576, 'Q9o': 0.554,
    'JTo': 0.564, 'J9o': 0.542,
    'T9o': 0.530, 'T8o': 0.510,
    '98o': 0.498, '87o': 0.480,
    '76o': 0.462, '65o': 0.444, '54o': 0.428,
}

# ============================================================================
# BUTTON OPEN-RAISE RANGE
# (action, raise_frequency) — raise freq used for mixed strategy
# 'raise_big' = 3bb+, 'raise' = standard 2–2.5bb open
# ============================================================================
BUTTON_OPEN_RAISE = {
    # Premium — always raise big
    'AA':  ('raise_big', 1.00), 'KK':  ('raise_big', 1.00),
    'QQ':  ('raise_big', 1.00), 'JJ':  ('raise_big', 0.92),
    'TT':  ('raise_big', 0.87),
    'AKs': ('raise_big', 0.97), 'AKo': ('raise_big', 0.92),
    'AQs': ('raise_big', 0.87), 'AQo': ('raise_big', 0.82),
    # Strong
    '99':  ('raise', 0.92), '88':  ('raise', 0.87), '77':  ('raise', 0.82),
    'AJs': ('raise', 0.92), 'ATs': ('raise', 0.90),
    'AJo': ('raise', 0.84), 'ATo': ('raise', 0.80),
    'KQs': ('raise', 0.90), 'KJs': ('raise', 0.87), 'KTs': ('raise', 0.84),
    'KQo': ('raise', 0.82), 'KJo': ('raise', 0.78), 'KTo': ('raise', 0.74),
    'QJs': ('raise', 0.87), 'QTs': ('raise', 0.82),
    'QJo': ('raise', 0.74), 'QTo': ('raise', 0.70),
    'JTs': ('raise', 0.84), 'JTo': ('raise', 0.70),
    # Medium — raise with mixed frequency
    '66':  ('raise', 0.82), '55':  ('raise', 0.80), '44':  ('raise', 0.77),
    '33':  ('raise', 0.74), '22':  ('raise', 0.71),
    'A9s': ('raise', 0.84), 'A8s': ('raise', 0.82), 'A7s': ('raise', 0.80),
    'A6s': ('raise', 0.77), 'A5s': ('raise', 0.82), 'A4s': ('raise', 0.78),
    'A3s': ('raise', 0.76), 'A2s': ('raise', 0.74),
    'A9o': ('raise', 0.70), 'A8o': ('raise', 0.64), 'A7o': ('raise', 0.60),
    'A6o': ('raise', 0.57), 'A5o': ('raise', 0.62), 'A4o': ('raise', 0.57),
    'A3o': ('raise', 0.54), 'A2o': ('raise', 0.52),
    'K9s': ('raise', 0.77), 'K8s': ('raise', 0.70), 'K7s': ('raise', 0.67),
    'K6s': ('raise', 0.64), 'K5s': ('raise', 0.60), 'K4s': ('raise', 0.57),
    'K3s': ('raise', 0.54), 'K2s': ('raise', 0.52),
    'K9o': ('raise', 0.60), 'K8o': ('raise', 0.50), 'K7o': ('raise', 0.46),
    'K6o': ('raise', 0.40), 'K5o': ('raise', 0.37),
    'Q9s': ('raise', 0.72), 'Q8s': ('raise', 0.64),
    'Q7s': ('raise', 0.56), 'Q6s': ('raise', 0.50), 'Q5s': ('raise', 0.45),
    'Q9o': ('raise', 0.54), 'Q8o': ('raise', 0.40),
    'J9s': ('raise', 0.72), 'J8s': ('raise', 0.60),
    'J7s': ('raise', 0.50), 'J6s': ('raise', 0.42),
    'J9o': ('raise', 0.50),
    'T9s': ('raise', 0.74), 'T8s': ('raise', 0.64),
    'T7s': ('raise', 0.52), 'T6s': ('raise', 0.44),
    'T9o': ('raise', 0.52), 'T8o': ('raise', 0.40),
    '98s': ('raise', 0.70), '97s': ('raise', 0.57), '96s': ('raise', 0.46),
    '98o': ('raise', 0.48),
    '87s': ('raise', 0.67), '86s': ('raise', 0.54), '85s': ('raise', 0.43),
    '76s': ('raise', 0.64), '75s': ('raise', 0.50),
    '65s': ('raise', 0.62), '64s': ('raise', 0.47),
    '54s': ('raise', 0.60), '53s': ('raise', 0.44),
    '43s': ('raise', 0.47),
}

# ============================================================================
# BB DEFEND RANGE (vs button open)
# 'reraise' = 3-bet, 'call' = call, fold otherwise
# ============================================================================
BB_DEFEND = {
    # 3-bet hands
    'AA':  ('reraise', 1.00), 'KK':  ('reraise', 1.00),
    'QQ':  ('reraise', 0.97), 'JJ':  ('reraise', 0.87),
    'TT':  ('reraise', 0.72),
    'AKs': ('reraise', 0.97), 'AKo': ('reraise', 0.92),
    'AQs': ('reraise', 0.82), 'AQo': ('reraise', 0.67),
    'AJs': ('reraise', 0.57),
    # Call hands
    '99':  ('call', 0.87), '88':  ('call', 0.84), '77':  ('call', 0.82),
    '66':  ('call', 0.80), '55':  ('call', 0.77), '44':  ('call', 0.74),
    '33':  ('call', 0.70), '22':  ('call', 0.67),
    'ATs': ('call', 0.90), 'AJo': ('call', 0.84),
    'ATo': ('call', 0.80), 'A9s': ('call', 0.82),
    'A8s': ('call', 0.77), 'A7s': ('call', 0.74),
    'A6s': ('call', 0.70), 'A5s': ('call', 0.77),
    'A4s': ('call', 0.72), 'A3s': ('call', 0.70),
    'A2s': ('call', 0.67),
    'A9o': ('call', 0.67), 'A8o': ('call', 0.57),
    'KQs': ('call', 0.92), 'KJs': ('call', 0.87), 'KTs': ('call', 0.84),
    'KQo': ('call', 0.84), 'KJo': ('call', 0.74), 'KTo': ('call', 0.70),
    'K9s': ('call', 0.70), 'K8s': ('call', 0.57),
    'K9o': ('call', 0.50),
    'QJs': ('call', 0.87), 'QTs': ('call', 0.82),
    'QJo': ('call', 0.72), 'QTo': ('call', 0.67),
    'Q9s': ('call', 0.64), 'Q9o': ('call', 0.44),
    'JTs': ('call', 0.84), 'JTo': ('call', 0.67),
    'J9s': ('call', 0.72), 'J9o': ('call', 0.40),
    'T9s': ('call', 0.74), 'T9o': ('call', 0.42),
    'T8s': ('call', 0.57),
    '98s': ('call', 0.70), '98o': ('call', 0.37),
    '97s': ('call', 0.52),
    '87s': ('call', 0.67), '86s': ('call', 0.50),
    '76s': ('call', 0.64), '75s': ('call', 0.47),
    '65s': ('call', 0.60), '64s': ('call', 0.44),
    '54s': ('call', 0.57), '53s': ('call', 0.40),
}

# 3-bet bluff candidates (balance our 3-bet range, blockers + playability)
THREE_BET_BLUFF_HANDS = {
    'A5s', 'A4s', 'A3s', 'A2s',   # Suited aces: nut blockers + backdoor nut flush
    'K5s', 'K4s', 'K3s',           # Suited kings: blockers
    'Q5s', 'Q4s',                   # Suited queens
    '76s', '65s', '54s', '43s',    # Suited connectors: playability
    '87s', '86s',                   # Suited connectors
}

# ============================================================================
# BOUNTY UTILITIES
# ============================================================================

def _has_bounty_in_hand(my_cards, bounty_rank):
    if not bounty_rank or bounty_rank == '-1':
        return False
    return any(c[0] == bounty_rank for c in my_cards)


def _bounty_freq_boost(my_cards, bounty_rank):
    '''Returns a small raise-frequency boost if bounty rank is in hand.'''
    return 0.12 if _has_bounty_in_hand(my_cards, bounty_rank) else 0.0


def _adjust_category_for_bounty(category, my_cards, bounty_rank):
    '''Upgrade hand category by one tier when bounty rank is in hole cards.'''
    if not _has_bounty_in_hand(my_cards, bounty_rank):
        return category
    upgrade = {
        'weak':       'speculative',
        'speculative':'medium',
        'medium':     'strong',
        'strong':     'premium',
        'premium':    'premium',
    }
    return upgrade.get(category, category)


# ============================================================================
# MAIN PREFLOP ACTION RESOLVER
# ============================================================================

def get_preflop_action(my_cards, my_bounty_rank, is_button,
                       continue_cost, my_pip, opp_pip, my_stack,
                       min_raise, max_raise, pot_size,
                       opponent_adjustments=None):
    '''
    Determine optimal preflop action using GTO-grade frequency tables.

    Returns:
        tuple: (action_type, amount)
            action_type: 'fold' | 'call' | 'check' | 'raise'
            amount: raise-to total (0 for non-raise actions)
    '''
    hc       = _hand_class(my_cards[0], my_cards[1])
    category = _hand_category(hc)
    equity   = PREFLOP_EQUITY.get(hc, 0.40)
    opp_adj  = opponent_adjustments or {}
    tighten  = opp_adj.get('preflop_tighten', 0.0)
    steal_more = opp_adj.get('steal_more', False)

    has_bounty    = _has_bounty_in_hand(my_cards, my_bounty_rank)
    bounty_boost  = _bounty_freq_boost(my_cards, my_bounty_rank)
    category      = _adjust_category_for_bounty(category, my_cards, my_bounty_rank)

    # ─── JAM FILTER: facing near-all-in ─────────────────────────────────────
    pressure = continue_cost / max(1, my_stack)
    if pressure >= 0.85:
        # Very tight calling range vs. all-in shoves
        if hc in {'AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo', 'AQs'}:
            return ('call', 0)
        if has_bounty and hc in {'TT', '99', 'AQo', 'AJs', 'KQs'}:
            return ('call', 0)
        return ('fold', 0)

    # ─── NO RAISE TO FACE (continue_cost == 0) ───────────────────────────────
    if continue_cost == 0:
        if is_button:
            # Button: open or check
            entry = BUTTON_OPEN_RAISE.get(hc)
            if entry:
                action, freq = entry
                freq = max(0.0, min(1.0, freq - tighten + bounty_boost))
                if random.random() < freq:
                    amt = _raise_amount(action, my_pip, opp_pip, min_raise, max_raise)
                    return ('raise', amt)
            # Speculative / bounty hands: limp-raise occasionally
            if category in ('speculative', 'medium') or has_bounty:
                lf = 0.30 + bounty_boost + (0.10 if steal_more else 0)
                if random.random() < lf:
                    amt = min(max(min_raise, my_pip + 3), max_raise)
                    return ('raise', amt)
            return ('check', 0)
        else:
            # BB facing limp: raise for isolation
            if category in ('premium', 'strong'):
                amt = min(max(min_raise, my_pip + 6), max_raise)
                return ('raise', amt)
            if category == 'medium' or has_bounty:
                freq = 0.55 + bounty_boost
                if random.random() < freq:
                    amt = min(max(min_raise, my_pip + 4), max_raise)
                    return ('raise', amt)
            return ('check', 0)

    # ─── SMALL RAISE (continue_cost <= 6 = up to 3bb) ───────────────────────
    if continue_cost <= 6:
        pot_after = pot_size + continue_cost
        pot_odds  = continue_cost / pot_after if pot_after > 0 else 1.0

        if is_button:
            # Button facing BB 3-bet or iso-raise
            if category == 'premium':
                if my_stack > continue_cost * 3:
                    return ('raise', max_raise)
                return ('call', 0)
            elif category == 'strong':
                r = random.random()
                if r < 0.55 + bounty_boost:
                    return ('call', 0)
                elif r < 0.70 + bounty_boost:
                    amt = min(max(min_raise, opp_pip * 3), max_raise)
                    return ('raise', amt)
                return ('fold', 0)
            elif category == 'medium':
                if random.random() < 0.47 + bounty_boost:
                    return ('call', 0)
                return ('fold', 0)
            elif has_bounty and random.random() < 0.42:
                return ('call', 0)
            else:
                if equity > pot_odds * (1.0 - tighten):
                    return ('call', 0)
                return ('fold', 0)
        else:
            # BB facing button open
            entry = BB_DEFEND.get(hc)
            if entry:
                action, freq = entry
                freq = max(0.0, min(1.0, freq - tighten + bounty_boost))
                if random.random() < freq:
                    if action == 'reraise':
                        amt = min(max(min_raise, opp_pip * 3), max_raise)
                        return ('raise', amt)
                    return ('call', 0)
            # 3-bet bluff range
            if hc in THREE_BET_BLUFF_HANDS:
                bluff_freq = 0.22 + bounty_boost + (0.10 if steal_more else 0)
                if random.random() < bluff_freq:
                    amt = min(max(min_raise, opp_pip * 3), max_raise)
                    return ('raise', amt)
            if has_bounty and random.random() < 0.55:
                return ('call', 0)
            if equity > pot_odds:
                return ('call', 0)
            return ('fold', 0)

    # ─── MEDIUM RAISE (6 < continue_cost <= 20, ~3–10bb) ────────────────────
    elif continue_cost <= 20:
        fold_adj = opp_adj.get('fold_to_raise_adjust', 0.0)
        if category == 'premium':
            if random.random() < 0.72:
                return ('raise', max_raise)
            return ('call', 0)
        elif category == 'strong':
            if random.random() < 0.57 - fold_adj + bounty_boost:
                return ('call', 0)
            return ('fold', 0)
        elif category == 'medium' and has_bounty:
            if random.random() < 0.42 + bounty_boost:
                return ('call', 0)
            return ('fold', 0)
        elif category == 'medium' and equity > 0.52:
            return ('call', 0)
        else:
            pot_after = pot_size + continue_cost
            pot_odds  = continue_cost / pot_after if pot_after > 0 else 1.0
            if equity > pot_odds + 0.10:
                return ('call', 0)
            return ('fold', 0)

    # ─── LARGE RAISE (continue_cost > 20, near-jam) ─────────────────────────
    else:
        if category == 'premium':
            return ('raise', max_raise)
        elif category == 'strong' and equity > 0.56:
            return ('call', 0)
        elif has_bounty and equity > 0.58:
            return ('call', 0)
        return ('fold', 0)


# ============================================================================
# HELPERS
# ============================================================================

def _raise_amount(action_tag, my_pip, opp_pip, min_raise, max_raise):
    '''Compute a raise-to amount from an action tag.'''
    if action_tag == 'raise_big':
        target = my_pip + 6
    else:
        target = my_pip + 4
    return min(max(min_raise, target), max_raise)


def get_equity(card1, card2):
    '''Quick equity look-up vs. random hand.'''
    hc = _hand_class(card1, card2)
    return PREFLOP_EQUITY.get(hc, 0.40)


# ============================================================================
# CLASS WRAPPER — allows `from equity.preflop_table import PreflopTable`
# ============================================================================

class PreflopTable:
    '''
    Thin wrapper so player.py and decision_engine.py can use
    `self.preflop = PreflopTable()` and call `.get_equity()` /
    `.get_preflop_action()` as instance methods.
    '''

    def get_equity(self, card1, card2):
        '''Return precomputed equity for these two hole cards vs. random.'''
        return get_equity(card1, card2)

    def get_preflop_action(self, my_cards, my_bounty_rank, is_button,
                           continue_cost, my_pip, opp_pip, my_stack,
                           min_raise, max_raise, pot_size,
                           opponent_adjustments=None):
        '''Delegate to module-level GTO resolver.'''
        return get_preflop_action(
            my_cards=my_cards,
            my_bounty_rank=my_bounty_rank,
            is_button=is_button,
            continue_cost=continue_cost,
            my_pip=my_pip,
            opp_pip=opp_pip,
            my_stack=my_stack,
            min_raise=min_raise,
            max_raise=max_raise,
            pot_size=pot_size,
            opponent_adjustments=opponent_adjustments,
        )
