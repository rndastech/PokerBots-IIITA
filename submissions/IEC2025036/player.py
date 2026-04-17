'''
Poker Bot — Tight-Aggressive (TAG) Strategy v0.3
=================================================
A deterministic, rule-based Texas Hold'em poker bot for the IIITA Bounty
Poker engine. Uses eval7 for post-flop hand evaluation.

v0.2: Added deterministic bluffing via hash-based pseudo-randomness.
v0.3: Added O(1) opponent modelling (Maniac / Nit detection).
      - vs Maniac (raise_rate > 60%): widen pre-flop call range, 3-bet light
      - vs Nit    (fold_rate  > 50%): suppress bluffing, steal blinds aggressively

Every decision is O(1) — frozenset lookups and integer comparisons only.
No randomness, no simulation, no learning.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

try:
    import eval7
except ImportError:
    import eval7_fallback as eval7  # Pure Python fallback for local dev


# ---------------------------------------------------------------------------
# RANK ORDERING — used for canonical hand notation & top-pair detection
# ---------------------------------------------------------------------------
_RANK_ORDER: dict[str, int] = {
    'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
    '9': 9, '8': 8, '7': 7, '6': 6, '5': 5,
    '4': 4, '3': 3, '2': 2,
}

# ---------------------------------------------------------------------------
# TOP ~20% STARTING HANDS — Premium TAG Range (frozenset for O(1) lookup)
# ---------------------------------------------------------------------------
_PREMIUM_HANDS: frozenset[str] = frozenset({
    # --- Pocket Pairs (9) ---
    'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66',
    # --- Suited Aces (10) ---
    'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A5s', 'A4s', 'A3s', 'A2s',
    # --- Suited Broadway + Connectors (17) ---
    'KQs', 'KJs', 'KTs', 'K9s',
    'QJs', 'QTs', 'Q9s',
    'JTs', 'J9s',
    'T9s', 'T8s',
    '98s', '97s',
    '87s', '76s', '65s', '54s',
    # --- Offsuit Broadway (7) ---
    'AKo', 'AQo', 'AJo', 'ATo',
    'KQo', 'KJo',
    'QJo',
})

# Extra hands to call/3-bet against a detected Maniac (~30% range)
_MANIAC_DEFEND_HANDS: frozenset[str] = frozenset({
    '55', '44', '33', '22',
    'K8s', 'K7s', 'K6s',
    'Q8s', 'J8s',
    'A9o', 'A8o', 'KTo', 'QTo', 'JTo',
    'T7s', '96s', '86s', '75s', '64s',
})

# ---------------------------------------------------------------------------
# POST-FLOP HAND STRENGTH TIERS
# ---------------------------------------------------------------------------
_STRONG_HAND_TYPES: frozenset[str] = frozenset({
    'Two Pair', 'Trips', 'Straight', 'Flush',
    'Full House', 'Quads', 'Straight Flush',
})


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS — module-level for O(1) access
# ---------------------------------------------------------------------------

def _canonicalize_hand(card1: str, card2: str) -> str:
    """
    Convert two hole cards into canonical hand notation.
    ('Ah', 'Kh') → 'AKs', ('Kd', 'Ah') → 'AKo', ('Jh', 'Jd') → 'JJ'
    """
    rank1, suit1 = card1[0], card1[1]
    rank2, suit2 = card2[0], card2[1]

    if _RANK_ORDER[rank1] < _RANK_ORDER[rank2]:
        rank1, suit1, rank2, suit2 = rank2, suit2, rank1, suit1

    if rank1 == rank2:
        return rank1 + rank2

    suffix = 's' if suit1 == suit2 else 'o'
    return rank1 + rank2 + suffix


def _is_top_pair_or_overpair(hole_cards: list[str], board_cards: list[str]) -> bool:
    """
    True if we have top pair (paired with highest board card)
    or an overpair (pocket pair above all board cards).
    """
    hole_rank1: int = _RANK_ORDER[hole_cards[0][0]]
    hole_rank2: int = _RANK_ORDER[hole_cards[1][0]]
    board_top: int = max(_RANK_ORDER[c[0]] for c in board_cards)

    if hole_rank1 == hole_rank2 and hole_rank1 > board_top:
        return True

    if hole_rank1 == board_top or hole_rank2 == board_top:
        return True

    return False


# ---------------------------------------------------------------------------
# PLAYER CLASS — integrates TAG strategy with the skeleton engine
# ---------------------------------------------------------------------------

class Player(Bot):
    '''
    Tight-Aggressive poker bot with deterministic, O(1) decision-making.

    Pre-flop:  Raise with top 20% hands, fold everything else.
    Post-flop: Raise with strong made hands (Two Pair+, Top Pair),
               check/fold with weak ones.
    Bounty:    Widen range slightly when we hold the bounty rank.
    v0.3:      Opponent model — detect Maniac/Nit and adapt.
    '''

    def __init__(self):
        '''Called when a new game starts. Called exactly once.'''
        self.bounty_rank = None

        # --- Opponent model counters (updated each round) ---
        self._opp_raises: int = 0    # how many times opponent raised pre-flop
        self._opp_folds: int  = 0    # how many times opponent folded pre-flop
        self._opp_rounds: int = 0    # total rounds observed

        # Derived archetype flags — recomputed each round in O(1)
        self._vs_maniac: bool = False
        self._vs_nit: bool    = False

    def handle_new_round(self, game_state, round_state, active):
        '''Called when a new round starts. Called NUM_ROUNDS times.'''
        self.bounty_rank = round_state.bounties[active]

        # Update archetype flags using previous round data
        # Warm-up: wait at least 20 rounds before trusting stats
        if self._opp_rounds >= 20:
            raise_rate = self._opp_raises / self._opp_rounds
            fold_rate  = self._opp_folds  / self._opp_rounds
            self._vs_maniac = raise_rate > 0.60   # raises >60% of hands
            self._vs_nit    = fold_rate  > 0.50   # folds  >50% of hands
        else:
            self._vs_maniac = False
            self._vs_nit    = False

    def handle_round_over(self, game_state, terminal_state, active):
        '''Called when a round ends. Update opponent model.'''
        previous_state = terminal_state.previous_state
        opp_actions = getattr(previous_state, 'opp_actions', None)

        # Infer opponent behaviour from pip deltas at end of pre-flop
        # If opponent put in more than big blind pre-flop → they raised
        # If opponent ended with full stack delta = 0 → they folded immediately
        opp_contribution = STARTING_STACK - terminal_state.previous_state.stacks[1 - active]

        self._opp_rounds += 1
        if opp_contribution > BIG_BLIND * 2:
            self._opp_raises += 1
        elif opp_contribution == 0 or opp_contribution <= BIG_BLIND:
            # Opponent put in at most the BB — likely folded or just called blind
            self._opp_folds += 1

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens — deterministic TAG decision tree.

        Returns: FoldAction(), CallAction(), CheckAction(), or RaiseAction(amount).
        '''
        legal_actions  = round_state.legal_actions()
        street         = round_state.street
        my_cards       = round_state.hands[active]
        board_cards    = round_state.deck[:street]
        my_pip         = round_state.pips[active]
        opp_pip        = round_state.pips[1 - active]
        my_stack       = round_state.stacks[active]
        opp_stack      = round_state.stacks[1 - active]
        continue_cost  = opp_pip - my_pip
        pot_size       = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)

        min_raise = 0
        max_raise = 0
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()

        has_bounty = (
            my_cards[0][0] == self.bounty_rank or
            my_cards[1][0] == self.bounty_rank
        )

        # Clock safety
        if game_state.game_clock < 3.0:
            if CheckAction in legal_actions:
                return CheckAction()
            if CallAction in legal_actions and continue_cost <= BIG_BLIND:
                return CallAction()
            if FoldAction in legal_actions:
                return FoldAction()
            return CheckAction()

        # =================================================================
        # PRE-FLOP (street == 0)
        # =================================================================
        if street == 0:
            return self._preflop(
                my_cards, legal_actions, continue_cost,
                my_pip, min_raise, max_raise, has_bounty, pot_size
            )

        # =================================================================
        # POST-FLOP (street >= 3)
        # =================================================================
        return self._postflop(
            my_cards, board_cards, legal_actions, continue_cost,
            my_pip, my_stack, min_raise, max_raise, has_bounty, pot_size
        )

    # -------------------------------------------------------------------
    # PRE-FLOP STRATEGY
    # -------------------------------------------------------------------

    def _preflop(self, my_cards, legal_actions, continue_cost,
                 my_pip, min_raise, max_raise, has_bounty, pot_size):
        """
        Pre-flop: RAISE premium hands, FOLD everything else.

        v0.3 adaptations:
          vs Maniac → also defend with _MANIAC_DEFEND_HANDS; 3-bet to isolate
          vs Nit    → steal pre-flop raises more aggressively (raise any two)
        """
        canonical  = _canonicalize_hand(my_cards[0], my_cards[1])
        is_premium = canonical in _PREMIUM_HANDS

        # --- vs MANIAC: widen defence range ---
        is_maniac_defend = self._vs_maniac and canonical in _MANIAC_DEFEND_HANDS

        if is_premium or is_maniac_defend:
            if RaiseAction in legal_actions:
                if self._vs_maniac:
                    # 3-bet big to isolate: 4x the continue cost
                    target = max(min_raise, my_pip + max(8, continue_cost * 4))
                else:
                    target = max(min_raise, my_pip + max(6, continue_cost * 3))
                return RaiseAction(min(target, max_raise))
            if CallAction in legal_actions:
                return CallAction()
            if CheckAction in legal_actions:
                return CheckAction()

        # --- vs NIT: steal blinds with any two cards when cheap ---
        if self._vs_nit and continue_cost <= BIG_BLIND and RaiseAction in legal_actions:
            # Min-raise steal — cheap risk, Nit often folds
            return RaiseAction(min(min_raise, max_raise))

        # --- Bounty rank in hand: call small bets ---
        if has_bounty:
            if continue_cost == 0 and CheckAction in legal_actions:
                return CheckAction()
            if continue_cost <= 4 and CallAction in legal_actions:
                return CallAction()

        # --- Junk hand: fold ---
        if continue_cost == 0 and CheckAction in legal_actions:
            return CheckAction()
        if FoldAction in legal_actions:
            return FoldAction()

        return CallAction()

    # -------------------------------------------------------------------
    # POST-FLOP STRATEGY
    # -------------------------------------------------------------------

    def _postflop(self, my_cards, board_cards, legal_actions, continue_cost,
                  my_pip, my_stack, min_raise, max_raise, has_bounty, pot_size):
        """
        Post-flop: eval7 hand evaluation → deterministic action.

        v0.3 adaptations:
          vs Nit: suppress bluffing (they fold anyway → bluffs waste chips)
          vs Maniac: call wider post-flop (they overbet with air)
        """
        all_cards = [eval7.Card(c) for c in my_cards + board_cards]
        hand_rank: int = eval7.evaluate(all_cards)
        hand_type: str = eval7.handtype(hand_rank)

        # --- DETERMINISTIC BLUFFING (suppressed vs Nit) ---
        # Nit folds pre-flop and folds to any post-flop bet.
        # Bluffing them is pointless — they already folded or have a monster.
        bluff_factor = hash("".join(board_cards) + str(pot_size)) % 100
        can_bluff = (not self._vs_nit) and bluff_factor < 15

        # --- STRONG HANDS: Two Pair or better → RAISE ---
        if hand_type in _STRONG_HAND_TYPES:
            if RaiseAction in legal_actions:
                bet_size = max(min_raise, my_pip + (pot_size * 2 // 3))
                return RaiseAction(min(bet_size, max_raise))
            if CallAction in legal_actions:
                return CallAction()
            if CheckAction in legal_actions:
                return CheckAction()

        # --- PAIR: check if top pair / overpair ---
        if hand_type == 'Pair':
            is_strong_pair = _is_top_pair_or_overpair(my_cards, board_cards)

            if is_strong_pair:
                if RaiseAction in legal_actions:
                    bet_size = max(min_raise, my_pip + (pot_size // 2))
                    return RaiseAction(min(bet_size, max_raise))
                if CallAction in legal_actions:
                    return CallAction()
                if CheckAction in legal_actions:
                    return CheckAction()
            else:
                # Weak pair
                if continue_cost == 0 and CheckAction in legal_actions:
                    if can_bluff and RaiseAction in legal_actions:
                        bet_size = max(min_raise, my_pip + (pot_size * 2 // 3))
                        return RaiseAction(min(bet_size, max_raise))
                    return CheckAction()

                # vs Maniac: call a wider range (they overbet air constantly)
                call_threshold = pot_size // 2 if self._vs_maniac else pot_size // 3
                if continue_cost <= max(call_threshold, 4) and CallAction in legal_actions:
                    return CallAction()
                if has_bounty and continue_cost <= max(pot_size // 2, 6) and CallAction in legal_actions:
                    return CallAction()
                if FoldAction in legal_actions:
                    return FoldAction()

        # --- HIGH CARD: check if free, fold to any bet ---
        if continue_cost == 0 and CheckAction in legal_actions:
            if can_bluff and RaiseAction in legal_actions:
                bet_size = max(min_raise, my_pip + (pot_size * 2 // 3))
                return RaiseAction(min(bet_size, max_raise))
            return CheckAction()
        if FoldAction in legal_actions:
            return FoldAction()

        # Fallback safety
        if CheckAction in legal_actions:
            return CheckAction()
        if CallAction in legal_actions:
            return CallAction()
        return FoldAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
