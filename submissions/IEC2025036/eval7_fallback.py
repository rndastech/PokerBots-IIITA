"""
eval7 Pure Python Fallback
==========================
Implements the subset of eval7 used by TAGPokerBot:
  - Card(string)
  - evaluate(cards) -> int (higher = better)
  - handtype(rank) -> str

When real eval7 is available (competition server), this module is NOT used.
This exists solely for local development/testing on systems without a C compiler.
"""

from itertools import combinations

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
RANKS = '23456789TJQKA'
SUITS = 'cdhs'

RANK_MAP: dict[str, int] = {r: i for i, r in enumerate(RANKS)}  # '2'=0 .. 'A'=12
SUIT_MAP: dict[str, int] = {s: i for i, s in enumerate(SUITS)}  # 'c'=0 .. 's'=3

# Hand type constants — encoded in upper bits of the rank integer
# Higher type = stronger hand
_HIGH_CARD      = 0
_PAIR           = 1
_TWO_PAIR       = 2
_THREE_OF_KIND  = 3
_STRAIGHT       = 4
_FLUSH          = 5
_FULL_HOUSE     = 6
_FOUR_OF_KIND   = 7
_STRAIGHT_FLUSH = 8

_HAND_TYPE_NAMES: dict[int, str] = {
    _HIGH_CARD:      'High Card',
    _PAIR:           'Pair',
    _TWO_PAIR:       'Two Pair',
    _THREE_OF_KIND:  'Trips',
    _STRAIGHT:       'Straight',
    _FLUSH:          'Flush',
    _FULL_HOUSE:     'Full House',
    _FOUR_OF_KIND:   'Quads',
    _STRAIGHT_FLUSH: 'Straight Flush',
}


# ---------------------------------------------------------------------------
# Card class
# ---------------------------------------------------------------------------
class Card:
    """Minimal Card compatible with eval7.Card interface."""
    __slots__ = ('rank', 'suit', '_rank_int', '_rank_char', '_suit_char', '_string')

    def __init__(self, string: str) -> None:
        if len(string) != 2 or string[0] not in RANK_MAP or string[1] not in SUITS:
            raise ValueError(f"Invalid card string: {string!r}")
        self._string = string
        self.rank = RANK_MAP[string[0]]  # integer 0-12, matches real eval7
        self.suit = SUIT_MAP[string[1]]  # integer 0-3, matches real eval7
        self._rank_int = self.rank
        self._rank_char = string[0]
        self._suit_char = string[1]

    def __repr__(self) -> str:
        return f"Card(\"{self._string}\")"

    def __str__(self) -> str:
        return self._string

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self._string == other._string

    def __hash__(self) -> int:
        return hash(self._string)


# ---------------------------------------------------------------------------
# Deck class — matches eval7.Deck interface used by the engine
# ---------------------------------------------------------------------------
class Deck:
    """A standard 52-card deck with shuffle, deal, and peek."""

    def __init__(self) -> None:
        self._cards: list[Card] = [
            Card(r + s) for s in SUITS for r in RANKS
        ]
        self._dealt: int = 0

    def shuffle(self) -> None:
        import random as _rng
        _rng.shuffle(self._cards)
        self._dealt = 0

    def deal(self, n: int) -> list[Card]:
        """Deal n cards off the top (removes them from the deck)."""
        cards = self._cards[self._dealt : self._dealt + n]
        self._dealt += n
        return cards

    def peek(self, n: int) -> list[Card]:
        """Peek at the top n cards already dealt (board cards)."""
        return self._cards[self._dealt : self._dealt + n]

    def __str__(self) -> str:
        remaining = self._cards[self._dealt:]
        return ' '.join(str(c) for c in remaining)


# ---------------------------------------------------------------------------
# 5-card hand evaluation
# ---------------------------------------------------------------------------
def _evaluate_5(cards: list[Card]) -> int:
    """
    Evaluate exactly 5 cards. Returns an integer where higher = better.

    Encoding: hand_type * 10_000_000 + kicker_value
    This ensures any higher hand type always beats any lower hand type.
    """
    ranks = sorted([c._rank_int for c in cards], reverse=True)
    suits = [c.suit for c in cards]

    is_flush = len(set(suits)) == 1

    # Check for straight
    is_straight = False
    straight_high = 0
    if ranks[0] - ranks[4] == 4 and len(set(ranks)) == 5:
        is_straight = True
        straight_high = ranks[0]
    elif ranks == [12, 3, 2, 1, 0]:  # A-2-3-4-5 (wheel)
        is_straight = True
        straight_high = 3  # 5-high straight

    # Count rank frequencies
    freq: dict[int, int] = {}
    for r in ranks:
        freq[r] = freq.get(r, 0) + 1

    counts = sorted(freq.values(), reverse=True)
    # Sort ranks by frequency (descending), then by rank (descending)
    ranked_by_freq = sorted(freq.keys(), key=lambda r: (freq[r], r), reverse=True)

    # Build kicker value: encode up to 5 ranks in base-13
    def kicker_val(rank_list: list[int]) -> int:
        val = 0
        for i, r in enumerate(rank_list):
            val += r * (13 ** (4 - i))
            if i >= 4:
                break
        return val

    # --- Straight Flush ---
    if is_flush and is_straight:
        return _STRAIGHT_FLUSH * 10_000_000 + straight_high

    # --- Four of a Kind ---
    if counts[0] == 4:
        return _FOUR_OF_KIND * 10_000_000 + kicker_val(ranked_by_freq)

    # --- Full House ---
    if counts[0] == 3 and counts[1] == 2:
        return _FULL_HOUSE * 10_000_000 + kicker_val(ranked_by_freq)

    # --- Flush ---
    if is_flush:
        return _FLUSH * 10_000_000 + kicker_val(ranks)

    # --- Straight ---
    if is_straight:
        return _STRAIGHT * 10_000_000 + straight_high

    # --- Three of a Kind ---
    if counts[0] == 3:
        return _THREE_OF_KIND * 10_000_000 + kicker_val(ranked_by_freq)

    # --- Two Pair ---
    if counts[0] == 2 and counts[1] == 2:
        return _TWO_PAIR * 10_000_000 + kicker_val(ranked_by_freq)

    # --- One Pair ---
    if counts[0] == 2:
        return _PAIR * 10_000_000 + kicker_val(ranked_by_freq)

    # --- High Card ---
    return _HIGH_CARD * 10_000_000 + kicker_val(ranks)


# ---------------------------------------------------------------------------
# Public API — matches eval7 interface
# ---------------------------------------------------------------------------
def evaluate(cards: list[Card]) -> int:
    """
    Evaluate a poker hand of 5-7 cards.
    Returns an integer rank where higher = better hand.
    For 6-7 cards, finds the best 5-card combination.
    """
    if len(cards) < 5 or len(cards) > 7:
        raise ValueError(f"Need 5-7 cards, got {len(cards)}")

    if len(cards) == 5:
        return _evaluate_5(cards)

    # For 6-7 cards, try all 5-card combinations and return the best
    best = -1
    for combo in combinations(cards, 5):
        score = _evaluate_5(list(combo))
        if score > best:
            best = score
    return best


def handtype(rank: int) -> str:
    """Convert an integer rank from evaluate() to a hand type string."""
    type_code = rank // 10_000_000
    return _HAND_TYPE_NAMES.get(type_code, 'High Card')


# Expose ranks and suits for compatibility
ranks = list(RANKS)
suits = list(SUITS)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Royal flush
    hand = [Card(s) for s in ['Ah', 'Kh', 'Qh', 'Jh', 'Th']]
    r = evaluate(hand)
    print(f"Royal Flush: {handtype(r)} (rank={r})")

    # Full house
    hand = [Card(s) for s in ['Ah', 'Ad', 'As', 'Kh', 'Kd']]
    r = evaluate(hand)
    print(f"Full House:  {handtype(r)} (rank={r})")

    # Pair
    hand = [Card(s) for s in ['Ah', 'Ad', '7s', '5h', '2d']]
    r = evaluate(hand)
    print(f"Pair:        {handtype(r)} (rank={r})")

    # High card
    hand = [Card(s) for s in ['Ah', 'Kd', '7s', '5h', '2d']]
    r = evaluate(hand)
    print(f"High Card:   {handtype(r)} (rank={r})")

    # 7-card hand (trips)
    hand = [Card(s) for s in ['Ah', 'Ad', 'As', '7c', '2d', 'Kh', 'Qd']]
    r = evaluate(hand)
    print(f"7-card Trips: {handtype(r)} (rank={r})")
