# cython: language_level=3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  HYDRA ENGINE — Cython-Accelerated Poker Computation Core          ║
║  Compiles to C for maximum throughput under 60-second game clock.   ║
║  Functions:                                                        ║
║    calc_equity    — Monte Carlo + exact river equity                ║
║    calc_call_ev   — Exact bounty-aware call EV                     ║
║    calc_raise_ev  — Exact bounty-aware raise EV with fold equity   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import eval7
from random import randint as _randint

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS & PRE-BUILT CARD CACHE
# ═══════════════════════════════════════════════════════════════

RANKS = '23456789TJQKA'
SUITS = 'dhcs'
FULL_DECK_STRS = tuple(r + s for r in RANKS for s in SUITS)

# Pre-build all 52 eval7.Card objects once at import time.
# Dict lookup is O(1) — avoids repeated Card() constructor calls.
_CARD_OBJ = {s: eval7.Card(s) for s in FULL_DECK_STRS}

# Bounty parameters (mirror config.py defaults)
_BOUNTY_RATIO = 1.5
_BOUNTY_CONSTANT = 10.0


# ═══════════════════════════════════════════════════════════════
#  EQUITY CALCULATION
# ═══════════════════════════════════════════════════════════════

def calc_equity(list my_hand_strs, list board_strs, int num_sims):
    """
    Monte Carlo equity estimation with exact river enumeration.

    On the river (5 board cards), enumerates ALL C(remaining,2) opponent
    hands for perfect accuracy (~990 evals, <2ms).
    On flop/turn, runs num_sims random rollouts.

    Args:
        my_hand_strs: list of 2 card strings, e.g. ['Ah', 'Kd']
        board_strs:   list of 0-5 board card strings
        num_sims:     Monte Carlo iterations (ignored on river)

    Returns:
        tuple(int, int, int): (wins, ties, total_simulations)
    """
    cdef int wins = 0
    cdef int ties = 0
    cdef int i, j, k
    cdef int n_rem, board_needed, sample_size

    # Convert strings → cached eval7.Card objects
    my_cards = [_CARD_OBJ[s] for s in my_hand_strs]
    board_cards = [_CARD_OBJ[s] for s in board_strs]

    # Build remaining deck (exclude known cards)
    dead = set(my_hand_strs)
    for s in board_strs:
        dead.add(s)
    remaining = [_CARD_OBJ[s] for s in FULL_DECK_STRS if s not in dead]
    n_rem = len(remaining)

    board_needed = 5 - len(board_cards)
    sample_size = 2 + board_needed  # opponent hand + remaining board

    # ── RIVER: exact enumeration ──
    if board_needed == 0:
        return _exact_river(my_cards, board_cards, remaining, n_rem)

    # ── FLOP / TURN: Monte Carlo ──
    # Use index-based partial Fisher-Yates shuffle for speed.
    # Only shuffles first `sample_size` elements per iteration.
    cdef list idx = list(range(n_rem))

    for i in range(num_sims):
        # Partial shuffle: O(sample_size) instead of O(n)
        for k in range(sample_size):
            j = _randint(k, n_rem - 1)
            if k != j:
                idx[k], idx[j] = idx[j], idx[k]

        opp = [remaining[idx[0]], remaining[idx[1]]]
        extra = [remaining[idx[m]] for m in range(2, sample_size)]
        full_board = board_cards + extra

        my_score = eval7.evaluate(my_cards + full_board)
        opp_score = eval7.evaluate(opp + full_board)

        if my_score > opp_score:
            wins += 1
        elif my_score == opp_score:
            ties += 1

    return (wins, ties, num_sims)


def _exact_river(list my_cards, list board_cards, list remaining, int n):
    """
    Exact equity on the river by enumerating all C(n,2) opponent hands.
    Pre-computes our score once; only evaluates opponent hands in the loop.
    Typically n≈45 → C(45,2)=990 evaluations. Takes <2ms.
    """
    cdef int wins = 0
    cdef int ties = 0
    cdef int total = 0
    cdef int i, j

    my_score = eval7.evaluate(my_cards + board_cards)

    for i in range(n):
        for j in range(i + 1, n):
            opp_score = eval7.evaluate([remaining[i], remaining[j]] + board_cards)
            if my_score > opp_score:
                wins += 1
            elif my_score == opp_score:
                ties += 1
            total += 1

    return (wins, ties, total)


# ═══════════════════════════════════════════════════════════════
#  BOUNTY-AWARE EV CALCULATIONS
# ═══════════════════════════════════════════════════════════════
#
#  ENGINE BOUNTY MATH (from engine.py get_delta):
#    If winner's bounty is hit:
#      delta = opponent_contribution × BOUNTY_RATIO + BOUNTY_CONSTANT
#    Otherwise:
#      delta = opponent_contribution
#
#  All EVs below are MARGINAL: relative to folding (EV_fold = 0).

def calc_call_ev(
    double equity,
    int continue_cost,
    int pot_total,
    int opp_contribution,
    bint bounty_hit
):
    """
    Exact marginal EV of calling vs folding.

    Derivation:
      delta_win  = opp_contribution  (or ×1.5 + 10 with bounty)
      delta_lose = -(my_contribution + continue_cost)
      delta_fold = -my_contribution

      M = eq × (delta_win - delta_fold) + (1-eq) × (delta_lose - delta_fold)
        = eq × (opp_contribution + my_contribution) - (1-eq) × continue_cost
        = eq × pot_total - (1-eq) × continue_cost
        = eq × (pot_total + continue_cost) - continue_cost

      With bounty:
        M += eq × (opp_contribution × 0.5 + BOUNTY_CONSTANT)

    Args:
        equity:           P(win) from Monte Carlo/exact [0, 1]
        continue_cost:    chips to call (opp_pip - my_pip)
        pot_total:        total chips in pot (my_contrib + opp_contrib)
        opp_contribution: opponent's total investment (STARTING_STACK - opp_stack)
        bounty_hit:       True if our bounty rank is in our hand + board

    Returns:
        float: positive → call is profitable, negative → fold
    """
    cdef double pot_after_call = <double>(pot_total + continue_cost)
    cdef double ev = equity * pot_after_call - <double>continue_cost

    if bounty_hit:
        ev += equity * (<double>opp_contribution * (_BOUNTY_RATIO - 1.0) + _BOUNTY_CONSTANT)

    return ev


def calc_raise_ev(
    double equity,
    double fold_equity,
    int my_contribution,
    int opp_contribution,
    int raise_cost,
    int opp_call_addition,
    bint bounty_hit
):
    """
    Exact marginal EV of raising vs folding.

    Model: opponent either folds (prob=fold_equity) or calls and we go
    to showdown (prob=1-fold_equity).  Re-raises are modeled as calls
    for simplicity — a conservative assumption.

    Derivation (all relative to delta_fold = -my_contribution):

      If opponent folds:
        delta = opp_contribution [×1.5 + 10 if bounty]
        gain  = delta - (-my_contribution) = delta + my_contribution

      If opponent calls and we win:
        opp_new = opp_contribution + opp_call_addition
        delta   = opp_new [×1.5 + 10 if bounty]
        gain    = delta + my_contribution

      If opponent calls and we lose:
        delta = -(my_contribution + raise_cost)
        gain  = -raise_cost

      M = FE × gain_fold + (1-FE) × [eq × gain_win + (1-eq) × (-raise_cost)]

    Args:
        equity:            P(win at showdown)
        fold_equity:       P(opponent folds to our raise)
        my_contribution:   our total investment so far (STARTING_STACK - my_stack)
        opp_contribution:  opponent total investment so far
        raise_cost:        additional chips we invest (raise_to - my_pip)
        opp_call_addition: chips opponent adds to call (raise_to - opp_pip)
        bounty_hit:        True if our bounty rank is connected

    Returns:
        float: marginal EV of raising vs folding
    """
    cdef double opp_cont_f = <double>opp_contribution
    cdef double my_cont_f  = <double>my_contribution
    cdef double rc_f       = <double>raise_cost

    # ── Gain if opponent folds ──
    cdef double delta_fold_gain
    if bounty_hit:
        delta_fold_gain = opp_cont_f * _BOUNTY_RATIO + _BOUNTY_CONSTANT
    else:
        delta_fold_gain = opp_cont_f
    cdef double gain_if_fold = delta_fold_gain + my_cont_f

    # ── Gain if opponent calls and we win ──
    cdef double opp_new = opp_cont_f + <double>opp_call_addition
    cdef double delta_call_win
    if bounty_hit:
        delta_call_win = opp_new * _BOUNTY_RATIO + _BOUNTY_CONSTANT
    else:
        delta_call_win = opp_new
    cdef double gain_if_call_win = delta_call_win + my_cont_f

    # ── EV if called (showdown) ──
    cdef double ev_if_called = equity * gain_if_call_win + (1.0 - equity) * (-rc_f)

    return fold_equity * gain_if_fold + (1.0 - fold_equity) * ev_if_called
