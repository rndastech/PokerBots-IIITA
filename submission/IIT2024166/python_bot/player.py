"""
╔══════════════════════════════════════════════════════════════════════╗
║  H Y D R A  —  Tournament Poker Bot  (IIITA Pokerbots 2024)       ║
║                                                                    ║
║  Architecture:                                                     ║
║    Layer 1  O(1) Preflop LUT ─── instant hand-strength ranking     ║
║    Layer 2  Cython Monte Carlo ─ eval7 C-backend equity engine     ║
║    Layer 3  Opponent Profiler ── VPIP/PFR/AF adaptive exploits     ║
║    Layer 4  Bounty Module ────── exact EV amplification math       ║
║    Layer 5  Clock Guardian ───── adaptive sim-budget scaling       ║
║                                                                    ║
║  Meta-strategy: exploit the field of LLM-generated bots that       ║
║  ignore bounty mechanics, play statically, and lack adaptation.    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

# ═══════════════════════════════════════════════════════════════════
#  CYTHON ENGINE — with pure-Python fallback for robustness
# ═══════════════════════════════════════════════════════════════════

try:
    from hydra_engine import calc_equity, calc_call_ev, calc_raise_ev
    _ENGINE = 'CYTHON'
except ImportError:
    _ENGINE = 'PYTHON'
    import eval7 as _e7
    from random import randint as _ri

    _FDECK = [r + s for r in '23456789TJQKA' for s in 'dhcs']
    _CC = {s: _e7.Card(s) for s in _FDECK}

    def calc_equity(my_hand_strs, board_strs, num_sims):
        my_c = [_CC[s] for s in my_hand_strs]
        bc = [_CC[s] for s in board_strs]
        dead = set(my_hand_strs) | set(board_strs)
        rem = [_CC[s] for s in _FDECK if s not in dead]
        n = len(rem)
        bn = 5 - len(bc)
        ss = 2 + bn
        if bn == 0:
            w, t, tot = 0, 0, 0
            ms = _e7.evaluate(my_c + bc)
            for i in range(n):
                for j in range(i + 1, n):
                    os = _e7.evaluate([rem[i], rem[j]] + bc)
                    if ms > os: w += 1
                    elif ms == os: t += 1
                    tot += 1
            return (w, t, tot)
        w, t = 0, 0
        idx = list(range(n))
        for _ in range(num_sims):
            for k in range(ss):
                j = _ri(k, n - 1)
                if k != j: idx[k], idx[j] = idx[j], idx[k]
            opp = [rem[idx[0]], rem[idx[1]]]
            extra = [rem[idx[m]] for m in range(2, ss)]
            fb = bc + extra
            ms = _e7.evaluate(my_c + fb)
            os = _e7.evaluate(opp + fb)
            if ms > os: w += 1
            elif ms == os: t += 1
        return (w, t, num_sims)

    def calc_call_ev(equity, continue_cost, pot_total, opp_contribution, bounty_hit):
        pac = float(pot_total + continue_cost)
        ev = equity * pac - float(continue_cost)
        if bounty_hit:
            # Optimal fractional bounty multiplier
            ev += equity * (float(opp_contribution) * 0.5 + 10.0)
        return ev

    def calc_raise_ev(equity, fold_equity, my_contribution, opp_contribution,
                      raise_cost, opp_call_addition, bounty_hit):
        mc = float(my_contribution); oc = float(opp_contribution); rc = float(raise_cost)
        dfg = (oc * 1.5 + 10.0) if bounty_hit else oc
        gif = dfg + mc
        on = oc + float(opp_call_addition)
        dcw = (on * 1.5 + 10.0) if bounty_hit else on
        gcw = dcw + mc
        eic = equity * gcw + (1.0 - equity) * (-rc)
        return fold_equity * gif + (1.0 - fold_equity) * eic


# ═══════════════════════════════════════════════════════════════════
#  OMEGA ADDITIONS (Board Texture / Dependencies)
# ═══════════════════════════════════════════════════════════════════
import random

def analyze_board(board_cards_str):
    if not board_cards_str: return {'wet': False, 'paired': False}
    suits = [c[1] for c in board_cards_str]
    ranks = [c[0] for c in board_cards_str]
    is_monotone = (len(set(suits)) == 1 and len(suits) >= 3)
    is_two_tone = (len(set(suits)) == 2 and len(suits) >= 3)
    is_paired = len(set(ranks)) < len(ranks)
    is_wet = (is_monotone or is_two_tone) and not is_paired
    return {'wet': is_wet, 'paired': is_paired}

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════

RANK_VAL = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
            '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}


# ═══════════════════════════════════════════════════════════════════
#  PLAYER CLASS
# ═══════════════════════════════════════════════════════════════════

class Player(Bot):
    """
    Hydra — a multi-layered exploitative poker bot.
    """

    def __init__(self):
        """Build preflop LUT and initialize opponent tracker."""

        # ── Preflop Strength LUT ──
        self.preflop_lut = {
            'AA': 'S', 'AKs': 'S', 'AKo': 'S', 'AQs': 'A', 'AQo': 'A', 'AJs': 'A', 'AJo': 'B', 'ATs': 'B',
            'ATo': 'C', 'A9s': 'B', 'A9o': 'C', 'A8s': 'B', 'A8o': 'C', 'A7s': 'B', 'A7o': 'C', 'A6s': 'B',
            'A6o': 'C', 'A5s': 'B', 'A5o': 'C', 'A4s': 'B', 'A4o': 'C', 'A3s': 'B', 'A3o': 'C', 'A2s': 'B',
            'A2o': 'C', 'KK': 'S', 'KQs': 'A', 'KQo': 'B', 'KJs': 'B', 'KJo': 'B', 'KTs': 'B', 'KTo': 'C',
            'K9s': 'C', 'K9o': 'C', 'K8s': 'C', 'K8o': 'C', 'K7s': 'C', 'K7o': 'C', 'K6s': 'C', 'K6o': 'C',
            'K5s': 'C', 'K5o': 'C', 'K4s': 'C', 'K4o': 'C', 'K3s': 'C', 'K3o': 'C', 'K2s': 'C', 'K2o': 'C',
            'QQ': 'S', 'QJs': 'B', 'QJo': 'C', 'QTs': 'B', 'QTo': 'C', 'Q9s': 'C', 'Q9o': 'C', 'Q8s': 'C',
            'Q8o': 'C', 'Q7s': 'C', 'Q7o': 'C', 'Q6s': 'C', 'Q6o': 'C', 'Q5s': 'C', 'Q5o': 'C', 'Q4s': 'C',
            'Q4o': 'C', 'Q3s': 'C', 'Q3o': 'C', 'Q2s': 'C', 'Q2o': 'C', 'JJ': 'S', 'JTs': 'B', 'JTo': 'C',
            'J9s': 'B', 'J9o': 'C', 'J8s': 'C', 'J8o': 'C', 'J7s': 'C', 'J7o': 'C', 'J6s': 'C', 'J6o': 'C',
            'J5s': 'C', 'J5o': 'C', 'J4s': 'C', 'J4o': 'C', 'J3s': 'C', 'J3o': 'C', 'J2s': 'C', 'J2o': 'C',
            'TT': 'A', 'T9s': 'B', 'T9o': 'C', 'T8s': 'B', 'T8o': 'C', 'T7s': 'C', 'T7o': 'C', 'T6s': 'C',
            'T6o': 'C', 'T5s': 'C', 'T5o': 'C', 'T4s': 'C', 'T4o': 'C', 'T3s': 'C', 'T3o': 'C', 'T2s': 'C',
            'T2o': 'C', '99': 'A', '98s': 'B', '98o': 'C', '97s': 'C', '97o': 'C', '96s': 'C', '96o': 'C',
            '95s': 'C', '95o': 'C', '94s': 'C', '94o': 'C', '93s': 'C', '93o': 'C', '92s': 'C', '92o': 'C',
            '88': 'A', '87s': 'B', '87o': 'C', '86s': 'C', '86o': 'C', '85s': 'C', '85o': 'C', '84s': 'C',
            '84o': 'C', '83s': 'C', '83o': 'C', '82s': 'C', '82o': 'C', '77': 'B', '76s': 'B', '76o': 'C',
            '75s': 'C', '75o': 'C', '74s': 'C', '74o': 'C', '73s': 'C', '73o': 'C', '72s': 'C', '72o': 'C',
            '66': 'B', '65s': 'B', '65o': 'C', '64s': 'C', '64o': 'C', '63s': 'C', '63o': 'C', '62s': 'C',
            '62o': 'C', '55': 'B', '54s': 'B', '54o': 'C', '53s': 'C', '53o': 'C', '52s': 'C', '52o': 'C',
            '44': 'B', '43s': 'C', '43o': 'C', '42s': 'C', '42o': 'C', '33': 'B', '32s': 'C', '32o': 'C',
            '22': 'B'
        }

        # ── Opponent Statistics ──
        self.total_hands = 0
        self.opp_vpip = 0           # hands opponent voluntarily entered
        self.opp_pfr = 0            # hands opponent raised preflop
        self.opp_fold_total = 0     # total folds observed
        self.opp_post_bets = 0      # postflop bets/raises by opponent
        self.opp_post_checks = 0    # postflop checks by opponent
        self.opp_fold_to_us = 0     # folds when we were aggressor
        self.our_aggression_hands = 0  # hands where we bet/raised

        # ── Per-round tracking ──
        self._r_opp_raised_pre = False
        self._r_opp_saw_flop = False
        self._r_we_aggressed = False
        self._r_opp_bets = 0
        self._r_opp_checks = 0
        self._r_street_actions = 0
        self._r_last_street = -1  # track street transitions

    # ───────────────────────────────────────────────────────────
    #  PREFLOP HAND STRENGTH
    # ───────────────────────────────────────────────────────────

    @staticmethod
    def _hand_str(high, low, htype):
        """
        Compute preflop hand strength ∈ [0.20, 0.95].
        Calibrated against known heads-up equities:
          AA≈0.95  KK≈0.91  QQ≈0.88  JJ≈0.84  TT≈0.80
          AKs≈0.81  AKo≈0.77  AQs≈0.76  99≈0.76  22≈0.50
        """
        if htype == 'p':
            return min(0.95, 0.50 + (high - 2) * 0.0375)

        gap = high - low
        score = (high + low - 4) * 0.018 + 0.28

        # High-card kickers
        if high == 14:   score += 0.08
        elif high == 13: score += 0.04
        elif high == 12: score += 0.02

        # Suited & connectivity
        if htype == 's': score += 0.03
        if gap <= 1:     score += 0.015
        elif gap <= 2:   score += 0.005
        elif gap >= 5:   score -= 0.01

        return max(0.20, min(0.90, score))

    def _get_strength(self, card1, card2):
        """O(1) preflop strength lookup from two card strings."""
        r1, r2 = RANK_VAL[card1[0]], RANK_VAL[card2[0]]
        high, low = (card1[0], card2[0]) if r1 >= r2 else (card2[0], card1[0])
        if card1[0] == card2[0]:
            h_str = f"{high}{low}"
        elif card1[1] == card2[1]:
            h_str = f"{high}{low}s"
        else:
            h_str = f"{high}{low}o"
        return self.preflop_lut.get(h_str, 'C')

    # ───────────────────────────────────────────────────────────
    #  OPPONENT PROFILING
    # ───────────────────────────────────────────────────────────

    def _classify(self):
        """
        Classify opponent into one of: precision_nit, nit, station, tag, maniac, unknown.
        """
        if self.total_hands < 25:
            return 'unknown'

        n = max(1, self.total_hands)
        vpip_r = self.opp_vpip / n
        pfr_r = self.opp_pfr / n
        fold_r = self.opp_fold_total / n
        agg = self.opp_post_bets / max(1, self.opp_post_bets + self.opp_post_checks)

        # Omega addition: Detect 167 (Precision Nit)
        if fold_r > 0.60 and vpip_r < 0.40:
            return 'precision_nit'

        if vpip_r < 0.30 or fold_r > 0.50:
            return 'nit'
        if vpip_r > 0.58 and agg < 0.38:
            return 'station'
        if vpip_r > 0.58 and agg > 0.55:
            return 'maniac'
        if pfr_r > 0.20:
            return 'tag'
        return 'unknown'

    def _fold_equity(self, street, opp_type):
        """Estimate probability opponent folds to our bet/raise."""
        if self.total_hands < 15:
            return 0.35

        if self.our_aggression_hands > 0:
            base = self.opp_fold_to_us / self.our_aggression_hands
        else:
            base = 0.35

        # Opponent-type adjustments
        if opp_type == 'precision_nit':
            base = max(base, 0.75) # They auto-fold to anything unless holding premium
        elif opp_type == 'nit':
            base = max(base, 0.50)
        elif opp_type == 'station':
            base = min(base, 0.18)
        elif opp_type == 'maniac':
            base = min(base, 0.12)

        if street >= 5:
            base *= 0.85

        return max(0.05, min(0.85, base))

    # ───────────────────────────────────────────────────────────
    #  BOUNTY CHECK
    # ───────────────────────────────────────────────────────────

    @staticmethod
    def _bounty_connected(bounty_rank, my_cards, board_cards):
        """
        Check if our bounty rank appears in hole cards OR board.
        If the rank is in our HAND, bounty is guaranteed on any win.
        This is the single biggest edge in the game.
        """
        if bounty_rank == '-1':
            return False
        for c in my_cards:
            if c[0] == bounty_rank:
                return True
        for c in board_cards:
            if c[0] == bounty_rank:
                return True
        return False

    # ───────────────────────────────────────────────────────────
    #  CLOCK MANAGEMENT
    # ───────────────────────────────────────────────────────────

    @staticmethod
    def _sim_budget(clock, street):
        """
        Scale Monte Carlo iterations by remaining clock.
        River always uses exact enumeration (ignores this).
        """
        if clock > 35:
            return 300 if street == 3 else 200
        if clock > 18:
            return 150
        if clock > 8:
            return 80
        if clock > 3:
            return 40
        return 20

    # ───────────────────────────────────────────────────────────
    #  BET SIZING ENGINE
    # ───────────────────────────────────────────────────────────

    def _value_bet_size(self, street, pot, min_r, max_r, equity, bh, opp_type):
        """Compute value bet/raise size as a fraction of pot."""
        if street == 3:
            frac = 0.55
        elif street == 4:
            frac = 0.66
        else:
            frac = 0.75

        # Bounty: pump the pot — our wins are 1.5× + 10
        if bh:
            frac = min(frac * 1.25, 1.0)

        # vs calling station: max extraction
        if opp_type == 'station':
            frac = min(frac * 1.15, 1.0)
        elif opp_type == 'nit':
            frac *= 0.80  # smaller to keep them in

        target = max(min_r, int(pot * frac))
        target = min(target, max_r)

        # Jam with monsters
        if equity > 0.82 and pot > 20:
            target = max_r

        return target

    def _raise_vs_bet_size(self, street, pot, opp_pip, min_r, max_r, equity, bh, opp_type):
        """Compute raise size when facing an opponent's bet."""
        # Standard: 2.5–3× their bet
        mult = 2.5 if street <= 4 else 3.0
        target = max(min_r, int(opp_pip * mult))

        if bh:
            target = max(target, int(target * 1.15))
        if opp_type == 'station':
            target = max(target, int(target * 1.1))

        target = max(min_r, min(target, max_r))

        if equity >= 0.78:
            target = max_r

        return target

    # ───────────────────────────────────────────────────────────
    #  ROUND LIFECYCLE HOOKS
    # ───────────────────────────────────────────────────────────

    def handle_new_round(self, game_state, round_state, active):
        """Reset per-round tracking at the start of each hand."""
        self._r_opp_raised_pre = False
        self._r_opp_saw_flop = False
        self._r_we_aggressed = False
        self._r_opp_bets = 0
        self._r_opp_checks = 0
        self._r_street_actions = 0
        self._r_last_street = -1

    def handle_round_over(self, game_state, terminal_state, active):
        """Update opponent model with data from the completed hand."""
        self.total_hands += 1
        prev = terminal_state.previous_state
        street = prev.street
        delta = terminal_state.deltas[active]
        opp_cards = prev.hands[1 - active]

        # ── VPIP: did opponent see a flop (or showdown preflop)? ──
        if street >= 3 or (len(opp_cards) > 0 and street == 0):
            self.opp_vpip += 1

        # ── PFR ──
        if self._r_opp_raised_pre:
            self.opp_pfr += 1

        # ── Folds ──
        if len(opp_cards) == 0 and delta > 0:
            # Opponent folded, we won
            self.opp_fold_total += 1
            if self._r_we_aggressed:
                self.opp_fold_to_us += 1

        # ── Track our aggression ──
        if self._r_we_aggressed:
            self.our_aggression_hands += 1

        # ── Postflop aggression stats ──
        self.opp_post_bets += self._r_opp_bets
        self.opp_post_checks += self._r_opp_checks

    # ───────────────────────────────────────────────────────────
    #  EMERGENCY MODE
    # ───────────────────────────────────────────────────────────

    @staticmethod
    def _emergency(legal_actions, continue_cost, my_cards):
        """
        Ultra-fast decision when clock < 2s.
        Zero computation — pure heuristic to avoid TLE forfeit.
        """
        if continue_cost == 0:
            if CheckAction in legal_actions:
                return CheckAction()
        r1 = RANK_VAL.get(my_cards[0][0], 2) if len(my_cards) > 0 else 2
        r2 = RANK_VAL.get(my_cards[1][0], 2) if len(my_cards) > 1 else 2
        high = max(r1, r2)
        if r1 == r2 or high >= 12:
            if CallAction in legal_actions:
                return CallAction()
        if FoldAction in legal_actions:
            return FoldAction()
        if CheckAction in legal_actions:
            return CheckAction()
        return CallAction()

    # ───────────────────────────────────────────────────────────
    #  PREFLOP DECISION ENGINE
    # ───────────────────────────────────────────────────────────

    def _preflop_action(self, round_state, active, legal_actions,
                        continue_cost, my_pip, opp_pip, my_stack, opp_stack,
                        my_bounty, my_cards):
        """
        O(1) exact TAG baseline decisions augmented with Anti-MIT trapping.
        """
        base_tier = self._get_strength(my_cards[0], my_cards[1])
        tier_order = ['C', 'B', 'A', 'S']

        # 1. Bounty Pre-Flop Modifier: upgrade tier if one card matches our bounty
        has_bounty = (my_cards[0][0] == my_bounty or my_cards[1][0] == my_bounty)
        if has_bounty:
            current_idx = tier_order.index(base_tier)
            if current_idx < len(tier_order) - 1:
                base_tier = tier_order[current_idx + 1]

        # 2. Trap the Aggressor logic
        pfr_freq = self.opp_pfr / max(1, self.total_hands)
        is_aggressive = pfr_freq > 0.25
        facing_raise = continue_cost > 0 and opp_pip > BIG_BLIND

        h_str = my_cards[0][0] + my_cards[1][0]
        if my_cards[0][1] == my_cards[1][1]: h_str += 's'
        elif my_cards[0][0] != my_cards[1][0]: h_str += 'o'

        if facing_raise and is_aggressive:
            self._r_opp_raised_pre = True
            # Smoothly flat-call with QQ/AKs to trap
            if base_tier == 'S' and any(x in h_str for x in ['QQ', 'AKs']) and CallAction in legal_actions:
                return CallAction()
            
            # Increase 4-bet jamming frequency with AA/KK
            if base_tier == 'S' and any(x in h_str for x in ['AA', 'KK']) and RaiseAction in legal_actions:
                self._r_we_aggressed = True
                mn, mx = round_state.raise_bounds()
                return RaiseAction(mx)

        # Baseline Strategy (Adaptive Pot Odds Defense)
        if facing_raise:
            self._r_opp_raised_pre = True

        pot_odds = continue_cost / max(1, continue_cost + my_pip + opp_pip)

        if base_tier == 'S':
            if RaiseAction in legal_actions:
                mn, mx = round_state.raise_bounds()
                tgt = my_pip + max(10, continue_cost * 4) + max(0, -continue_cost + 10)
                self._r_we_aggressed = True
                return RaiseAction(int(max(mn, min(mx, tgt))))
            elif CallAction in legal_actions:
                return CallAction()

        elif base_tier == 'A':
            # Tier A defends easily against normal 3-bets/4-bets
            if RaiseAction in legal_actions and pot_odds <= 0.55:
                mn, mx = round_state.raise_bounds()
                tgt = my_pip + max(8, continue_cost * 3)
                self._r_we_aggressed = True
                return RaiseAction(int(max(mn, min(mx, tgt))))
            if CallAction in legal_actions:
                return CallAction()

        elif base_tier == 'B':
            # Tier B calls raises smoothly up to high pot odds
            if continue_cost == 0 and RaiseAction in legal_actions:
                mn, mx = round_state.raise_bounds()
                tgt = my_pip + max(6, int(opp_pip * 2.5))
                self._r_we_aggressed = True
                return RaiseAction(int(max(mn, min(mx, tgt))))
            elif pot_odds <= 0.45 and CallAction in legal_actions:
                return CallAction()
            elif FoldAction in legal_actions:
                return FoldAction()
            elif CallAction in legal_actions:
                return CallAction()

        else: # Tier C
            if continue_cost == 0:
                # Occasional steal from SB with C-tier
                if RaiseAction in legal_actions and my_pip == SMALL_BLIND and random.random() < 0.15:
                    mn, mx = round_state.raise_bounds()
                    self._r_we_aggressed = True
                    return RaiseAction(max(mn, min(mx, my_pip + 4)))
                if CheckAction in legal_actions:
                    return CheckAction()
            elif FoldAction in legal_actions:
                return FoldAction()

        # Fallbacks
        if CheckAction in legal_actions and continue_cost == 0:
            return CheckAction()
        if CallAction in legal_actions:
            return CallAction()
        return FoldAction()

    # ───────────────────────────────────────────────────────────
    #  MAIN ACTION ENTRY POINT
    # ───────────────────────────────────────────────────────────

    def get_action(self, game_state, round_state, active):
        """
        Main decision engine. Called every time the engine needs our action.

        Decision flow:
          1. Emergency mode if clock < 2s
          2. Preflop: O(1) LUT with bounty boost
          3. Postflop: Monte Carlo equity → bounty-adjusted EV → exploit
        """
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
        clock = game_state.game_clock

        my_contribution = STARTING_STACK - my_stack
        opp_contribution = STARTING_STACK - opp_stack
        pot = my_contribution + opp_contribution

        # ═══════ LAYER 5: CLOCK GUARDIAN ═══════
        if clock < 2.0:
            return self._emergency(legal_actions, continue_cost, my_cards)

        # ═══════ PREFLOP ═══════
        if street == 0:
            return self._preflop_action(
                round_state, active, legal_actions, continue_cost,
                my_pip, opp_pip, my_stack, opp_stack, my_bounty, my_cards)

        # ═══════ POST-FLOP ═══════

        # ── Track opponent's last action ──
        # Reset per-street action counter on street transition
        if street != self._r_last_street:
            self._r_street_actions = 0
            self._r_last_street = street

        if continue_cost > 0:
            self._r_opp_bets += 1
        elif self._r_street_actions > 0:
            # They checked back (not the first action of the street)
            self._r_opp_checks += 1
        self._r_street_actions += 1

        # ── Bounty check ──
        bounty_hit = self._bounty_connected(my_bounty, my_cards, board_cards)

        # ── Equity calculation (Cython-accelerated or fallback) ──
        sims = self._sim_budget(clock, street)
        wins, ties, total = calc_equity(list(my_cards), list(board_cards), sims)
        equity = (wins + 0.5 * ties) / max(1, total)

        # ── Opponent classification ──
        opp_type = self._classify()

        # ────────── OMEGA OMNI-ADAPTIVE ENGINES ──────────
        bext = analyze_board(board_cards)
        spr = my_stack / max(pot, 1)
        rng = random.random()

        if continue_cost > 0:
            ev_call = calc_call_ev(equity, continue_cost, pot, opp_contribution, bounty_hit)

            # ── 167 PRECISION NIT EXPLOIT (FOLD vs MONSTER SIZES) ──
            if opp_type == 'precision_nit':
                if continue_cost >= pot * 0.40 and equity < 0.70:
                    return FoldAction() if FoldAction in legal_actions else CheckAction()

            if ev_call > 0:
                if RaiseAction in legal_actions:
                    raise_thresh = 0.65 if not bounty_hit else 0.58
                    # Against stations, value raise heavier
                    if opp_type == 'station': raise_thresh -= 0.05
                    
                    if equity >= raise_thresh:
                        mn, mx = round_state.raise_bounds()
                        base_tgt = pot * (0.85 if bext['wet'] else 0.60)
                        if equity > 0.85 and spr < 4:
                            base_tgt = mx # Jam strictly
                        tgt = max(mn, min(mx, my_pip + int(base_tgt)))
                        
                        fe = self._fold_equity(street, opp_type)
                        ev_raise = calc_raise_ev(equity, fe, my_contribution, opp_contribution, tgt - my_pip, tgt - opp_pip, bounty_hit)
                        
                        if ev_raise > ev_call:
                            self._r_we_aggressed = True
                            return RaiseAction(tgt)
                return CallAction()

            # Semi-bluff raises when EV call is negative but we have fold equity
            if RaiseAction in legal_actions and equity > 0.25 and opp_type in ('nit', 'tag', 'unknown'):
                fe = self._fold_equity(street, opp_type)
                if fe > 0.50:
                    mn, mx = round_state.raise_bounds()
                    bluff = max(mn, min(mx, my_pip + int(pot * 0.45)))
                    ev_bluff = calc_raise_ev(equity, fe, my_contribution, opp_contribution, bluff - my_pip, bluff - opp_pip, bounty_hit)
                    if ev_bluff > 0:
                        self._r_we_aggressed = True
                        return RaiseAction(bluff)

            if FoldAction in legal_actions: return FoldAction()
            return CheckAction() if CheckAction in legal_actions else CallAction()

        # ────────── NOT FACING A BET ──────────
        else:
            if RaiseAction in legal_actions:
                # ── 167 PRECISION NIT EXPLOIT (CHECK-TRAP STRONG HANDS) ──
                if opp_type == 'precision_nit' and equity > 0.65:
                    return CheckAction() # They will auto-fire pot*0.22+7

                bet_thresh = 0.50 if not bounty_hit else 0.42
                
                # ── POLARIZED WET/DRY SIZING ──
                if equity >= bet_thresh:
                    mn, mx = round_state.raise_bounds()
                    b_size = 0.75 if bext['wet'] else 0.45
                    if equity > 0.82 and spr < 3.0: 
                        b_size = 3.0 # Shove bounds
                    
                    # ── MIXED STRATEGY: Random check back strong hands to induce ──
                    if equity >= 0.75 and rng < 0.15 and opp_type != 'station':
                        return CheckAction()

                    sz = max(mn, min(mx, my_pip + int(pot * b_size)))
                    self._r_we_aggressed = True
                    return RaiseAction(sz)

                # ── 167 PRECISION NIT EXPLOIT (MIN-BET BLUFF WEAK HANDS) ──
                if opp_type == 'precision_nit' and equity < 0.40:
                    mn, mx = round_state.raise_bounds()
                    sz = max(mn, min(mx, my_pip + max(mn, 2))) # absolute minimum
                    self._r_we_aggressed = True
                    return RaiseAction(sz)

                # Standard Bluffing
                if opp_type in ('nit', 'unknown', 'tag') and equity >= 0.20:
                    fe = self._fold_equity(street, opp_type)
                    if fe > 0.40 and rng < 0.30: # 30% mixed strategy bluff freq
                        mn, mx = round_state.raise_bounds()
                        bluff = max(mn, min(mx, my_pip + int(pot * 0.35))) # small bluff
                        self._r_we_aggressed = True
                        return RaiseAction(bluff)

            return CheckAction()


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    run_bot(Player(), parse_args())
