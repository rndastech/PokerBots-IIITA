"""
Mahavir V3: Precision Exploit + Monte Carlo (No CFR)
====================================================
Key improvements over V2:
  1. REMOVED CFR (was hurting, not helping)
  2. BLUFF DETECTION: small bets (< pot*0.35) = likely bluff → call wider
  3. BET EVERY STREET: baseline folds non-premium to any bet > pot//5
  4. SMARTER DEFENSE: only discount equity vs passive opponents, not vs bluffers
  5. HIGHER BLUFF FREQUENCY: barrel all 3 streets consistently
  6. WIDER PREFLOP CALLS: don't overfold to 3-bets with playable hands
"""

from __future__ import annotations

import random
import eval7

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

RANKS = "23456789TJQKA"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
BIG_BLIND = 2


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ═══════════════════════════════════════════════════════════════════════════
# Preflop equity — hand-class tiers + gap penalty (from playersu.py)
# ═══════════════════════════════════════════════════════════════════════════
def preflop_equity(hole, bounty=None):
    r1, s1 = hole[0][0], hole[0][1]
    r2, s2 = hole[1][0], hole[1][1]
    i1, i2 = RANKS.index(r1), RANKS.index(r2)
    high, low = max(i1, i2), min(i1, i2)
    suited = (s1 == s2)
    gap = high - low

    if r1 == r2:
        eq = 0.52 + (high / 12) * 0.33
    elif high >= 12 and low >= 10:
        eq = 0.63 + (min(low, high) / 12) * 0.09
    elif high == 12:
        eq = 0.50 + (low / 12) * 0.08
    elif high == 11:
        eq = 0.45 + (low / 12) * 0.08
    elif high == 10 and low >= 8:
        eq = 0.50 + (low / 12) * 0.06
    elif high >= 9 and low >= 7:
        eq = 0.46 + (low / 12) * 0.05
    elif high >= 9:
        eq = 0.38 + (high / 12) * 0.06
    elif high >= 6:
        eq = 0.30 + (high / 12) * 0.06
    else:
        eq = 0.20 + (high / 12) * 0.05

    if gap >= 8:    eq -= 0.08
    elif gap >= 6:  eq -= 0.05
    elif gap >= 4:  eq -= 0.02

    if suited:
        eq += 0.06 if gap <= 3 else 0.03

    if gap == 1: eq += 0.03
    if gap == 0 and r1 != r2: eq += 0.01

    if bounty and (r1 == bounty or r2 == bounty):
        eq += 0.07

    return max(0.15, min(eq, 0.96))


# ═══════════════════════════════════════════════════════════════════════════
# Main Bot
# ═══════════════════════════════════════════════════════════════════════════
class Player(Bot):
    def __init__(self):
        self._seen_state_keys = set()
        self.opp_pre_actions = 0
        self.opp_pre_raises = 0
        self.opp_post_actions = 0
        self.opp_post_aggr = 0
        self._i_raised_pre = False
        random.seed(9917231)

    # ─────────────────────────────────────────────────────────────
    # Engine callbacks
    # ─────────────────────────────────────────────────────────────
    def handle_new_round(self, game_state, round_state, active):
        self._seen_state_keys = set()
        self._i_raised_pre = False

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    # ─────────────────────────────────────────────────────────────
    # Main decision
    # ─────────────────────────────────────────────────────────────
    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:street] if street > 0 else []

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        continue_cost = opp_pip - my_pip
        pot = 800 - (round_state.stacks[0] + round_state.stacks[1])

        bounty_rank = round_state.bounties[active]
        opp_bounty_rank = round_state.bounties[1 - active]

        self._observe_opponent_action(round_state, active)

        # Variance control: protect lead late
        if self._should_preserve_lead(game_state.bankroll, game_state.round_num):
            if CheckAction in legal: return CheckAction()
            if FoldAction in legal: return FoldAction()

        # Clock safety
        if game_state.game_clock < 1.0:
            if CheckAction in legal: return CheckAction()
            if continue_cost <= 2 and CallAction in legal: return CallAction()
            if FoldAction in legal: return FoldAction()
            return CallAction() if CallAction in legal else CheckAction()

        mn = mx = 0
        if RaiseAction in legal:
            mn, mx = round_state.raise_bounds()

        agg_pre, agg_post = self._opponent_rates()
        has_bounty = self._has_bounty(my_cards + board_cards, bounty_rank)

        # ════════════════════════════════════════════════════════
        #  PRE-FLOP
        # ════════════════════════════════════════════════════════
        if street == 0:
            eq = preflop_equity(my_cards, bounty_rank)
            # Bonus if we hold opponent's bounty card
            if opp_bounty_rank and (my_cards[0][0] == opp_bounty_rank or my_cards[1][0] == opp_bounty_rank):
                eq = min(0.96, eq + 0.06)

            # HIGH AGGRESSION: 4-bet+ (cost > 20)
            if continue_cost > BIG_BLIND * 10:
                if eq >= 0.74:
                    self._i_raised_pre = True
                    return RaiseAction(clamp(mx, mn, mx))
                if eq >= 0.55 and CallAction in legal:
                    return CallAction()
                return FoldAction() if FoldAction in legal else self._safe_check(legal)

            # MEDIUM raise (3-bet, cost 3-20)
            elif continue_cost > BIG_BLIND:
                if eq >= 0.76:
                    self._i_raised_pre = True
                    return RaiseAction(clamp(mx, mn, mx))
                elif eq >= 0.62:
                    target = int(continue_cost * 3)
                    if RaiseAction in legal and target >= mn:
                        self._i_raised_pre = True
                        return RaiseAction(clamp(target, mn, mx))
                    return CallAction() if CallAction in legal else FoldAction()
                elif eq >= 0.48:
                    return CallAction() if CallAction in legal else FoldAction()
                elif eq >= 0.42 and continue_cost <= BIG_BLIND * 4:
                    return CallAction() if CallAction in legal else FoldAction()
                else:
                    return FoldAction() if FoldAction in legal else self._safe_check(legal)

            # NO RAISE / limped
            else:
                if eq >= 0.65:
                    self._i_raised_pre = True
                    return self._raise_to(BIG_BLIND * 4, legal, mn, mx)
                elif eq >= 0.50:
                    self._i_raised_pre = True
                    return self._raise_to(my_pip + 4, legal, mn, mx)
                elif eq >= 0.38:
                    if CheckAction in legal: return CheckAction()
                    return CallAction() if CallAction in legal else FoldAction()
                else:
                    if continue_cost == 0 and RaiseAction in legal and random.random() < 0.14:
                        self._i_raised_pre = True
                        return self._raise_to(my_pip + 4, legal, mn, mx)
                    if continue_cost == 0:
                        return CheckAction() if CheckAction in legal else FoldAction()
                    return FoldAction() if FoldAction in legal else self._safe_check(legal)

        # ════════════════════════════════════════════════════════
        #  POST-FLOP
        # ════════════════════════════════════════════════════════
        samples = self._time_samples(game_state.game_clock, street)
        eq = self._mc_equity(my_cards, board_cards, samples)

        # Bounty adjustments
        vis = [c[0] for c in board_cards]
        if bounty_rank in vis:    eq = min(1.0, eq + 0.10)
        elif has_bounty:          eq += 0.04

        # Opponent bounty evasion (only if they're actually betting)
        if opp_bounty_rank in vis and continue_cost > pot * 0.4:
            eq -= 0.12

        # Made-hand floor
        top_pair, overpair, set_made, flush_made = self._made_flags(my_cards, board_cards)
        if set_made or flush_made: eq = max(eq, 0.83)
        elif top_pair or overpair: eq = max(eq, 0.63)

        eq = clamp(eq, 0.02, 0.98)
        pot_odds = continue_cost / max(1, pot + continue_cost) if continue_cost > 0 else 0.0

        # Baseline fold threshold: they fold non-premium to any bet > max(4, pot//5)
        # So we need to bet at least: int(pot * 0.22) + 7  to exceed this
        fold_threshold = int(pot * 0.22) + 7

        # ════════════════════════════════════════════════════════
        #  WE CAN BET (cost == 0) — BET EVERY STREET
        # ════════════════════════════════════════════════════════
        if continue_cost <= 0:
            if RaiseAction in legal and pot >= 4:
                # NUTS: max raise (value hammer)
                if eq >= 0.85:
                    return RaiseAction(mx)

                # Strong: big value bet
                elif eq >= 0.72:
                    target = max(fold_threshold, int(pot * 0.80))
                    return self._raise_to(target, legal, mn, mx)

                # Good: medium value bet
                elif eq >= 0.58:
                    target = max(fold_threshold, int(pot * 0.60))
                    return self._raise_to(target, legal, mn, mx)

                # Medium: cheapest bet that forces baseline to fold (~70% fold equity)
                elif eq >= 0.38:
                    return self._raise_to(fold_threshold, legal, mn, mx)

                # Weak: bluff barrel (continuation bet logic)
                elif eq >= 0.20:
                    should_bluff = (
                        (self._i_raised_pre and street == 3) or           # c-bet always
                        (street == 3 and random.random() < 0.65) or       # flop bluff 65%
                        (street == 4 and random.random() < 0.55) or       # turn barrel 55%
                        (street == 5 and random.random() < 0.35)          # river barrel 35%
                    )
                    if should_bluff:
                        return self._raise_to(fold_threshold, legal, mn, mx)

                # Trash: occasional min-raise steal
                elif random.random() < 0.12:
                    return RaiseAction(mn)

            if CheckAction in legal: return CheckAction()
            return CallAction() if CallAction in legal else FoldAction()

        # ════════════════════════════════════════════════════════
        #  FACING A BET — smart bluff detection
        # ════════════════════════════════════════════════════════
        is_small_bet = continue_cost < pot * 0.35
        is_medium_bet = pot * 0.35 <= continue_cost < pot * 0.65
        is_big_bet = continue_cost >= pot * 0.65

        # Huge all-in type bet
        if continue_cost > 130:
            if eq >= 0.80 and CallAction in legal:
                return CallAction()
            if FoldAction in legal: return FoldAction()

        # NUTS: re-raise all-in
        if RaiseAction in legal and eq >= 0.85 and continue_cost < my_stack * 0.75:
            return RaiseAction(mx)

        # Strong: value re-raise vs passive
        if (RaiseAction in legal and eq >= 0.70
                and agg_post < 0.30 and continue_cost <= pot * 0.45):
            target = max(fold_threshold, int(pot * 0.80))
            return self._raise_to(target, legal, mn, mx)

        # ── BLUFF DETECTION ──
        # Small bet = likely bluff → call with wider range
        if is_small_bet:
            if eq >= 0.35 and CallAction in legal:
                return CallAction()
        # Medium bet → call with decent hand
        elif is_medium_bet:
            if eq >= 0.45 and CallAction in legal:
                return CallAction()
            if (set_made or flush_made or top_pair) and CallAction in legal:
                return CallAction()
        # Big bet → only call with strong hand
        else:
            if eq >= 0.55 and CallAction in legal:
                return CallAction()
            if (set_made or flush_made) and eq >= 0.45 and CallAction in legal:
                return CallAction()

        # Cheap call safety net
        if CallAction in legal and continue_cost <= max(4, pot * 0.12) and eq >= 0.28:
            return CallAction()

        # River hero-call vs aggressive opponent
        if (CallAction in legal and street == 5 and eq >= pot_odds - 0.02
                and continue_cost <= pot * 0.40 and agg_post >= 0.40):
            return CallAction()

        if FoldAction in legal: return FoldAction()
        if CheckAction in legal: return CheckAction()
        return CallAction()

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _raise_to(amount, legal, mn, mx):
        if RaiseAction in legal:
            return RaiseAction(max(mn, min(mx, int(amount))))
        if CallAction in legal: return CallAction()
        if CheckAction in legal: return CheckAction()
        return FoldAction()

    @staticmethod
    def _safe_check(legal):
        if CheckAction in legal: return CheckAction()
        if CallAction in legal: return CallAction()
        return FoldAction()

    # ─────────────────────────────────────────────────────────────
    # Opponent modeling
    # ─────────────────────────────────────────────────────────────
    def _record_opp_action(self, action_type, is_preflop):
        if is_preflop:
            self.opp_pre_actions += 1
            if action_type == "raise": self.opp_pre_raises += 1
        else:
            self.opp_post_actions += 1
            if action_type in ("raise", "bet"): self.opp_post_aggr += 1

    def _state_key(self, state):
        if state is None: return None
        hands = tuple(tuple(h) for h in state.hands)
        deck = tuple(state.deck[:state.street])
        return (state.button, state.street, tuple(state.pips),
                tuple(state.stacks), hands, tuple(state.bounties), deck)

    def _observe_opponent_action(self, round_state, active):
        prev = round_state.previous_state
        if prev is None: return
        key = self._state_key(round_state)
        if key in self._seen_state_keys: return
        self._seen_state_keys.add(key)
        opp = 1 - active

        if round_state.street != prev.street:
            if prev.button % 2 != opp: return
            act = "call" if prev.pips[opp] != prev.pips[active] else "check"
            self._record_opp_action(act, prev.street == 0)
            return

        if round_state.button != prev.button + 1: return
        if prev.button % 2 != opp: return

        opp_prev, opp_now = prev.pips[opp], round_state.pips[opp]
        my_prev = prev.pips[active]

        if opp_now > opp_prev:
            cc = my_prev - opp_prev
            act = "call" if (cc > 0 and opp_now == my_prev) else "raise"
        elif round_state.pips == prev.pips and round_state.stacks == prev.stacks:
            act = "check"
        else:
            return
        self._record_opp_action(act, round_state.street == 0)

    def _opponent_rates(self):
        pre = (self.opp_pre_raises + 2) / (self.opp_pre_actions + 5)
        post = (self.opp_post_aggr + 2) / (self.opp_post_actions + 5)
        return pre, post

    # ─────────────────────────────────────────────────────────────
    # Monte Carlo equity
    # ─────────────────────────────────────────────────────────────
    def _mc_equity(self, hero_str, board_str, samples):
        hero = [eval7.Card(c) for c in hero_str]
        board = [eval7.Card(c) for c in board_str]
        dead = {str(c) for c in hero + board}
        rem = [c for c in eval7.Deck().cards if str(c) not in dead]
        board_rem = 5 - len(board)
        wins = ties = 0

        for _ in range(samples):
            random.shuffle(rem)
            opp = rem[:2]
            run = rem[2:2 + board_rem]
            full_board = board + run if board_rem else board
            s1 = eval7.evaluate(hero + full_board)
            s2 = eval7.evaluate(opp + full_board)
            if s1 > s2: wins += 1
            elif s1 == s2: ties += 1
        return (wins + 0.5 * ties) / max(1, samples)

    # ─────────────────────────────────────────────────────────────
    # Hand features
    # ─────────────────────────────────────────────────────────────
    def _made_flags(self, hero, board):
        if not board:
            return False, False, hero[0][0] == hero[1][0], False
        hr = [c[0] for c in hero]
        br = [c[0] for c in board]
        hs = [c[1] for c in hero]
        bs = [c[1] for c in board]

        top_board = max(RANK_VALUE[r] for r in br)
        top_pair = any(RANK_VALUE[r] == top_board for r in hr)
        overpair = set_made = False
        if hr[0] == hr[1]:
            pair_rank = RANK_VALUE[hr[0]]
            overpair = pair_rank > top_board
            set_made = br.count(hr[0]) > 0
        suit_counts = {}
        for s in hs + bs:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        flush_made = any(v >= 5 for v in suit_counts.values())
        return top_pair, overpair, set_made, flush_made

    def _has_bounty(self, cards, bounty_rank):
        return any(c and c[0] == bounty_rank for c in cards)

    # ─────────────────────────────────────────────────────────────
    # Time and risk controls
    # ─────────────────────────────────────────────────────────────
    def _time_samples(self, game_clock, street):
        if game_clock < 2.5: base = 30
        elif game_clock < 6.0: base = 70
        elif game_clock < 15.0: base = 120
        elif game_clock < 30.0: base = 170
        else: base = 220
        if street == 5: base = int(base * 0.8)
        return max(20, base)

    def _should_preserve_lead(self, bankroll, round_num):
        return round_num >= 430 and bankroll >= 160


if __name__ == "__main__":
    run_bot(Player(), parse_args())