from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import eval7
from collections import deque

# ═══════════════════════════════════════════════════════════════════════
#  PRECISION EXPLOIT BOT  vs  Baseline
#
#  Baseline post-flop call logic (key insight):
#    Calls post-flop bet ONLY when:
#      - premium_preflop (always), OR
#      - strong_preflop AND cost <= max(4, pot//5), OR
#      - has_bounty_rank AND cost <= max(6, pot//6)
#
#  EXPLOIT:
#    If we raised 4x BB pre (~8 chips), pot ~16. pot//5 = 3.
#    → baseline calls our ANY-SIZE c-bet only with premium.
#    → baseline folds ~65-75% of time to our c-bet!
#
#    So: bluff SMALL (saves money on misses, same fold rate)
#        value-bet BIG (extract max from premium calling range)
#        JAM or FOLD vs 4-bets: no more re-raise wars
#        never call their all-in without 60%+ equity
# ═══════════════════════════════════════════════════════════════════════

RANKS = "23456789TJQKA"


def rank_idx(r):
    return RANKS.index(r)


def preflop_equity(hole, bounty=None):
    """
    Accurate heads-up preflop equity estimate.
    Key fix: gap penalty so Q3s / K2o aren't overrated.
    """
    r1, s1 = hole[0][0], hole[0][1]
    r2, s2 = hole[1][0], hole[1][1]
    i1, i2 = rank_idx(r1), rank_idx(r2)
    high, low = max(i1, i2), min(i1, i2)
    suited = (s1 == s2)
    gap = high - low

    # ── Base equity by hand class ──
    if r1 == r2:                                # Pairs: 22→0.52, AA→0.85
        eq = 0.52 + (high / 12) * 0.33

    elif high >= 12 and low >= 10:              # AK AQ AJ KQ KJ → 0.63-0.72
        eq = 0.63 + (min(low, high) / 12) * 0.09

    elif high == 12:                            # Ax (A-high non-pair)
        eq = 0.50 + (low / 12) * 0.08          # A2→0.51, AJ→~0.58

    elif high == 11:                            # Kx
        eq = 0.45 + (low / 12) * 0.08          # K2→0.46, KJ→~0.52

    elif high == 10 and low >= 8:               # QJ QT JT → broadway connectors
        eq = 0.50 + (low / 12) * 0.06

    elif high >= 9 and low >= 7:                # T9 98 87 → mid connectors
        eq = 0.46 + (low / 12) * 0.05

    elif high >= 9:                             # T-x, 9-x gapped
        eq = 0.38 + (high / 12) * 0.06

    elif high >= 6:                             # mid-low cards
        eq = 0.30 + (high / 12) * 0.06

    else:
        eq = 0.20 + (high / 12) * 0.05

    # ── Gap penalty (large gaps are weak speculative hands) ──
    if gap >= 8:    eq -= 0.08   # e.g. A2, K3, Q4
    elif gap >= 6:  eq -= 0.05   # e.g. Q6, J5
    elif gap >= 4:  eq -= 0.02   # e.g. J7, T6

    # ── Suited bonus (only meaningful when gap is small) ──
    if suited:
        if gap <= 3: eq += 0.06   # connected suited: big bonus
        else:        eq += 0.03   # gapped suited: small bonus

    # ── Connector bonus ──
    if gap == 1: eq += 0.03
    if gap == 0 and r1 != r2: eq += 0.01   # same-rank different suits

    # ── Bounty card in hand ──
    if bounty and (r1 == bounty or r2 == bounty):
        eq += 0.07

    return max(0.15, min(eq, 0.96))


def mc_equity(hole_cards, community_cards, sims=150):
    """Monte-Carlo equity vs random opponent hand."""
    try:
        hole  = [eval7.Card(c) for c in hole_cards]
        board = [eval7.Card(c) for c in community_cards if c]
    except Exception:
        return 0.5

    deck = eval7.Deck()
    known = {str(c) for c in hole + board}
    deck.cards = [c for c in deck.cards if str(c) not in known]

    remain = 5 - len(board)
    wins = ties = total = 0

    for _ in range(sims):
        random.shuffle(deck.cards)
        need = remain + 2
        if len(deck.cards) < need:
            continue
        cards     = deck.cards[:need]
        run_board = board + cards[:remain]
        opp_hole  = cards[remain:]
        if len(hole + run_board) != 7:
            continue
        try:
            my  = eval7.evaluate(hole + run_board)
            opp = eval7.evaluate(opp_hole + run_board)
            total += 1
            if my > opp:    wins += 1
            elif my == opp: ties += 1
        except Exception:
            continue

    return (wins + 0.5 * ties) / total if total > 0 else 0.5


class Player(Bot):

    def __init__(self):
        self.hands_played  = 0
        self.recent        = deque(maxlen=50)

        # Per-hand state
        self._i_raised_pre      = False
        self._they_raised_pre   = False
        self._aggression_level  = 0    # how many raises have happened pre-flop
        self._prev_street        = -1

    # ── safe helpers ──────────────────────────────────────────────
    @staticmethod
    def _chk_fold(legal):
        if CheckAction in legal: return CheckAction()
        return FoldAction()

    @staticmethod
    def _call_chk(legal):
        if CallAction in legal: return CallAction()
        if CheckAction in legal: return CheckAction()
        return FoldAction()

    def _raise_to(self, amount, legal, mn, mx):
        if RaiseAction in legal:
            return RaiseAction(max(mn, min(mx, int(amount))))
        return self._call_chk(legal)

    # ─────────────────────────────────────────────────────────────
    def get_action(self, game_state, round_state, active):
        try:
            legal      = round_state.legal_actions()
            hole       = round_state.hands[active]
            street     = round_state.street
            community  = round_state.deck[:street]

            my_pip     = round_state.pips[active]
            opp_pip    = round_state.pips[1 - active]
            cost       = opp_pip - my_pip          # chips to call

            my_stack   = round_state.stacks[active]
            opp_stack  = round_state.stacks[1 - active]
            pot        = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)

            my_bounty  = round_state.bounties[active]
            opp_bounty = round_state.bounties[1 - active]

            mn = mx = 0
            if RaiseAction in legal:
                mn, mx = round_state.raise_bounds()

            eff   = min(my_stack, opp_stack)
            allin = (eff == 0)

            if street != self._prev_street:
                self._prev_street = street

            # Count raise aggression pre-flop to detect war
            if street == 0 and cost > 0:
                self._aggression_level = max(self._aggression_level,
                                             int(opp_pip / max(BIG_BLIND, 1)))
                if cost > BIG_BLIND and not self._they_raised_pre:
                    self._they_raised_pre = True

            # ── Already all-in ──
            if allin:
                return self._chk_fold(legal)

            # ════════════════════════════════════════
            #  PRE-FLOP
            # ════════════════════════════════════════
            if street == 0:
                eq = preflop_equity(hole, my_bounty)
                # Extra if we hold opp's bounty card
                if opp_bounty and (hole[0][0] == opp_bounty or hole[1][0] == opp_bounty):
                    eq = min(0.96, eq + 0.06)

                # ── HIGH AGGRESSION: they re-raised our 3-bet (4-bet+) ──
                # Baseline actually jams with a very wide "premium" range (incl. KTo, QJs)
                # So we can actually call their all-ins with a wider range to capture EV!
                if cost > BIG_BLIND * 10:
                    if eq >= 0.76:
                        self._i_raised_pre = True
                        return self._raise_to(mx, legal, mn, mx)  # Re-jam 
                    if eq >= 0.58 and CallAction in legal:
                        return CallAction() # Call their wide jam with strong hands
                    return FoldAction()

                # ── MEDIUM raise (6–20 chips) ──
                elif cost > BIG_BLIND:
                    if eq >= 0.78:
                        self._i_raised_pre = True
                        return self._raise_to(mx, legal, mn, mx)
                    elif eq >= 0.64:
                        target = int(cost * 3)
                        if RaiseAction in legal and target >= mn:
                            self._i_raised_pre = True
                            return RaiseAction(max(mn, min(mx, target)))
                        return CallAction() if CallAction in legal else FoldAction()
                    elif eq >= 0.50:
                        # Playable: just call
                        if CallAction in legal:
                            return CallAction()
                        return FoldAction()
                    elif eq >= 0.44 and cost <= BIG_BLIND * 4:
                        # Marginal hand, cheap call
                        return CallAction() if CallAction in legal else FoldAction()
                    else:
                        return FoldAction()

                # ── No raise / just big blind to call ──
                else:
                    # Baseline folds everything non-premium if cost >= 4 preflop.
                    # Raising to 4 mathematically forces folds for minimum risk!
                    if eq >= 0.65:
                        # Strong: raise 4x BB for value
                        self._i_raised_pre = True
                        return self._raise_to(BIG_BLIND * 4, legal, mn, mx)
                    elif eq >= 0.50:
                        # Good: minimum raise to force fold (cost=4 -> bet=my_pip+4)
                        self._i_raised_pre = True
                        return self._raise_to(my_pip + 4, legal, mn, mx)
                    elif eq >= 0.38:
                        # Speculative: check/limp
                        if CheckAction in legal: return CheckAction()
                        return CallAction() if CallAction in legal else FoldAction()
                    else:
                        # Trash: steal 12% from SB position with mathematically cheapest steal
                        if cost == 0 and RaiseAction in legal and random.random() < 0.12:
                            self._i_raised_pre = True
                            return self._raise_to(my_pip + 4, legal, mn, mx)
                        if cost == 0:
                            return CheckAction() if CheckAction in legal else FoldAction()
                        return FoldAction()

            # ════════════════════════════════════════
            #  POST-FLOP  (streets 3, 4, 5)
            # ════════════════════════════════════════
            sims = {3: 180, 4: 150, 5: 130}.get(street, 150)
            eq   = mc_equity(hole, community, sims)

            # Bounty adjustments
            vis = [c[0] for c in community]
            if my_bounty in vis:    eq = min(1.0, eq + 0.10)
            if opp_bounty in vis:   eq = max(0.0, eq - 0.04)

            pot_odds = cost / (pot + cost) if cost > 0 else 0.0

            # ── FACING A BET ────────────────────────
            if cost > 0:
                # AGAINST BASELINE: A bet always means they have a premium hand!
                # Raw MC equity vs a random hand is misleading here; penalize our equity heavily.
                real_eq = eq - 0.15 

                if real_eq >= 0.65:
                    if RaiseAction in legal and cost < my_stack * 0.4:
                        return self._raise_to(pot * 0.85, legal, mn, mx)
                    return CallAction() if CallAction in legal else self._chk_fold(legal)

                elif real_eq >= 0.50:
                    # Good: call if not a huge bet
                    if cost <= pot * 0.60:
                        return CallAction() if CallAction in legal else self._chk_fold(legal)
                    return FoldAction()

                else:
                    # FOLD! They have a premium hand and our real equity is too low.
                    return FoldAction()

            # ── WE CAN BET (cost == 0) ───────────────
            # Baseline checks non-premium always → bet every street
            # Sizing:
            #   Strong  → 65-80% pot (extract from premium callers)
            #   Medium  → 33% pot (cheap steal, same fold rate vs baseline)
            #   Bluff   → min(pot//5 + 2, 7) baseline folds non-premium to anything > pot//5
            else:
                fold_threshold = int(pot * 0.22) + 7
                
                if eq >= 0.84:
                    return self._raise_to(mx, legal, mn, mx)

                elif eq >= 0.72:
                    target = max(fold_threshold, int(pot * 0.85))
                    return self._raise_to(target, legal, mn, mx)

                elif eq >= 0.58:
                    target = max(fold_threshold, int(pot * 0.65))
                    return self._raise_to(target, legal, mn, mx)

                elif eq >= 0.42:
                    # Medium: mathematically cheapest bet to force baseline to fold non-premium
                    # Baseline calls if cost <= max(4, pot//5) or max(6, pot//6).
                    # A cost of `int(pot * 0.22) + 7` guarantees it trips their fold logic!
                    bluff_amount = int(pot * 0.22) + 7
                    if RaiseAction in legal:
                        return self._raise_to(bluff_amount, legal, mn, mx)
                    return CheckAction() if CheckAction in legal else self._chk_fold(legal)

                elif eq >= 0.26:
                    # Weak: bluff cheapest possible amount on flop if raised pre; barrels on turns
                    should_bluff = (
                        (self._i_raised_pre and street == 3) or
                        (street == 4 and random.random() < 0.60) or
                        (street == 5 and random.random() < 0.40)
                    )
                    if should_bluff and RaiseAction in legal:
                        bluff_amount = int(pot * 0.22) + 7
                        return self._raise_to(bluff_amount, legal, mn, mx)
                    return CheckAction() if CheckAction in legal else self._chk_fold(legal)

                else:
                    # Trash: check down
                    return CheckAction() if CheckAction in legal else self._chk_fold(legal)

        except Exception:
            legal = round_state.legal_actions()
            if CheckAction in legal: return CheckAction()
            if CallAction  in legal: return CallAction()
            return FoldAction()

    # ── lifecycle ────────────────────────────────────────────────
    def handle_new_round(self, game_state, round_state, active):
        self._i_raised_pre     = False
        self._they_raised_pre  = False
        self._aggression_level = 0
        self._prev_street      = -1

    def handle_round_over(self, game_state, terminal_state, active):
        self.recent.append(terminal_state.deltas[active])
        self.hands_played += 1


if __name__ == '__main__':
    run_bot(Player(), parse_args())