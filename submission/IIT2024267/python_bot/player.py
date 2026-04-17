"""
================================================================================
  Adaptive Poker Exploit Agent
  IIT2024267 | PokerBots-IIITA 2026
================================================================================

Motivation & Research Foundation
----------------------------------
This bot is inspired by two key bodies of work in computational poker:

1. **Exploitative Play via Opponent Modelling** (Johanson et al., 2007):
   Rather than seeking a Nash Equilibrium (GTO), this agent identifies and
   exploits structural leaks in specific opponent programs. The core idea:
   against a fixed, non-adaptive opponent, maximum EV comes from exploiting
   their exact weaknesses, not from balancing our own range.

2. **Monte Carlo Counterfactual Regret Minimisation (MCCFR)** — simplified:
   We use Monte Carlo sampling (eval7) for equity estimation rather than
   full CFR, following the "information-set sampling" principle from
   Lanctot et al. (2009). This gives us real-time equity with controllable
   accuracy/speed tradeoff.

Design Philosophy
------------------
The agent uses a **dual-mode opponent classifier** to switch between two
exploit profiles mid-session:

  Mode A — "Fold-Heavy" (e.g. Baseline):
    Bet cheap (pot//5 + 3) on every street with any equity >= 0.38.
    Their folding threshold is "any bet > pot//5" → our cheap bet triggers
    ~90% fold equity. Even with trash we profit (0.9 * pot >> call_loss).

  Mode B — "Call-Station / Sophisticated" (e.g. Player2):
    Check medium hands (no wasted bluffs), bet for value (65% pot) when
    clearly ahead (eq >= 0.65), overbet (1.3x pot) with monsters to exploit
    their "fold one-pair to overbets" heuristic.

The classifier needs only ~10 bets to converge via Bayesian fold-rate
estimation. Before convergence, we default to Mode A (cheap bets) which
is conservative and works against both opponent types.

Preflop: We 3-bet strong hands (PF strength >= 0.65) and open wide (>= 0.38),
pressuring both opponents preflop. Against large 3-bets, we push/fold using
Nash equilibrium thresholds when stack-to-BB ratio falls below 12.

Clock Safety: MC iterations are budgeted by game_clock, preventing TLEs.
================================================================================
"""
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import STARTING_STACK, BIG_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
import eval7
import random

# ── Rank lookup ──────────────────────────────────────────────────────────────
RANK_MAP = {r: i for i, r in enumerate("23456789TJQKA")}

# ── Nash push/fold thresholds (HU, effective BB) ─────────────────────────────
# Source: Sklansky-Malmuth push/fold charts, simplified for HU.
# push_thresh = minimum preflop strength to open-jam
# call_thresh = minimum preflop strength to call a jam
_NASH = [
    (12, 0.50, 0.60),   # 12+ BB: push top 50%, call top 60%
    ( 8, 0.38, 0.50),   # 8-11 BB
    ( 5, 0.20, 0.40),   # 5-7 BB: push wider, call tighter
    ( 3, 0.00, 0.25),   # 3-4 BB: push any, call top 25%
    ( 0, 0.00, 0.00),   # <3 BB:  always push, always call (pot committed)
]

def _nash_thresholds(eff_bb):
    """Return (push_thresh, call_thresh) for given effective BB count."""
    for cutoff, pt, ct in _NASH:
        if eff_bb >= cutoff:
            return pt, ct
    return 0.0, 0.0


def hand_strength(hole):
    """
    Fast O(1) preflop hand strength estimate [0.10, 0.95].

    Based on a simplified Chen formula:
      - Pair strength scales with rank
      - High-card strength from top two ranks
      - Suited/connected bonuses
    Returns a normalised score comparable to percentile in the hand range.
    """
    r1, r2 = hole[0][0], hole[1][0]
    i1, i2 = RANK_MAP[r1], RANK_MAP[r2]
    high, low = max(i1, i2), min(i1, i2)
    suited = hole[0][1] == hole[1][1]
    pair = r1 == r2
    gap = high - low

    if pair:
        s = 0.50 + high * 0.035           # AA ≈ 0.94, 22 ≈ 0.57
    elif high >= 12 and low >= 10:
        s = 0.65 + low * 0.01             # Broadway combos
    elif high == 12:
        s = 0.48 + low * 0.01
    elif high == 11:
        s = 0.43 + low * 0.01
    else:
        s = 0.25 + high * 0.02

    if suited: s += 0.04
    if gap <= 1 and not pair: s += 0.03   # connectors bonus
    if gap >= 6: s -= 0.05                # big-gap penalty

    return max(0.10, min(0.95, s))


def mc_equity(my_cards, board_cards, iters=300):
    """
    Monte Carlo equity estimation via random runout sampling.

    Samples `iters` random opponent hands + board runouts and measures
    win/tie rate. O(iters) with eval7's optimised hand evaluator.
    Returns float in [0, 1].
    """
    deck = eval7.Deck()
    hero = [eval7.Card(c) for c in my_cards]
    board = [eval7.Card(c) for c in board_cards]
    known = set(hero + board)
    # Build remaining deck excluding known cards
    remaining = [c for c in deck.cards if c not in known]
    need = 5 - len(board)   # cards still to come
    wins = 0
    for _ in range(iters):
        random.shuffle(remaining)
        sim_board = board + remaining[:need]
        opp_cards = remaining[need:need + 2]
        mv = eval7.evaluate(hero + sim_board)
        ov = eval7.evaluate(opp_cards + sim_board)
        if mv > ov:
            wins += 2       # win = 2 points
        elif mv == ov:
            wins += 1       # tie = 1 point (split pot)
    return wins / (2.0 * iters)


class Player(Bot):
    """
    Adaptive exploit agent with real-time opponent profiling.

    State tracked across hands:
      our_bets   : total times we bet/raised postflop
      opp_folds  : times opponent folded after we bet
      fold_rate  : rolling fold-to-our-bet ratio → opponent classifier
    """

    def __init__(self):
        # Fold tracking for opponent classification
        self.our_bets = 0
        self.opp_folds = 0
        self._bet_this_hand = False

    def handle_new_round(self, game_state, round_state, active):
        """Reset per-hand state."""
        self._bet_this_hand = False

    def handle_round_over(self, game_state, terminal_state, active):
        """
        Detect opponent folds after our bets.
        If we bet this hand, check whether opponent's hole cards are hidden
        at terminal state — hidden = they folded (standard skeleton behaviour:
        folded player's cards are set to [] in terminal state).
        """
        if not self._bet_this_hand:
            return
        # Detect fold: opponent's cards are empty in terminal state
        try:
            opp_idx = 1 - active
            opp_cards = terminal_state.previous_state.hands[opp_idx]
            if opp_cards is None or len(opp_cards) == 0:
                self.opp_folds += 1
        except Exception:
            # Fallback: if terminal delta is strongly positive and we bet,
            # they likely folded (conservative heuristic)
            try:
                delta = terminal_state.deltas[active]
                if delta > 0 and self.our_bets > 0:
                    self.opp_folds += 1
            except Exception:
                pass

    # ── Opponent classifier ─────────────────────────────────────────────────

    def _fold_rate(self):
        """
        Bayesian fold rate: (folds + 1) / (bets + 2).
        The Laplace smoothing (+1/+2) ensures sensible defaults before
        we have enough data — returns 0.50 until we see ~10+ bets.
        """
        return (self.opp_folds + 1) / (self.our_bets + 2)

    def _is_fold_heavy(self):
        """
        Returns True if opponent folds frequently (baseline-like).
        Threshold: fold_rate > 0.55.
        We need at least 8 bets before trusting this classification.
        Before that, default to True (cheap bets are safe vs both opponents).
        """
        if self.our_bets < 8:
            return True     # conservative default: cheap bets harm no one
        return self._fold_rate() > 0.55

    def _record_bet(self):
        """Mark that we bet/raised this hand (for fold tracking)."""
        self._bet_this_hand = True
        self.our_bets += 1

    # ── Main action logic ───────────────────────────────────────────────────

    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        street = round_state.street
        hole = round_state.hands[active]
        board = round_state.deck[:street]
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]
        cost = opp_pip - my_pip              # chips to call
        pot = 2 * STARTING_STACK - my_stack - opp_stack
        my_bounty = round_state.bounties[active]

        # Raise bounds (0 if raise is not a legal action)
        mn, mx = 0, 0
        if RaiseAction in legal:
            mn, mx = round_state.raise_bounds()

        clk = game_state.game_clock

        # ── Clock safety: ultra-fast fallback if clock is nearly empty ──────
        if clk < 2.0:
            if cost == 0:
                return CheckAction() if CheckAction in legal else CallAction()
            return CallAction() if CallAction in legal else CheckAction()

        # ── Bounty bonus: bump hand strength if we hold the bounty rank ─────
        has_bounty = (hole[0][0] == my_bounty or hole[1][0] == my_bounty)

        # ════════════════════════════════════════════════════════════════════
        #  PREFLOP
        # ════════════════════════════════════════════════════════════════════
        if street == 0:
            hs = hand_strength(hole)
            if has_bounty:
                hs = min(0.95, hs + 0.05)   # bounty card makes hand more playable

            eff_bb = min(my_stack, opp_stack) / BIG_BLIND

            # ── Nash push/fold when stacks are short (< 12 BB) ───────────
            if eff_bb < 12:
                push_t, call_t = _nash_thresholds(eff_bb)
                if cost == 0 and RaiseAction in legal:
                    # Short stack: open-jam or fold
                    if hs >= push_t:
                        return RaiseAction(mx)
                    return CheckAction() if CheckAction in legal else FoldAction()
                elif cost > 0:
                    # Facing a raise with short stack: call or fold based on Nash
                    if hs >= call_t:
                        return CallAction()
                    return FoldAction() if FoldAction in legal else CallAction()

            # ── Deep stack preflop ───────────────────────────────────────
            # Facing a big 3-bet or jam: narrow call/raise range
            if cost > BIG_BLIND * 10:
                if hs >= 0.75 and RaiseAction in legal:
                    return RaiseAction(mx)                   # 4-bet/jam premiums
                if hs >= 0.65:
                    return CallAction()                      # call strong hands
                return FoldAction() if FoldAction in legal else CallAction()

            # Facing a standard 3-bet (4-10 BB)
            if cost > BIG_BLIND * 4:
                if hs >= 0.70 and RaiseAction in legal:
                    return RaiseAction(max(mn, min(mx, int(opp_pip * 3))))
                if hs >= 0.58:
                    return CallAction()
                return FoldAction() if FoldAction in legal else CallAction()

            # Facing an open raise (2-4 BB):
            # 3-bet with strong hands to exploit player2's ~55% fold-to-3bet rate
            if cost > 0:
                if hs >= 0.65 and RaiseAction in legal:
                    # 3-bet to 3.5x their open
                    return RaiseAction(max(mn, min(mx, max(14, int(opp_pip * 3.5)))))
                if hs >= 0.48:
                    return CallAction()
                if cost <= BIG_BLIND and hs >= 0.40:
                    return CallAction()   # cheap defend vs min-raise
                return FoldAction() if FoldAction in legal else CallAction()

            # First to act: open wide — pressure both opponent types preflop
            # Baseline folds to any open. Player2 folds tier 4-5 hands.
            if hs >= 0.38 and RaiseAction in legal:
                return RaiseAction(max(mn, min(mx, int(BIG_BLIND * 2.5))))
            return CallAction() if CallAction in legal else CheckAction()

        # ════════════════════════════════════════════════════════════════════
        #  POSTFLOP: MC equity + opponent-adaptive bet sizing
        # ════════════════════════════════════════════════════════════════════

        # Budget MC iterations by clock: more time = more accuracy
        iters = 400 if clk > 20 else (250 if clk > 8 else 120)
        eq = mc_equity(hole, board, iters=iters)

        # Bounty boosts effective equity (hitting it adds real chip value)
        bounty_hit = any(c[0] == my_bounty for c in hole) or any(c[0] == my_bounty for c in board)
        if bounty_hit:
            eq = min(0.99, eq * 1.10 + 0.05)

        # ── Facing opponent's bet ────────────────────────────────────────────
        # Both opponents bet for value (neither is a pure bluffer at this stage).
        # → Tight calling; only continue with meaningful equity.
        if cost > 0:
            # Pot odds: the minimum equity needed to break even on a call
            pot_odds = cost / max(1.0, pot + 2 * cost)
            bet_frac = cost / max(1.0, pot)

            # Monster re-raise: extract maximum value
            if eq >= 0.80 and RaiseAction in legal:
                self._record_bet()
                return RaiseAction(max(mn, min(mx, int(pot * 0.85) + my_pip)))

            # +EV call: our equity meaningfully exceeds pot odds
            # Extra margin (0.07) accounts for variance and multi-street risk
            if eq >= pot_odds + 0.07:
                return CallAction()

            # Cheap peel: tiny bets are worth calling with any draw (small risk)
            if bet_frac <= 0.20 and eq >= 0.35:
                return CallAction()

            # Fold: math says we're not getting the right price
            return FoldAction() if FoldAction in legal else CallAction()

        # ── Checked to us (no cost to act) ──────────────────────────────────
        if RaiseAction not in legal:
            return CheckAction() if CheckAction in legal else CallAction()

        # ── DUAL-MODE BETTING ────────────────────────────────────────────────
        #
        # We select bet sizing based on opponent type:
        #
        #   cheap = pot//5 + 3:  Triggers baseline's "fold if bet > pot//5" rule.
        #                        Against player2, pot_odds ≈ 0.15 → they call
        #                        with equity > 0.18 (i.e., almost always).
        #
        #   value = 65% pot:     Against player2, pot_odds ≈ 0.39 → they need
        #                        equity > 0.42 to call. Extracts value while
        #                        pricing out their weaker hands.
        #
        #   overbet = 130% pot:  Against player2, pot_odds ≈ 0.57 → they fold
        #                        one-pair hands (their threshold for overbets is
        #                        "fold unless equity > pot_odds + 0.08 = 0.65").
        #                        Use this with monsters.
        #
        cheap  = max(mn, min(mx, pot // 5 + 3 + my_pip))
        value  = max(mn, min(mx, int(pot * 0.65) + my_pip))
        overbet = max(mn, min(mx, int(pot * 1.30) + my_pip))

        fold_heavy = self._is_fold_heavy()

        if fold_heavy:
            # ── Mode A: Fold-heavy opponent (baseline-like) ──────────────────
            # Bet frequently with cheap sizing. Their fold rate is ~90% to any
            # bet, so even weak hands are profitable: 0.9 * pot - 0.1 * eq_loss.

            if eq >= 0.60:
                # Clear value: bet larger to extract from their rare calls
                self._record_bet()
                return RaiseAction(value)

            if eq >= 0.38:
                # Thin value / bluff: cheap bet, they fold ~90% of the time
                self._record_bet()
                return RaiseAction(cheap)

            if self._fold_rate() > 0.70 and random.random() < 0.50:
                # Pure steal: confirmed very high fold rate → exploitable with trash
                self._record_bet()
                return RaiseAction(cheap)

            # Weak hand against a caller — check and see a free card
            return CheckAction()

        else:
            # ── Mode B: Call-station / sophisticated opponent (player2-like) ──
            # They call too wide for cheap bluffs to be profitable (need equity
            # > 0.18, i.e., almost never fold). Only bet for value.

            if eq >= 0.80:
                # Monster: overbet to exploit their one-pair folding threshold
                self._record_bet()
                return RaiseAction(overbet)

            if eq >= 0.65:
                # Strong value: standard 65% pot sizing
                self._record_bet()
                return RaiseAction(value)

            if eq >= 0.55:
                # Thin value: cheap bet — we're ahead of their calling range,
                # and the small size keeps us profitable even when called
                self._record_bet()
                return RaiseAction(cheap)

            # Below 55% equity: check back.
            # Player2 also checks 35-60% equity hands back (by their own logic),
            # so we get free cards and avoid losing chips as a bluff.
            return CheckAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())