"""
Tournament Poker Bot — Hybrid GTO/Exploit Architecture
=======================================================
Design goals:
  * GTO-like baseline vs unknowns (balanced, unexploitable)
  * Rapid exploit detection with confidence gating
  * Five targeted exploit strategies that auto-switch
  * Adaptive bet sizing calibrated to opponent thresholds
  * Robust under tournament clock pressure
"""

import os
import random
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from skeleton.actions import CallAction, CheckAction, FoldAction, RaiseAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.states import STARTING_STACK

# ---------------------------------------------------------------------------
# 1. CARD PRIMITIVES
# ---------------------------------------------------------------------------

class Suit:
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


@dataclass(frozen=True)
class Card:
    rank: int   # 2..14  (14 = Ace)
    suit: int   # 0..3


def valid_card(c: Card) -> bool:
    return 2 <= c.rank <= 14 and 0 <= c.suit <= 3


def card_id(c: Card) -> int:
    return c.suit * 13 + (c.rank - 2)


def _rank(ch: str) -> int:
    return (
        ord(ch) - ord("0") if "2" <= ch <= "9" else
        10 if ch in "Tt" else
        11 if ch in "Jj" else
        12 if ch in "Qq" else
        13 if ch in "Kk" else
        14 if ch in "Aa" else -1
    )


def _suit(ch: str) -> int:
    return (
        Suit.CLUBS if ch in "cC" else
        Suit.DIAMONDS if ch in "dD" else
        Suit.HEARTS if ch in "hH" else
        Suit.SPADES if ch in "sS" else -1
    )


def parse_card(token: str) -> Optional[Card]:
    if not token or len(token) < 2:
        return None
    r, s = _rank(token[0]), _suit(token[1])
    return Card(rank=r, suit=s) if r >= 2 and s >= 0 else None


# ---------------------------------------------------------------------------
# 2. GAME STATE
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    hero_hole: List[Card] = field(default_factory=list)
    board: List[Card] = field(default_factory=list)
    pot: int = 0
    to_call: int = 0
    hero_stack: int = 0
    starting_stack: int = STARTING_STACK
    num_opponents: int = 1

    def valid(self) -> bool:
        if len(self.hero_hole) != 2:
            return False
        used = [False] * 52
        for c in self.hero_hole + self.board:
            if not valid_card(c):
                return False
            cid = card_id(c)
            if used[cid]:
                return False
            used[cid] = True
        return True


def adapt_state(game_state, round_state, active: int) -> GameState:
    st = GameState()
    st.starting_stack = STARTING_STACK
    st.num_opponents = 1

    hand = round_state.hands[active]
    if len(hand) >= 2:
        c0, c1 = parse_card(hand[0]), parse_card(hand[1])
        if c0 and c1 and c0 != c1:
            st.hero_hole = [c0, c1]

    for i in range(min(5, max(0, round_state.street))):
        if i < len(round_state.deck):
            bc = parse_card(round_state.deck[i])
            if bc:
                st.board.append(bc)

    my_pip = round_state.pips[active]
    opp_pip = round_state.pips[1 - active]
    my_stack = round_state.stacks[active]
    opp_stack = round_state.stacks[1 - active]

    st.to_call = max(0, opp_pip - my_pip)
    st.pot = max(0, (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack))
    st.hero_stack = max(0, my_stack)
    return st


# ---------------------------------------------------------------------------
# 3. HAND EVALUATOR  (rank-based, no lookup tables)
# ---------------------------------------------------------------------------

class Eval:
    @staticmethod
    def _has(mask: int, r: int) -> bool:
        return 2 <= r <= 14 and bool((mask >> (r - 2)) & 1)

    @staticmethod
    def _straight_high(mask: int) -> int:
        for h in range(14, 4, -1):
            if all(Eval._has(mask, h - i) for i in range(5)):
                return h
        if all(Eval._has(mask, r) for r in (14, 2, 3, 4, 5)):
            return 5
        return 0

    @staticmethod
    def _pack(*args: int) -> int:
        result = 0
        for v in args:
            result = (result << 4) | (v & 0xF)
        return result

    @staticmethod
    def evaluate(cards: List[Card]) -> int:
        n = len(cards)
        assert 5 <= n <= 7

        rc = [0] * 15
        sc = [0] * 4
        sm = [0, 0, 0, 0]
        rm = 0

        for c in cards:
            rc[c.rank] += 1
            sc[c.suit] += 1
            bit = 1 << (c.rank - 2)
            sm[c.suit] |= bit
            rm |= bit

        # Flush / straight-flush
        for s in range(4):
            if sc[s] >= 5:
                sh = Eval._straight_high(sm[s])
                if sh:
                    return (8 << 20) | Eval._pack(sh)

        quads = [r for r in range(14, 1, -1) if rc[r] == 4]
        trips = [r for r in range(14, 1, -1) if rc[r] == 3]
        pairs = [r for r in range(14, 1, -1) if rc[r] == 2]
        sings = [r for r in range(14, 1, -1) if rc[r] == 1]

        if quads:
            k = next((r for r in range(14, 1, -1) if r != quads[0] and rc[r] > 0), 0)
            return (7 << 20) | Eval._pack(quads[0], k)

        if trips and (pairs or len(trips) > 1):
            p = trips[1] if len(trips) > 1 else pairs[0]
            return (6 << 20) | Eval._pack(trips[0], p)

        for s in range(4):
            if sc[s] >= 5:
                fc = [r for r in range(14, 1, -1) if Eval._has(sm[s], r)][:5]
                return (5 << 20) | Eval._pack(*fc)

        sh = Eval._straight_high(rm)
        if sh:
            return (4 << 20) | Eval._pack(sh)

        if trips:
            ks = [r for r in range(14, 1, -1) if r != trips[0] and rc[r] > 0][:2]
            return (3 << 20) | Eval._pack(trips[0], *ks)

        if len(pairs) >= 2:
            k = next((r for r in range(14, 1, -1) if r not in pairs[:2] and rc[r] > 0), 0)
            return (2 << 20) | Eval._pack(pairs[0], pairs[1], k)

        if pairs:
            ks = [r for r in range(14, 1, -1) if r != pairs[0] and rc[r] > 0][:3]
            return (1 << 20) | Eval._pack(pairs[0], *ks)

        hs = [r for r in range(14, 1, -1) if rc[r] > 0][:5]
        return Eval._pack(*hs)


# ---------------------------------------------------------------------------
# 4. BOARD TEXTURE
# ---------------------------------------------------------------------------

@dataclass
class BoardTexture:
    flush_draw: bool = False    # 3+ same suit
    straight_draw: bool = False # connected ranks
    paired: bool = False
    wet: bool = False           # flush_draw OR straight_draw


def analyze_texture(board: List[Card]) -> BoardTexture:
    t = BoardTexture()
    if len(board) < 3:
        return t

    sc = [0, 0, 0, 0]
    rc = [0] * 15
    ranks: List[int] = []

    for c in board:
        sc[c.suit] += 1
        rc[c.rank] += 1

    t.flush_draw = any(v >= 3 for v in sc)
    t.paired = any(v >= 2 for v in rc)

    ranks = sorted(set(c.rank for c in board))
    for i in range(len(ranks) - 2):
        if ranks[i + 2] - ranks[i] <= 4:
            t.straight_draw = True
            break

    t.wet = t.flush_draw or t.straight_draw
    return t


# ---------------------------------------------------------------------------
# 5. EQUITY ENGINE
# ---------------------------------------------------------------------------

class EquityEngine:
    """
    Two-tier equity estimation:
      1. Heuristic shortcut (O(1)) for extreme hands
      2. Time-bounded Monte Carlo fallback
    """

    def __init__(self, seed=None):
        self._rng = random.Random(seed)
        self._timed_out = False
        self._sims_done = 0

    # ---- 5a. Heuristic shortcut -----------------------------------------

    @staticmethod
    def preflop_heuristic(a: Card, b: Card, n_opp: int) -> Tuple[bool, float]:
        hi, lo = max(a.rank, b.rank), min(a.rank, b.rank)
        pair = a.rank == b.rank
        suited = a.suit == b.suit
        gap = hi - lo

        if pair:
            eq = 0.50 + (hi - 2) * 0.028
        else:
            eq = 0.30
            if hi == 14:
                eq += 0.08
            if hi >= 13:
                eq += 0.04
            if suited:
                eq += 0.03
            eq -= gap * 0.012

        eq -= 0.07 * (n_opp - 1)
        eq = max(0.08, min(0.92, eq))

        # Only shortcut on extremes
        is_extreme = (pair and hi >= 12) or (hi == 14 and lo >= 13) or (not pair and hi <= 9 and gap >= 5)
        return is_extreme, eq

    @staticmethod
    def postflop_heuristic(st: GameState) -> Tuple[bool, float]:
        if len(st.board) < 3:
            return False, 0.5

        a, b = st.hero_hole[0], st.hero_hole[1]
        rc = [0] * 15
        sc = [0] * 4
        rm = 0
        for c in [a, b] + st.board:
            rc[c.rank] += 1
            sc[c.suit] += 1
            rm |= 1 << (c.rank - 2)

        quads = sum(1 for r in range(2, 15) if rc[r] == 4)
        trips = sum(1 for r in range(2, 15) if rc[r] == 3)
        pairs = sum(1 for r in range(2, 15) if rc[r] == 2)
        flush_made = any(v >= 5 for v in sc)
        straight_made = Eval._straight_high(rm) > 0
        flush_draw = any(v == 4 for v in sc)

        if len(st.board) == 5:
            score = Eval.evaluate(st.hero_hole + st.board)
            cat = score >> 20
            if cat >= 6:
                return True, 0.94
            if cat >= 4:
                return True, 0.83
            if cat == 3:
                return True, 0.74
            if cat <= 1:
                return True, 0.22
            return False, 0.5

        if quads or flush_made or straight_made:
            return True, 0.90
        if trips or pairs >= 2:
            return True, 0.78
        hi = max(a.rank, b.rank)
        if pairs == 0 and not flush_draw and hi <= 11:
            return True, 0.20
        return False, 0.5

    def estimate(self, st: GameState, sims: int, time_limit_us: int = 600_000) -> float:
        self._timed_out = False
        self._sims_done = 0

        # --- Heuristic shortcut first ---
        if len(st.board) == 0:
            extreme, eq = self.preflop_heuristic(st.hero_hole[0], st.hero_hole[1], st.num_opponents)
            if extreme:
                return eq

        if len(st.board) >= 3:
            shortcut, eq = self.postflop_heuristic(st)
            if shortcut:
                return eq

        # --- Monte Carlo ---
        return self._monte_carlo(st, sims, time_limit_us)

    def _monte_carlo(self, st: GameState, sims: int, time_limit_us: int) -> float:
        if sims <= 0 or not st.valid():
            return 0.5

        used = [False] * 52
        for c in st.hero_hole + st.board:
            used[card_id(c)] = True

        pool = [Card(rank=r, suit=s) for s in range(4) for r in range(2, 15)
                if not used[card_id(Card(rank=r, suit=s))]]

        n_opp = st.num_opponents
        board_needed = 5 - len(st.board)
        if n_opp * 2 + board_needed > len(pool):
            return 0.5

        wins = ties = 0
        start = time.perf_counter_ns()

        fixed = (board_needed == 0)
        full_board = list(st.board) + [None] * board_needed  # type: ignore

        hero_fixed = 0
        if fixed:
            hero_fixed = Eval.evaluate(st.hero_hole + st.board)

        for sim in range(sims):
            if time_limit_us > 0 and sim > 0 and sim % 20 == 0:
                elapsed = (time.perf_counter_ns() - start) // 1000
                if elapsed >= time_limit_us:
                    self._timed_out = True
                    break

            live = pool[:]
            live_n = len(live)

            def draw() -> Card:
                nonlocal live_n
                idx = self._rng.randrange(live_n)
                card = live[idx]
                live[idx] = live[live_n - 1]
                live_n -= 1
                return card

            opp_holes = [[draw(), draw()] for _ in range(n_opp)]

            if not fixed:
                base = len(st.board)
                for i in range(base, 5):
                    full_board[i] = draw()

            hero_score = hero_fixed if fixed else Eval.evaluate(st.hero_hole + full_board)

            best = True
            tied = False
            for oh in opp_holes:
                os_ = Eval.evaluate(oh + full_board)
                if os_ > hero_score:
                    best = False
                    tied = False
                    break
                if os_ == hero_score:
                    tied = True

            if best:
                (ties if tied else wins).__add__  # noqa
                if tied:
                    ties += 1
                else:
                    wins += 1

            # Early stopping
            done = sim + 1
            if done >= 60 and done % 20 == 0:
                eq = (wins + 0.5 * ties) / done
                if eq >= 0.92 or eq <= 0.08:
                    self._sims_done = done
                    return eq

        total = sim + 1
        self._sims_done = total
        return (wins + 0.5 * ties) / total if total > 0 else 0.5

    def timed_out(self) -> bool:
        return self._timed_out

    def sims_done(self) -> int:
        return self._sims_done


# ---------------------------------------------------------------------------
# 6. OPPONENT MODEL + CLASSIFICATION
# ---------------------------------------------------------------------------

class OpponentType(Enum):
    UNKNOWN = auto()
    BALANCED = auto()       # GTO-like, hard to exploit
    CALLING_STATION = auto()# call_rate > 0.60
    NIT = auto()            # fold_rate > 0.55, raise_rate < 0.25
    AGGRESSIVE = auto()     # raise_rate > 0.45
    EXPLOIT_BOT = auto()    # small_bet_ratio > 0.40 (likely exploitative bot)
    MANIAC = auto()         # raise_rate > 0.60, no fold


class ConfidenceLevel(Enum):
    LOW = auto()     # < 20 actions
    MEDIUM = auto()  # 20–60
    HIGH = auto()    # > 60


@dataclass
class OpponentProfile:
    total: int = 0
    folds: int = 0
    calls: int = 0
    raises: int = 0
    small_bets: int = 0          # raises ≤ 33% pot
    large_bets: int = 0          # raises ≥ 75% pot
    fold_to_3bet: int = 0
    fold_to_3bet_opps: int = 0
    recent: List[int] = field(default_factory=list)   # 0=fold, 1=call, 2=raise
    RECENT_N: int = 12

    def record(self, action: int, is_small_bet: bool = False, is_large_bet: bool = False, faced_raise: bool = False):
        self.total += 1
        if action == 0:
            self.folds += 1
        elif action == 1:
            self.calls += 1
        else:
            self.raises += 1
            if is_small_bet:
                self.small_bets += 1
            if is_large_bet:
                self.large_bets += 1

        if faced_raise:
            self.fold_to_3bet_opps += 1
            if action == 0:
                self.fold_to_3bet += 1

        self.recent.append(action)
        if len(self.recent) > self.RECENT_N:
            self.recent.pop(0)

    # -- Rates --
    def fold_rate(self) -> float:
        return self.folds / self.total if self.total else 0.0

    def call_rate(self) -> float:
        return self.calls / self.total if self.total else 0.0

    def raise_rate(self) -> float:
        return self.raises / self.total if self.total else 0.0

    def small_bet_ratio(self) -> float:
        return self.small_bets / max(1, self.raises)

    def fold_to_3bet_rate(self) -> float:
        return self.fold_to_3bet / self.fold_to_3bet_opps if self.fold_to_3bet_opps else 0.35

    def recent_fold_rate(self) -> float:
        if not self.recent:
            return self.fold_rate()
        return self.recent.count(0) / len(self.recent)

    def recent_raise_rate(self) -> float:
        if not self.recent:
            return self.raise_rate()
        return self.recent.count(2) / len(self.recent)

    # -- Classification --
    def confidence(self) -> ConfidenceLevel:
        if self.total < 20:
            return ConfidenceLevel.LOW
        if self.total < 60:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.HIGH

    def classify(self) -> OpponentType:
        if self.total < 8:
            return OpponentType.UNKNOWN

        fr = self.fold_rate()
        cr = self.call_rate()
        rr = self.raise_rate()
        sbr = self.small_bet_ratio()

        if rr > 0.60 and fr < 0.20:
            return OpponentType.MANIAC
        if sbr > 0.40 and rr > 0.25:
            return OpponentType.EXPLOIT_BOT
        if cr > 0.60:
            return OpponentType.CALLING_STATION
        if rr > 0.45:
            return OpponentType.AGGRESSIVE
        if fr > 0.55 and rr < 0.25:
            return OpponentType.NIT
        if 0.25 <= fr <= 0.50 and 0.25 <= cr <= 0.55 and rr <= 0.35:
            return OpponentType.BALANCED
        return OpponentType.UNKNOWN


class OpponentRegistry:
    def __init__(self):
        self._profiles: Dict[str, OpponentProfile] = {}

    def get(self, name: str) -> OpponentProfile:
        if name not in self._profiles:
            self._profiles[name] = OpponentProfile()
        return self._profiles[name]

    def observe(self, name: str, action: int, pot: int = 0, bet: int = 0, faced_raise: bool = False):
        p = self.get(name)
        is_small = action == 2 and pot > 0 and bet <= pot * 0.34
        is_large = action == 2 and pot > 0 and bet >= pot * 0.75
        p.record(action, is_small, is_large, faced_raise)


# ---------------------------------------------------------------------------
# 7. STRATEGY MODULES
# ---------------------------------------------------------------------------

class StrategyMode(Enum):
    BALANCED = auto()
    VALUE_HEAVY = auto()     # vs calling station
    TRAP = auto()            # vs aggressive / maniac
    STEAL = auto()           # vs nit
    ANTI_EXPLOIT = auto()    # vs exploit bot
    MANIAC_CONTROL = auto()  # vs maniac (tighter trapping)


def select_strategy(opp_type: OpponentType, confidence: ConfidenceLevel) -> StrategyMode:
    if confidence == ConfidenceLevel.LOW:
        return StrategyMode.BALANCED
    mapping = {
        OpponentType.UNKNOWN: StrategyMode.BALANCED,
        OpponentType.BALANCED: StrategyMode.BALANCED,
        OpponentType.CALLING_STATION: StrategyMode.VALUE_HEAVY,
        OpponentType.NIT: StrategyMode.STEAL,
        OpponentType.AGGRESSIVE: StrategyMode.TRAP,
        OpponentType.EXPLOIT_BOT: StrategyMode.ANTI_EXPLOIT,
        OpponentType.MANIAC: StrategyMode.MANIAC_CONTROL,
    }
    return mapping.get(opp_type, StrategyMode.BALANCED)


# ---------------------------------------------------------------------------
# 8. ACTION TYPES & BET SIZING
# ---------------------------------------------------------------------------

class Act(Enum):
    FOLD = auto()
    CHECK_CALL = auto()
    RAISE_SMALL = auto()    # ~25-33% pot
    RAISE_MED = auto()      # ~55-70% pot
    RAISE_BIG = auto()      # ~85-100% pot
    ALL_IN = auto()


@dataclass
class Decision:
    act: Act = Act.FOLD
    amount: int = 0


class BetSizer:
    """
    Adaptive bet sizing calibrated to opponent tendencies.
    Key insight: use minimum bet that achieves strategic goal.
    """

    def __init__(self, rng: random.Random):
        self._rng = rng

    def _jitter(self, base: float, pct: float = 0.12) -> float:
        return base * (1.0 + (self._rng.random() - 0.5) * pct)

    def size_raise(
        self,
        act: Act,
        st: GameState,
        profile: OpponentProfile,
        raise_min: int,
        raise_max: int,
    ) -> int:
        """Return raise-to amount (total, including call)."""
        pot = max(st.pot, 1)
        call = st.to_call
        stack = st.hero_stack

        # Adapt to opponent: against callers, bet bigger; against folders, smaller
        scale = 1.0
        if profile.call_rate() > 0.55:
            scale = 1.15
        elif profile.fold_rate() > 0.55:
            scale = 0.82

        if act == Act.RAISE_SMALL:
            extra = int(self._jitter(pot * 0.28) * scale)
        elif act == Act.RAISE_MED:
            extra = int(self._jitter(pot * 0.62) * scale)
        elif act == Act.RAISE_BIG:
            extra = int(self._jitter(pot * 0.90) * scale)
        elif act == Act.ALL_IN:
            return min(stack + call, raise_max)  # raise_to = pip + stack
        else:
            extra = int(pot * 0.28)

        # Exploit: find minimum bet that folds an opponent with known fold threshold
        if profile.fold_to_3bet_rate() > 0.60 and act in (Act.RAISE_SMALL, Act.RAISE_MED):
            # Sufficient pressure at 1/3 pot already folds them
            extra = max(extra, int(pot * 0.28))

        target = call + extra
        target = max(raise_min - (st.pips_active if hasattr(st, 'pips_active') else 0), target) if raise_min else target
        return max(raise_min, min(target, raise_max))

    def make(
        self,
        act: Act,
        st: GameState,
        profile: OpponentProfile,
        raise_min: int,
        raise_max: int,
        my_pip: int,
    ) -> Decision:
        if act == Act.FOLD:
            if st.to_call == 0:
                return Decision(Act.CHECK_CALL, 0)
            return Decision(Act.FOLD, 0)

        if act == Act.CHECK_CALL:
            return Decision(Act.CHECK_CALL, min(st.to_call, st.hero_stack))

        if act == Act.ALL_IN:
            if raise_max <= 0:
                return Decision(Act.CHECK_CALL, min(st.to_call, st.hero_stack))
            return Decision(Act.ALL_IN, raise_max)

        # Raise actions
        if raise_max <= 0 or raise_min <= 0:
            return Decision(Act.CHECK_CALL, min(st.to_call, st.hero_stack))

        target = self.size_raise(act, st, profile, raise_min, raise_max)
        target = max(raise_min, min(target, raise_max))

        if target >= st.hero_stack + my_pip:
            return Decision(Act.ALL_IN, raise_max)

        return Decision(act, target)


# ---------------------------------------------------------------------------
# 9. ACTION ENGINE  (strategy-aware decision logic)
# ---------------------------------------------------------------------------

class ActionEngine:
    """
    Core decision logic. Given equity + opponent profile + strategy mode,
    returns an abstract Decision.
    """

    def __init__(self, rng: random.Random):
        self._rng = rng
        self._sizer = BetSizer(rng)

        # Self-image tracking (to detect over-aggressive loops)
        self._recent_aggressive = 0
        self._recent_passive = 0
        self._history: List[int] = []   # 1=agg, 0=pass, stored last 20
        self._HIST_N = 20

        # Delayed trap state
        self._trap_credit: float = 0.0  # accumulated trap credit
        self._barrel_pending: bool = False

    def rand(self) -> float:
        return self._rng.random()

    # ---- self-image ---

    def _record(self, agg: bool):
        style = 1 if agg else 0
        self._history.append(style)
        if len(self._history) > self._HIST_N:
            old = self._history.pop(0)
            if old == 1:
                self._recent_aggressive = max(0, self._recent_aggressive - 1)
            else:
                self._recent_passive = max(0, self._recent_passive - 1)
        if style == 1:
            self._recent_aggressive += 1
        else:
            self._recent_passive += 1

    def _image_aggression(self) -> float:
        denom = self._recent_aggressive + self._recent_passive
        return self._recent_aggressive / denom if denom else 0.5

    # ---- pot odds ---

    @staticmethod
    def pot_odds(st: GameState) -> float:
        if st.to_call <= 0:
            return 0.0
        return st.to_call / max(1, st.pot + st.to_call)

    # ---- main entry ---

    def decide(
        self,
        st: GameState,
        equity: float,
        profile: OpponentProfile,
        mode: StrategyMode,
        texture: BoardTexture,
        in_position: bool,
        raise_min: int,
        raise_max: int,
        my_pip: int,
    ) -> Decision:

        # Micro-randomize equity to prevent pattern exploitation
        equity = max(0.0, min(1.0, equity + (self.rand() - 0.5) * 0.015))

        po = self.pot_odds(st)
        street = len(st.board)  # 0=preflop, 3=flop, 4=turn, 5=river

        # Stack ratios
        spr = st.hero_stack / max(1, st.pot)   # stack-to-pot
        stack_pct = st.hero_stack / max(1, st.starting_stack)
        short_stack = stack_pct <= 0.20

        # Facing large bet → safety fold threshold
        if st.to_call > 0:
            call_fraction = st.to_call / max(1, st.hero_stack)
            if call_fraction >= 0.80 and equity < 0.72:
                act = Act.FOLD
                dec = self._sizer.make(act, st, profile, raise_min, raise_max, my_pip)
                self._record(False)
                return dec
            if call_fraction >= 0.40 and equity < 0.62:
                act = Act.FOLD
                dec = self._sizer.make(act, st, profile, raise_min, raise_max, my_pip)
                self._record(False)
                return dec

        # Route to strategy
        if mode == StrategyMode.VALUE_HEAVY:
            act = self._value_heavy(st, equity, profile, texture, po, street, short_stack)
        elif mode == StrategyMode.STEAL:
            act = self._steal(st, equity, profile, texture, po, street, short_stack)
        elif mode == StrategyMode.TRAP:
            act = self._trap(st, equity, profile, texture, po, street, short_stack, in_position)
        elif mode == StrategyMode.ANTI_EXPLOIT:
            act = self._anti_exploit(st, equity, profile, texture, po, street, short_stack)
        elif mode == StrategyMode.MANIAC_CONTROL:
            act = self._maniac_control(st, equity, profile, texture, po, street, short_stack)
        else:
            act = self._balanced(st, equity, profile, texture, po, street, short_stack, in_position)

        # ---- Universal post-routing safety checks ----

        # Never call off a huge chunk of stack without equity
        if act == Act.CHECK_CALL and st.to_call > 0:
            if equity + 0.02 < po:
                act = Act.FOLD

        # Short-stack mode: polarize to all-in or fold
        if short_stack:
            if act in (Act.RAISE_SMALL, Act.RAISE_MED, Act.RAISE_BIG) and equity >= 0.58:
                act = Act.ALL_IN
            elif act == Act.ALL_IN and equity < 0.58:
                act = Act.CHECK_CALL if equity >= po + 0.03 else Act.FOLD

        # Very high stack: avoid accidental all-in
        if stack_pct >= 1.7 and act == Act.ALL_IN and equity < 0.88:
            act = Act.RAISE_BIG

        # Anti-image: if too aggressive lately, occasionally check back
        if self._image_aggression() > 0.68 and act in (Act.RAISE_SMALL, Act.RAISE_MED) and self.rand() < 0.20:
            act = Act.CHECK_CALL

        dec = self._sizer.make(act, st, profile, raise_min, raise_max, my_pip)
        self._record(dec.act in (Act.RAISE_SMALL, Act.RAISE_MED, Act.RAISE_BIG, Act.ALL_IN))
        return dec

    # ----------------------------------------------------------------
    # 9a. BALANCED (GTO baseline)
    # ----------------------------------------------------------------

    def _balanced(
        self,
        st: GameState,
        eq: float,
        p: OpponentProfile,
        tex: BoardTexture,
        po: float,
        street: int,
        short: bool,
        ip: bool,
    ) -> Act:
        # Position adjustments
        eq_adj = eq + (0.02 if ip else -0.02)

        if street == 0:
            return self._preflop_balanced(st, eq_adj, p, short)

        # Postflop thresholds
        t_allin = 0.85
        t_big   = 0.75
        t_med   = 0.65
        t_small = 0.55

        # Adjust for position and opponent
        if ip:
            t_big -= 0.02; t_med -= 0.02
        if p.call_rate() > 0.55:
            t_big += 0.02
        if p.fold_rate() > 0.55:
            t_big -= 0.03; t_med -= 0.02

        po_real = max(po, 0.0)

        if eq_adj >= t_allin:
            return Act.ALL_IN if self.rand() < 0.35 else Act.RAISE_BIG
        if eq_adj >= t_big:
            return Act.RAISE_BIG if self.rand() < 0.55 else Act.RAISE_MED
        if eq_adj >= t_med:
            return Act.RAISE_MED if self.rand() < 0.45 else Act.CHECK_CALL
        if eq_adj >= t_small:
            return Act.CHECK_CALL if self.rand() < 0.65 else Act.RAISE_SMALL
        if eq_adj >= po_real + 0.04:
            return Act.CHECK_CALL
        if st.to_call == 0:
            return Act.CHECK_CALL   # free check
        return Act.FOLD

    def _preflop_balanced(self, st: GameState, eq: float, p: OpponentProfile, short: bool) -> Act:
        if short:
            if eq >= 0.58:
                return Act.ALL_IN
            return Act.CHECK_CALL if eq >= 0.45 else Act.FOLD

        if eq >= 0.72:
            return Act.RAISE_BIG if self.rand() < 0.60 else Act.RAISE_MED
        if eq >= 0.60:
            return Act.RAISE_MED if self.rand() < 0.55 else Act.CHECK_CALL
        if eq >= 0.48:
            return Act.CHECK_CALL
        if st.to_call > 0:
            return Act.FOLD
        return Act.CHECK_CALL

    # ----------------------------------------------------------------
    # 9b. VALUE_HEAVY (vs calling station)
    # ----------------------------------------------------------------

    def _value_heavy(self, st: GameState, eq: float, p: OpponentProfile, tex: BoardTexture, po: float, street: int, short: bool) -> Act:
        # Against callers: no bluffing, bet big with strong hands
        if street == 0:
            if eq >= 0.65:
                return Act.RAISE_BIG
            if eq >= 0.52:
                return Act.CHECK_CALL
            if st.to_call > 0 and eq < po + 0.02:
                return Act.FOLD
            return Act.CHECK_CALL

        if eq >= 0.78:
            return Act.RAISE_BIG
        if eq >= 0.68:
            return Act.RAISE_MED
        if eq >= 0.55:
            return Act.CHECK_CALL
        if eq < po + 0.03 and st.to_call > 0:
            return Act.FOLD
        if st.to_call == 0:
            return Act.CHECK_CALL  # never bluff into callers
        return Act.FOLD

    # ----------------------------------------------------------------
    # 9c. STEAL (vs nit)
    # ----------------------------------------------------------------

    def _steal(self, st: GameState, eq: float, p: OpponentProfile, tex: BoardTexture, po: float, street: int, short: bool) -> Act:
        # Exploit high fold rate: apply frequent small pressure
        ftr = p.fold_to_3bet_rate()

        if street == 0:
            if eq >= 0.58:
                return Act.RAISE_BIG
            if eq >= 0.45:
                return Act.RAISE_MED if self.rand() < 0.60 else Act.CHECK_CALL
            # Steal attempt preflop
            if st.to_call == 0 and self.rand() < 0.35:
                return Act.RAISE_SMALL
            if st.to_call > 0 and eq < po + 0.03:
                return Act.FOLD
            return Act.CHECK_CALL

        # Postflop: pressure dry boards
        if not tex.wet:
            if eq >= 0.55:
                return Act.RAISE_MED
            if eq >= 0.40 and ftr > 0.55 and st.to_call == 0:
                bluff_p = 0.28 + (ftr - 0.55) * 0.40
                if self.rand() < bluff_p:
                    return Act.RAISE_SMALL
        else:
            # Wet board: more cautious
            if eq >= 0.65:
                return Act.RAISE_MED
            if eq >= 0.50:
                return Act.CHECK_CALL

        if eq >= po + 0.04 and st.to_call > 0:
            return Act.CHECK_CALL
        if st.to_call == 0:
            return Act.CHECK_CALL
        return Act.FOLD

    # ----------------------------------------------------------------
    # 9d. TRAP (vs aggressive opponent)
    # ----------------------------------------------------------------

    def _trap(self, st: GameState, eq: float, p: OpponentProfile, tex: BoardTexture, po: float, street: int, short: bool, ip: bool) -> Act:
        # Let them hang themselves; induce aggression; raise strong hands
        if street == 0:
            if eq >= 0.75:
                # Slow play occasionally
                if self.rand() < 0.45:
                    return Act.CHECK_CALL
                return Act.RAISE_MED
            if eq >= 0.55:
                return Act.CHECK_CALL
            if st.to_call > 0 and eq < po + 0.02:
                return Act.FOLD
            return Act.CHECK_CALL

        # Trap credit: release accumulated equity suddenly
        if self._trap_credit > 0.5 and eq >= 0.60:
            self._trap_credit *= 0.5
            return Act.RAISE_BIG

        if eq >= 0.80:
            # Strong hand: trap or strike
            if self.rand() < 0.55 and st.to_call == 0:
                return Act.CHECK_CALL   # check-raise bait
            return Act.RAISE_BIG
        if eq >= 0.65:
            if self.rand() < 0.60:
                self._trap_credit = min(1.0, self._trap_credit + 0.3)
                return Act.CHECK_CALL
            return Act.RAISE_MED
        if eq >= 0.50:
            return Act.CHECK_CALL
        if eq >= po + 0.03 and st.to_call > 0:
            return Act.CHECK_CALL
        if st.to_call == 0:
            return Act.CHECK_CALL
        return Act.FOLD

    # ----------------------------------------------------------------
    # 9e. ANTI_EXPLOIT (vs exploit bot)
    # ----------------------------------------------------------------

    def _anti_exploit(self, st: GameState, eq: float, p: OpponentProfile, tex: BoardTexture, po: float, street: int, short: bool) -> Act:
        # Counter small-bet probing: call wider, raise back often
        sbr = p.small_bet_ratio()

        if street == 0:
            # Play tighter preflop to avoid getting caught out of position
            if eq >= 0.62:
                return Act.RAISE_MED
            if eq >= 0.50:
                return Act.CHECK_CALL
            return Act.FOLD if st.to_call > 0 else Act.CHECK_CALL

        # Postflop: punish small bets
        is_facing_small = st.to_call > 0 and st.pot > 0 and st.to_call <= st.pot * 0.34

        if is_facing_small:
            # Call or re-raise with any reasonable equity
            if eq >= 0.55:
                return Act.RAISE_MED if self.rand() < 0.55 else Act.CHECK_CALL
            if eq >= 0.40:
                return Act.CHECK_CALL   # call widely vs small bets
            return Act.FOLD

        # Standard postflop
        if eq >= 0.70:
            return Act.RAISE_BIG
        if eq >= 0.58:
            return Act.RAISE_MED if self.rand() < 0.50 else Act.CHECK_CALL
        if eq >= po + 0.04:
            return Act.CHECK_CALL
        return Act.FOLD if st.to_call > 0 else Act.CHECK_CALL

    # ----------------------------------------------------------------
    # 9f. MANIAC_CONTROL (vs maniac)
    # ----------------------------------------------------------------

    def _maniac_control(self, st: GameState, eq: float, p: OpponentProfile, tex: BoardTexture, po: float, street: int, short: bool) -> Act:
        # Tighter trapping: let them bluff into us
        if street == 0:
            if eq >= 0.70:
                # Occasional smooth call to induce
                return Act.CHECK_CALL if self.rand() < 0.40 else Act.RAISE_BIG
            if eq >= 0.55:
                return Act.CHECK_CALL
            return Act.FOLD if st.to_call > 0 else Act.CHECK_CALL

        if eq >= 0.75:
            if st.to_call == 0 and self.rand() < 0.60:
                return Act.CHECK_CALL   # check-raise
            return Act.RAISE_BIG
        if eq >= 0.60:
            return Act.CHECK_CALL       # induce bluffs
        if eq >= po + 0.05:
            return Act.CHECK_CALL
        return Act.FOLD if st.to_call > 0 else Act.CHECK_CALL


# ---------------------------------------------------------------------------
# 10. SIMULATION BUDGET
# ---------------------------------------------------------------------------

def sim_budget(st: GameState, rng: random.Random, clock: float) -> int:
    board = len(st.board)
    base = 240 if board == 0 else (180 if board == 3 else (140 if board == 4 else 100))
    budget = rng.randint(int(base * 0.85), int(base * 1.15))

    if clock < 10.0:
        budget = min(budget, 100)
    if clock < 4.0:
        budget = min(budget, 60)
    if clock < 1.5:
        budget = min(budget, 30)

    return budget


# ---------------------------------------------------------------------------
# 11. ENGINE → SKELETON ACTION MAPPER
# ---------------------------------------------------------------------------

def to_engine_action(dec: Decision, legal_actions, continue_cost: int, raise_bounds: Tuple[int, int], my_pip: int):
    can_raise = RaiseAction in legal_actions
    can_call  = CallAction in legal_actions
    can_check = CheckAction in legal_actions

    def call_or_check():
        if continue_cost == 0 and can_check:
            return CheckAction()
        if can_call:
            return CallAction()
        return CheckAction() if can_check else FoldAction()

    if dec.act == Act.FOLD:
        if continue_cost == 0 and can_check:
            return CheckAction()
        return FoldAction()

    if dec.act == Act.CHECK_CALL:
        return call_or_check()

    # Raise
    if not can_raise or raise_bounds[1] <= 0 or raise_bounds[0] > raise_bounds[1]:
        return call_or_check()

    target = max(raise_bounds[0], min(dec.amount, raise_bounds[1]))
    return RaiseAction(target)


# ---------------------------------------------------------------------------
# 12. BOT ENTRY POINT
# ---------------------------------------------------------------------------

class Player(Bot):

    def __init__(self):
        self._equity_engine = EquityEngine(seed=None)
        self._registry = OpponentRegistry()
        self._action_engine = ActionEngine(random.Random())
        self._budget_rng = random.Random()
        self._debug = os.getenv("POKER_DEBUG", "0") != "0"
        self._last_strategy: StrategyMode = StrategyMode.BALANCED

    def handle_new_round(self, game_state, round_state, active):
        pass  # Reset per-round state if needed

    def handle_round_over(self, game_state, terminal_state, active):
        """Record round outcome to refine opponent model."""
        prev = terminal_state.previous_state
        delta = terminal_state.deltas[active]

        if prev is None:
            return

        try:
            opp_cards = prev.hands[1 - active]
            opp_hidden = len(opp_cards) < 2 or opp_cards[0] == "" or opp_cards[1] == ""
        except Exception:
            opp_hidden = True

        profile = self._registry.get("villain")
        if opp_hidden and delta > 0:
            # Opponent folded
            profile.record(0)
        elif not opp_hidden:
            # Went to showdown
            profile.record(1)

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()

        my_pip  = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        continue_cost = max(0, opp_pip - my_pip)

        # Observe opponent's action this street
        profile = self._registry.get("villain")
        if continue_cost > 0:
            pot = max(1, (STARTING_STACK - round_state.stacks[active]) + (STARTING_STACK - round_state.stacks[1 - active]))
            self._registry.observe("villain", 2, pot=pot, bet=continue_cost)
        elif round_state.button > 0:
            self._registry.observe("villain", 1)

        raise_bounds = round_state.raise_bounds() if RaiseAction in legal_actions else (0, 0)

        # Build internal state
        st = adapt_state(game_state, round_state, active)
        if not st.valid():
            if CheckAction in legal_actions:
                return CheckAction()
            if CallAction in legal_actions:
                return CallAction()
            return FoldAction()

        # Classify opponent + select strategy
        opp_type = profile.classify()
        confidence = profile.confidence()
        mode = select_strategy(opp_type, confidence)

        # Equity
        sims = sim_budget(st, self._budget_rng, game_state.game_clock)
        equity = self._equity_engine.estimate(st, sims, time_limit_us=550_000)

        # Texture
        texture = analyze_texture(st.board)
        in_position = (round_state.button == active)

        # Decide
        dec = self._action_engine.decide(
            st, equity, profile, mode, texture, in_position,
            raise_bounds[0], raise_bounds[1], my_pip
        )

        out = to_engine_action(dec, legal_actions, continue_cost, raise_bounds, my_pip)

        if self._debug:
            print(
                f"[DBG] r={game_state.round_num} st={round_state.street}"
                f" eq={equity:.3f} opp={opp_type.name} conf={confidence.name}"
                f" mode={mode.name} act={type(out).__name__}"
                f" ip={in_position} pot={st.pot} call={st.to_call}",
                file=sys.stderr,
            )

        return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args()
    run_bot(Player(), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())