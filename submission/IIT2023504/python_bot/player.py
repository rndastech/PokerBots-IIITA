"""
Python port of IIT2023504 C++ bot logic.
The behavior mirrors the C++ implementation as closely as possible.
"""

import os
import random
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.states import STARTING_STACK


class Suit:
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


@dataclass(frozen=True)
class Card:
    rank: int  # 2..14 where 14 = Ace
    suit: int


def valid_card(c: Card) -> bool:
    return 2 <= c.rank <= 14 and 0 <= c.suit <= 3


def card_id(c: Card) -> int:
    return c.suit * 13 + (c.rank - 2)


def parse_rank(r: str) -> int:
    if "2" <= r <= "9":
        return ord(r) - ord("0")
    if r in ("T", "t"):
        return 10
    if r in ("J", "j"):
        return 11
    if r in ("Q", "q"):
        return 12
    if r in ("K", "k"):
        return 13
    if r in ("A", "a"):
        return 14
    return -1


def parse_suit(s: str) -> int:
    if s in ("c", "C"):
        return Suit.CLUBS
    if s in ("d", "D"):
        return Suit.DIAMONDS
    if s in ("h", "H"):
        return Suit.HEARTS
    if s in ("s", "S"):
        return Suit.SPADES
    return -1


def parse_card_token(token: str):
    if token is None or len(token) < 2:
        return None
    r = parse_rank(token[0])
    s = parse_suit(token[1])
    if r < 2 or s < 0:
        return None
    return Card(rank=r, suit=s)


@dataclass
class MyGameState:
    hero_hole: List[Card] = field(default_factory=list)
    board: List[Card] = field(default_factory=list)
    num_opponents: int = 1
    pot: int = 0
    to_call: int = 0
    hero_stack: int = 0
    starting_stack: int = STARTING_STACK
    focus_opponent: str = "villain"

    def valid(self) -> bool:
        if len(self.hero_hole) != 2:
            return False
        if len(self.board) > 5:
            return False
        if not (1 <= self.num_opponents <= 9):
            return False

        if not valid_card(self.hero_hole[0]) or not valid_card(self.hero_hole[1]):
            return False

        used = [False] * 52
        h0 = card_id(self.hero_hole[0])
        h1 = card_id(self.hero_hole[1])
        if h0 < 0 or h0 >= 52 or h1 < 0 or h1 >= 52 or h0 == h1:
            return False
        used[h0] = True
        used[h1] = True

        for c in self.board:
            if not valid_card(c):
                return False
            cid = card_id(c)
            if cid < 0 or cid >= 52 or used[cid]:
                return False
            used[cid] = True

        return True


class Eval:
    @staticmethod
    def has_rank(mask: int, rank: int) -> bool:
        if rank < 2 or rank > 14:
            return False
        return ((mask >> (rank - 2)) & 1) != 0

    @staticmethod
    def highest_straight_high_card(rank_mask: int) -> int:
        for high in range(14, 4, -1):
            ok = True
            for r in range(high, high - 5, -1):
                if not Eval.has_rank(rank_mask, r):
                    ok = False
                    break
            if ok:
                return high

        if (
            Eval.has_rank(rank_mask, 14)
            and Eval.has_rank(rank_mask, 2)
            and Eval.has_rank(rank_mask, 3)
            and Eval.has_rank(rank_mask, 4)
            and Eval.has_rank(rank_mask, 5)
        ):
            return 5
        return 0

    @staticmethod
    def pack5(a: int, b: int = 0, c: int = 0, d: int = 0, e: int = 0) -> int:
        return (a << 16) | (b << 12) | (c << 8) | (d << 4) | e

    @staticmethod
    def evaluate_n(cards: List[Card]) -> int:
        n = len(cards)
        if n < 5 or n > 7:
            raise ValueError("evaluate_n expects 5..7 cards")

        rank_count = [0] * 15
        suit_count = [0] * 4
        suit_mask = [0, 0, 0, 0]
        rank_mask = 0

        for c in cards:
            r = c.rank
            s = c.suit
            rank_count[r] += 1
            suit_count[s] += 1
            bit = 1 << (r - 2)
            suit_mask[s] |= bit
            rank_mask |= bit

        flush_suit = -1
        for s in range(4):
            if suit_count[s] >= 5:
                flush_suit = s
                break

        if flush_suit != -1:
            sf_high = Eval.highest_straight_high_card(suit_mask[flush_suit])
            if sf_high > 0:
                return (8 << 20) | Eval.pack5(sf_high)

        quads: List[int] = []
        trips: List[int] = []
        pairs: List[int] = []
        singles: List[int] = []

        for r in range(14, 1, -1):
            if rank_count[r] == 4:
                quads.append(r)
            elif rank_count[r] == 3:
                trips.append(r)
            elif rank_count[r] == 2:
                pairs.append(r)
            elif rank_count[r] == 1:
                singles.append(r)

        if quads:
            kicker = 0
            for r in range(14, 1, -1):
                if r != quads[0] and rank_count[r] > 0:
                    kicker = r
                    break
            return (7 << 20) | Eval.pack5(quads[0], kicker)

        if trips and (pairs or len(trips) > 1):
            trip_rank = trips[0]
            pair_rank = trips[1] if len(trips) > 1 else pairs[0]
            return (6 << 20) | Eval.pack5(trip_rank, pair_rank)

        if flush_suit != -1:
            flush_cards = []
            for r in range(14, 1, -1):
                if Eval.has_rank(suit_mask[flush_suit], r):
                    flush_cards.append(r)
                    if len(flush_cards) == 5:
                        break
            return (5 << 20) | Eval.pack5(
                flush_cards[0], flush_cards[1], flush_cards[2], flush_cards[3], flush_cards[4]
            )

        st_high = Eval.highest_straight_high_card(rank_mask)
        if st_high > 0:
            return (4 << 20) | Eval.pack5(st_high)

        if trips:
            k1 = 0
            k2 = 0
            for r in range(14, 1, -1):
                if r == trips[0]:
                    continue
                if rank_count[r] > 0:
                    if k1 == 0:
                        k1 = r
                    else:
                        k2 = r
                        break
            return (3 << 20) | Eval.pack5(trips[0], k1, k2)

        if len(pairs) >= 2:
            kicker = 0
            for r in range(14, 1, -1):
                if r != pairs[0] and r != pairs[1] and rank_count[r] > 0:
                    kicker = r
                    break
            return (2 << 20) | Eval.pack5(pairs[0], pairs[1], kicker)

        if len(pairs) == 1:
            ks = [0, 0, 0]
            idx = 0
            for r in range(14, 1, -1):
                if r != pairs[0] and rank_count[r] > 0:
                    ks[idx] = r
                    idx += 1
                    if idx == 3:
                        break
            return (1 << 20) | Eval.pack5(pairs[0], ks[0], ks[1], ks[2])

        hs = [0, 0, 0, 0, 0]
        idx = 0
        for r in range(14, 1, -1):
            if rank_count[r] > 0:
                hs[idx] = r
                idx += 1
                if idx == 5:
                    break
        return Eval.pack5(hs[0], hs[1], hs[2], hs[3], hs[4])

    @staticmethod
    def evaluate7(cards7: List[Card]) -> int:
        return Eval.evaluate_n(cards7)


class MonteCarloEngine:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self._timed_out_last = False
        self._sims_done_last = 0

    def estimate_equity(self, state: MyGameState, simulations: int, enable_early_stop=True, max_time_us=700000) -> float:
        self._timed_out_last = False
        self._sims_done_last = 0

        if (not state.valid()) or simulations <= 0:
            return 0.0

        start_ns = time.perf_counter_ns()

        used = [False] * 52
        pool: List[Card] = []

        for c in state.hero_hole:
            cid = card_id(c)
            if used[cid]:
                return 0.0
            used[cid] = True

        for c in state.board:
            cid = card_id(c)
            if used[cid]:
                return 0.0
            used[cid] = True

        for s in range(4):
            for r in range(2, 15):
                c = Card(rank=r, suit=s)
                if not used[card_id(c)]:
                    pool.append(c)

        opp_count = state.num_opponents
        board_missing = 5 - len(state.board)
        need_cards = opp_count * 2 + board_missing
        if need_cards > len(pool):
            return 0.0

        wins = 0
        ties = 0

        fixed_board = board_missing == 0
        full_board: List[Card] = [None] * 5  # type: ignore
        if fixed_board:
            for i in range(5):
                full_board[i] = state.board[i]

        hero_fixed_score = 0
        if fixed_board:
            hero7 = [state.hero_hole[0], state.hero_hole[1]] + full_board
            hero_fixed_score = Eval.evaluate7(hero7)

        total_done = 0
        min_samples = min(simulations, 80)
        check_interval = 24

        for sim in range(simulations):
            if max_time_us > 0:
                elapsed_us = (time.perf_counter_ns() - start_ns) // 1000
                if elapsed_us >= max_time_us and sim >= 24:
                    self._timed_out_last = True
                    break

            sim_pool = pool.copy()
            live_size = len(sim_pool)

            def draw_random() -> Card:
                nonlocal live_size
                idx = self.rng.randrange(live_size)
                out = sim_pool[idx]
                sim_pool[idx] = sim_pool[live_size - 1]
                live_size -= 1
                return out

            opp_holes = []
            for _ in range(opp_count):
                opp_holes.append([draw_random(), draw_random()])

            if not fixed_board:
                for i in range(len(state.board)):
                    full_board[i] = state.board[i]
                for i in range(len(state.board), 5):
                    full_board[i] = draw_random()

            hero_score = hero_fixed_score
            if not fixed_board:
                hero7 = [state.hero_hole[0], state.hero_hole[1]] + full_board
                hero_score = Eval.evaluate7(hero7)

            hero_best = True
            hero_tied = False

            for o in range(opp_count):
                opp7 = [opp_holes[o][0], opp_holes[o][1]] + full_board
                opp_score = Eval.evaluate7(opp7)
                if opp_score > hero_score:
                    hero_best = False
                    hero_tied = False
                    break
                if opp_score == hero_score:
                    hero_tied = True

            if hero_best:
                if hero_tied:
                    ties += 1
                else:
                    wins += 1

            total_done = sim + 1
            if enable_early_stop and total_done >= min_samples and (total_done % check_interval == 0):
                eq = (wins + 0.5 * ties) / float(total_done)
                if eq >= 0.90 or eq <= 0.10:
                    break

        self._sims_done_last = total_done
        if total_done <= 0:
            return -1.0
        return (wins + 0.5 * ties) / float(total_done)

    def timed_out_last(self) -> bool:
        return self._timed_out_last

    def sims_done_last(self) -> int:
        return self._sims_done_last


@dataclass
class OpponentStats:
    total_actions: int = 0
    folds: int = 0
    calls: int = 0
    raises: int = 0

    fold_to_raise: int = 0
    fold_to_raise_opportunities: int = 0

    street_actions: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    street_raises: List[int] = field(default_factory=lambda: [0, 0, 0, 0])

    recent_actions: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    recent_count: int = 0
    recent_pos: int = 0

    def push_recent(self, action: int):
        self.recent_actions[self.recent_pos] = action
        self.recent_pos = (self.recent_pos + 1) % len(self.recent_actions)
        if self.recent_count < len(self.recent_actions):
            self.recent_count += 1

    def fold_rate(self) -> float:
        return (self.folds / self.total_actions) if self.total_actions > 0 else 0.0

    def aggression(self) -> float:
        return (self.raises / self.total_actions) if self.total_actions > 0 else 0.0

    def call_rate(self) -> float:
        return (self.calls / self.total_actions) if self.total_actions > 0 else 0.0

    def fold_to_raise_rate(self) -> float:
        if self.fold_to_raise_opportunities <= 0:
            return 0.0
        return self.fold_to_raise / self.fold_to_raise_opportunities

    def recent_aggression(self) -> float:
        if self.recent_count <= 0:
            return 0.0
        raises_recent = 0
        for i in range(self.recent_count):
            if self.recent_actions[i] == 2:
                raises_recent += 1
        return raises_recent / self.recent_count

    def is_tight(self) -> bool:
        return self.fold_rate() > 0.45 and self.aggression() < 0.30

    def is_loose(self) -> bool:
        return self.fold_rate() < 0.28 and self.call_rate() > 0.48

    def is_aggressive(self) -> bool:
        return self.aggression() > 0.42 or self.recent_aggression() > 0.50


class ObservedActionType(Enum):
    FOLD = auto()
    CALL = auto()
    RAISE = auto()


class OpponentModel:
    def __init__(self):
        self.stats: Dict[str, OpponentStats] = {}

    def observe_action(self, name: str, action: ObservedActionType, street_idx: int = -1, faced_raise: bool = False):
        s = self.stats.setdefault(name, OpponentStats())
        s.total_actions += 1

        if 0 <= street_idx <= 3:
            s.street_actions[street_idx] += 1

        if faced_raise:
            s.fold_to_raise_opportunities += 1

        if action == ObservedActionType.FOLD:
            s.folds += 1
            s.push_recent(0)
            if faced_raise:
                s.fold_to_raise += 1
        elif action == ObservedActionType.CALL:
            s.calls += 1
            s.push_recent(1)
        else:
            s.raises += 1
            s.push_recent(2)
            if 0 <= street_idx <= 3:
                s.street_raises[street_idx] += 1

    def get_stats(self, name: str) -> OpponentStats:
        return self.stats.get(name, OpponentStats())


class BotActionType(Enum):
    FOLD = auto()
    CALL = auto()
    RAISE_THIRD_POT = auto()
    RAISE_HALF_POT = auto()
    RAISE_TWO_THIRD_POT = auto()
    RAISE_POT = auto()
    ALL_IN = auto()


def is_raise_action(t: BotActionType) -> bool:
    return t in {
        BotActionType.RAISE_THIRD_POT,
        BotActionType.RAISE_HALF_POT,
        BotActionType.RAISE_TWO_THIRD_POT,
        BotActionType.RAISE_POT,
    }


def is_aggressive_action(t: BotActionType) -> bool:
    return is_raise_action(t) or t == BotActionType.ALL_IN


@dataclass
class BoardTexture:
    flush_heavy: bool = False
    straight_heavy: bool = False
    paired: bool = False
    dangerous: bool = False


@dataclass
class QuickEquityHint:
    use_shortcut: bool = False
    equity: float = 0.5


@dataclass
class BotAction:
    type: BotActionType = BotActionType.FOLD
    amount: int = 0


def street_index_from_board_count(board_count: int) -> int:
    if board_count <= 0:
        return 0
    if board_count == 3:
        return 1
    if board_count == 4:
        return 2
    return 3


def analyze_board_texture(st: MyGameState) -> BoardTexture:
    t = BoardTexture()
    if len(st.board) < 3:
        return t

    suit_count = [0, 0, 0, 0]
    rank_count = [0] * 15
    uniq_ranks: List[int] = []

    for c in st.board:
        suit_count[c.suit] += 1
        rank_count[c.rank] += 1

    for s in range(4):
        if suit_count[s] >= 3:
            t.flush_heavy = True
            break

    for r in range(2, 15):
        if rank_count[r] > 0:
            uniq_ranks.append(r)
        if rank_count[r] >= 2:
            t.paired = True

    if len(uniq_ranks) >= 3:
        uniq_ranks.sort()
        for i in range(0, len(uniq_ranks) - 2):
            if uniq_ranks[i + 2] - uniq_ranks[i] <= 4:
                t.straight_heavy = True
                break

    t.dangerous = t.flush_heavy or t.straight_heavy
    return t


def quick_equity_shortcut(st: MyGameState) -> QuickEquityHint:
    hint = QuickEquityHint()
    if not st.valid():
        return hint

    a = st.hero_hole[0]
    b = st.hero_hole[1]
    hi = max(a.rank, b.rank)
    lo = min(a.rank, b.rank)
    pair = a.rank == b.rank
    suited = a.suit == b.suit
    gap = hi - lo

    board_count = len(st.board)

    if board_count == 0:
        eq = 0.50
        if pair:
            eq = 0.56 + (hi - 2) * 0.025
        else:
            eq = 0.34
            if hi >= 13:
                eq += 0.06
            if hi == 14 and lo >= 10:
                eq += 0.06
            if suited:
                eq += 0.03
            eq -= min(0.07, float(gap) * 0.015)
            if gap >= 5 and not suited:
                eq -= 0.04

        eq -= 0.075 * float(st.num_opponents - 1)
        eq = max(0.08, min(0.92, eq))

        if (pair and hi >= 12) or (hi == 14 and lo >= 13 and suited):
            hint.use_shortcut = True
            hint.equity = min(0.92, eq + 0.06)
            return hint

        if (not pair) and (not suited) and hi <= 10 and gap >= 5 and eq < 0.25:
            hint.use_shortcut = True
            hint.equity = max(0.08, eq - 0.03)
            return hint

        return hint

    rank_count = [0] * 15
    suit_count = [0, 0, 0, 0]
    rank_mask = 0

    rank_count[a.rank] += 1
    rank_count[b.rank] += 1
    suit_count[a.suit] += 1
    suit_count[b.suit] += 1
    rank_mask |= 1 << (a.rank - 2)
    rank_mask |= 1 << (b.rank - 2)

    for c in st.board:
        rank_count[c.rank] += 1
        suit_count[c.suit] += 1
        rank_mask |= 1 << (c.rank - 2)

    pairs = 0
    trips = 0
    quads = 0
    for r in range(2, 15):
        if rank_count[r] == 4:
            quads += 1
        elif rank_count[r] == 3:
            trips += 1
        elif rank_count[r] == 2:
            pairs += 1

    flush_draw = False
    flush_made = False
    for s in range(4):
        if suit_count[s] >= 5:
            flush_made = True
        if suit_count[s] == 4 and board_count < 5:
            flush_draw = True

    straight_made = Eval.highest_straight_high_card(rank_mask) > 0
    straight_draw = False
    if (not straight_made) and board_count < 5:
        for high in range(14, 4, -1):
            cnt = 0
            for r in range(high, high - 5, -1):
                if Eval.has_rank(rank_mask, r):
                    cnt += 1
            if cnt >= 4:
                straight_draw = True
                break

    if board_count == 5:
        hero7 = [a, b] + st.board[:5]
        score = Eval.evaluate7(hero7)
        cat = score >> 20

        if cat >= 6:
            hint.use_shortcut = True
            hint.equity = 0.93
        elif cat >= 4:
            hint.use_shortcut = True
            hint.equity = 0.82
        elif cat == 3:
            hint.use_shortcut = True
            hint.equity = 0.74
        elif cat <= 1:
            hint.use_shortcut = True
            hint.equity = 0.24

        return hint

    if quads > 0 or trips > 0 or pairs >= 2 or flush_made or straight_made:
        hint.use_shortcut = True
        hint.equity = 0.80
        return hint

    weak_no_draw = pairs == 0 and (not flush_draw) and (not straight_draw) and hi <= 11
    if weak_no_draw and board_count >= 3:
        hint.use_shortcut = True
        hint.equity = 0.20
        return hint

    return hint


class DecisionEngine:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.bluff_attempts = 0
        self.bluff_successes = 0

    def rand01(self) -> float:
        return self.rng.random()

    def record_bluff_result(self, success: bool):
        self.bluff_attempts += 1
        if success:
            self.bluff_successes += 1

    def bluff_success_rate(self) -> float:
        return (self.bluff_successes / self.bluff_attempts) if self.bluff_attempts > 0 else 0.5

    def clamp_bet(self, st: MyGameState, target: int) -> int:
        if target <= st.to_call:
            return st.to_call
        return min(target, st.hero_stack)

    def compute_pot_odds(self, st: MyGameState) -> float:
        if st.to_call <= 0:
            return 0.0
        denom = float(st.pot + st.to_call)
        if denom <= 0.0:
            return 1.0
        return float(st.to_call) / denom

    def make_fold_or_check(self, st: MyGameState) -> BotAction:
        if st.to_call <= 0:
            return BotAction(BotActionType.CALL, 0)
        return BotAction(BotActionType.FOLD, 0)

    def make_call_or_check(self, st: MyGameState) -> BotAction:
        amt = min(st.to_call, st.hero_stack)
        return BotAction(BotActionType.CALL, amt)

    def make_raise_fraction(self, st: MyGameState, num: int, den: int, action_type: BotActionType) -> BotAction:
        base_extra = max(1, (st.pot * num) // den)
        raise_to = st.to_call + base_extra
        jitter = 0.90 + 0.25 * self.rand01()
        raise_to = int(raise_to * jitter)
        raise_to = self.clamp_bet(st, raise_to)

        if raise_to >= st.hero_stack:
            return BotAction(BotActionType.ALL_IN, st.hero_stack)
        if raise_to <= st.to_call:
            return self.make_call_or_check(st)
        return BotAction(action_type, raise_to)

    def make_raise_third_pot(self, st: MyGameState) -> BotAction:
        return self.make_raise_fraction(st, 1, 3, BotActionType.RAISE_THIRD_POT)

    def make_raise_half_pot(self, st: MyGameState) -> BotAction:
        return self.make_raise_fraction(st, 1, 2, BotActionType.RAISE_HALF_POT)

    def make_raise_two_thirds_pot(self, st: MyGameState) -> BotAction:
        return self.make_raise_fraction(st, 2, 3, BotActionType.RAISE_TWO_THIRD_POT)

    def make_raise_pot(self, st: MyGameState) -> BotAction:
        return self.make_raise_fraction(st, 1, 1, BotActionType.RAISE_POT)

    def choose_aggressive_value_bet(self, st: MyGameState, equity: float) -> BotAction:
        if equity > 0.85 and self.rand01() < 0.30:
            return BotAction(BotActionType.ALL_IN, st.hero_stack)
        if equity >= 0.70:
            if self.rand01() < 0.55:
                return self.make_raise_pot(st)
            return self.make_raise_two_thirds_pot(st)
        if self.rand01() < 0.55:
            return self.make_raise_two_thirds_pot(st)
        return self.make_raise_half_pot(st)

    def weighted_mix(
        self,
        st: MyGameState,
        w_fold: float,
        w_call: float,
        w_r33: float,
        w_r66: float,
        w_r100: float,
        w_allin: float,
    ) -> BotAction:
        total = w_fold + w_call + w_r33 + w_r66 + w_r100 + w_allin
        x = self.rand01() * (total if total > 0.0 else 1.0)

        x -= w_fold
        if x <= 0.0:
            return self.make_fold_or_check(st)

        x -= w_call
        if x <= 0.0:
            return self.make_call_or_check(st)

        x -= w_r33
        if x <= 0.0:
            return self.make_raise_third_pot(st)

        x -= w_r66
        if x <= 0.0:
            return self.make_raise_two_thirds_pot(st)

        x -= w_r100
        if x <= 0.0:
            return self.make_raise_pot(st)

        if w_allin > 0.0:
            return BotAction(BotActionType.ALL_IN, st.hero_stack)

        return self.make_call_or_check(st)

    def random_legal_mix(self, st: MyGameState, equity: float, pot_odds: float) -> BotAction:
        if equity < 0.35:
            if pot_odds > 0.30:
                return self.weighted_mix(st, 0.78, 0.20, 0.02, 0.00, 0.00, 0.00)
            return self.weighted_mix(st, 0.68, 0.22, 0.10, 0.00, 0.00, 0.00)
        if equity < 0.70:
            return self.weighted_mix(st, 0.16, 0.58, 0.08, 0.14, 0.04, 0.00)
        return self.weighted_mix(st, 0.04, 0.28, 0.05, 0.26, 0.30, 0.07)

    def decide_preflop(self, st: MyGameState, equity: float, opp: OpponentStats, short_stack: bool) -> BotAction:
        a = st.hero_hole[0]
        b = st.hero_hole[1]
        hi = max(a.rank, b.rank)
        lo = min(a.rank, b.rank)
        pair = a.rank == b.rank
        suited = a.suit == b.suit

        strong = (pair and hi >= 12) or (hi == 14 and lo == 13)
        medium = (pair and hi >= 9) or (hi == 14 and lo >= 11) or (hi == 13 and lo == 12)
        weak = (not strong) and (not medium)

        if short_stack:
            if strong or equity >= 0.58:
                return BotAction(BotActionType.ALL_IN, st.hero_stack)
            if st.to_call > 0 and equity < 0.44:
                return self.make_fold_or_check(st)
            return self.make_call_or_check(st)

        if strong:
            if st.to_call == 0 and self.rand01() < 0.18:
                return self.make_raise_half_pot(st)
            return self.make_raise_pot(st) if self.rand01() < 0.60 else self.make_raise_two_thirds_pot(st)

        if medium:
            if opp.is_loose():
                return self.make_raise_two_thirds_pot(st) if self.rand01() < 0.58 else self.make_call_or_check(st)
            if opp.is_aggressive():
                return self.make_call_or_check(st) if self.rand01() < 0.58 else self.make_raise_half_pot(st)
            return self.make_raise_half_pot(st) if self.rand01() < 0.52 else self.make_call_or_check(st)

        if weak:
            if st.to_call > 0 and (equity < 0.42 or st.to_call > st.hero_stack // 5):
                return self.make_fold_or_check(st)
            if suited and hi >= 11 and self.rand01() < 0.22 and st.to_call == 0:
                return self.make_raise_third_pot(st)
            return self.make_call_or_check(st)

        return self.make_call_or_check(st)

    def finalize_action(self, st: MyGameState, action: BotAction) -> BotAction:
        if action.type == BotActionType.ALL_IN:
            action.amount = st.hero_stack
            return action

        if action.type == BotActionType.CALL:
            action.amount = min(st.to_call, st.hero_stack)
            return action

        if is_raise_action(action.type):
            if action.amount >= st.hero_stack:
                return BotAction(BotActionType.ALL_IN, st.hero_stack)
            if action.amount <= st.to_call:
                return self.make_call_or_check(st)

        return action

    def decide(self, st: MyGameState, equity: float, opp: OpponentStats) -> BotAction:
        equity = max(0.0, min(1.0, equity + (self.rand01() - 0.5) * 0.02))

        pot_odds = self.compute_pot_odds(st)
        texture = analyze_board_texture(st)
        action = self.make_fold_or_check(st)

        facing_big_bet = st.to_call > 0 and st.hero_stack > 0 and (st.to_call * 10 >= 4 * st.hero_stack)
        if facing_big_bet and equity < 0.75:
            return self.make_fold_or_check(st)

        facing_near_allin = st.to_call > 0 and st.hero_stack > 0 and (st.to_call * 10 >= 8 * st.hero_stack)
        if facing_near_allin and equity < 0.80:
            return self.make_fold_or_check(st)

        very_low_stack = st.starting_stack > 0 and (st.hero_stack * 100 <= 12 * st.starting_stack)
        if very_low_stack and st.to_call > 0 and equity < 0.50:
            return self.make_fold_or_check(st)

        big_pot = st.hero_stack > 0 and st.pot > st.hero_stack / 2
        very_big_pot = st.hero_stack > 0 and st.pot >= (st.hero_stack * 3) // 4
        short_stack = st.starting_stack > 0 and st.hero_stack * 10 <= 2 * st.starting_stack

        if len(st.board) == 0:
            action = self.decide_preflop(st, equity, opp, short_stack)
            if big_pot and st.to_call > 0 and equity < max(0.72, pot_odds + 0.12):
                action = self.make_fold_or_check(st)
            if action.type == BotActionType.ALL_IN and equity < 0.62:
                action = self.make_fold_or_check(st) if (st.to_call > 0 and equity < pot_odds + 0.02) else self.make_call_or_check(st)
            return self.finalize_action(st, action)

        if short_stack:
            if equity >= 0.62:
                action = BotAction(BotActionType.ALL_IN, st.hero_stack)
            elif st.to_call > 0 and equity + 0.02 < pot_odds:
                action = self.make_fold_or_check(st)

        risk_premium = 0.06 if big_pot else 0.0
        if st.to_call > 0 and equity + 0.015 + risk_premium < pot_odds:
            action = self.make_fold_or_check(st)
        elif equity >= 0.85:
            action = self.choose_aggressive_value_bet(st, equity)
        elif equity >= 0.75:
            action = self.weighted_mix(st, 0.01, 0.18, 0.04, 0.38, 0.30, 0.09)
        elif equity >= 0.70:
            action = self.weighted_mix(st, 0.02, 0.22, 0.06, 0.39, 0.24, 0.07)
        elif equity >= max(0.60, pot_odds + 0.10):
            action = self.weighted_mix(st, 0.08, 0.48, 0.14, 0.20, 0.10, 0.00)
        elif equity >= pot_odds + 0.02:
            action = self.weighted_mix(st, 0.20, 0.68, 0.05, 0.07, 0.00, 0.00)
        else:
            action = self.make_fold_or_check(st)

        fold_rate = opp.fold_rate()
        call_rate = opp.call_rate()
        aggression = opp.aggression()
        fold_to_raise = opp.fold_to_raise_rate()
        bluff_sr = self.bluff_success_rate()

        if st.to_call == 0 and 0.50 < equity < 0.70 and fold_to_raise > 0.56 and (not texture.dangerous):
            if self.rand01() < 0.09:
                action = self.make_raise_third_pot(st)

        pot_small_for_bluff = st.pot <= int(0.36 * st.hero_stack)
        allow_bluff = equity < 0.35 and fold_rate > 0.58 and pot_small_for_bluff and (not texture.dangerous)

        if allow_bluff or (equity < 0.40 and fold_to_raise > 0.55 and pot_small_for_bluff and (not texture.dangerous)):
            bluff_p = 0.30 if fold_to_raise > 0.60 else 0.22
            if opp.is_tight():
                bluff_p += 0.10
            if call_rate > 0.55 or opp.is_loose():
                bluff_p -= 0.16
            if call_rate > 0.60:
                bluff_p = 0.0
            if texture.paired:
                bluff_p -= 0.03
            if self.bluff_attempts >= 5:
                if bluff_sr < 0.45:
                    bluff_p *= 0.45
                elif bluff_sr > 0.60:
                    bluff_p *= 1.20

            bluff_p = max(0.04, min(0.38, bluff_p))
            if self.rand01() < bluff_p:
                action = self.make_raise_third_pot(st) if self.rand01() < 0.60 else self.make_raise_two_thirds_pot(st)

        if call_rate > 0.55 or opp.is_loose():
            if equity < 0.55:
                action = self.make_call_or_check(st)
                if equity < 0.35:
                    action = self.make_fold_or_check(st)
            elif equity > 0.66:
                action = self.choose_aggressive_value_bet(st, equity)

        if (aggression > 0.45 or opp.is_aggressive()) and equity > 0.72:
            if self.rand01() < 0.60:
                action = self.make_call_or_check(st)

        if opp.is_tight() and equity < 0.38 and pot_small_for_bluff and (not texture.dangerous) and self.rand01() < 0.20:
            action = self.make_raise_third_pot(st)

        if texture.dangerous and equity < 0.72 and is_raise_action(action.type) and self.rand01() < 0.65:
            action = self.make_call_or_check(st)

        dev = 0.10
        if equity < 0.35:
            dev = 0.10
        elif equity < 0.70:
            dev = 0.14
        else:
            dev = 0.12

        if texture.dangerous and equity < 0.70:
            dev *= 0.65

        if self.rand01() < dev:
            action = self.random_legal_mix(st, equity, pot_odds)

        if action.type == BotActionType.CALL and st.to_call > 0 and equity + 0.01 < pot_odds:
            action = self.make_fold_or_check(st)

        if st.to_call > 0 and st.hero_stack > 0 and st.to_call * 10 >= 6 * st.hero_stack and equity < 0.78:
            action = self.make_fold_or_check(st)

        if big_pot and st.to_call > 0 and equity < max(0.73, pot_odds + 0.10):
            action = self.make_fold_or_check(st)

        if very_big_pot and st.to_call > 0 and equity < max(0.79, pot_odds + 0.14):
            action = self.make_fold_or_check(st)

        if equity < 0.40 and big_pot and (is_raise_action(action.type) or action.type == BotActionType.ALL_IN):
            action = self.make_fold_or_check(st)

        very_high_stack = st.starting_stack > 0 and st.hero_stack * 10 >= 17 * st.starting_stack
        if very_high_stack and action.type == BotActionType.ALL_IN and equity < 0.90:
            action = self.choose_aggressive_value_bet(st, equity)
            if action.type == BotActionType.ALL_IN:
                action = self.make_raise_pot(st)

        if action.type == BotActionType.ALL_IN and equity < 0.70:
            action = self.make_fold_or_check(st) if (st.to_call > 0 and equity < pot_odds + 0.02) else self.make_call_or_check(st)

        return self.finalize_action(st, action)


def adapt_state(game_state, round_state, active: int) -> MyGameState:
    st = MyGameState()
    st.num_opponents = 1
    st.starting_stack = STARTING_STACK

    if len(round_state.hands[active]) >= 2:
        c0 = parse_card_token(round_state.hands[active][0])
        c1 = parse_card_token(round_state.hands[active][1])
        if c0 is not None and c1 is not None and c0 != c1:
            st.hero_hole = [c0, c1]

    board_count_limit = max(0, min(5, round_state.street))
    st.board = []
    for i in range(board_count_limit):
        if i >= len(round_state.deck):
            break
        bc = parse_card_token(round_state.deck[i])
        if bc is not None:
            st.board.append(bc)

    my_pip = round_state.pips[active]
    opp_pip = round_state.pips[1 - active]
    my_stack = round_state.stacks[active]
    opp_stack = round_state.stacks[1 - active]

    continue_cost = max(0, opp_pip - my_pip)
    my_contrib = STARTING_STACK - my_stack
    opp_contrib = STARTING_STACK - opp_stack

    st.pot = max(0, my_contrib + opp_contrib)
    st.to_call = continue_cost
    st.hero_stack = max(0, my_stack)
    st.starting_stack = STARTING_STACK
    st.focus_opponent = "villain"

    _ = game_state
    return st


def compute_sim_budget(st: MyGameState, rng: random.Random) -> int:
    lo, hi = 80, 120
    board_count = len(st.board)
    if board_count == 0:
        lo, hi = 200, 250
    elif board_count == 3:
        lo, hi = 150, 200
    elif board_count == 4:
        lo, hi = 120, 150
    return rng.randint(lo, hi)


def map_to_engine_action(
    action: BotAction,
    st: MyGameState,
    legal_actions,
    my_pip: int,
    continue_cost: int,
    raise_bounds: Tuple[int, int],
):
    can_raise = RaiseAction in legal_actions
    can_call = CallAction in legal_actions
    can_check = CheckAction in legal_actions

    def fallback_call_or_check():
        if continue_cost == 0 and can_check:
            return CheckAction()
        if can_call:
            return CallAction()
        if can_check:
            return CheckAction()
        return FoldAction()

    if action.type == BotActionType.FOLD:
        if continue_cost == 0 and can_check:
            return CheckAction()
        return FoldAction()

    if action.type == BotActionType.CALL:
        return fallback_call_or_check()

    if not can_raise:
        return fallback_call_or_check()

    if raise_bounds[0] <= my_pip + continue_cost:
        return fallback_call_or_check()

    if raise_bounds[0] < 0 or raise_bounds[1] < raise_bounds[0]:
        return fallback_call_or_check()

    target = raise_bounds[0]
    if action.type == BotActionType.RAISE_THIRD_POT:
        target = my_pip + continue_cost + max(1, st.pot // 3)
    elif action.type == BotActionType.RAISE_HALF_POT:
        target = my_pip + continue_cost + max(1, st.pot // 2)
    elif action.type == BotActionType.RAISE_TWO_THIRD_POT:
        target = my_pip + continue_cost + max(1, (2 * st.pot) // 3)
    elif action.type == BotActionType.RAISE_POT:
        target = my_pip + continue_cost + max(1, st.pot)
    elif action.type == BotActionType.ALL_IN:
        target = my_pip + st.hero_stack

    target = max(raise_bounds[0], min(target, raise_bounds[1]))
    return RaiseAction(target)


class Player(Bot):
    def __init__(self):
        self.mc_engine = MonteCarloEngine()
        self.opp_model = OpponentModel()
        self.decision_engine = DecisionEngine()
        self.budget_rng = random.Random()
        self.debug_mode = False

        env = os.getenv("POKER_DEBUG")
        if env is not None:
            self.debug_mode = env != "0"

    def handle_new_round(self, game_state, round_state, active):
        _ = (game_state, round_state, active)

    def handle_round_over(self, game_state, terminal_state, active):
        prev = terminal_state.previous_state
        my_delta = terminal_state.deltas[active]

        opp_cards_hidden = False
        if prev is not None:
            try:
                opp_cards = prev.hands[1 - active]
                opp_cards_hidden = (len(opp_cards) < 2) or (opp_cards[0] == "") or (opp_cards[1] == "")
            except Exception:
                opp_cards_hidden = True

        if prev is not None and opp_cards_hidden and my_delta > 0:
            self.opp_model.observe_action(
                "villain",
                ObservedActionType.FOLD,
                street_index_from_board_count(prev.street),
                False,
            )
            self.decision_engine.record_bluff_result(True)
        elif prev is not None and (not opp_cards_hidden):
            self.opp_model.observe_action(
                "villain",
                ObservedActionType.CALL,
                street_index_from_board_count(prev.street),
                False,
            )
            if my_delta < 0:
                self.decision_engine.record_bluff_result(False)

        _ = game_state

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        continue_cost = max(0, opp_pip - my_pip)

        if continue_cost > 0:
            self.opp_model.observe_action(
                "villain",
                ObservedActionType.RAISE,
                street_index_from_board_count(round_state.street),
                False,
            )
        elif round_state.button > 0:
            self.opp_model.observe_action(
                "villain",
                ObservedActionType.CALL,
                street_index_from_board_count(round_state.street),
                False,
            )

        raise_bounds = (0, 0)
        if RaiseAction in legal_actions:
            raise_bounds = round_state.raise_bounds()

        st = adapt_state(game_state, round_state, active)
        if not st.valid():
            if CheckAction in legal_actions:
                return CheckAction()
            if CallAction in legal_actions:
                return CallAction()
            return FoldAction()

        sims = compute_sim_budget(st, self.budget_rng)
        if game_state.game_clock < 10.0:
            sims = min(sims, 100)
        if game_state.game_clock < 4.0:
            sims = min(sims, 60)
        if game_state.game_clock < 1.5:
            sims = min(sims, 30)

        hint = quick_equity_shortcut(st)
        equity = hint.equity
        used_mc_fallback = False

        if not hint.use_shortcut:
            equity = self.mc_engine.estimate_equity(st, sims, True, 550000)
            if equity < 0.0 or self.mc_engine.timed_out_last():
                used_mc_fallback = True
                equity = max(0.10, min(0.90, hint.equity))

        opp = self.opp_model.get_stats("villain")
        my_action = self.decision_engine.decide(st, equity, opp)

        out = map_to_engine_action(my_action, st, legal_actions, my_pip, continue_cost, raise_bounds)

        if self.debug_mode:
            print(
                "[DEBUG]"
                f" round={game_state.round_num}"
                f" street={round_state.street}"
                f" eq={equity}"
                f" shortcut={1 if hint.use_shortcut else 0}"
                f" sims_req={0 if hint.use_shortcut else sims}"
                f" sims_done={0 if hint.use_shortcut else self.mc_engine.sims_done_last()}"
                f" mc_timeout={0 if hint.use_shortcut else (1 if self.mc_engine.timed_out_last() else 0)}"
                f" mc_fallback={1 if used_mc_fallback else 0}"
                f" pot={st.pot}"
                f" to_call={st.to_call}"
                f" opp_fold={opp.fold_rate()}"
                f" opp_f2r={opp.fold_to_raise_rate()}"
                f" opp_agg={opp.aggression()}"
                f" bluff_sr={self.decision_engine.bluff_success_rate()}"
                f" action={type(out).__name__}"
                f" amt={out.amount if isinstance(out, RaiseAction) else 0}",
                file=sys.stderr,
            )

        return out


def build_full_deck() -> List[Card]:
    deck = []
    for s in range(4):
        for r in range(2, 15):
            deck.append(Card(rank=r, suit=s))
    return deck


def random_state_for_test(rng: random.Random) -> MyGameState:
    deck = build_full_deck()
    rng.shuffle(deck)

    st = MyGameState()
    st.hero_hole = [deck[0], deck[1]]

    board_count_options = [0, 3, 4, 5]
    bc = board_count_options[rng.randint(0, 3)]
    st.board = [deck[2 + i] for i in range(bc)]

    st.starting_stack = STARTING_STACK
    st.hero_stack = rng.randint(25, STARTING_STACK)
    st.pot = rng.randint(4, STARTING_STACK * 2)
    st.to_call = rng.randint(0, st.hero_stack)
    st.num_opponents = 1
    st.focus_opponent = "villain"
    return st


def make_profile(fold_rate: float, call_rate: float, raise_rate: float) -> OpponentStats:
    _ = raise_rate
    s = OpponentStats()
    n = 200
    s.total_actions = n
    s.folds = int(fold_rate * n)
    s.calls = int(call_rate * n)
    s.raises = max(0, n - s.folds - s.calls)
    if s.folds + s.calls + s.raises < n:
        s.calls += n - (s.folds + s.calls + s.raises)
    s.fold_to_raise_opportunities = 100
    s.fold_to_raise = int(fold_rate * 100.0)
    return s


def eval_action_ev(action: BotAction, st: MyGameState, eq: float, opp_fold_prob: float, rng: random.Random) -> float:
    aggressive = is_aggressive_action(action.type)
    invest_call = max(0, min(st.to_call, st.hero_stack))
    invest = invest_call

    if aggressive:
        invest = max(invest_call, min(st.hero_stack, max(st.to_call, action.amount)))

    if action.type == BotActionType.FOLD:
        if eq > 0.60:
            return -0.20 * st.pot
        return 0.0

    if aggressive and eq < 0.45:
        if rng.random() < opp_fold_prob:
            return 0.55 * st.pot

    return eq * (st.pot + invest) - (1.0 - eq) * invest


def run_internal_test_mode(games: int) -> int:
    rng = random.Random(20260416)
    mc = MonteCarloEngine(1337)
    de = DecisionEngine(2026)

    profiles = [
        ("tight", make_profile(0.62, 0.24, 0.14)),
        ("loose", make_profile(0.20, 0.58, 0.22)),
        ("aggressive", make_profile(0.22, 0.26, 0.52)),
    ]

    print(f"[SELFTEST] games={games}")
    for name, opp in profiles:
        wins = 0
        losses = 0
        total_ev = 0.0

        for _ in range(games):
            st = random_state_for_test(rng)
            hint = quick_equity_shortcut(st)
            if hint.use_shortcut:
                eq = hint.equity
            else:
                eq = mc.estimate_equity(st, compute_sim_budget(st, rng), True, 250000)
            if eq < 0.0:
                eq = max(0.10, min(0.90, hint.equity))

            action = de.decide(st, eq, opp)
            ev = eval_action_ev(action, st, eq, opp.fold_rate(), rng)
            total_ev += ev
            if ev >= 0.0:
                wins += 1
            else:
                losses += 1

        print(f"[SELFTEST] vs={name} wins={wins} losses={losses} avg_ev={total_ev / max(1, games)}")

    return 0


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    env = os.getenv("POKER_SELFTEST")
    if env is not None and env != "0":
        return run_internal_test_mode(30)

    for i, arg in enumerate(argv):
        if arg == "--selftest":
            games = 30
            if i + 1 < len(argv):
                try:
                    games = int(argv[i + 1])
                except Exception:
                    games = 30
            games = max(20, min(50, games))
            return run_internal_test_mode(games)

    args = parse_args()
    run_bot(Player(), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
