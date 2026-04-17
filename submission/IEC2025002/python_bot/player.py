"""
APEX Bot - IIITA PokerBots 2026
Team: Raiseby2(IEC2025002)
Strategy: Monte Carlo equity + opponent profiling + Nash push/fold
"""
import sys, socket, random, os, time
from collections import defaultdict

try:
    import eval7
except ImportError:
    print("eval7 not installed. Run: pip install eval7", file=sys.stderr)
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS  (mirror engine config.py)
# ═══════════════════════════════════════════════════════════════════════════════
STARTING_STACK  = 400
BIG_BLIND       = 2
SMALL_BLIND     = 1
BOUNTY_RATIO    = 1.5
BOUNTY_CONSTANT = 10

# M1: Import NUM_ROUNDS from skeleton (set by engine) — fall back gracefully
try:
    from skeleton.states import NUM_ROUNDS
except Exception:
    try:
        from states import NUM_ROUNDS
    except Exception:
        NUM_ROUNDS = 1000

# M1: Phase detection — dynamic based on round count
IS_GATEKEEPER_MODE = (NUM_ROUNDS == 300)         # CI qual — beat baseline
IS_FINALS_MODE     = not IS_GATEKEEPER_MODE       # everything else → full exploit

RANK_NAMES = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
RANK_IDX   = {r: i for i, r in enumerate(RANK_NAMES)}

# ═══════════════════════════════════════════════════════════════════════════════
#  M2 — TLE GUARD STATE
# ═══════════════════════════════════════════════════════════════════════════════
_SERVER_LAGGING = False

# ═══════════════════════════════════════════════════════════════════════════════
#  M3 — OPPONENT COMPLEXITY SEED (safe default, no filesystem reads)
# ═══════════════════════════════════════════════════════════════════════════════
_OPP_COMPLEXITY = 'UNKNOWN'   # assume unknown → defaults to TAG profile

# ═══════════════════════════════════════════════════════════════════════════════
#  M4 — SIDE-CHANNEL TIMING STATE
# ═══════════════════════════════════════════════════════════════════════════════
_response_times  = []
_last_send_time  = 0.0
_slow_resp_count = 0
_bluff_boost     = 0.0   # 0.0 – 0.15 additive to fold_freq

def _record_response(recv_time: float):
    global _slow_resp_count, _bluff_boost
    try:
        delta_ms = (recv_time - _last_send_time) * 1000
        if _last_send_time <= 0: return
        _response_times.append(delta_ms)
        if len(_response_times) > 20: _response_times.pop(0)
        _slow_resp_count = min(_slow_resp_count+1, 5) if delta_ms > 40 else max(_slow_resp_count-1, 0)
        _bluff_boost = min(_slow_resp_count * 0.03, 0.15)
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════════
#  L2 — PREFLOP TABLE  (Chen formula, 169 hands, O(1))
# ═══════════════════════════════════════════════════════════════════════════════
def _build_pf():
    pts = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,1.0,1.5,2.0,2.5,3.0,4.0]
    t = {}
    for r1 in range(13):
        for r2 in range(r1+1):
            for s in (True, False):
                g = r1-r2
                sc = pts[r1]*2+5 if r1==r2 else (
                    pts[r1]+pts[r2] + (2 if s else 0)
                    + (1 if g==0 else (-1 if g==2 else (-2 if g==3 else (-(g-2)*1.5 if g>3 else 0))))
                    - (1 if g>=1 and r2<3 else 0)
                )
                t[(r1,r2,s)] = max(sc, 0.0)
    mx = max(t.values())
    return {k: v/mx for k,v in t.items()}

_PF = _build_pf()

def preflop_str(hole):
    if len(hole) < 2: return 0.5
    c1, c2 = hole
    r1, r2  = c1.rank, c2.rank
    if r1 < r2: r1, r2 = r2, r1
    return _PF.get((r1, r2, c1.suit==c2.suit), 0.0)

# ═══════════════════════════════════════════════════════════════════════════════
#  L4 — NASH PUSH/FOLD  (HU exact equilibrium)
# ═══════════════════════════════════════════════════════════════════════════════
_NASH = [
    (15, 0.32, 0.50),
    (12, 0.25, 0.46),
    (10, 0.20, 0.42),
    ( 8, 0.12, 0.38),
    ( 6, 0.04, 0.33),
    ( 4, 0.00, 0.12),   # FIX: push any, call top 88% (was call_t=0.28)
    ( 2, 0.00, 0.00),   # FIX: push AND call 100% — pot committed (was call_t=0.20)
]

def nash_thresholds(eff_bb):
    for cutoff, push_t, call_t in _NASH:
        if eff_bb >= cutoff: return push_t, call_t
    return 0.0, 0.20

# ═══════════════════════════════════════════════════════════════════════════════
#  CARD UTILS
# ═══════════════════════════════════════════════════════════════════════════════
def parse_cards(raw):
    try:
        return [eval7.Card(c.strip()) for c in raw.split(',') if c.strip()]
    except Exception:
        return []

# ═══════════════════════════════════════════════════════════════════════════════
#  L1 — PIMC  (Range-Weighted Monte Carlo, time-guarded)
# ═══════════════════════════════════════════════════════════════════════════════
def mc_equity(hole, board, n=1500, opp_range_min=0.0):
    """
    (win_prob, tie_prob) vs opponent filtered to opp_range_min.
    M2: sets _SERVER_LAGGING if wall-time > 35 ms.
    """
    global _SERVER_LAGGING
    if len(hole) < 2: return 0.5, 0.0
    t0 = time.monotonic()

    if _SERVER_LAGGING:
        n = min(n, 300)
    if opp_range_min > 0.40:
        n = max(n // 2, 400)   # FIX: tight range = low accept rate, keep more samples
    elif opp_range_min > 0.02:
        n = max(n // 3, 250)   # moderate range filter

    seen  = set(str(c) for c in hole+board)
    deck  = [c for c in eval7.Deck().cards if str(c) not in seen]
    need  = 5 - len(board)
    wins  = ties = total = 0
    limit = max(n*14, 2500)
    att   = 0

    while total < n and att < limit:
        # TLE CIRCUIT BREAKER: bail if loop exceeds 35ms AND we have 100+ samples
        # (Need 100+ samples for ±10% accuracy; fewer = random play = disaster)
        if att % 100 == 0 and att > 0 and total >= 100 and (time.monotonic() - t0) * 1000 > 35:
            break

        att += 1
        try:
            s   = random.sample(deck, 2+need)
            opp = s[:2]
            if opp_range_min > 0.02 and preflop_str(opp) < opp_range_min:
                continue
            brd = board+s[2:]
            ms  = eval7.evaluate(brd+hole)
            os_ = eval7.evaluate(brd+opp)
            if   ms > os_: wins += 1
            elif ms == os_: ties += 1
            total += 1
        except Exception:
            pass

    elapsed_ms = (time.monotonic()-t0)*1000
    # M2: If this call was slow, throttle remaining calls THIS hand only.
    # Reset happens in OppModel.new_hand() at the start of each new hand.
    if elapsed_ms > 35 and not _SERVER_LAGGING:
        _SERVER_LAGGING = True

    return (wins/total, ties/total) if total else (0.5, 0.0)

# ═══════════════════════════════════════════════════════════════════════════════
#  L3 — BOARD TEXTURE  (ALL streets, FIXED)
# ═══════════════════════════════════════════════════════════════════════════════
def board_texture(board):
    if not board:
        return dict(wet=0, paired=False, trips=False,
                    monotone=False, two_tone=False, rainbow=True, connected=0)
    ranks = [c.rank for c in board]
    suits = [c.suit for c in board]
    sc    = {}
    for s in suits: sc[s] = sc.get(s,0)+1
    ms = max(sc.values())

    # FIXED: works for 3/4/5-card boards
    mono   = ms >= 3
    t_tone = (ms == 2) or (ms >= 3 and len(board) > 3)
    rain   = ms == 1 and len(board) <= 3

    rc = {}
    for r in ranks: rc[r] = rc.get(r,0)+1
    paired = any(v >= 2 for v in rc.values())
    trips  = any(v >= 3 for v in rc.values())

    uniq = sorted(set(ranks))
    conn = sum(1 for i in range(len(uniq)-1) if uniq[i+1]-uniq[i] <= 2)

    wet = 0
    if mono:        wet += 6
    elif t_tone:    wet += 2
    if len(board)==4 and ms>=3: wet += 2
    wet += conn*2
    if not paired:  wet += 1
    return dict(wet=min(wet,10), paired=paired, trips=trips,
                monotone=mono, two_tone=t_tone, rainbow=rain, connected=conn)

# ═══════════════════════════════════════════════════════════════════════════════
#  L6 — BOUNTY MECHANICS
# ═══════════════════════════════════════════════════════════════════════════════
def b_hit(hole, board, rc):
    if not rc: return False
    ri = RANK_IDX.get(rc, -1)
    return ri >= 0 and any(c.rank == ri for c in hole+board)

def b_prob(hole, board, rc):
    if not rc: return 0.0
    ri = RANK_IDX.get(rc, -1)
    if ri < 0: return 0.0
    seen = hole+board
    if any(c.rank == ri for c in seen): return 1.0
    rem = 5-len(board)
    if rem == 0: return 0.0
    tu = 52-len(seen)
    ru = 4-sum(1 for c in seen if c.rank == ri)
    if ru <= 0: return 0.0
    pm = 1.0
    for i in range(rem):
        pm *= max(0, tu-ru-i) / max(1, tu-i)
    return 1.0-pm

def b_bonus(wp, pot, bh, bp):
    p = 1.0 if bh else (bp if bp > 0.05 else 0.0)
    if p == 0.0: return 0.0
    extra = wp * p * (pot*(BOUNTY_RATIO-1) + BOUNTY_CONSTANT)
    return min(extra / max(pot+BIG_BLIND, 1), 0.25)  # capped lower: RATIO=1.5 gives less bonus

# ═══════════════════════════════════════════════════════════════════════════════
#  L5 — OPPONENT MODEL  (HU-calibrated VPIP/PFR/AF + Markov)
# ═══════════════════════════════════════════════════════════════════════════════
class OppModel:
    """
    HU norms: VPIP≈55%, PFR≈40%
    NIT:     VPIP<38%,  PFR<28%
    TAG:     VPIP 38-72%, PFR 28-62%
    LAG:     VPIP≥70%,  PFR≥48%
    Station: VPIP>72%,  PFR<20%
    Maniac:  PFR>68%  OR  (AF>4.5 AND PFR>58%)
    """
    def __init__(self):
        self.vpip_hands   = 0
        self.vpip_vol     = 0
        self.pfr_total    = 0
        self.bets         = defaultdict(int)
        self.calls        = defaultdict(int)
        self.folds        = defaultdict(int)
        self.obs          = defaultdict(int)
        self.sizes        = defaultdict(list)
        self.fold_to_3b   = 0
        self.faced_3b     = 0
        self.cbet_did     = 0
        self.cbet_opp     = 0
        self.fold_to_cbet = 0
        self.faced_cbet   = 0
        self._markov      = defaultdict(lambda: defaultdict(int))
        self._prev_act    = None
        self._showdown_hands = []
        self._pf_raises   = 0
        self._hands_obs   = 0
        self.is_baseline  = False
        self._pf_aggressor = False
        self._vpip_counted = False  # init for pf_action guard
        # M3: seed initial profile from file-recon complexity
        self._initial_profile = (
            'BASIC'    if _OPP_COMPLEXITY == 'BASIC'
            else 'TAG' if _OPP_COMPLEXITY == 'MEDIUM'
            else 'TAG'   # ADVANCED or UNKNOWN → assume TAG
        )

    def new_hand(self):
        global _SERVER_LAGGING
        _SERVER_LAGGING = False   # M2: recover full MC power each new hand
        # VPIP Ghost-Hand Fix: only count when opponent actually acts
        self._hands_obs   += 1
        self._pf_aggressor = False
        self._prev_act     = None
        self._vpip_counted = False  # reset per-hand VPIP flag

    def pf_action(self, action, was_raise_in=False):
        # FIX: Only count VPIP opportunity when opponent actually acts
        if not self._vpip_counted:
            self.vpip_hands += 1
            self._vpip_counted = True
        if action in ('call','raise'): self.vpip_vol += 1
        if action == 'raise':
            self.pfr_total    += 1
            self._pf_aggressor = True
            self._pf_raises   += 1
        if was_raise_in and action != 'raise':
            self.faced_3b += 1
            if action == 'fold': self.fold_to_3b += 1
        self._check_baseline()

    def postflop_action(self, street, action, amount=0, pot=1,
                        is_first=False, we_raised_pf=False):
        self.obs[street] += 1
        if action == 'raise':
            self.bets[street] += 1
            if pot > 0: self.sizes[street].append(amount/pot)
            if is_first and self._pf_aggressor:
                self.cbet_did += 1; self.cbet_opp += 1
        elif action == 'call':
            self.calls[street] += 1
            if is_first and we_raised_pf: self.faced_cbet += 1
        elif action == 'fold':
            self.folds[street] += 1
            if is_first and we_raised_pf:
                self.faced_cbet += 1; self.fold_to_cbet += 1
        if self._prev_act is not None:
            self._markov[(self._prev_act, street)][action] += 1
        self._prev_act = action

    def record_showdown(self, opp_hole, opp_won):
        if len(opp_hole) == 2:
            self._showdown_hands.append((preflop_str(opp_hole), opp_won))

    def _check_baseline(self):
        if self._hands_obs >= 15 and not self.is_baseline:
            if self._pf_raises == 0:
                self.is_baseline = True
            elif self._hands_obs >= 25:
                all_s = [s for v in self.sizes.values() for s in v]
                if len(all_s) >= 10:
                    var = sum((s-0.5)**2 for s in all_s) / len(all_s)
                    if var < 0.002: self.is_baseline = True

    def _pct(self, num, den, default):
        return num/den if den >= 6 else default

    @property
    def vpip(self):      return self._pct(self.vpip_vol,     self.vpip_hands, 0.50)
    @property
    def pfr(self):       return self._pct(self.pfr_total,    self.vpip_hands, 0.28)
    @property
    def fold_3bet(self): return self._pct(self.fold_to_3b,   self.faced_3b,   0.55)
    @property
    def fold_cbet(self): return self._pct(self.fold_to_cbet, self.faced_cbet,  0.40)

    def af(self, street=None):
        if street is not None:
            return self.bets[street] / max(self.calls[street], 1)
        return sum(self.bets.values()) / max(sum(self.calls.values()), 1)

    def fold_freq(self, street):
        o = self.obs[street]
        return self.folds[street]/o if o >= 6 else 0.33

    def range_min_pf(self):
        if self.vpip_hands < 15: return 0.0
        if self.is_nit:     return 0.52   # FIX: was 0.48, slightly tighter for NIT range
        if self.is_tag:     return 0.22
        if self.is_station: return 0.05
        if self.is_maniac:  return 0.00
        if self.is_lag:     return 0.12
        return 0.00

    # HU-calibrated type classification
    @property
    def is_nit(self):     return self.vpip < 0.38 and self.pfr < 0.28
    @property
    def is_tag(self):     return 0.38 <= self.vpip < 0.72 and 0.28 <= self.pfr < 0.62
    @property
    def is_station(self): return self.vpip > 0.72 and self.pfr < 0.20
    @property
    def is_maniac(self):  return self.pfr > 0.68 or (self.af() > 4.5 and self.pfr > 0.58)
    @property
    def is_lag(self):     return self.vpip >= 0.70 and self.pfr >= 0.48

    def profile(self):
        if self.is_maniac:  return 'MANIAC'
        if self.is_nit:     return 'NIT'
        if self.is_station: return 'STATION'
        if self.is_lag:     return 'LAG'
        if self.is_tag:     return 'TAG'
        if self.vpip_hands < 40: return 'TAG'  # M6: default TAG (acts last)
        return 'UNKNOWN'

    def call_threshold(self, street):
        agg = self.af(street)
        if agg > 3.0: return 0.26
        elif agg < 0.5: return 0.43
        return 0.34

# ═══════════════════════════════════════════════════════════════════════════════
#  TRACKER
# ═══════════════════════════════════════════════════════════════════════════════
class Tracker:
    def __init__(self):
        self.bankroll       = 0
        self.round_num      = 0
        self.opp            = OppModel()
        self._cache         = {}
        self._we_pf_raised  = False
        self._opp_pf_raised = False
        self._init()

    def _init(self):
        self.hole       = []
        self.board      = []
        self.my_bounty  = ''
        self.active     = 0
        self.button     = 0
        self.street     = 0
        self.pips       = [SMALL_BLIND, BIG_BLIND]
        self.stacks     = [STARTING_STACK-SMALL_BLIND, STARTING_STACK-BIG_BLIND]
        self._cache     = {}
        self._we_pf_raised  = False
        self._opp_pf_raised = False

    @property
    def pot(self):
        return (STARTING_STACK-self.stacks[0]) + (STARTING_STACK-self.stacks[1])

    @property
    def cc(self):
        return max(0, self.pips[1-self.active] - self.pips[self.active])

    @property
    def in_position(self):
        return (self.active==0) if self.street > 0 else (self.active==1)

    @property
    def eff_bb(self):
        return min(self.stacks[0], self.stacks[1]) / BIG_BLIND

    def can_raise(self):
        """Mirrors engine's legal_actions: raises forbidden when opponent all-in or we must go all-in to call."""
        a = self.active
        cc = self.cc
        if cc > 0:
            # Facing a bet: raises forbidden if calling puts us all-in OR opponent has 0 stack
            return cc != self.stacks[a] and self.stacks[1-a] > 0
        else:
            # Not facing a bet: bets forbidden if either stack is 0
            return self.stacks[0] > 0 and self.stacks[1] > 0

    def raise_bounds(self):
        a  = self.active
        cc = self.cc
        mc = min(self.stacks[a], self.stacks[1-a]+cc)
        mi = min(mc, cc+max(cc, BIG_BLIND))
        return self.pips[a]+mi, self.pips[a]+mc

    def equity(self, n=1500):
        rng = self.opp.range_min_pf()
        key = (tuple(str(c) for c in self.hole),
               tuple(str(c) for c in self.board),
               round(rng, 2))
        if key not in self._cache:
            self._cache[key] = mc_equity(self.hole, self.board, n, rng)
        return self._cache[key]

    def apply_action(self, code, amount=0):
        a = self.button % 2
        if code == 'C':
            if self.button==0 and self.street==0:
                d = BIG_BLIND-self.pips[0]
                self.stacks[0] -= d; self.pips[0] = BIG_BLIND
            else:
                c = self.pips[1-a]-self.pips[a]
                self.stacks[a] -= c; self.pips[a] += c
        elif code == 'R':
            contrib = amount-self.pips[a]
            self.stacks[a] -= contrib; self.pips[a] = amount
        self.button += 1

    def new_street(self, n):
        self.street = {3:3, 4:4, 5:5}.get(n, self.street)
        self.pips = [0,0]; self.button = 1

# ═══════════════════════════════════════════════════════════════════════════════
#  BET SIZING + M8 VARIANCE WARP
# ═══════════════════════════════════════════════════════════════════════════════
def optimal_bet_size(eff_eq, pot, spr, tex, is_river, in_pos):
    if   eff_eq >= 0.85: base = 1.00 if (is_river or spr<1.5) else 0.85
    elif eff_eq >= 0.72: base = 0.80 if is_river else 0.70
    elif eff_eq >= 0.58: base = 0.60
    elif eff_eq >= 0.45: base = 0.50
    else:                base = 0.65
    wet = tex.get('wet', 0)
    if wet >= 7:   base *= 0.80
    elif wet <= 2: base *= 1.10
    if in_pos and eff_eq < 0.80: base *= 0.90
    if tex.get('paired') and eff_eq >= 0.75: base *= 1.10
    return min(base, 1.50)

def _warp(amount, lo, hi):
    """
    M8: ±1-2 chip noise 40% of time. FIXED: clamps result to [lo,hi].
    This prevents illegal raise amounts from warp pushing past hi_r.
    """
    if random.random() < 0.40:
        noise = random.choice([-2, -1, 1, 2])
        return max(lo, min(hi, amount + noise))
    return max(lo, min(hi, amount))

# ═══════════════════════════════════════════════════════════════════════════════
#  CORE DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def decide(t):
    """
    Public entry point. Immortal: never raises, never returns None.
    SANITIZER: 'K' is illegal when facing a bet (cc > 0). Auto-convert to C/F.
    """
    try:
        result = _decide_inner(t)
        if result is None:
            raise ValueError('None result')
        # SANITIZE: K is only legal when cc == 0. If facing a bet, convert.
        if result == 'K' and t.cc > 0:
            result = 'C' if t.cc <= BIG_BLIND * 6 else 'F'  # wider call threshold for 200 BB deep
        return result
    except Exception:
        try:
            return 'K' if t.cc == 0 else ('C' if t.cc <= BIG_BLIND*3 else 'F')
        except Exception:
            return 'F'

def _decide_inner(t):
    a          = t.active
    pot        = t.pot
    cc         = t.cc
    facing_bet = cc > 0

    lo_r, hi_r = t.raise_bounds()
    can_raise   = lo_r <= hi_r and hi_r > 0

    eff_stk = min(t.stacks[a], t.stacks[1-a])
    spr     = eff_stk / max(pot, 1)
    eff_bb  = t.eff_bb

    # ── L1 + M2: range-weighted equity ───────────────────────────────────────
    n_sim = (600  if t.street==0 else
             1200 if t.street==3 else
             1500 if t.street==4 else 2000)
    if _SERVER_LAGGING: n_sim = min(n_sim, 300)

    win, tie = t.equity(n_sim)
    raw_eq   = win + tie*0.5

    # ── L6: Bounty ────────────────────────────────────────────────────────────
    bh     = b_hit(t.hole, t.board, t.my_bounty)
    bp     = b_prob(t.hole, t.board, t.my_bounty)
    bb_val = b_bonus(win, pot, bh, bp)
    eff    = min(raw_eq + bb_val, 1.0)

    # ── Pot odds ──────────────────────────────────────────────────────────────
    po = cc/(pot+cc) if (pot+cc) > 0 else 0.0

    # ── L3: Board texture ─────────────────────────────────────────────────────
    tex      = board_texture(t.board)
    is_river = t.street == 5
    ip       = t.in_position

    # ── L5: Opponent reads ───────────────────────────────────────────────────
    opp        = t.opp
    prof       = opp.profile()
    fold_freq  = opp.fold_freq(t.street)
    is_nit     = opp.is_nit
    is_maniac  = opp.is_maniac
    is_station = opp.is_station
    is_tag     = (prof == 'TAG')
    is_baseline = opp.is_baseline

    # M12/M13: Detect limp-trappers and donk-bettors from aggregate stats
    is_limp_trap = (opp.vpip > 0.50 and opp.pfr < 0.18
                    and opp.af(3) > 1.5 and opp._hands_obs >= 15)
    is_donk_bot  = (opp.bets.get(3, 0) > 4
                    and opp.obs.get(3, 0) >= 8
                    and opp.bets[3] / opp.obs[3] > 0.45
                    and opp.pfr < 0.30)

    # COMPETITION: Unknown opponent guard — be tighter until we profile them
    is_unknown = (opp._hands_obs < 30 and prof in ('TAG', 'UNKNOWN'))

    dumb_mode = False

    # ── Fold equity (M4: boosted by timing) ───────────────────────────────────
    eff_fold_freq = min(fold_freq + _bluff_boost, 0.90)
    # COMPETITION: don't bluff unknowns or stations; require confirmed fold tendency
    bluff_ok      = eff_fold_freq > 0.394 and not is_station and not is_unknown
    is_draw       = 0.29 <= raw_eq <= 0.53 and not is_river

    # ── Raise helpers (FIXED: _warp takes lo,hi for safe clamping) ───────────
    def clamp(v):
        if not can_raise: return None
        v = max(lo_r, min(hi_r, int(v)))
        return v if lo_r <= v <= hi_r else None

    def safe_raise(amount):
        """Warp then clamp — FIXED version, always legal."""
        if not can_raise: return None
        raw    = max(lo_r, min(hi_r, int(amount)))
        warped = _warp(raw, lo_r, hi_r)
        return warped if lo_r <= warped <= hi_r else None

    def bet(frac=0.70):
        raw = max(int(pot*frac), lo_r)
        v   = safe_raise(raw)
        if v is None: v = clamp(raw)   # fallback unwarped
        return f'R{v}' if v else None

    def boc(frac=0.70):  return bet(frac) or 'K'
    def bocl(frac=0.70): return bet(frac) or 'C'

    def sized_bet():
        frac = optimal_bet_size(eff, pot, spr, tex, is_river, ip)
        return boc(frac)

    def preflop_raise(chips):
        """Safe preflop raise with warp applied after clamp. FIXED."""
        v = safe_raise(chips)
        return f'R{v}' if v else None

    rng = random.random()

    # ══════════════════════════════════════════════════════════════════════════
    #  L4 — NASH PUSH/FOLD  (shallow stacks, mathematically unexploitable)
    # ══════════════════════════════════════════════════════════════════════════
    if t.street == 0 and eff_bb <= 15:
        pf = preflop_str(t.hole)
        if bh:        pf = min(pf+0.16, 0.98)
        elif bp>0.40: pf = min(pf+0.08, 0.98)

        push_t, call_t = nash_thresholds(eff_bb)

        if t.button == 0:          # SB: push or fold
            if pf >= push_t:
                v = clamp(hi_r)    # shove
                return f'R{v}' if v else 'C'
            return 'F'             # fold

        if facing_bet and cc >= eff_stk * 0.65:   # BB facing near-shove
            return 'C' if pf >= call_t else 'F'

    # ══════════════════════════════════════════════════════════════════════════
    #  PRE-FLOP  (deep stacks)
    # ══════════════════════════════════════════════════════════════════════════
    if t.street == 0:
        pf = preflop_str(t.hole)
        if bh:        pf = min(pf+0.16, 0.98)
        elif bp>0.40: pf = min(pf+0.08, 0.98)

        # ── SB open ───────────────────────────────────────────────────────────
        if t.button == 0:

            # M1: Gatekeeper — steal top 75% of hands, never bluff
            if IS_GATEKEEPER_MODE:
                if pf >= 0.25:
                    r = preflop_raise(BIG_BLIND*3)
                    return r or 'C'
                return 'F'

            # L5: NIT steal comes BEFORE dumb mode (don't waste EV vs free money)
            if is_nit:
                r = preflop_raise(BIG_BLIND*3)
                return r or 'C'

            # L7: Baseline exploit comes before dumb mode too
            if is_baseline:
                r = preflop_raise(BIG_BLIND*3)
                return r or 'C'

            # M5: Dumb Mode — appear as maniac to poison enemy tracker
            if dumb_mode:
                if pf >= 0.35 or rng < 0.30:
                    # FIXED: stronger hands bet bigger (was accidentally backwards)
                    chips = BIG_BLIND*4 if pf < 0.55 else BIG_BLIND*3
                    r = preflop_raise(chips)
                    return r or 'C'
                return 'F'

            # L5: MANIAC — limp-trap medium, open strong
            if is_maniac:
                if pf >= 0.65:
                    r = preflop_raise(BIG_BLIND*3); return r or 'C'
                elif pf >= 0.45:
                    return 'C'   # limp-trap: let maniac bomb in, re-raise

            # GTO opens
            if   pf >= 0.80:
                r = preflop_raise(BIG_BLIND*4); return r or 'C'
            elif pf >= 0.65:
                r = preflop_raise(BIG_BLIND*3); return r or 'C'
            elif pf >= 0.48:
                r = preflop_raise(max(lo_r, int(BIG_BLIND*2.5))); return r or 'C'
            elif pf >= 0.32 or bh:
                r = preflop_raise(lo_r); return r or 'C'
            else:
                return 'F'

        # ── BB free look ──────────────────────────────────────────────────────
        if not facing_bet:
            return 'K'

        # ── BB facing a raise ─────────────────────────────────────────────────
        raise_frac = cc / max(pot, 1)

        # PATCH: HYPER-AGGRO COUNTER — if opponent 3-bets/raises > 60% of hands,
        # they're bluffing with huge range. Counter: call VERY wide, trap with monsters.
        opp_3bet_freq = opp.pfr  # PFR is a proxy for 3-bet aggression
        is_hyper_aggro = (opp_3bet_freq > 0.60 or is_maniac) and opp._hands_obs >= 12

        if is_hyper_aggro:
            if pf >= 0.80:
                # TRAP: flat-call premiums to let them c-bet bluff postflop
                if rng < 0.55:
                    return 'C'
                # Sometimes 4-bet for balance
                target = min(hi_r, max(lo_r, t.pips[a]+cc*3+pot))
                v = clamp(target); return f'R{v}' if v else 'C'
            elif pf >= 0.55:
                # 4-bet occasionally for balance, mostly flat
                if rng < 0.20 and can_raise:
                    target = min(hi_r, max(lo_r, t.pips[a]+cc*3))
                    v = clamp(target); return f'R{v}' if v else 'C'
                return 'C'
            elif pf >= 0.30:
                # WIDE CALL — they're bluffing most of the time
                if cc <= BIG_BLIND * 8:
                    return 'C'
                elif pf >= 0.40:
                    return 'C'
                else:
                    return 'F'
            else:
                return 'F' if cc > BIG_BLIND * 4 else 'C'

        # BB facing a raise — dumb_mode removed, play real strategy

        # M7: 3-BET SQUEEZE — unconditional with pf>=0.55
        if pf >= 0.55:
            squeeze = int(cc*2.5 + BIG_BLIND)  # bigger 3-bet at 200 BB deep
            v = clamp(squeeze)
            if v:
                r = preflop_raise(squeeze); return r or 'C'
            return 'C' if eff > po+0.04 else 'F'

        # M10: 3-BET LIGHT DEFENSE — detect aggressive 3-bettors
        # If opponent 3-bets >55% of the time, tighten calling range
        if opp_3bet_freq > 0.55 and raise_frac > 0.40:
            # They 3-bet light: 4-bet premium, fold marginals faster
            if pf >= 0.85:
                target = min(hi_r, max(lo_r, t.pips[a]+cc*3+pot))
                v = clamp(target); return f'R{v}' if v else 'C'
            elif pf >= 0.70: return 'C'
            elif pf >= 0.55 and eff > po+0.12: return 'C'
            else: return 'F'

        # GTO 3-bet/call ranges
        if pf >= 0.85:
            target = min(hi_r, max(lo_r, t.pips[a]+cc*3+pot))
            v = clamp(target); return f'R{v}' if v else 'C'
        elif pf >= 0.72:
            if raise_frac < 0.55 and opp.fold_3bet > 0.42:
                target = min(hi_r, max(lo_r, t.pips[a]+cc*3))
                v = clamp(target); return f'R{v}' if v else 'C'
            return 'C'
        elif pf >= 0.55 and eff > po+0.07: return 'C'  # wider defense at 200 BB (implied odds)
        elif pf >= 0.40 and eff > po+0.12 and cc <= BIG_BLIND*6: return 'C'  # wider at depth
        else: return 'F'

    # ══════════════════════════════════════════════════════════════════════════
    #  POST-FLOP
    # ══════════════════════════════════════════════════════════════════════════

    # Baseline simplified postflop — pure value, no bluffs
    if is_baseline:
        if not facing_bet:
            if eff >= 0.52: return sized_bet()
            return 'K'
        else:
            if eff >= 0.52: return 'C'
            elif eff > po+0.06: return 'C'
            elif eff > po+0.03 and bh: return 'C'
            else: return 'F'

    # L7 PATCH: MDF guard — fires BEFORE equity checks
    if facing_bet and cc < pot*0.20 and eff > 0.27:
        mdf = 1.0 - cc/(pot+cc)
        if rng < mdf:
            if eff >= 0.65 and can_raise: return bocl(1.5)
            return 'C'

    # ── Not facing a bet ──────────────────────────────────────────────────────
    if not facing_bet:

        # L5: Baseline exploit
        if is_baseline and eff >= 0.52:
            return sized_bet()

        # L5: MANIAC/STATION thin value — bet any edge since they call everything
        if (is_maniac or is_station) and eff >= 0.36 and not is_river:
            return boc(0.60)
        if (is_maniac or is_station) and eff >= 0.48 and is_river:
            return boc(0.70)

        # M12: Limp-trap — they limp, then trap postflop
        # When they limp+call preflop and board is dry, check behind unless strong
        if is_limp_trap and not facing_bet and not t._we_pf_raised:
            wet = tex.get('wet', 0)
            if wet <= 3:  # dry board — they may be slow-playing
                if eff >= 0.75: return sized_bet()
                return 'K'   # check back, avoid walking into their trap

        # NIT POSTFLOP FIX: NIT called our raise → their range is top 30% (strong).
        # Standard eff=0.58 betting threshold is too loose vs their tight calling range.
        # After NIT calls our PF open, require eff>=0.70 to bet (not the usual 0.58).
        # This prevents over-continuation-betting into their strong calling range.
        if is_nit and t._we_pf_raised and not is_river:
            if eff >= 0.75: return sized_bet()   # clear value only
            elif eff >= 0.60 and is_draw and bluff_ok: return boc(0.50)  # semi-bluff draws
            else: return 'K'   # everything else: pot control, let them bluff
        if is_nit and t._we_pf_raised and is_river:
            if eff >= 0.80: return boc(0.70)   # only bet strong value on river
            return 'K'   # no river bluffs vs NIT (they call with real hands)

        # M6: TAG check-raise setup (check here, raise when they bet turn)
        # Logic is applied in the "facing a bet" section below

        # M9: BINARY GAME-TREE STARVATION  FIXED: removed the EV-losing 1.5x overbet
        # Marginal equity (0.40-0.55) on non-river: min-bet only (free card + fold pressure)
        # EV(min-bet) ≈ fold_freq*pot > 0 when fold_freq > 0 → always at least marginally +EV
        if 0.40 <= raw_eq <= 0.55 and not is_river and bluff_ok:
            if rng < 0.45:   # 45% of the time min-bet for free card / fold pressure
                v = clamp(lo_r)
                return f'R{v}' if v else 'K'
            # 55%: check (pot control, let them bluff into us)

        # Standard value / semi-bluff ladder
        if eff >= 0.83:
            return sized_bet()
        elif eff >= 0.70:
            if ip and tex.get('wet',0) <= 2 and eff >= 0.82 and rng < 0.15:
                return 'K'   # slow-play IP dry board
            return sized_bet()
        elif eff >= 0.58:
            if ip and not is_river and rng < 0.20:
                return 'K'   # pot control
            return boc(0.55)
        elif eff >= 0.46 and bh:
            return boc(0.75)
        # COMPETITION: River probe bet — with mixed frequency
        elif is_river and ip and 0.40 <= eff <= 0.57 and not is_station:
            if rng < 0.35:   # probe 35% of time with mediocre river hands
                return boc(0.40 + random.random() * 0.20)  # varied sizing
            return 'K'
        elif is_draw and bluff_ok:
            frac = 0.62 if tex.get('wet',0) >= 5 else 0.55
            return boc(frac)
        elif not is_river and bluff_ok and is_nit:
            if rng < 0.40:
                return boc(0.55)
            return 'K'
        elif is_river and raw_eq < 0.37 and bluff_ok and is_nit and spr > 0.4:
            if not t._opp_pf_raised and rng < 0.18:
                return boc(0.65)
            return 'K'
        else:
            return 'K'

    # ── Facing a bet ──────────────────────────────────────────────────────────
    else:
        # M13: Donk-bet counter — opponent leads into us as PF aggressor
        if (is_donk_bot and facing_bet and t._we_pf_raised
                and not opp._pf_aggressor and t.street == 3):
            if eff >= 0.78: return bocl(1.5)   # raise donk with strong hands
            elif eff >= 0.58 or (is_draw and eff > po): return 'C'
            return 'F'   # fold marginal — donk-bettor usually has something

        # M10: DONK-BET AWARENESS — opponent leads into us (PF aggressor)
        # Donk bets are typically weak/medium hands probing. Raise to exploit.
        is_donk = facing_bet and t._we_pf_raised and not t._opp_pf_raised
        if is_donk and t.street >= 3:
            bet_frac = cc / max(pot, 1)
            if bet_frac < 0.45:  # small donk = weak probe → raise for value
                if can_raise and eff >= 0.55: return bocl(1.30)  # raise to exploit
                if eff >= 0.45: return 'C'  # call with decent hands
            elif bet_frac < 0.80:  # medium donk — respect slightly more
                if can_raise and eff >= 0.68: return bocl(1.20)
                if eff >= 0.52: return 'C'

        # M6: TAG/UNKNOWN check-raise OOP on dry turn
        if (is_tag or is_unknown) and not ip and t.street==4 and facing_bet:
            wet = tex.get('wet', 0)
            if wet <= 3 and eff >= 0.38 and can_raise and rng < 0.35:
                return bocl(1.20)

        # COMPETITION: Flop check-raise OOP with strong hands (any opponent)
        if not ip and t.street==3 and facing_bet and eff >= 0.78:
            if can_raise and rng < 0.55:  # check-raise 55% of nutted hands OOP
                return bocl(1.40)  # pot-sized raise

        # COMPETITION: River check-raise with monsters
        if is_river and facing_bet and eff >= 0.90 and can_raise:
            return bocl(2.0)  # overbet raise with near-nuts

        # M9: Min-raise counter-pressure on marginal equity
        if 0.40 <= raw_eq <= 0.55 and not is_river:
            if rng < 0.35 and can_raise:
                v = clamp(lo_r)
                return f'R{v}' if v else 'C'
            if eff > po - 0.05:
                return 'C'

        # M11: NIT facing bet — they almost never bluff, so fold more often
        if is_nit and facing_bet and t.street >= 4:
            # NIT betting turn/river = real hand. Tighten call threshold.
            if eff < 0.65: return 'F'

        if eff >= 0.83:
            frac = 2.2 if spr > 1.5 else 1.5
            return bocl(frac)
        elif eff >= 0.72:
            if can_raise and eff >= 0.77 and spr > 1.0: return bocl(1.80)
            return 'C'
        elif eff >= 0.55:                    return 'C'
        elif eff > po+0.10:                  return 'C'
        elif eff > po+0.05 and bh:           return 'C'
        elif is_maniac and eff > po-0.02:    return 'C'  # call WIDE vs maniac (they bluff tons)
        elif eff > po+0.04 and is_maniac:    return 'C'
        elif is_station and eff > po+0.02:   return 'C'
        elif not is_river and eff > po-0.05 and spr > 3 and is_draw: return 'C'
        elif not is_river:
            mdf = 1.0 - cc/(pot+cc)
            if rng < mdf*0.30 and eff > 0.25: return 'C'  # slightly more MDF defense
            return 'F'
        else:
            return 'F'

# ═══════════════════════════════════════════════════════════════════════════════
#  PROTOCOL PARSER
# ═══════════════════════════════════════════════════════════════════════════════
class Parser:
    def __init__(self, t):
        self.t             = t
        self._pf_saw_raise = False
        self._first_flop   = True

    def parse(self, line):
        t  = self.t
        gd = False

        for p in line.strip().split():
            if not p: continue
            ch = p[0]

            if   ch == 'Q': return 'quit'
            elif ch == 'T': pass
            elif ch == 'P':
                try: t.active = int(p[1:])
                except: pass

            elif ch == 'H':
                saved_active = t.active  # P# already set this — preserve it
                t._init()
                t.active = saved_active  # restore after _init reset
                t.hole = parse_cards(p[1:])
                t.round_num += 1
                t.opp.new_hand()
                self._pf_saw_raise = False
                self._first_flop   = True

            elif ch == 'G': t.my_bounty = p[1:]

            elif ch == 'B':
                t.board = parse_cards(p[1:])
                t.new_street(len(t.board))
                self._first_flop = (t.street == 3)

            elif ch == 'O':
                try:
                    opp_hole = parse_cards(p[1:])
                    t.opp.record_showdown(opp_hole, False)
                except Exception: pass

            elif ch == 'D':
                try: t.bankroll += int(p[1:]); gd = True
                except: pass

            elif ch == 'Y': pass

            elif ch in ('F','C','K','R'):
                try:
                    amount = int(p[1:]) if ch=='R' else 0
                    actor  = t.button % 2
                    is_opp = (actor != t.active)
                    nm     = {'F':'fold','C':'call','K':'check','R':'raise'}[ch]

                    if is_opp:
                        if t.street == 0:
                            t.opp.pf_action(nm, was_raise_in=self._pf_saw_raise)
                            if ch == 'R':
                                self._pf_saw_raise = True
                                t._opp_pf_raised   = True
                        elif t.street == 3:
                            t.opp.postflop_action(3, nm, amount, t.pot,
                                is_first=self._first_flop,
                                we_raised_pf=t._we_pf_raised)
                            self._first_flop = False
                        else:
                            t.opp.postflop_action(t.street, nm, amount, t.pot,
                                                  we_raised_pf=t._we_pf_raised)
                    else:
                        if t.street == 0 and ch == 'R':
                            self._pf_saw_raise = True
                            t._we_pf_raised    = True

                    t.apply_action(ch, amount)
                except Exception: pass

        if gd:     return 'ack'
        if t.hole: return 'action'
        return 'none'

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN  (Immortal wrapper + M4 timing)
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    global _last_send_time

    if len(sys.argv) < 2:
        sys.exit(1)

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', int(sys.argv[1])))
        f = sock.makefile('rw')
    except Exception:
        sys.exit(1)

    t = Tracker()
    p = Parser(t)
    _awaiting_response = False

    while True:
        try:
            line = f.readline()
        except Exception:
            break
        if not line:
            break

        recv_time = time.monotonic()

        # M4: Timing signal
        try:
            if _awaiting_response:
                _record_response(recv_time)
                _awaiting_response = False
        except Exception:
            pass

        # Parse
        try:
            tag = p.parse(line)
        except Exception:
            tag = 'none'

        if   tag == 'quit':
            break
        elif tag == 'ack':
            try:
                f.write('K\n'); f.flush()
            except Exception:
                break
        elif tag == 'action':
            # IMMORTAL WRAPPER — never crashes, always responds
            try:
                action = decide(t)
                if not action: action = 'K'
            except Exception:
                try:
                    action = 'K' if t.cc == 0 else ('C' if t.cc <= BIG_BLIND*2 else 'F')
                except Exception:
                    action = 'F'
            # FINAL SANITIZER: K is illegal when facing a bet
            if action == 'K' and t.cc > 0:
                action = 'C' if t.cc <= BIG_BLIND * 6 else 'F'
            # RAISE VALIDATOR: check legality + clamp to bounds
            if action.startswith('R'):
                try:
                    if not t.can_raise():
                        # Raise is ILLEGAL in this state — fallback
                        action = 'C' if t.cc > 0 else 'K'
                    else:
                        amt = int(float(action[1:]))
                        lo, hi = t.raise_bounds()
                        if lo <= hi and hi > 0:
                            amt = max(lo, min(hi, amt))
                            action = f'R{amt}'
                        else:
                            action = 'C' if t.cc > 0 else 'K'
                except Exception:
                    action = 'C' if t.cc > 0 else 'K'
            try:
                f.write(action + '\n'); f.flush()
                _last_send_time    = time.monotonic()
                _awaiting_response = True
            except Exception:
                break

if __name__ == '__main__':
    main()
