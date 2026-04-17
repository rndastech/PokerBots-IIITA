'''
Upgraded opponent tracker.
Blends raw-count accuracy (first 30 hands) with EMA smoothing
(α=0.08, ~12-hand window) thereafter, for fast opponent adaptation.

Tracks per-street aggression, preflop jam frequency, and bet sizing
patterns to enable fine-grained exploitation.
'''


class OpponentTracker:
    '''Tracks all relevant opponent statistics across hands.'''

    ALPHA = 0.08  # EMA weight — faster adaptation than old 0.04

    def __init__(self):
        self.reset()

    def reset(self):
        # ── hand counters ──────────────────────────────────────────────────
        self.hands_played = 0

        # Preflop raw counts (for accurate early-game stats)
        self.vpip_count          = 0   # voluntarily put $ in
        self.pfr_count           = 0   # preflop raise
        self.preflop_faced_raise = 0
        self.preflop_fold_count  = 0
        self.preflop_jam_count   = 0   # raises >= 100 chips (near all-in)

        # Postflop raw counts
        self.bets_and_raises   = 0
        self.calls             = 0
        self.checks            = 0
        self.folds_postflop    = 0
        self.postflop_actions  = 0

        # C-bet tracking
        self.cbet_opportunities = 0
        self.cbet_made          = 0
        self.faced_cbet         = 0
        self.folded_to_cbet     = 0

        # Showdown
        self.saw_flop          = 0
        self.went_to_showdown  = 0

        # Per-street aggression
        self.flop_bets   = 0; self.flop_checks   = 0
        self.turn_bets   = 0; self.turn_checks   = 0
        self.river_bets  = 0; self.river_checks  = 0

        # Bet sizing: list of (bet_amount, pot_size) tuples
        self.bet_sizes_history = []
        self.raise_sizes_preflop = []

        # ── EMA smoothed stats (initialised at reasonable priors) ──────────
        self._ema_vpip     = 0.50
        self._ema_pfr      = 0.28
        self._ema_cbet     = 0.50
        self._ema_fc_cbet  = 0.50
        self._ema_aggr     = 1.00
        self._ema_wtsd     = 0.35

        # ── per-hand working state ─────────────────────────────────────────
        self._h_vpip   = False
        self._h_pfr    = False
        self._h_agg    = 0
        self._h_pass   = 0
        self._h_cbet_opp = False   # were they PF aggressor?
        self._h_sd     = False

    # ─────────────────────────────────────────────────────────────────────────
    def _ema(self, current, obs):
        return self.ALPHA * obs + (1 - self.ALPHA) * current

    # ─────────────────────────────────────────────────────────────────────────
    def observe_action(self, action_type, is_preflop,
                       opponent_action=True,
                       is_cbet_opportunity=False,
                       facing_cbet=False,
                       amount=0, pot_size=0):
        '''
        Record a single opponent action.

        Args:
            action_type: 'fold' | 'call' | 'check' | 'raise'
            is_preflop: bool
            opponent_action: always True when called for opponent
            is_cbet_opportunity: True if opponent was PF aggressor and this is flop
            facing_cbet: True if opponent is facing our c-bet
            amount: chip amount of the bet/raise
            pot_size: current pot size when action taken
        '''
        if is_preflop:
            if action_type in ('call', 'raise'):
                if not self._h_vpip:
                    self.vpip_count += 1
                self._h_vpip = True
            if action_type == 'raise':
                if not self._h_pfr:
                    self.pfr_count += 1
                self._h_pfr = True
                if amount > 0:
                    self.raise_sizes_preflop.append(amount)
                    if amount >= 100:
                        self.preflop_jam_count += 1
            if action_type == 'fold':
                self.preflop_fold_count  += 1
                self.preflop_faced_raise += 1
            elif action_type == 'call':
                self.preflop_faced_raise += 1
        else:
            # Postflop
            self.postflop_actions += 1
            if action_type == 'raise':
                self.bets_and_raises += 1
                self._h_agg += 1
                if pot_size > 0 and amount > 0:
                    self.bet_sizes_history.append((amount, pot_size))
            elif action_type == 'call':
                self.calls += 1
                self._h_pass += 1
            elif action_type == 'check':
                self.checks += 1
            elif action_type == 'fold':
                self.folds_postflop += 1

        # C-bet tracking
        if is_cbet_opportunity:
            self.cbet_opportunities += 1
            self._h_cbet_opp = True
            if action_type in ('raise',):
                self.cbet_made += 1
                self._ema_cbet = self._ema(self._ema_cbet, 1.0)
            else:
                self._ema_cbet = self._ema(self._ema_cbet, 0.0)

        if facing_cbet:
            self.faced_cbet += 1
            if action_type == 'fold':
                self.folded_to_cbet += 1
                self._ema_fc_cbet = self._ema(self._ema_fc_cbet, 1.0)
            else:
                self._ema_fc_cbet = self._ema(self._ema_fc_cbet, 0.0)

    # ─────────────────────────────────────────────────────────────────────────
    def round_over(self, went_to_showdown):
        '''Called at end of each hand.'''
        self.hands_played += 1

        # Update EMA stats
        self._ema_vpip = self._ema(self._ema_vpip, 1.0 if self._h_vpip else 0.0)
        self._ema_pfr  = self._ema(self._ema_pfr,  1.0 if self._h_pfr  else 0.0)
        self._ema_wtsd = self._ema(self._ema_wtsd, 1.0 if went_to_showdown else 0.0)

        if self._h_pass > 0:
            ratio = self._h_agg / self._h_pass
        elif self._h_agg > 0:
            ratio = 3.0
        else:
            ratio = 0.5
        self._ema_aggr = self._ema(self._ema_aggr, ratio)

        if went_to_showdown:
            self.went_to_showdown += 1

        # Reset per-hand state
        self._h_vpip  = False
        self._h_pfr   = False
        self._h_agg   = 0
        self._h_pass  = 0
        self._h_cbet_opp = False
        self._h_sd    = False

    # ─────────────────────────────────────────────────────────────────────────
    # Derived properties — blend raw counts (accurate early) + EMA (smooth later)
    # ─────────────────────────────────────────────────────────────────────────
    @property
    def vpip(self):
        n = self.hands_played
        if n < 5:
            return 0.50
        raw = self.vpip_count / n
        if n < 30:
            return raw
        return 0.5 * raw + 0.5 * self._ema_vpip

    @property
    def pfr(self):
        n = self.hands_played
        if n < 5:
            return 0.25
        raw = self.pfr_count / n
        if n < 30:
            return raw
        return 0.5 * raw + 0.5 * self._ema_pfr

    @property
    def aggression(self):
        if self.calls == 0:
            return 3.0 if self.bets_and_raises > 0 else 0.5
        raw = self.bets_and_raises / self.calls
        if self.hands_played < 30:
            return raw
        return 0.5 * raw + 0.5 * self._ema_aggr

    @property
    def fold_to_cbet(self):
        if self.faced_cbet < 3:
            return 0.50
        return self.folded_to_cbet / self.faced_cbet

    @property
    def cbet_freq(self):
        if self.cbet_opportunities < 3:
            return 0.50
        return self._ema_cbet

    @property
    def wtsd(self):
        if self.hands_played < 5:
            return 0.35
        raw = self.went_to_showdown / self.hands_played
        if self.hands_played < 30:
            return raw
        return 0.5 * raw + 0.5 * self._ema_wtsd

    @property
    def fold_to_raise_pct(self):
        if self.preflop_faced_raise < 3:
            return 0.50
        return self.preflop_fold_count / self.preflop_faced_raise

    @property
    def postflop_fold_pct(self):
        if self.postflop_actions < 3:
            return 0.35
        return self.folds_postflop / max(1, self.postflop_actions)

    @property
    def avg_bet_size_ratio(self):
        '''Average bet size as fraction of pot.'''
        if not self.bet_sizes_history:
            return 0.60
        return sum(b / max(p, 1) for b, p in self.bet_sizes_history) / len(self.bet_sizes_history)

    @property
    def is_preflop_jammer(self):
        '''True if opponent has jammed preflop >= 2 times.'''
        return self.preflop_jam_count >= 2

    # ─────────────────────────────────────────────────────────────────────────
    def get_stats(self):
        return {
            'vpip':              self.vpip,
            'pfr':               self.pfr,
            'aggression':        self.aggression,
            'fold_to_cbet':      self.fold_to_cbet,
            'cbet_freq':         self.cbet_freq,
            'wtsd':              self.wtsd,
            'fold_to_raise_pct': self.fold_to_raise_pct,
            'postflop_fold_pct': self.postflop_fold_pct,
            'avg_bet_size_ratio':self.avg_bet_size_ratio,
            'is_preflop_jammer': self.is_preflop_jammer,
            'hands_played':      self.hands_played,
        }
