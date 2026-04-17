'''
Opponent classifier — maps tracked stats to named archetypes
and returns a rich strategy adjustment dict.

Archetypes: LAG | TAG | CALLING_STATION | ROCK | UNKNOWN

Exploit logic per archetype is grounded in established HU poker theory:
  LAG  — Loose Aggressive: trap, call wider, don't over-bluff
  TAG  — Tight Aggressive: near-GTO, 3-bet bluff, steal blinds
  CALLING_STATION — Loose Passive: bet for value always, never bluff
  ROCK — Tight Passive: steal constantly, fold to their jams
'''


class OpponentClassifier:

    MIN_HANDS = 15  # need at least this many hands before trusting classification

    def classify(self, stats):
        '''
        Classify opponent into an archetype.

        Returns:
            str: 'LAG' | 'TAG' | 'CALLING_STATION' | 'ROCK' | 'UNKNOWN'
        '''
        if stats['hands_played'] < self.MIN_HANDS:
            return 'UNKNOWN'

        vpip = stats['vpip']
        aggr = stats['aggression']

        if vpip > 0.50:
            return 'LAG' if aggr > 1.5 else 'CALLING_STATION'
        else:
            return 'TAG' if aggr > 1.5 else 'ROCK'

    def get_adjustments(self, archetype, stats=None):
        '''
        Return a strategy adjustment dict for the classified archetype.

        All adjustments are relative deltas from baseline GTO play.

        Keys used by DecisionEngine:
            bluff_freq_mult     — multiplier on bluff frequency   (1.0 = baseline)
            min_value_equity    — delta added to value-bet equity threshold
            call_equity_shift   — delta added to call equity threshold
            bet_size_mult       — multiplier on bet sizing fraction
            preflop_tighten     — delta subtracted from open/defend frequency
            steal_more          — True → widen blind-steal range
            fold_to_raise_adjust — delta on fold vs. their raises
            trap_more           — True → check-call monsters vs. this type
            bluff_less          — True → suppress bluffing
        '''
        stats = stats or {}

        if archetype == 'CALLING_STATION':
            # They call too much. Value bet every decent hand, never bluff.
            # Bet sizing can be larger to extract maximum.
            return {
                'bluff_freq_mult':    0.05,   # almost never bluff
                'min_value_equity':  -0.10,   # value bet with thinner hands
                'call_equity_shift':  0.05,   # respect their rare raises
                'bet_size_mult':      1.20,   # bet bigger for value
                'preflop_tighten':    0.05,
                'steal_more':         False,
                'fold_to_raise_adjust': 0.05,
                'trap_more':          False,
                'bluff_less':         True,
            }

        elif archetype == 'ROCK':
            # Tight Passive (nit). Steal constantly. Fold to their jams.
            # Small-bet probes work well; large bets get folds.
            jam_extra_tighten = 0.20 if (stats.get('is_preflop_jammer') or False) else 0.0
            return {
                'bluff_freq_mult':    2.20,   # bluff very frequently
                'min_value_equity':   0.05,   # only bet strong hands for value
                'call_equity_shift':  0.10 + jam_extra_tighten,  # respect their strong raises
                'bet_size_mult':      0.85,   # small bets — they fold anyway
                'preflop_tighten':   -0.15,   # widen our own open range (steal)
                'steal_more':         True,
                'fold_to_raise_adjust': 0.30 + jam_extra_tighten,
                'trap_more':          False,
                'bluff_less':         False,
            }

        elif archetype == 'LAG':
            # Loose Aggressive. Set traps. Call down wider.
            # They bluff frequently: our medium hands are actually good.
            return {
                'bluff_freq_mult':    0.50,   # bluff less (they call too often)
                'min_value_equity':  -0.05,   # value bet wider
                'call_equity_shift': -0.08,   # call wider vs their aggression
                'bet_size_mult':      1.10,   # slightly larger for value
                'preflop_tighten':    0.10,
                'steal_more':         False,
                'fold_to_raise_adjust':-0.10, # don't over-fold to their raises
                'trap_more':          True,   # check-call with monsters
                'bluff_less':         True,
            }

        elif archetype == 'TAG':
            # Tight Aggressive. Close to GTO. Mix in 3-bet bluffs.
            # Steal blinds more; they're disciplined but foldable.
            return {
                'bluff_freq_mult':    1.20,
                'min_value_equity':   0.00,
                'call_equity_shift':  0.00,
                'bet_size_mult':      1.00,
                'preflop_tighten':    0.00,
                'steal_more':         True,
                'fold_to_raise_adjust': 0.08,
                'trap_more':          False,
                'bluff_less':         False,
            }

        else:  # UNKNOWN
            return {
                'bluff_freq_mult':    1.00,
                'min_value_equity':   0.00,
                'call_equity_shift':  0.00,
                'bet_size_mult':      1.00,
                'preflop_tighten':    0.00,
                'steal_more':         False,
                'fold_to_raise_adjust': 0.00,
                'trap_more':          False,
                'bluff_less':         False,
            }

    def should_bluff(self, stats, street, pot_size, bet_size):
        '''
        Data-driven bluff profitability check.

        A bluff is +EV when the opponent's fold frequency exceeds the
        breakeven threshold: bet_size / (pot_size + bet_size).

        Args:
            stats: From OpponentTracker.get_stats()
            street: 0=preflop, 3=flop, 4=turn, 5=river
            pot_size, bet_size: chip amounts

        Returns:
            bool: True if bluffing is likely profitable.
        '''
        if stats['hands_played'] < 10:
            return False

        if street == 0:
            fold_freq = stats['fold_to_raise_pct']
        else:
            fold_freq = stats['postflop_fold_pct']

        denom = pot_size + bet_size
        if denom <= 0:
            return False
        breakeven = bet_size / denom
        return fold_freq > breakeven + 0.05   # 5% margin of safety

    def bet_size_adjust(self, stats):
        '''
        Return a bet-size multiplier based on opponent's average bet size.

        If opponent overbets (avg > 80% pot) → they're polarized →
        call their raises lighter with strong hands (don't need to adjust ours).
        If opponent underbets (<35% pot) → small sizing → they're weak →
        we can bet larger for value.
        '''
        ratio = stats.get('avg_bet_size_ratio', 0.60)
        if ratio > 0.80:
            return 1.0   # no change — just call lighter with monsters
        elif ratio < 0.35:
            return 1.15  # they underbet; exploit with larger value bets
        return 1.0
