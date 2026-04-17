'''
Upgraded bet sizer.
5-tier EHS → sizing fraction, with board texture and opponent type modifiers.
Supports turn/river overbet mode for value hands on later streets.
Geometric sizing for multi-street value extraction.
'''
import math


class BetSizer:
    def __init__(self):
        pass

    def _geometric_fraction(self, pot, my_stack, streets_remaining):
        '''
        Compute geometric bet sizing fraction.

        The idea: choose a single bet fraction f such that if we bet f × pot
        on each remaining street, we naturally get all-in by the river.

        Formula: f = ((1 + 2S/P)^(1/N) - 1) / 2
        Clamped to [0.25, 1.25].
        '''
        if streets_remaining <= 0 or pot <= 0 or my_stack <= 0:
            return 0.55  # fallback
        ratio = my_stack / pot
        if ratio <= 0.0:
            return 1.0   # already pot-committed, just jam
        
        try:
            # Correct formula for geometric pot growth
            f = (math.pow(1 + 2 * ratio, 1.0 / streets_remaining) - 1.0) / 2.0
        except (ValueError, OverflowError):
            f = 0.55
            
        return max(0.25, min(1.25, f))

    def get_bet_size(self, equity, pot, min_raise, max_raise,
                     board_wet, street=3, is_bluff=False,
                     opponent_type='UNKNOWN', my_pip=0):
        '''
        Compute a bet-to raise amount.

        Args:
            equity:        Hand equity / EHS (0–1)
            pot:           Current pot size in chips
            min_raise:     Engine minimum legal raise
            max_raise:     Engine maximum legal raise
            board_wet:     Bool — wet board (flush/straight draws present)
            street:        3=flop, 4=turn, 5=river
            is_bluff:      True → use bluff sizing
            opponent_type: 'CALLING_STATION' | 'ROCK' | 'LAG' | 'TAG' | 'UNKNOWN'
            my_pip:        Our current pip (used to convert pot% → raise-to total)

        Returns:
            int: raise-to amount (clamped to [min_raise, max_raise])
        '''
        if is_bluff:
            # Bluffs: smaller sizing to risk less, still credible
            if opponent_type == 'CALLING_STATION':
                fraction = 0.25
            else:
                fraction = 0.33  # ~1/3 pot bluff sizing
        else:
            # Geometric sizing for strong+ hands on flop/turn
            streets_remaining = max(1, (5 - street))  # flop=2, turn=1, river=0→1
            stack_behind = max_raise - my_pip  # approximate remaining stack

            if equity >= 0.75 and streets_remaining >= 2 and stack_behind > 0:
                # Use geometric sizing for multi-street value
                fraction = self._geometric_fraction(pot, stack_behind, streets_remaining)
            elif equity >= 0.85:
                # Monster — overbet on later streets
                if street >= 4:
                    fraction = 0.90
                else:
                    fraction = 0.72
            elif equity >= 0.72:
                # Strong value
                if street >= 4:
                    fraction = 0.65
                else:
                    fraction = 0.55
            elif equity >= 0.58:
                # Thin value / c-bet
                fraction = 0.42
            elif equity >= 0.45:
                # Probe / merge bet
                fraction = 0.30
            else:
                fraction = 0.30

        # Board texture modifier
        if board_wet:
            fraction += 0.10   # charge draws a premium on wet boards
        else:
            fraction -= 0.05   # smaller sizing on dry boards

        # Opponent type modifier
        if opponent_type == 'CALLING_STATION':
            fraction += 0.18   # bet much larger — they'll call anyway
        elif opponent_type == 'ROCK':
            fraction -= 0.10   # small bets — they fold to large ones
        elif opponent_type == 'LAG':
            fraction += 0.08   # slightly larger for value vs. aggro callers

        # Clamp fraction
        fraction = max(0.25, min(1.50, fraction))

        # Compute chip amount (fraction of pot, then convert to raise-to)
        chip_amount  = int(pot * fraction)
        chip_amount  = max(chip_amount, 2)
        raise_to     = my_pip + chip_amount

        # Clamp to legal bounds
        raise_to = max(min_raise, min(raise_to, max_raise))
        return raise_to

    def get_bluff_size(self, pot, min_raise, max_raise, my_pip=0,
                       opponent_type='UNKNOWN'):
        '''Convenience wrapper for bluff sizing.'''
        return self.get_bet_size(
            equity=0.0, pot=pot, min_raise=min_raise, max_raise=max_raise,
            board_wet=False, street=3, is_bluff=True,
            opponent_type=opponent_type, my_pip=my_pip
        )

