'''
Time management module.
Ensures we never drain the 60-second shared game clock across 1000 rounds.
Also manages lead-preservation endgame mode.
'''


class TimeManager:
    """
    Manages computation budget across the entire 1000-round match.

    Game clock: 60 seconds shared. Average budget: 60ms/round.
    Any complex board (multi-street, multi-raise) can easily eat 200ms
    without adaptive sampling → clock drain → hard timeout.

    Modes:
        DEEP:      > 25s remaining   — 400 MC samples (~12ms)
        NORMAL:    > 10s remaining   — 150 MC samples (~5ms)
        FAST:      > 3s  remaining   — 50  MC samples (~2ms)
        EMERGENCY: <= 3s remaining   — 0   MC samples (table lookup only)
    """

    TOTAL_ROUNDS = 1000

    def get_mode(self, game_clock, round_num):
        """
        Determine computation mode based on remaining clock and rounds left.

        Args:
            game_clock: Remaining seconds on the shared game clock.
            round_num:  Current round number (1-indexed).

        Returns:
            str: One of 'EMERGENCY', 'FAST', 'NORMAL', 'DEEP'
        """
        rounds_remaining = max(1, self.TOTAL_ROUNDS - round_num + 1)
        time_per_round = game_clock / rounds_remaining

        if game_clock < 3.0 or time_per_round < 0.015:
            return 'EMERGENCY'
        elif game_clock < 10.0 or time_per_round < 0.030:
            return 'FAST'
        elif game_clock < 25.0 or time_per_round < 0.060:
            return 'NORMAL'
        else:
            return 'DEEP'

    def get_mc_samples(self, mode):
        """Return number of Monte Carlo samples for the given mode."""
        return {
            'EMERGENCY': 0,
            'FAST':      50,
            'NORMAL':    150,
            'DEEP':      400,
        }.get(mode, 150)

    def should_use_opponent_model(self, mode):
        """Whether to run the (slightly expensive) opponent model queries."""
        return mode in ('NORMAL', 'DEEP')

    def should_preserve_lead(self, current_bankroll, round_num,
                             lead_threshold=400, late_game_start=900):
        """
        Lead Preservation Shield (Pillar 7 — Variance Management).

        Elastic threshold: scales with rounds remaining instead of using
        a single fixed cutoff. This locks in moderate leads earlier and
        doesn't over-protect tiny leads late.

        Formula:
            rounds_left = 1000 - round_num
            threshold = max(100, rounds_left * 2)
            Trigger if round >= 850 AND bankroll >= threshold
            OR if round >= 900 AND bankroll >= 200

        Args:
            current_bankroll: Our running bankroll delta vs. opponent.
            round_num: Current round number.
            lead_threshold: (unused, kept for API compat)
            late_game_start: (unused, kept for API compat)

        Returns:
            bool: True → play check/fold this hand.
        """
        rounds_left = max(1, self.TOTAL_ROUNDS - round_num)

        # Elastic threshold: need less lead when fewer rounds remain
        elastic_threshold = max(100, rounds_left * 2)

        # Primary trigger: moderate+ lead with few rounds left
        if round_num >= 850 and current_bankroll >= elastic_threshold:
            return True

        # Secondary trigger: solid lead in endgame
        if round_num >= 900 and current_bankroll >= 200:
            return True

        # Massive lead: protect earlier
        if round_num >= 800 and current_bankroll >= 600:
            return True

        return False
