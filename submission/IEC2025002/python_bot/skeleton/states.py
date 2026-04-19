from collections import namedtuple

# Game constants — must match engine config.py
NUM_ROUNDS      = 1000
STARTING_STACK  = 400
BIG_BLIND       = 2
SMALL_BLIND     = 1
BOUNTY_RATIO    = 1.5
BOUNTY_CONSTANT = 10

GameState     = namedtuple('GameState',     ['bankroll', 'game_clock', 'round_num'])
TerminalState = namedtuple('TerminalState', ['deltas', 'bounty_hits', 'previous_state'])
RoundState    = namedtuple('RoundState',    ['button', 'street', 'pips', 'stacks',
                                             'hands', 'bounties', 'board_cards', 'previous_state'])
