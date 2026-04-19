import treys
import random

class Card:
    def __init__(self, s):
        self.s = s
        try:
            self.t_card = treys.Card.new(s)
        except:
            self.t_card = None
    @property
    def rank(self):
        r = self.s[0]
        return ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'].index(r)
    def __str__(self): return self.s
    def __repr__(self): return self.s
    def __eq__(self, other): return self.s == other.s
    def __hash__(self): return hash(self.s)

class Deck:
    def __init__(self):
        self.cards = [Card(r+s) for r in '23456789TJQKA' for s in 'shcd']
    def shuffle(self):
        random.shuffle(self.cards)
    def deal(self, n):
        res = self.cards[:n]
        self.cards = self.cards[n:]
        return res
    def peek(self, n):
        return self.cards[:n]

_treys_evaluator = treys.Evaluator()
def evaluate(cards):
    t_cards = [c.t_card for c in cards]
    return 8000 - _treys_evaluator.evaluate(t_cards[:5], t_cards[5:])

class HandRange:
    def __init__(self, s):
        self.s = s

def py_hand_vs_range_monte_carlo(hand, opp_range, board, iters):
    iters = min(iters, 5000)  # High limit for final stress testing
    all_cards = [Card(r+s) for r in '23456789TJQKA' for s in 'shcd']
    known = set([c.s for c in hand] + [c.s for c in board])
    rem_deck = [c for c in all_cards if c.s not in known]
    win = 0.0
    for _ in range(iters):
        random.shuffle(rem_deck)
        opp_hand = rem_deck[:2]
        rem_b = 5 - len(board)
        sim_board = board + rem_deck[2:2+rem_b]
        s_me = evaluate(sim_board + hand)
        s_opp = evaluate(sim_board + opp_hand)
        if s_me > s_opp:
            win += 1.0
        elif s_me == s_opp:
            win += 0.5
    return win / iters
