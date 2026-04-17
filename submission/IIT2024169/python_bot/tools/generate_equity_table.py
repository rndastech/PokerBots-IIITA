import json
import os
import random
import eval7

# Rank ordering and suits
RANK_ORDER = '23456789TJQKA'
SUITS = 'shdc'
ALL_CARDS_STR = [r + s for r in RANK_ORDER for s in SUITS]
ALL_CARDS_OBJ = [eval7.Card(c) for c in ALL_CARDS_STR]
CARD_STR_TO_OBJ = {str(c): c for c in ALL_CARDS_OBJ}

def canonical_hands():
    '''Returns a dictionary of Canonical Name -> Sample hole cards (strings)'''
    hands = {}
    for i in range(len(RANK_ORDER)):
        r1 = RANK_ORDER[i]
        for j in range(i, len(RANK_ORDER)):
            r2 = RANK_ORDER[j]
            high, low = r2, r1 # Since j >= i, r2 is higher or equal rank
            
            if high == low:
                # Pair
                name = high + low
                hands[name] = [high + 's', low + 'h']
            else:
                # Suited
                name_s = high + low + 's'
                hands[name_s] = [high + 's', low + 's']
                # Offsuit
                name_o = high + low + 'o'
                hands[name_o] = [high + 's', low + 'h']
    return hands

def mc_equity(hero_hc_strs, sims=10000):
    hero_hc = [CARD_STR_TO_OBJ[c] for c in hero_hc_strs]
    
    # Deck is everything but hero's cards
    used = set(hero_hc_strs)
    deck = [c for c in ALL_CARDS_OBJ if str(c) not in used]
    
    wins = 0
    ties = 0
    
    for _ in range(sims):
        sampled = random.sample(deck, 7)
        villain_hc = sampled[:2]
        board = sampled[2:]
        
        hero_score = eval7.evaluate(hero_hc + board)
        villain_score = eval7.evaluate(villain_hc + board)
        
        if hero_score > villain_score:
            wins += 1
        elif hero_score == villain_score:
            ties += 1
            
    return (wins + 0.5 * ties) / sims

def generate():
    hands = canonical_hands()
    results = {}
    total = len(hands)
    print(f"Generating table for {total} starting hands...")
    
    # We use a modest number of sims (10,000) so it's relatively quick.
    # 100k would be better but takes 10x longer. 10k gives ~1% accuracy.
    # For a real competition submission, you can run this with 100k-500k sims.
    
    i = 0
    for name, sample_cards in hands.items():
        i += 1
        eq = mc_equity(sample_cards, sims=10000)
        results[name] = round(eq, 4)
        if i % 10 == 0:
            print(f"Progress: {i}/{total} ({name} equity: {results[name]})")
            
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'equity', 'preflop_equity.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Done! Saved to {os.path.abspath(out_path)}")

if __name__ == '__main__':
    generate()
