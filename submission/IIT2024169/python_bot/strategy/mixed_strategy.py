'''
Adds controlled randomness to prevent deterministic exploitation.
'''
import random

class MixedStrategy:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def should_bluff(self, required_equity, multiplier=1.0):
        '''
        Probability of bluffing should roughly map to alpha = B / (P + 2B) (GTO concept)
        but we simplify here using the multiplier from opponent classifier.
        Args:
            required_equity: Pot odds (0 to 1)
            multiplier: from opponent classifier
        '''
        # Baseline: If pot odds are 33%, we theoretically want to bluff ~33% of our betting range.
        # Here we just use a flat probability adjusted by odds and opponent type.
        base_bluff_prob = required_equity * 0.5 # rough heuristic
        final_prob = min(0.35, base_bluff_prob * multiplier)
        return random.random() < final_prob

    def should_slowplay(self):
        '''
        Occasionally trap with monster hands (equity > 85%).
        Check/call instead of raising.
        '''
        return random.random() < 0.15 # 15% slowplay

    def add_noise_to_sizing(self, target_size, min_raise, max_raise, noise_pct=0.10):
        '''
        Add random noise to bet size to hide patterns.
        '''
        if target_size <= min_raise:
            return min_raise
        if target_size >= max_raise:
            return max_raise

        noise_amt = target_size * noise_pct
        noisy_size = target_size + random.uniform(-noise_amt, noise_amt)

        return int(max(min_raise, min(max_raise, noisy_size)))
