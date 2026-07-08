"""Word-order shuffle perturbation."""

import random
from edel.robustness.base import RobustnessTest

class WordOrderShuffle(RobustnessTest):
    """Randomly permutes word order within a text to destroy syntax while preserving tokens."""
    
    name = "word_order_shuffle"
    label = "Word-order shuffle"
    priority = "M"
    requires_reembed = True
    
    def perturb(self, texts: list[str], n: int) -> list[str]:
        """Shuffle words.
        
        Since this test is boolean (shuffled or not), we interpret N as the number
        of times to apply a partial shuffle, or as a percentage of words to swap.
        To keep it simple and consistent with the displacement sweep:
        If n=0, return unchanged.
        If n>0, we will swap n pairs of random words (up to max possible swaps).
        """
        if n <= 0:
            return texts
            
        perturbed_texts = []
        for text in texts:
            if not text or not text.strip():
                perturbed_texts.append(text)
                continue
                
            words = text.split()
            if len(words) < 2:
                perturbed_texts.append(text)
                continue
                
            # Perform up to n random swaps
            # We use a fixed seed per text length and n to make it deterministic 
            # for the same text and n, but different across texts.
            rng = random.Random(hash(text) + n)
            
            # Make a copy to shuffle
            shuffled = words[:]
            
            num_swaps = min(n, len(words) // 2)
            for _ in range(num_swaps):
                idx1 = rng.randint(0, len(words) - 1)
                idx2 = rng.randint(0, len(words) - 1)
                shuffled[idx1], shuffled[idx2] = shuffled[idx2], shuffled[idx1]
                
            perturbed_texts.append(" ".join(shuffled))
            
        return perturbed_texts
