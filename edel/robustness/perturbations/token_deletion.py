"""Token deletion perturbation."""

import random
from edel.robustness.base import RobustnessTest

class RandomTokenDeletion(RobustnessTest):
    """Randomly delete N tokens from the text to test length influence."""
    
    name = "random_token_deletion"
    label = "Random Token Deletion"
    priority = "M"
    requires_reembed = True
    
    def perturb(self, texts: list[str], n: int) -> list[str]:
        if n <= 0:
            return texts
            
        perturbed_texts = []
        for text in texts:
            if not text or not text.strip():
                perturbed_texts.append(text)
                continue
                
            words = text.split()
            if len(words) <= n:
                # If we want to delete more words than exist, we could return empty,
                # but an empty string might crash the embedding model.
                # Let's return just one remaining word or empty string if it's fine.
                perturbed_texts.append("")
                continue
                
            rng = random.Random(hash(text) + n)
            indices_to_delete = set(rng.sample(range(len(words)), n))
            
            kept_words = [w for i, w in enumerate(words) if i not in indices_to_delete]
            perturbed_texts.append(" ".join(kept_words))
            
        return perturbed_texts
