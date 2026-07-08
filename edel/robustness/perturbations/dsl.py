"""DSL injection perturbations (Within-field and Out-of-field)."""

import random
from edel.robustness.base import RobustnessTest

class WithinFieldDSLInjection(RobustnessTest):
    """Inject N domain-specific terms (within-field) into random positions in the text."""
    
    name = "within_field_dsl_injection"
    label = "Within-field DSL Injection"
    priority = "S"
    requires_reembed = True
    
    def perturb(self, texts: list[str], n: int) -> list[str]:
        if n <= 0:
            return texts
            
        # Get lexicon from runner attribute, or use fallback Isabelle terms
        lexicon = getattr(self, "within_lexicon", None)
        if not lexicon:
            lexicon = ["theorem", "proof", "lemma", "induction", "simplifier", "tactic", "formalization", "isabelle", "hol", "correctness"]
            
        perturbed_texts = []
        for text in texts:
            if not text or not text.strip():
                perturbed_texts.append(text)
                continue
                
            words = text.split()
            rng = random.Random(hash(text) + n)
            
            # Inject N terms
            for _ in range(n):
                insert_idx = rng.randint(0, len(words))
                term = rng.choice(lexicon)
                words.insert(insert_idx, term)
                
            perturbed_texts.append(" ".join(words))
            
        return perturbed_texts


class OutOfFieldDSLInjection(RobustnessTest):
    """Inject N out-of-field terms into random positions in the text."""
    
    name = "out_of_field_dsl_injection"
    label = "Out-of-field DSL Injection"
    priority = "S"
    requires_reembed = True
    
    def perturb(self, texts: list[str], n: int) -> list[str]:
        if n <= 0:
            return texts
            
        # Get lexicon from runner attribute, or use fallback biology/finance terms
        lexicon = getattr(self, "out_lexicon", None)
        if not lexicon:
            lexicon = ["mitochondria", "photosynthesis", "derivative", "portfolio", "arbitrage", "cardiovascular", "pathogen", "genome", "liquidity", "antibiotic"]
            
        perturbed_texts = []
        for text in texts:
            if not text or not text.strip():
                perturbed_texts.append(text)
                continue
                
            words = text.split()
            rng = random.Random(hash(text) + n)
            
            # Inject N terms
            for _ in range(n):
                insert_idx = rng.randint(0, len(words))
                term = rng.choice(lexicon)
                words.insert(insert_idx, term)
                
            perturbed_texts.append(" ".join(words))
            
        return perturbed_texts
