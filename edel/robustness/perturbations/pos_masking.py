"""POS masking perturbations."""

import random
from edel.robustness.base import RobustnessTest
from edel.robustness.nlp import tokenize_and_tag

class NounMasking(RobustnessTest):
    """Replace nouns with a blank token, holding length constant."""
    
    name = "noun_masking"
    label = "Noun Masking"
    priority = "M"
    requires_reembed = True
    mask_token = "[MASK]"
    
    def perturb(self, texts: list[str], n: int) -> list[str]:
        if n <= 0:
            return texts
            
        perturbed_texts = []
        for text in texts:
            if not text or not text.strip():
                perturbed_texts.append(text)
                continue
                
            try:
                tagged = tokenize_and_tag(text)
                # Find noun indices
                noun_indices = [i for i, (word, tag) in enumerate(tagged) if tag.startswith('N')]
                
                if not noun_indices:
                    perturbed_texts.append(text)
                    continue
                    
                rng = random.Random(hash(text) + n)
                # Select up to n nouns to mask
                num_to_mask = min(n, len(noun_indices))
                to_mask = set(rng.sample(noun_indices, num_to_mask))
                
                new_words = []
                for i, (word, tag) in enumerate(tagged):
                    if i in to_mask:
                        new_words.append(self.mask_token)
                    else:
                        new_words.append(word)
                        
                # Simple detokenization (just join by space)
                perturbed_texts.append(" ".join(new_words))
            except Exception as e:
                print(f"Error masking nouns: {e}")
                perturbed_texts.append(text)
                
        return perturbed_texts


class VerbMasking(RobustnessTest):
    """Replace verbs with a blank token, holding length constant."""
    
    name = "verb_masking"
    label = "Verb Masking"
    priority = "S"
    requires_reembed = True
    mask_token = "[MASK]"
    
    def perturb(self, texts: list[str], n: int) -> list[str]:
        if n <= 0:
            return texts
            
        perturbed_texts = []
        for text in texts:
            if not text or not text.strip():
                perturbed_texts.append(text)
                continue
                
            try:
                tagged = tokenize_and_tag(text)
                # Find verb indices
                verb_indices = [i for i, (word, tag) in enumerate(tagged) if tag.startswith('V')]
                
                if not verb_indices:
                    perturbed_texts.append(text)
                    continue
                    
                rng = random.Random(hash(text) + n)
                # Select up to n verbs to mask
                num_to_mask = min(n, len(verb_indices))
                to_mask = set(rng.sample(verb_indices, num_to_mask))
                
                new_words = []
                for i, (word, tag) in enumerate(tagged):
                    if i in to_mask:
                        new_words.append(self.mask_token)
                    else:
                        new_words.append(word)
                        
                # Simple detokenization
                perturbed_texts.append(" ".join(new_words))
            except Exception as e:
                print(f"Error masking verbs: {e}")
                perturbed_texts.append(text)
                
        return perturbed_texts


class AdjectiveMasking(RobustnessTest):
    """Replace adjectives with a blank token, holding length constant."""
    
    name = "adjective_masking"
    label = "Adjective Masking"
    priority = "S"
    requires_reembed = True
    mask_token = "[MASK]"
    
    def perturb(self, texts: list[str], n: int) -> list[str]:
        if n <= 0:
            return texts
            
        perturbed_texts = []
        for text in texts:
            if not text or not text.strip():
                perturbed_texts.append(text)
                continue
                
            try:
                tagged = tokenize_and_tag(text)
                # Find adjective indices
                adj_indices = [i for i, (word, tag) in enumerate(tagged) if tag.startswith('J')]
                
                if not adj_indices:
                    perturbed_texts.append(text)
                    continue
                    
                rng = random.Random(hash(text) + n)
                # Select up to n adjectives to mask
                num_to_mask = min(n, len(adj_indices))
                to_mask = set(rng.sample(adj_indices, num_to_mask))
                
                new_words = []
                for i, (word, tag) in enumerate(tagged):
                    if i in to_mask:
                        new_words.append(self.mask_token)
                    else:
                        new_words.append(word)
                        
                # Simple detokenization
                perturbed_texts.append(" ".join(new_words))
            except Exception as e:
                print(f"Error masking adjectives: {e}")
                perturbed_texts.append(text)
                
        return perturbed_texts
