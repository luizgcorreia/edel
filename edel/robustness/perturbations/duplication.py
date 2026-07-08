"""POS-based token duplication perturbations."""

import random
from edel.robustness.base import RobustnessTest
from edel.robustness.nlp import tokenize_and_tag

class NounDuplication(RobustnessTest):
    """Repeat nouns in the text N additional times in-place."""
    
    name = "noun_duplication"
    label = "Noun Duplication"
    priority = "C"
    requires_reembed = True
    
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
                new_words = []
                for word, tag in tagged:
                    if tag.startswith('N'):
                        new_words.extend([word] * (n + 1))
                    else:
                        new_words.append(word)
                perturbed_texts.append(" ".join(new_words))
            except Exception as e:
                print(f"Error in NounDuplication: {e}")
                perturbed_texts.append(text)
                
        return perturbed_texts


class AdjectiveDuplication(RobustnessTest):
    """Repeat adjectives in the text N additional times in-place."""
    
    name = "adjective_duplication"
    label = "Adjective Duplication"
    priority = "C"
    requires_reembed = True
    
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
                new_words = []
                for word, tag in tagged:
                    if tag.startswith('J'):
                        new_words.extend([word] * (n + 1))
                    else:
                        new_words.append(word)
                perturbed_texts.append(" ".join(new_words))
            except Exception as e:
                print(f"Error in AdjectiveDuplication: {e}")
                perturbed_texts.append(text)
                
        return perturbed_texts
