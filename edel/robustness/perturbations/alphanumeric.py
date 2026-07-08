"""Numeral-to-word substitution perturbation."""

import re
import random
from edel.robustness.base import RobustnessTest

class NumeralToWordSubstitution(RobustnessTest):
    """Replace numbers (numerals) with their word equivalents.
    
    N is interpreted as the percentage of numerals in the text to replace.
    """
    
    name = "numeral_to_word"
    label = "Numeral-to-word Substitution"
    priority = "S"
    requires_reembed = True
    
    def perturb(self, texts: list[str], n: int) -> list[str]:
        if n <= 0:
            return texts
            
        ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        
        def num_to_words(num_str: str) -> str:
            # Simple digit-by-digit mapping to keep it robust and general
            # E.g. "123" -> "one two three"
            return " ".join(ones[int(d)] for d in num_str if d.isdigit())
            
        perturbed_texts = []
        for text in texts:
            if not text or not text.strip():
                perturbed_texts.append(text)
                continue
                
            # Find all numbers/digit sequences
            matches = list(re.finditer(r'\b\d+\b', text))
            if not matches:
                perturbed_texts.append(text)
                continue
                
            rng = random.Random(hash(text) + n)
            
            # Select which matches to replace based on percentage n
            # We want to replace n% of numbers, at least 1 if n > 0
            num_to_replace = int(len(matches) * (n / 100.0))
            if num_to_replace == 0 and n > 0:
                num_to_replace = 1
            num_to_replace = min(num_to_replace, len(matches))
            
            indices_to_replace = set(rng.sample(range(len(matches)), num_to_replace))
            
            # Build the new string by replacing the selected matches
            new_text_parts = []
            last_idx = 0
            for idx, match in enumerate(matches):
                start, end = match.span()
                new_text_parts.append(text[last_idx:start])
                
                num_str = match.group(0)
                if idx in indices_to_replace:
                    new_text_parts.append(num_to_words(num_str))
                else:
                    new_text_parts.append(num_str)
                    
                last_idx = end
                
            new_text_parts.append(text[last_idx:])
            perturbed_texts.append("".join(new_text_parts))
            
        return perturbed_texts
