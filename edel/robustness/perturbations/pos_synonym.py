"""POS synonym substitution perturbations."""

import random
from edel.robustness.base import RobustnessTest
from edel.robustness.nlp import tokenize_and_tag, get_wordnet_pos, get_synonym

class NounSynonymSubstitution(RobustnessTest):
    """Replace each noun with a synonym, preserving semantic meaning and token count."""
    
    name = "noun_synonym"
    label = "Noun Synonym Substitution"
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
                
            try:
                tagged = tokenize_and_tag(text)
                # Find noun indices
                noun_indices = [i for i, (word, tag) in enumerate(tagged) if tag.startswith('N')]
                
                if not noun_indices:
                    perturbed_texts.append(text)
                    continue
                    
                rng = random.Random(hash(text) + n)
                # Select up to n nouns to substitute
                num_to_sub = min(n, len(noun_indices))
                to_sub = set(rng.sample(noun_indices, num_to_sub))
                
                new_words = []
                for i, (word, tag) in enumerate(tagged):
                    if i in to_sub:
                        wn_pos = get_wordnet_pos(tag)
                        synonym = get_synonym(word, wn_pos, rng) if wn_pos else word
                        new_words.append(synonym)
                    else:
                        new_words.append(word)
                        
                # Simple detokenization
                perturbed_texts.append(" ".join(new_words))
            except Exception as e:
                print(f"Error substituting synonyms: {e}")
                perturbed_texts.append(text)
                
        return perturbed_texts


class VerbSynonymSubstitution(RobustnessTest):
    """Replace each verb with a synonym, preserving semantic meaning and token count."""
    
    name = "verb_synonym"
    label = "Verb Synonym Substitution"
    priority = "S"
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
                # Find verb indices
                verb_indices = [i for i, (word, tag) in enumerate(tagged) if tag.startswith('V')]
                
                if not verb_indices:
                    perturbed_texts.append(text)
                    continue
                    
                rng = random.Random(hash(text) + n)
                # Select up to n verbs to substitute
                num_to_sub = min(n, len(verb_indices))
                to_sub = set(rng.sample(verb_indices, num_to_sub))
                
                new_words = []
                for i, (word, tag) in enumerate(tagged):
                    if i in to_sub:
                        wn_pos = get_wordnet_pos(tag)
                        synonym = get_synonym(word, wn_pos, rng) if wn_pos else word
                        new_words.append(synonym)
                    else:
                        new_words.append(word)
                        
                # Simple detokenization
                perturbed_texts.append(" ".join(new_words))
            except Exception as e:
                print(f"Error substituting verb synonyms: {e}")
                perturbed_texts.append(text)
                
        return perturbed_texts


class AdjectiveSynonymSubstitution(RobustnessTest):
    """Replace each adjective with a synonym, preserving semantic meaning and token count."""
    
    name = "adjective_synonym"
    label = "Adjective Synonym Substitution"
    priority = "S"
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
                # Find adjective indices
                adj_indices = [i for i, (word, tag) in enumerate(tagged) if tag.startswith('J')]
                
                if not adj_indices:
                    perturbed_texts.append(text)
                    continue
                    
                rng = random.Random(hash(text) + n)
                # Select up to n adjectives to substitute
                num_to_sub = min(n, len(adj_indices))
                to_sub = set(rng.sample(adj_indices, num_to_sub))
                
                new_words = []
                for i, (word, tag) in enumerate(tagged):
                    if i in to_sub:
                        wn_pos = get_wordnet_pos(tag)
                        synonym = get_synonym(word, wn_pos, rng) if wn_pos else word
                        new_words.append(synonym)
                    else:
                        new_words.append(word)
                        
                # Simple detokenization
                perturbed_texts.append(" ".join(new_words))
            except Exception as e:
                print(f"Error substituting adjective synonyms: {e}")
                perturbed_texts.append(text)
                
        return perturbed_texts
