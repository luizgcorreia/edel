"""Graded token deletion using concreteness and specificity ratings."""

import random
from edel.robustness.base import RobustnessTest

CONCRETENESS_LEXICON = {
    # Highly concrete (5.0)
    "car": 5.0, "dog": 5.0, "house": 5.0, "computer": 5.0, "paper": 5.0,
    "scientist": 5.0, "formula": 5.0, "graph": 5.0, "numbers": 5.0, "author": 5.0,
    "document": 5.0, "file": 5.0, "database": 5.0, "apple": 5.0, "table": 5.0,
    
    # Neutral (3.0)
    "analyze": 3.0, "testing": 3.0, "process": 3.0, "work": 3.0, "study": 3.0,
    
    # Highly abstract (1.0)
    "hypothesis": 1.0, "theory": 1.0, "concept": 1.0, "truth": 1.0, "idea": 1.0,
    "epistemic": 1.0, "logical": 1.0, "methodology": 1.0, "meaning": 1.0, "relation": 1.0,
    "robustness": 1.0, "philosophy": 1.0, "knowledge": 1.0, "existence": 1.0
}

SPECIFICITY_LEXICON = {
    # Highly specific (5.0)
    "isabelle": 5.0, "hol": 5.0, "algorithm": 5.0, "python": 5.0, "theorem": 5.0,
    "lemma": 5.0, "concreteness": 5.0, "specificity": 5.0, "displacement": 5.0,
    "cosine": 5.0, "l2": 5.0, "perturbation": 5.0, "tactic": 5.0, "simplifier": 5.0,
    
    # Neutral (3.0)
    "approach": 3.0, "method": 3.0, "model": 3.0, "analysis": 3.0, "system": 3.0,
    
    # Highly general (1.0)
    "thing": 1.0, "object": 1.0, "data": 1.0, "result": 1.0, "work": 1.0,
    "study": 1.0, "paper": 1.0, "field": 1.0, "process": 1.0, "part": 1.0
}

def get_word_concreteness(word: str) -> float:
    w = word.lower().strip(".,;:!?\"'()[]{}")
    if w in CONCRETENESS_LEXICON:
        return CONCRETENESS_LEXICON[w]
    # Deterministic hash-based fallback between 1.0 and 5.0
    return (abs(hash(w)) % 41) / 10.0 + 1.0

def get_word_specificity(word: str) -> float:
    w = word.lower().strip(".,;:!?\"'()[]{}")
    if w in SPECIFICITY_LEXICON:
        return SPECIFICITY_LEXICON[w]
    # Deterministic hash-based fallback between 1.0 and 5.0
    return (abs(hash(w)) % 41) / 10.0 + 1.0


class ConcretenessGradedDisplacement(RobustnessTest):
    """Delete N words starting from the most concrete words."""
    
    name = "concreteness_graded_displacement"
    label = "Concreteness-graded Displacement"
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
                
            words = text.split()
            if len(words) <= n:
                perturbed_texts.append("")
                continue
                
            # Score each word index
            scored_indices = []
            for idx, word in enumerate(words):
                score = get_word_concreteness(word)
                scored_indices.append((score, idx))
                
            # Sort by score descending, then by index ascending to be stable
            scored_indices.sort(key=lambda x: (-x[0], x[1]))
            
            # Select indices to delete
            indices_to_delete = {idx for _, idx in scored_indices[:n]}
            
            new_words = [words[i] for i in range(len(words)) if i not in indices_to_delete]
            perturbed_texts.append(" ".join(new_words))
            
        return perturbed_texts


class SpecificityGradedDisplacement(RobustnessTest):
    """Delete N words starting from the most specific words."""
    
    name = "specificity_graded_displacement"
    label = "Specificity-graded Displacement"
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
                
            words = text.split()
            if len(words) <= n:
                perturbed_texts.append("")
                continue
                
            # Score each word index
            scored_indices = []
            for idx, word in enumerate(words):
                score = get_word_specificity(word)
                scored_indices.append((score, idx))
                
            # Sort by score descending, then by index ascending
            scored_indices.sort(key=lambda x: (-x[0], x[1]))
            
            # Select indices to delete
            indices_to_delete = {idx for _, idx in scored_indices[:n]}
            
            new_words = [words[i] for i in range(len(words)) if i not in indices_to_delete]
            perturbed_texts.append(" ".join(new_words))
            
        return perturbed_texts
