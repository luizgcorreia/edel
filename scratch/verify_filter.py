import sys
from pathlib import Path
import math

def get_vowel_ratio(token: str) -> float:
    token = token.lower()
    # Including 'y' as it is often a vowel in English/technical terms
    vowels = sum(1 for c in token if c in "aeiouy")
    return vowels / len(token) if token else 0.0

STRICT_STOPWORDS = {
    "get", "set", "add", "is", "of", "to", "in", "on", "for", "as", "by", "new",
    "init", "start", "top", "bot", "empty", "proj", "update", "clear", "local",
    "less", "makes", "other", "eq", "do", "val", "fun", "let", "where"
}

def is_meaningful_semantic_token(nt, meaningful_acronyms=None):
    # Layer 1: Minimum total length
    if len(nt) < 4:
        return False
        
    # Layer 2: Stopwords
    if nt in STRICT_STOPWORDS:
        return False
        
    words = nt.split()
    
    # Layer 3: Multi-word tokens are usually phrases (like "hash tree")
    if len(words) > 1:
        # Check if it's not just a soup of single letters like "f g r"
        if all(len(w) < 3 for w in words):
            return False
        return True
        
    # Layer 4: Single word tokens must be "vowel-rich" or "frequent"
    w = words[0]
    
    # Threshold 0.3 catches words with 1 vowel in 4 chars (0.25) like "math", "extg", "hash"
    # These will require promotion via frequency
    if get_vowel_ratio(w) < 0.3:
        if meaningful_acronyms and w in meaningful_acronyms:
            return True
        return False
        
    # Standard length check for single words
    if len(w) < 4:
        return False
        
    return True

# Test Cases
test_tokens = [
    "hash tree",      # KEEP (multi-word)
    "zero sharing",   # KEEP (multi-word)
    "reconstruct",    # KEEP (ratio 4/11 = 0.36 >= 0.3)
    "f",              # FILTER (length)
    "extg",           # FILTER (ratio 0.25 < 0.3, needs promotion)
    "proj7",          # FILTER (length 5, ratio 1/5 = 0.2 < 0.3, needs promotion)
    "spmf",           # FILTER (ratio 0.0 < 0.3, needs promotion)
    "get party",      # KEEP (multi-word)
    "init",           # FILTER (stopword)
    "aby3",           # FILTER (ratio 1/4 = 0.25 < 0.3, needs promotion)
    "math",           # FILTER (ratio 1/4 = 0.25 < 0.3, needs promotion)
]

print("--- Verification without promotion ---")
for t in test_tokens:
    res = is_meaningful_semantic_token(t)
    print(f"{t:15} -> {'KEEP' if res else 'FILTER'}")

print("\n--- Verification with promotion ---")
# Promote math and spmf
promoted = {"spmf", "math", "aby3"}
for t in test_tokens:
    res = is_meaningful_semantic_token(t, meaningful_acronyms=promoted)
    print(f"{t:15} -> {'KEEP' if res else 'FILTER'}")
