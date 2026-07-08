"""NLP utilities for robustness tests."""

import string

_NLTK_INITIALIZED = False

def init_nltk():
    """Lazy initialize NLTK and download required resources."""
    global _NLTK_INITIALIZED
    if _NLTK_INITIALIZED:
        return
        
    import nltk
    try:
        # Check if already downloaded
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
    _NLTK_INITIALIZED = True

def get_wordnet_pos(treebank_tag: str) -> str | None:
    """Map treebank POS tag to WordNet POS tag."""
    from nltk.corpus import wordnet
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def get_synonym(word: str, pos: str, rng) -> str:
    """Get a random synonym for a word from WordNet."""
    from nltk.corpus import wordnet
    
    synsets = wordnet.synsets(word, pos=pos)
    if not synsets:
        return word
        
    lemmas = []
    for synset in synsets:
        for lemma in synset.lemmas():
            name = lemma.name()
            if name.lower() != word.lower() and '_' not in name:
                lemmas.append(name)
                
    if not lemmas:
        return word
        
    return rng.choice(lemmas)

def tokenize_and_tag(text: str) -> list[tuple[str, str]]:
    """Tokenize and POS tag a text."""
    init_nltk()
    from nltk import word_tokenize, pos_tag
    tokens = word_tokenize(text)
    return pos_tag(tokens)
