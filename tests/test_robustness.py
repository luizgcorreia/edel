"""Tests for the Null Model Robustness Module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from edel.robustness.base import displacement_cosine
from edel.robustness.perturbations.word_order import WordOrderShuffle
from edel.robustness.registry import ROBUSTNESS_REGISTRY, get_test
from edel.robustness.runner import run_robustness_sweep


@pytest.fixture
def synthetic_texts():
    return [
        "This is a simple test sentence for shuffling.",
        "Another sentence that has some words to move around.",
        "Short one.",
        ""  # Empty string handling
    ]

@pytest.fixture
def synthetic_dataframe():
    """Generate a small synthetic dataset for testing the sweep engine."""
    N = 10
    rng = np.random.default_rng(42)
    dim = 8

    def fake_emb_col():
        import json
        return [json.dumps(rng.standard_normal(dim).tolist()) for _ in range(N)]

    df = pd.DataFrame({
        'problem':        [f"Problem sentence {i} with some words to shuffle" for i in range(N)],
        'method':         [f"Method sentence {i} testing the pipeline" for i in range(N)],
        'finding':        [f"Finding sentence {i} is here" for i in range(N)],
        'interpretation': [f"Interpretation sentence {i}" for i in range(N)],
        'problem_embedding':        fake_emb_col(),
        'method_embedding':         fake_emb_col(),
        'finding_embedding':        fake_emb_col(),
        'interpretation_embedding': fake_emb_col(),
    })
    return df, dim


def test_word_order_perturb_n0_identity(synthetic_texts):
    test = WordOrderShuffle()
    perturbed = test.perturb(synthetic_texts, n=0)
    assert perturbed == synthetic_texts


def test_word_order_perturb_n5_changes(synthetic_texts):
    test = WordOrderShuffle()
    perturbed = test.perturb(synthetic_texts, n=5)
    
    # The first two sentences should be changed
    assert perturbed[0] != synthetic_texts[0]
    assert perturbed[1] != synthetic_texts[1]
    
    # "Short one." has only two words, so it might just swap them or stay same
    # Empty string should stay empty
    assert perturbed[3] == ""


def test_word_order_length_preserved(synthetic_texts):
    test = WordOrderShuffle()
    perturbed = test.perturb(synthetic_texts, n=10)
    
    for orig, pert in zip(synthetic_texts, perturbed):
        assert len(orig.split()) == len(pert.split())


def test_displacement_metrics():
    # Two identical vectors -> distance 0
    v1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    v2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = displacement_cosine(v1, v2)
    assert np.allclose(d, 0.0)
    
    # Orthogonal vectors -> distance 1.0
    v3 = np.array([[0.0, 1.0], [1.0, 0.0]])
    d2 = displacement_cosine(v1, v3)
    assert np.allclose(d2, 1.0)
    
    # Opposite vectors -> distance 2.0
    v4 = np.array([[-1.0, 0.0], [0.0, -1.0]])
    d3 = displacement_cosine(v1, v4)
    assert np.allclose(d3, 2.0)


def test_sweep_engine_shape(synthetic_dataframe):
    df, dim = synthetic_dataframe
    test = WordOrderShuffle()
    n_values = [0, 2]
    
    # Mock embed_fn that just returns random vectors
    rng = np.random.default_rng(42)
    def mock_embed(texts):
        return rng.standard_normal((len(texts), dim)).tolist()
        
    result = run_robustness_sweep(test, df, mock_embed, dim, n_values)
    
    assert result["test_name"] == test.name
    assert result["n_values"] == n_values
    
    # Check aspects
    aspects = ["problem", "method", "finding", "interpretation"]
    for aspect in aspects:
        assert len(result["mean_displacement"][aspect]) == len(n_values)
        assert len(result["std_displacement"][aspect]) == len(n_values)
        
        # At n=0, displacement should be exactly 0
        assert np.isclose(result["mean_displacement"][aspect][0], 0.0)
        
    # Check per-document traces
    for doc_id in df.index:
        for aspect in aspects:
            assert len(result["per_document"][doc_id][aspect]) == len(n_values)


def test_cache_roundtrip(tmp_path):
    from edel.robustness.cache import save_robustness_result, load_robustness_result
    
    result = {"test": "data"}
    exp_id = "test_exp"
    sample_ids = [1, 5, 2]  # Should be sorted internally
    
    save_robustness_result(exp_id, sample_ids, "test_name", result, tmp_path)
    
    loaded = load_robustness_result(exp_id, sample_ids, "test_name", tmp_path)
    assert loaded == result
    
    # Different sample IDs should not match
    assert load_robustness_result(exp_id, [1, 2, 3], "test_name", tmp_path) is None


def test_registry():
    tests = ROBUSTNESS_REGISTRY
    assert len(tests) >= 18
    
    word_order = get_test("word_order_shuffle")
    assert word_order.priority == "M"
    assert word_order.requires_reembed is True
    
    verb_mask = get_test("verb_masking")
    assert verb_mask.priority == "S"
    assert verb_mask.requires_reembed is True


def test_pos_synonym_substitution():
    from edel.robustness.nlp import init_nltk
    from edel.robustness.nlp import tokenize_and_tag
    init_nltk()
    
    verb_test = get_test("verb_synonym")
    adj_test = get_test("adjective_synonym")
    
    text = "Scientists analyze complex patterns."
    
    # Verb synonyms
    perturbed_verb = verb_test.perturb([text], n=1)[0]
    assert perturbed_verb != text
    assert len(tokenize_and_tag(perturbed_verb)) == len(tokenize_and_tag(text))
    
    # Adjective synonyms
    perturbed_adj = adj_test.perturb([text], n=1)[0]
    assert perturbed_adj != text
    assert len(tokenize_and_tag(perturbed_adj)) == len(tokenize_and_tag(text))


def test_pos_masking():
    verb_test = get_test("verb_masking")
    adj_test = get_test("adjective_masking")
    
    text = "The rapid runner quickly finished the long race."
    
    # Mask verb (finished)
    perturbed_verb = verb_test.perturb([text], n=1)[0]
    assert "[MASK]" in perturbed_verb
    
    # Mask adjective (rapid, long)
    perturbed_adj = adj_test.perturb([text], n=1)[0]
    assert "[MASK]" in perturbed_adj


def test_targeted_deletion():
    head_test = get_test("head_deletion")
    tail_test = get_test("tail_deletion")
    
    text = "One two three four five"
    
    assert head_test.perturb([text], n=2)[0] == "three four five"
    assert tail_test.perturb([text], n=2)[0] == "One two three"
    
    # Corner case: n greater than length
    assert head_test.perturb([text], n=10)[0] == ""
    assert tail_test.perturb([text], n=10)[0] == ""


def test_extensions():
    rand_ext = get_test("random_extension")
    gen_ext = get_test("generative_extension")
    
    text = "This is a sentence"
    
    # Random extension
    perturbed_rand = rand_ext.perturb([text], n=3)[0]
    assert len(perturbed_rand.split()) == len(text.split()) + 3
    
    # Generative extension fallback (no client attached)
    perturbed_gen = gen_ext.perturb([text], n=3)[0]
    assert "[generative extension of 3 words]" in perturbed_gen
    
    # Mock LLM Client
    class MockLLMClient:
        def generate(self, prompt):
            return '{"continuation": "additional words here"}'
            
    gen_ext.llm_client = MockLLMClient()
    perturbed_mock_gen = gen_ext.perturb([text], n=3)[0]
    assert "additional words here" in perturbed_mock_gen


def test_numeral_to_word():
    num_test = get_test("numeral_to_word")
    text = "I have 3 items and 25 boxes."
    
    # 100% replacement
    perturbed = num_test.perturb([text], n=100)[0]
    assert "three" in perturbed.lower()
    assert "two five" in perturbed.lower()
    assert "3" not in perturbed
    assert "25" not in perturbed


def test_dsl_injections():
    within_test = get_test("within_field_dsl_injection")
    out_test = get_test("out_of_field_dsl_injection")
    
    text = "The hypothesis was tested in the experiment."
    
    # Custom lexicons
    within_test.within_lexicon = ["isabelle", "hol"]
    out_test.out_lexicon = ["mitochondria"]
    
    perturbed_within = within_test.perturb([text], n=2)[0]
    assert len(perturbed_within.split()) == len(text.split()) + 2
    assert "isabelle" in perturbed_within or "hol" in perturbed_within
    
    perturbed_out = out_test.perturb([text], n=1)[0]
    assert len(perturbed_out.split()) == len(text.split()) + 1
    assert "mitochondria" in perturbed_out


def test_structural_metric():
    struct_test = get_test("sentence_count")
    assert struct_test.requires_reembed is False
    text = "This is sentence one. This is sentence two."
    assert struct_test.perturb([text], n=5)[0] == text


def test_duplication():
    noun_dup = get_test("noun_duplication")
    adj_dup = get_test("adjective_duplication")
    
    text = "The quick runner finished the race."
    # Nouns: runner, race. Adjectives: quick
    
    # Noun duplication with n=1 (repeats once -> 2 copies)
    perturbed_noun = noun_dup.perturb([text], n=1)[0]
    assert "runner runner" in perturbed_noun
    assert "race race" in perturbed_noun
    assert "quick quick" not in perturbed_noun
    
    # Adjective duplication with n=2 (repeats twice -> 3 copies)
    perturbed_adj = adj_dup.perturb([text], n=2)[0]
    assert "quick quick quick" in perturbed_adj
    assert "runner runner" not in perturbed_adj


def test_graded_displacement():
    concrete_del = get_test("concreteness_graded_displacement")
    spec_del = get_test("specificity_graded_displacement")
    
    # car (concrete=5.0), concept (concrete=1.0)
    text1 = "car concept"
    perturbed_concrete = concrete_del.perturb([text1], n=1)[0]
    assert perturbed_concrete == "concept"
    
    # algorithm (specific=5.0), thing (specific=1.0)
    text2 = "algorithm thing"
    perturbed_spec = spec_del.perturb([text2], n=1)[0]
    assert perturbed_spec == "thing"


def test_ratios_and_distributions():
    pmfi = get_test("pmfi_ratio")
    desc_noun = get_test("descriptive_noun_ratio")
    
    assert pmfi.requires_reembed is False
    assert desc_noun.requires_reembed is False
    
    text = "Isabelle is a logical system."
    assert pmfi.perturb([text], n=5)[0] == text
    assert desc_noun.perturb([text], n=5)[0] == text
