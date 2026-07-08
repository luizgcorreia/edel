"""Runner engine for robustness tests."""

import json
import numpy as np
import pandas as pd
from typing import Callable, Any
from sklearn.preprocessing import normalize as sk_normalize

from edel.pipeline.projection import load_embeddings_to_matrix
from edel.robustness.base import RobustnessTest, displacement_cosine


def run_robustness_sweep(
    test: RobustnessTest,
    df: pd.DataFrame,
    embed_fn: Callable[[list[str]], list[list[float]]],
    dimensions: int,
    n_values: list[int],
    aspects: list[str] = None,
    llm_client: Any = None,
    within_lexicon: list[str] = None,
    out_lexicon: list[str] = None,
) -> dict:
    """Run a single perturbation test across N values.
    
    Args:
        test: The robustness test to run.
        df: DataFrame subset containing original texts and embeddings.
        embed_fn: Function to generate embeddings for a list of texts.
        dimensions: Embedding dimensions.
        n_values: List of perturbation intensity values.
        aspects: List of aspects to test (e.g., ["problem", "method", "finding", "interpretation"]).
        llm_client: Active LLM client for generative perturbations.
        within_lexicon: Lexicon of domain terms for within-field injection.
        out_lexicon: Lexicon of domain terms for out-of-field injection.
        
    Returns:
        Sweep result dictionary.
    """
    if aspects is None:
        aspects = ["problem", "method", "finding", "interpretation"]
        
    # Inject llm_client and lexicons to test instance if it supports them
    if llm_client is not None:
        test.llm_client = llm_client
    if within_lexicon is not None:
        test.within_lexicon = within_lexicon
    if out_lexicon is not None:
        test.out_lexicon = out_lexicon
        
    result = {
        "test_name": test.name,
        "n_values": n_values,
        "mean_displacement": {a: [] for a in aspects},
        "std_displacement": {a: [] for a in aspects},
        "per_document": {doc_id: {a: [] for a in aspects} for doc_id in df.index},
    }
    
    # Pre-load original embeddings and normalize them
    orig_embs = {}
    for aspect in aspects:
        col = f"{aspect}_embedding"
        if col in df.columns:
            # load_embeddings_to_matrix returns (N, D) matrix
            mat = load_embeddings_to_matrix(df, col, dimensions)
            orig_embs[aspect] = sk_normalize(mat)
        else:
            orig_embs[aspect] = None

    for n in n_values:
        for aspect in aspects:
            if orig_embs[aspect] is None:
                # Missing original embeddings for this aspect
                result["mean_displacement"][aspect].append(0.0)
                result["std_displacement"][aspect].append(0.0)
                for doc_id in df.index:
                    result["per_document"][doc_id][aspect].append(0.0)
                continue
                
            orig_texts = df[aspect].fillna("").astype(str).tolist()
            
            # Perturb
            perturbed_texts = test.perturb(orig_texts, n)
            
            if not test.requires_reembed or n == 0:
                # No re-embedding required, or n=0 (baseline)
                # Displacement is zero
                displacements = np.zeros(len(df))
            else:
                # Re-embed
                try:
                    # Filter out empty texts for embedding API
                    valid_indices = [i for i, text in enumerate(perturbed_texts) if text.strip()]
                    valid_texts = [perturbed_texts[i] for i in valid_indices]
                    
                    if valid_texts:
                        new_embs = embed_fn(valid_texts)
                    else:
                        new_embs = []
                        
                    # Reconstruct full matrix
                    new_mat = np.zeros((len(df), dimensions))
                    for i, emb in zip(valid_indices, new_embs):
                        new_mat[i] = emb
                        
                    new_mat = sk_normalize(new_mat)
                    
                    # Compute displacement
                    displacements = displacement_cosine(orig_embs[aspect], new_mat)
                except Exception as e:
                    print(f"Error re-embedding for {aspect} at N={n}: {e}")
                    displacements = np.zeros(len(df))
            
            # Save results
            result["mean_displacement"][aspect].append(float(np.mean(displacements)))
            result["std_displacement"][aspect].append(float(np.std(displacements)))
            
            for i, doc_id in enumerate(df.index):
                result["per_document"][doc_id][aspect].append(float(displacements[i]))
                
    return result
