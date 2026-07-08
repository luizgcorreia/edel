"""Base definitions for robustness tests."""

import numpy as np
from abc import ABC, abstractmethod


class RobustnessTest(ABC):
    """Abstract base class for all robustness perturbation tests."""
    
    name: str = "base_test"
    label: str = "Base Test"
    priority: str = "M"
    requires_reembed: bool = True
    
    @abstractmethod
    def perturb(self, texts: list[str], n: int) -> list[str]:
        """Apply the transformation at intensity n to a list of texts.
        
        n=0 MUST return the original texts unchanged.
        
        Args:
            texts: List of original text strings.
            n: Intensity of the perturbation (e.g. number of tokens to modify).
            
        Returns:
            List of perturbed text strings.
        """
        pass


def displacement_cosine(orig: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    """Compute per-document cosine distance between original and perturbed embeddings.
    
    Cosine distance is 1 - cosine_similarity.
    
    Args:
        orig: Original embedding matrix (N, D), should be L2-normalized.
        perturbed: Perturbed embedding matrix (N, D), should be L2-normalized.
        
    Returns:
        1D array of length N containing the cosine distance for each document.
    """
    # Assuming embeddings are already L2-normalized
    sims = np.sum(orig * perturbed, axis=1)
    # Clip to avoid floating point precision issues > 1.0 or < -1.0
    sims = np.clip(sims, -1.0, 1.0)
    return 1.0 - sims


def displacement_l2(orig: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    """Compute per-document L2 distance between original and perturbed embeddings.
    
    Args:
        orig: Original embedding matrix (N, D).
        perturbed: Perturbed embedding matrix (N, D).
        
    Returns:
        1D array of length N containing the L2 distance for each document.
    """
    diffs = orig - perturbed
    return np.linalg.norm(diffs, axis=1)
