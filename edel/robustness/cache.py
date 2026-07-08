"""Caching layer for robustness tests."""

import pickle
import hashlib
from pathlib import Path


def _get_cache_dir(experiment_id: str, sample_ids: list, base_path: Path) -> Path:
    """Get the directory for caching robustness results for a specific sample."""
    # Create a deterministic hash of the sample IDs
    sorted_ids = sorted([str(sid) for sid in sample_ids])
    sample_hash = hashlib.md5(",".join(sorted_ids).encode('utf-8')).hexdigest()[:12]
    
    cache_dir = base_path / "experiments" / experiment_id / "robustness" / sample_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def save_robustness_result(experiment_id: str, sample_ids: list, test_name: str, result: dict, base_path: Path | str) -> None:
    """Save a robustness sweep result to the cache.
    
    Args:
        experiment_id: The ID of the experiment.
        sample_ids: List of document IDs in the sample.
        test_name: The name of the robustness test.
        result: The result dictionary to cache.
        base_path: The root artifacts directory.
    """
    base_path = Path(base_path)
    cache_dir = _get_cache_dir(experiment_id, sample_ids, base_path)
    file_path = cache_dir / f"{test_name}.pkl"
    
    with open(file_path, "wb") as f:
        pickle.dump(result, f)


def load_robustness_result(experiment_id: str, sample_ids: list, test_name: str, base_path: Path | str) -> dict | None:
    """Load a robustness sweep result from the cache.
    
    Args:
        experiment_id: The ID of the experiment.
        sample_ids: List of document IDs in the sample.
        test_name: The name of the robustness test.
        base_path: The root artifacts directory.
        
    Returns:
        The cached result dictionary, or None if not found.
    """
    base_path = Path(base_path)
    cache_dir = _get_cache_dir(experiment_id, sample_ids, base_path)
    file_path = cache_dir / f"{test_name}.pkl"
    
    if not file_path.exists():
        return None
        
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Failed to load robustness cache for {test_name}: {e}")
        return None
