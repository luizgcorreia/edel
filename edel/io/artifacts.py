"""Artifact persistence helpers for pipeline outputs."""

from pathlib import Path
import pickle


def save_artifacts(artifacts: dict, filename: str | Path) -> Path:
    """Serialize artifacts dictionary to disk with pickle."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(artifacts, f)
    return path


def load_artifacts(filename: str | Path) -> dict:
    """Load a serialized artifacts dictionary from disk."""
    with Path(filename).open("rb") as f:
        return pickle.load(f)
