"""Dataset loading helpers."""

from pathlib import Path

import pandas as pd


def load_dataset(filename: str | Path) -> pd.DataFrame:
    """Load a CSV dataset into a pandas DataFrame."""
    return pd.read_csv(filename)
