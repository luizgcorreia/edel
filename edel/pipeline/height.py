"""Height metric computations for epistemic landscape surfaces."""

import numpy as np


def compute_height_metric(df, metric: str = "cited_by_count", log_scale: bool = True):
    """Compute Z values and label for the selected height metric."""
    if metric == "cited_by_count":
        z = df["cited_by_count"].fillna(0).values
        if log_scale:
            z = np.log10(z + 1)
        label = "Citations (log10)" if log_scale else "Citations"
    else:
        z = df[metric].values
        label = metric

    return z, label
