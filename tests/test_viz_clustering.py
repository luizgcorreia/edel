"""Tests for Stage 6 Visualizations."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from edel.viz.clustering import (
    plot_cluster_trajectories,
    plot_clusters_on_landscape,
    plot_field_clusters,
)


@pytest.fixture
def df_clust():
    """Create a mock clustering DataFrame."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "proj_problem_umap_x": np.random.rand(10),
            "proj_problem_umap_y": np.random.rand(10),
            "proj_method_umap_x": np.random.rand(10),
            "proj_method_umap_y": np.random.rand(10),
            "cluster_domain": [0, 1] * 5,
            "cluster_style": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        }
    )


@pytest.fixture
def field_clust():
    """Create a mock field clustering DataFrame."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "cell_px": np.random.rand(5),
            "cell_py": np.random.rand(5),
            "cluster_field": [0, 1, 0, 1, 0],
        }
    )


def test_viz_clustering_plots(df_clust, field_clust):
    """Verify that Stage 6 visualization functions run without errors."""
    plt.switch_backend("Agg")

    try:
        # Test domain clusters
        plot_clusters_on_landscape(df_clust, method="umap", cluster_key="domain")

        # Test field clusters
        plot_field_clusters(field_clust, cluster_key="field")

        # Test style trajectories
        plot_cluster_trajectories(df_clust, method="umap", cluster_key="style", step=2)

    except Exception as e:
        pytest.fail(f"Clustering visualization failed with error: {e}")
    finally:
        plt.close("all")
