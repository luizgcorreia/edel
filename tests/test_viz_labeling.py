"""Tests for Stage 7 Visualizations."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from edel.viz.labeling import plot_epistemic_map, print_cluster_summaries


@pytest.fixture
def label_results():
    """Mock results from labeling stage."""
    return {
        "clusters": {
            "domain": {
                0: {"proposed_label": "Topic A", "cluster_topics": "Description A"},
                1: {"proposed_label": "Topic B", "cluster_topics": "Description B"},
            }
        },
        "axes": [{"axis_label": "Semantic X"}, {"axis_label": "Semantic Y"}],
    }


@pytest.fixture
def df_labeled():
    """Create a mock labeled DataFrame."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "proj_problem_umap_x": np.random.rand(10),
            "proj_problem_umap_y": np.random.rand(10),
            "cluster_domain": [0, 1] * 5,
        }
    )


def test_viz_labeling_plots(df_labeled, label_results):
    """Verify that Stage 7 visualization functions run without errors."""
    plt.switch_backend("Agg")

    try:
        # Test summary printing
        print_cluster_summaries(label_results, cluster_key="domain")

        # Test epistemic map
        plot_epistemic_map(
            df_labeled,
            label_results,
            method="umap",
            cluster_key="domain",
            topic_name="Scientometrics",
        )

    except Exception as e:
        pytest.fail(f"Labeling visualization failed with error: {e}")
    finally:
        plt.close("all")
