"""Unit tests for the hypothesis testing metrics plugin."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from edel.experiments.metrics.hypothesis_tests import (
    hypothesis_metrics,
    compute_wasserstein,
    compute_morans_i,
)


@pytest.fixture
def synthetic_artifacts() -> dict:
    """Generate mock pipeline artifacts for testing."""
    np.random.seed(42)
    N = 30
    dims = 16

    # Generate synthetic embeddings with some transition structure
    prob = np.random.randn(N, dims)
    # create step transitions
    meth = prob + np.random.randn(N, dims) * 0.1
    find = meth + np.random.randn(N, dims) * 0.1
    interp = find + np.random.randn(N, dims) * 0.1

    data = {
        "id": [f"https://openalex.org/W{i}" for i in range(N)],
        "title": [f"Paper {i}" for i in range(N)],
        "publication_year": [2020 + (i % 5) for i in range(N)],
        "problem_embedding": [json.dumps(p.tolist()) for p in prob],
        "method_embedding": [json.dumps(m.tolist()) for m in meth],
        "finding_embedding": [json.dumps(f.tolist()) for f in find],
        "interpretation_embedding": [json.dumps(i.tolist()) for i in interp],
    }
    df = pd.DataFrame(data)

    return {
        "embedding": df,
        "_dimensions": dims,
    }


def test_wasserstein_distance():
    """Verify that Wasserstein distance calculation executes correctly."""
    X = np.array([[1.0, 0.0], [2.0, 0.0]])
    Y = np.array([[1.0, 0.0], [2.0, 0.0]])
    # Identical distributions should have a Wasserstein distance of 0
    w_dist = compute_wasserstein(X, Y)
    assert pytest.approx(w_dist, abs=1e-5) == 0.0

    Z = np.array([[3.0, 0.0], [4.0, 0.0]])
    w_dist_diff = compute_wasserstein(X, Z)
    assert w_dist_diff > 0.0


def test_morans_i():
    """Verify Moran's I spatial correlation logic."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    w = np.array([
        [0.0, 1.0, 0.5],
        [1.0, 0.0, 1.0],
        [0.5, 1.0, 0.0]
    ])
    val = compute_morans_i(x, y, w)
    assert isinstance(val, float)


def test_hypothesis_metrics(synthetic_artifacts):
    """Run full hypothesis validation metrics plugin on mock data."""
    res = hypothesis_metrics(synthetic_artifacts)
    assert "metrics" in res
    assert "features" in res

    metrics = res["metrics"]
    
    # H1: energy distance primary test
    assert "h1a_energy_stat" in metrics
    assert "h1a_energy_pvalue" in metrics
    assert 0.0 <= metrics["h1a_energy_pvalue"] <= 1.0

    # H1a: per-edge Wasserstein effect sizes
    for key in ["norm_pm", "norm_mf", "norm_fi", "cos_pm_mf", "cos_pm_fi", "cos_mf_fi"]:
        assert f"h1a_w_{key}" in metrics

    # H1a: KS diagnostics (secondary)
    for key in ["norm_pm", "norm_mf", "norm_fi", "cos_pm_mf", "cos_pm_fi", "cos_mf_fi"]:
        assert f"h1a_ks_stat_{key}" in metrics
        assert f"h1a_ks_pvalue_{key}" in metrics
        assert 0.0 <= metrics[f"h1a_ks_pvalue_{key}"] <= 1.0

    # H1a: feature stores
    assert "h1a_obs_features" in res["features"]
    assert "h1a_shuf_features" in res["features"]
    assert "h1_edge_norms" in res["features"]

    # H2 Wasserstein keys (primary local transition test)
    h2_keys = ["pm", "pf", "pi", "mp", "mf", "mi", "fp", "fm", "fi", "ip", "im", "if"]
    for key in h2_keys:
        assert f"h2_w_dist_{key}" in metrics
        assert f"h2_pvalue_{key}" in metrics
        assert f"h2_z_{key}" in metrics
        assert 0.0 <= metrics[f"h2_pvalue_{key}"] <= 1.0

    # H2b Transition Asymmetry keys
    h2b_keys = ["pm", "mf", "fi", "pf", "pi", "mi"]
    for key in h2b_keys:
        assert f"h2b_entropy_forward_{key}" in metrics
        assert f"h2b_entropy_reverse_{key}" in metrics
        assert f"h2b_branching_forward_{key}" in metrics
        assert f"h2b_branching_reverse_{key}" in metrics
        assert f"h2b_diff_{key}" in metrics
        assert f"h2b_pvalue_{key}" in metrics
        assert 0.0 <= metrics[f"h2b_pvalue_{key}"] <= 1.0

    # H3 Predictive keys
    assert "h3_w_edel" in metrics
    assert "h3_w_baseline" in metrics
    assert "h3_predictive_gain" in metrics
    assert "h3_gain_pvalue" in metrics
    assert "h3_moran_i" in metrics
    assert "h3_moran_pvalue" in metrics
    assert -1.0 <= metrics["h3_moran_i"] <= 1.0
    assert 0.0 <= metrics["h3_moran_pvalue"] <= 1.0
    assert 0.0 <= metrics["h3_gain_pvalue"] <= 1.0
