"""Metrics plugin registry for the EDEL experiments engine.

METRIC_REGISTRY is an ordered list of metric functions. Each function has the
contract:

    fn(artifacts: dict) -> {"metrics": dict[str, float], "features": dict[str, np.ndarray]}

The `artifacts` dict is the shared mutable context — metric functions may
read pipeline artifacts from it and write intermediate results for downstream
functions (e.g. operator_metrics writes pm/mf/fi for structure_metrics).

Order matters: operator_metrics must run before structure_metrics.
"""

from edel.experiments.metrics.segmentation import segmentation_metrics
from edel.experiments.metrics.embedding import embedding_metrics
from edel.experiments.metrics.operators import operator_metrics
from edel.experiments.metrics.structure import structure_metrics
from edel.experiments.metrics.hypothesis_tests import hypothesis_metrics

METRIC_REGISTRY = [
    segmentation_metrics,
    embedding_metrics,
    operator_metrics,   # must precede structure_metrics (writes _operators to context)
    structure_metrics,
    hypothesis_metrics,
]

__all__ = [
    "METRIC_REGISTRY",
    "segmentation_metrics",
    "embedding_metrics",
    "operator_metrics",
    "structure_metrics",
    "hypothesis_metrics",
]
