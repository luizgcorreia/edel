"""EDEL Robustness Testing Module.

Provides document-level semantic displacement tests against structural perturbations.
"""

from .base import RobustnessTest
from .registry import ROBUSTNESS_REGISTRY, list_tests, get_test
from .runner import run_robustness_sweep
from .cache import load_robustness_result, save_robustness_result

__all__ = [
    "RobustnessTest",
    "ROBUSTNESS_REGISTRY",
    "list_tests",
    "get_test",
    "run_robustness_sweep",
    "load_robustness_result",
    "save_robustness_result",
]
