"""Experiments engine for EDEL — public API."""

from edel.experiments.runner import run_experiments
from edel.experiments.analyzer import analyze_experiments, compare_experiments
from edel.experiments.registry import EXPERIMENTS, get_experiment, list_experiments, register_experiment

__all__ = [
    "run_experiments",
    "analyze_experiments",
    "compare_experiments",
    "EXPERIMENTS",
    "get_experiment",
    "list_experiments",
    "register_experiment",
]
