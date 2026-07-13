"""Callback registration for the EDEL Dashboard."""

from pathlib import Path
from dash import Dash

from .config import register_config_callbacks
from .experiments import register_experiment_callbacks
from .metrics import register_metrics_callbacks
from .landscape import register_landscape_callbacks
from .trajectory import register_trajectory_callbacks
from .hypothesis_callbacks import register_hypothesis_callbacks
from .convergence_callbacks import register_convergence_callbacks
from .report_generator import register_report_generator_callbacks
from .robustness_callbacks import register_robustness_callbacks
from .paper_figures import register_paper_figures_callbacks

def register_callbacks(app: Dash, base_path: Path) -> None:
    """Register all callbacks with the Dash app."""
    register_config_callbacks(app)
    register_experiment_callbacks(app, base_path)
    register_metrics_callbacks(app, base_path)
    register_landscape_callbacks(app, base_path)
    register_trajectory_callbacks(app, base_path)
    register_hypothesis_callbacks(app, base_path)
    register_convergence_callbacks(app, base_path)
    register_report_generator_callbacks(app, base_path)
    register_robustness_callbacks(app, base_path)
    register_paper_figures_callbacks(app, base_path)
