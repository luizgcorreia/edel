"""Callback registration for the EDEL Dashboard."""

from pathlib import Path
from dash import Dash

from .config import register_config_callbacks
from .experiments import register_experiment_callbacks
from .metrics import register_metrics_callbacks
from .landscape import register_landscape_callbacks

def register_callbacks(app: Dash, base_path: Path) -> None:
    """Register all callbacks with the Dash app."""
    register_config_callbacks(app)
    register_experiment_callbacks(app, base_path)
    register_metrics_callbacks(app, base_path)
    register_landscape_callbacks(app, base_path)
