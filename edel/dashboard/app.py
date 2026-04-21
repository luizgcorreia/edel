"""Main entry point for the EDEL Research Dashboard."""

import argparse
from pathlib import Path

import dash
import dash_bootstrap_components as dbc

from edel.dashboard.layout import create_layout
from edel.dashboard.callbacks import register_callbacks

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------

def create_app(base_path: str = "artifacts") -> dash.Dash:
    """Initialize and configure the Dash application."""
    
    # We use a Bootstrap theme for standard styling
    app = dash.Dash(
        __name__, 
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
        title="⛰️ EDEL Dashboard"
    )
    
    # Initialize experiment registry and snippets with persistence
    from edel.experiments.registry import init_registry
    from edel.experiments.snippets import init_snippets
    configs_dir = Path(base_path) / "configs"
    init_registry(configs_dir)
    init_snippets(configs_dir)
    
    # Set the layout as a function to ensure it's re-evaluated on every page load
    app.layout = create_layout
    
    # Register callbacks
    register_callbacks(app, Path(base_path))
    
    return app

# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    """Run the Dash server."""
    parser = argparse.ArgumentParser(description="Run the EDEL Research Dashboard.")
    parser.add_argument("--base-path", type=str, default="artifacts", help="Root directory for artifacts and jobs.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP to bind to.")
    parser.add_argument("--port", type=int, default=8050, help="Port to listen on.")
    parser.add_argument("--debug", action="store_true", help="Run dash in debug mode.")
    
    args = parser.parse_args()
    
    app = create_app(args.base_path)
    
    print(f"Starting EDEL Dashboard on http://{args.host}:{args.port}")
    print(f"Using artifact base path: {Path(args.base_path).resolve()}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
