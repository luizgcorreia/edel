"""Top-level layout and state stores for the EDEL Dashboard."""

from dash import html, dcc
import dash_bootstrap_components as dbc

from edel.dashboard.components import (
    config_manager_layout,
    job_panel_layout,
    metrics_panel_layout,
    landscape_panel_layout,
    debugger_panel_layout,
)

def create_layout() -> html.Div:
    """Create the root layout with navigation tabs and global stores."""
    return html.Div([
        # Global State Stores
        dcc.Store(id='config-store'),
        dcc.Store(id='experiment-store'),
        dcc.Store(id='selected-paper-store'),
        
        # Navigation Bar
        dbc.NavbarSimple(
            brand="EDEL Research Dashboard",
            brand_href="#",
            color="dark",
            dark=True,
            className="mb-4",
        ),
        
        # Main Tabs
        dbc.Container([
            dbc.Tabs([
                dbc.Tab(config_manager_layout(), label="1. Config Manager", tab_id="tab-config"),
                dbc.Tab(job_panel_layout(), label="2. Experiment Runner", tab_id="tab-runner"),
                dbc.Tab(metrics_panel_layout(), label="3. Metrics Analysis", tab_id="tab-metrics"),
                dbc.Tab(landscape_panel_layout(), label="4. Interactive Landscape", tab_id="tab-landscape"),
                dbc.Tab(debugger_panel_layout(), label="5. Stage Debugger", tab_id="tab-debugger"),
            ], id="main-tabs", active_tab="tab-metrics"),
        ], fluid=True, className="px-4"),
        
        # Hidden div to pass base_path to callbacks if needed
        html.Div(id='base-path-store', style={'display': 'none'})
    ])
