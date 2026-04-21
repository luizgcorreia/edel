"""Interactive Landscape Map Component for the EDEL Dashboard."""

from dash import html, dcc
import dash_bootstrap_components as dbc
from edel.dashboard.utils import get_registry_options

def landscape_panel_layout() -> dbc.Container:
    """Layout for the Interactive Landscape Map panel."""
    return dbc.Container([
        html.H3("Interactive Landscape Map", className="mt-3 mb-4"),
        
        dbc.Row([
            # Left Column: Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Map Controls"),
                    dbc.CardBody([
                        html.Label("Select Experiment:"),
                        dcc.Dropdown(
                            id="map-experiment-select",
                            options=get_registry_options(),
                            value="scientometrics_baseline",
                            clearable=False,
                            persistence=False,
                            className="mb-3"
                        ),
                        
                        html.Label("View Mode:"),
                        dcc.RadioItems(
                            id="map-view-mode",
                            options=[
                                {"label": " 3D Surface", "value": "3d"},
                                {"label": " 2D Contour", "value": "2d"}
                            ],
                            value="3d",
                            inline=True,
                            className="mb-3"
                        ),
                        
                        html.Label("Projection Method:"),
                        dcc.RadioItems(
                            id="map-method-select",
                            options=[
                                {"label": " UMAP", "value": "umap"},
                                {"label": " Diffusion", "value": "diffusion"}
                            ],
                            value="diffusion",
                            inline=True,
                            className="mb-3"
                        ),
                        
                        html.Label("Layers:", className="fw-bold mt-2"),
                        dbc.Checklist(
                            options=[
                                {"label": "Terrain Surface (3D)", "value": "surface"},
                                {"label": "Paper Scatter", "value": "scatter"},
                                {"label": "Vector Field", "value": "vectors"},
                                {"label": "Cluster Labels", "value": "clusters"}
                            ],
                            value=["surface", "scatter", "clusters"],
                            id="map-layer-toggles",
                            switch=True,
                            className="mb-3"
                        ),
                        
                        html.Hr(),
                        
                        html.Div([
                            html.H6("Search Paper", className="mb-2"),
                            dcc.Dropdown(
                                id="map-paper-search",
                                placeholder="Search by title or ID (type to search)...",
                                searchable=True,
                                clearable=True,
                                options=[], # Empty by default, populated lazily
                                className="mb-3"
                            )
                        ]),
                        
                        html.Div([
                            html.Strong("Selected Paper Trajectory:"),
                            html.Div(id="map-selected-paper-info", children="Click a paper on the map to view its trajectory.", className="text-muted small mt-2")
                        ])
                    ])
                ])
            ], md=3),
            
            # Right Column: The Map
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            dcc.Loading(
                                id="loading-map",
                                type="circle",
                                children=dcc.Graph(id='landscape-graph', responsive=False)
                            )
                        ], style={"minHeight": "800px", "display": "flex", "alignItems": "center", "justifyContent": "center"})
                    ])
                ])
            ], md=9)
        ])
    ], fluid=True, className="mb-5")
