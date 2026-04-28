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
                                {"label": "Cluster Labels", "value": "clusters"},
                                {"label": "Domain Regions (2D)", "value": "regions"},
                                {"label": "Knowledge Frontier (2D)", "value": "frontier"},
                                {"label": "Relevant Papers", "value": "papers"}
                            ],
                            value=["surface", "scatter", "clusters", "regions", "papers"],
                            id="map-layer-toggles",
                            switch=True,
                            className="mb-3"
                        ),
                        
                        html.Hr(),
                        
                        html.Div([
                            html.H6("Relevant Papers Configuration", className="mb-2"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Metric:", className="small text-muted"),
                                    dcc.Dropdown(
                                        id="map-papers-metric",
                                        options=[
                                            {"label": "Absolute Citations", "value": "cited_by_count"},
                                            {"label": "Citations per Year", "value": "citation_velocity"},
                                            {"label": "Normalized Percentile", "value": "citation_normalized_percentile"},
                                            {"label": "Landscape Peaks", "value": "local_peaks"},
                                            {"label": "Cluster Capitals", "value": "cluster_centroids"}
                                        ],
                                        value="cited_by_count",
                                        clearable=False,
                                        className="mb-2"
                                    )
                                ], width=8),
                                dbc.Col([
                                    html.Label("Count:", className="small text-muted"),
                                    dbc.Input(
                                        id="map-top-papers",
                                        type="number",
                                        min=0,
                                        step=1,
                                        value=10,
                                        className="mb-2"
                                    )
                                ], width=4)
                            ])
                        ]),
                        
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
                        ], className="mb-4"),
                        
                        html.Hr(),
                        
                        html.Div([
                            html.H6("Export Map", className="mb-2"),
                            dcc.Loading(
                                id="loading-export",
                                type="circle",
                                children=[
                                    dbc.ButtonGroup([
                                        dbc.Button("Download PNG", id="btn-download-png", color="outline-primary", size="sm"),
                                        dbc.Button("Download HTML", id="btn-download-html", color="outline-primary", size="sm"),
                                    ], className="w-100"),
                                ]
                            ),
                            dcc.Download(id="download-map-file")
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
                                children=[
                                    html.Div(
                                        dcc.Graph(id='landscape-graph-3d', responsive=False, style={"height": "800px"}),
                                        id='map-container-3d'
                                    ),
                                    html.Div(
                                        dcc.Graph(id='landscape-graph-2d', responsive=False, style={"height": "800px"}),
                                        id='map-container-2d',
                                        style={'display': 'none'}
                                    )
                                ]
                            )
                        ], style={"minHeight": "800px"})
                    ])
                ])
            ], md=9)
        ])
    ], fluid=True, className="mb-5")
