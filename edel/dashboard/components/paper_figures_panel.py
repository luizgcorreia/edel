"""Paper-ready figures generator component for the EDEL Dashboard."""

from dash import html, dcc
import dash_bootstrap_components as dbc
from edel.dashboard.utils import get_registry_options

def paper_figures_panel_layout() -> dbc.Container:
    """Layout for the Paper Figures panel (Tab 11)."""
    
    # 10 paper figures options grouped by hypothesis
    figure_options = [
        {"label": "H1.1: Discourse Trajectory & Simplex (3D)", "value": "fig-h1-trajectory-simplex"},
        {"label": "H1.2: Energy Distance Results (All Runs)", "value": "fig-h1-energy-distance"},
        {"label": "H1.3: Example Simplex Visualizations (Unified)", "value": "fig-h1-example-simplex"},
        {"label": "H2.1: Transition Operators Heatmap (z-scores)", "value": "fig-h2-heatmap"},
        {"label": "H2.2: Representative Transition Neighborhoods", "value": "fig-h2-neighborhoods"},
        {"label": "H2.3: 3D Connected Simplices (Transitions)", "value": "fig-h2-connected-3d"},
        {"label": "H2.4: Discourse Transition Space (PCA on Displacement Vectors)", "value": "fig-h2-transition-space"},
        {"label": "H3.1: Predictive Gain (EDEL vs Persistence)", "value": "fig-h3-predictive-gain"},
        {"label": "H3.2: Wasserstein Transport Null Distribution", "value": "fig-h3-wasserstein-null"},
        {"label": "H3.3: Predicted vs Observed Density Maps", "value": "fig-h3-density-maps"},
        {"label": "H3.4: Bivariate Moran Scatterplot", "value": "fig-h3-moran-scatterplot"},
    ]

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("📚 Paper Figures Generator", className="mt-3 mb-1"),
                html.P(
                    "Configure and export publication-ready high-quality figures for the paper. "
                    "Toggle the 'Paper Style' to apply academic styling (serif fonts, white background, high contrast).",
                    className="text-muted mb-4"
                ),
            ], width=12)
        ]),

        dbc.Row([
            # ----------------------------------------------------------------
            # Left Column — Configuration Controls
            # ----------------------------------------------------------------
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Strong("Figure Configuration"), className="bg-primary text-white"),
                    dbc.CardBody([
                        # Select Figure
                        html.Label("Target Paper Figure:", className="fw-bold"),
                        dcc.Dropdown(
                            id="paper-fig-select",
                            options=figure_options,
                            value="fig-h1-trajectory-simplex",
                            clearable=False,
                            className="mb-3"
                        ),

                        # Select Experiment
                        html.Div(id="paper-fig-exp-group", children=[
                            html.Label("Experiment:", className="fw-bold"),
                            dcc.Dropdown(
                                id="paper-fig-exp-select",
                                options=get_registry_options(),
                                placeholder="Select experiment...",
                                clearable=False,
                                className="mb-3"
                            ),
                            html.Label("Custom Experiment Name (for print):", className="fw-bold"),
                            dbc.Input(
                                id="paper-fig-custom-name",
                                type="text",
                                placeholder="e.g. AFP Baseline (leave blank to use ID)",
                                className="mb-3"
                            ),
                        ]),

                        # Select Paper (only visible for certain figures)
                        html.Div(id="paper-fig-paper-group", children=[
                            html.Label("Representative Paper:", className="fw-bold"),
                            dcc.Dropdown(
                                id="paper-fig-paper-select",
                                options=[],
                                placeholder="Select paper...",
                                clearable=True,
                                className="mb-3"
                            ),
                            html.Small(
                                "Choose a specific paper from the dataset to visualize its trajectory/neighborhood.",
                                className="text-muted d-block mb-3"
                            ),
                            html.Div([
                                html.Label("Select Top N Most Cited:", className="small fw-bold me-2"),
                                dbc.Input(
                                    id="paper-fig-top-n-input",
                                    type="number",
                                    value=5,
                                    min=1,
                                    max=100,
                                    size="sm",
                                    style={"width": "70px", "display": "inline-block", "verticalAlign": "middle"},
                                    className="me-2"
                                ),
                                dbc.Button(
                                    "Select",
                                    id="paper-fig-btn-top-n",
                                    color="secondary",
                                    size="sm",
                                    className="py-0 px-2",
                                    style={"verticalAlign": "middle"}
                                )
                            ], className="mb-3", style={"display": "flex", "alignItems": "center"})
                        ], style={"display": "block"}),

                        # Transition Operator Selector (only for neighborhood/H2 plots)
                        html.Div(id="paper-fig-transition-group", children=[
                            html.Label("Transition Operator:", className="fw-bold"),
                            dcc.Dropdown(
                                id="paper-fig-transition-select",
                                options=[
                                    {"label": "Problem → Method (D(M|p))", "value": "pm"},
                                    {"label": "Method → Finding (D(F|m))", "value": "mf"},
                                    {"label": "Finding → Interpretation (D(I|f))", "value": "fi"},
                                    {"label": "Problem → Finding (D(F|p))", "value": "pf"},
                                    {"label": "Problem → Interpretation (D(I|p))", "value": "pi"},
                                    {"label": "Method → Interpretation (D(I|m))", "value": "mi"},
                                ],
                                value="pm",
                                clearable=False,
                                className="mb-3"
                            ),
                        ], style={"display": "none"}),

                        # Neighbors Selector (only for neighborhood/H2 plots)
                        html.Div(id="paper-fig-neighbors-group", children=[
                            html.Label("Number of Neighbors (k):", className="fw-bold"),
                            dcc.Slider(
                                id="paper-fig-neighbors-slider",
                                min=1,
                                max=10,
                                step=1,
                                value=4,
                                marks={i: str(i) for i in range(1, 11)},
                                className="mb-3"
                            ),
                        ], style={"display": "none"}),

                        html.Hr(),

                        # Paper Style Options
                        html.Label("Aesthetic / Paper Styling:", className="fw-bold mb-2"),
                        
                        dbc.Checklist(
                            id="paper-fig-style-options",
                            options=[
                                {"label": "Apply Paper Style (Serif, White BG, Academic Ticks)", "value": "paper-style"},
                                {"label": "Show Gridlines", "value": "gridlines"},
                                {"label": "Colorblind-Friendly / High-Contrast Color Scheme", "value": "high-contrast"},
                                {"label": "Display Vertex Labels (Hide Legend)", "value": "vertex-labels"},
                            ],
                            value=["paper-style", "gridlines"],
                            switch=True,
                            className="mb-3"
                        ),

                        html.Label("Serif Font:", className="small text-muted"),
                        dcc.Dropdown(
                            id="paper-fig-font-select",
                            options=[
                                {"label": "Georgia", "value": "Georgia"},
                                {"label": "Times New Roman", "value": "Times New Roman"},
                                {"label": "Courier New (Monospace)", "value": "Courier New"},
                                {"label": "Computer Modern Style (Serif)", "value": "serif"},
                            ],
                            value="Georgia",
                            clearable=False,
                            className="mb-3"
                        ),

                        html.Label("Base Font Size (px):", className="small text-muted mb-1 d-block"),
                        dcc.Slider(
                            id="paper-fig-font-size",
                            min=8,
                            max=24,
                            step=1,
                            value=11,
                            marks={i: str(i) for i in range(8, 25, 2)},
                            className="mb-3"
                        ),

                        html.Hr(),

                        # Download Options
                        html.Label("HTML Download Options (Local Export):", className="fw-bold mb-2"),
                        
                        html.Label("Target Column Layout:", className="small text-muted"),
                        dcc.Dropdown(
                            id="paper-fig-aspect-ratio",
                            options=[
                                {"label": "Single Column (4:3)", "value": "4:3"},
                                {"label": "Double Column (16:9)", "value": "16:9"},
                                {"label": "Square (1:1)", "value": "1:1"},
                            ],
                            value="4:3",
                            clearable=False,
                            className="mb-3"
                        ),

                        html.Label("Local Export Format (via Modebar):", className="small text-muted"),
                        dcc.Dropdown(
                            id="paper-fig-export-format",
                            options=[
                                {"label": "SVG (Vector, Recommended)", "value": "svg"},
                                {"label": "PNG (High-Resolution Raster)", "value": "png"},
                            ],
                            value="svg",
                            clearable=False,
                            className="mb-3"
                        ),

                        dbc.Button(
                            "📥 Download Figure HTML",
                            id="btn-download-paper-fig",
                            color="success",
                            className="w-100 mt-2"
                        )
                    ])
                ], className="mb-4"),

                dbc.Card([
                    dbc.CardHeader(html.Strong("LaTeX Caption / Description")),
                    dbc.CardBody([
                        html.Div(id="paper-fig-latex-desc", className="small text-muted font-monospace", style={"whiteSpace": "pre-wrap"}),
                        dbc.Button("📋 Copy to Clipboard", id="btn-copy-latex", color="secondary", size="sm", className="mt-3 w-100")
                    ])
                ])
            ], md=4),

            # ----------------------------------------------------------------
            # Right Column — Figure Display
            # ----------------------------------------------------------------
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Span(id="paper-fig-title", children="Figure Preview"),
                        dbc.Button("⛶ Fullscreen", id="paper-fig-fullscreen-btn", size="sm", color="link", className="float-end p-0 text-decoration-none text-info")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            id="paper-fig-loading",
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id="paper-fig-graph",
                                    figure={"data": [], "layout": {"template": "plotly_white"}},
                                    responsive=True,
                                    style={"height": "500px"},
                                    config={
                                        "toImageButtonOptions": {
                                            "format": "svg",
                                            "filename": "paper_figure",
                                            "height": 500,
                                            "width": 700,
                                            "scale": 2
                                        }
                                    }
                                )
                            ]
                        )
                    ])
                ], id="paper-fig-card", className="mb-4"),

                dbc.Card([
                    dbc.CardHeader(html.Strong("Figure Details & Associated Statistics")),
                    dbc.CardBody([
                        html.Div(id="paper-fig-stats-container", children=[
                            html.P("Select a figure and an experiment to view associated statistics.", className="text-muted mb-0")
                        ])
                    ])
                ])
            ], md=8)
        ]),
        
        # Hidden components
        dcc.Clipboard(id="paper-fig-clipboard", style={"display": "none"}),
        dcc.Download(id="paper-fig-downloader")
    ], fluid=True, className="mb-5")
