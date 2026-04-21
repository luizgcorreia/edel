"""Callbacks for the Metrics Analysis panel."""

from pathlib import Path
from dash import Dash, Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from edel.dashboard.cache import get_results_df, rebuild_results_cache
from edel.dashboard.utils import df_to_dash_columns, df_to_dash_records
from edel.experiments.analyzer import compare_experiments
from edel.experiments.runner import load_registry

def register_metrics_callbacks(app: Dash, base_path: Path) -> None:
    
    @app.callback(
        [Output("metrics-table", "columns"),
         Output("metrics-table", "data"),
         Output("metrics-bar-chart", "figure")],
        [Input("btn-rebuild-cache", "n_clicks"),
         Input("metric-y-axis", "value")]
    )
    def update_metrics_data(n_clicks, selected_metric):
        """Load results.parquet (or rebuild if requested) and update table/chart."""
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        if triggered_id == "btn-rebuild-cache":
            df = rebuild_results_cache(base_path, delta_only=True)
        else:
            df = get_results_df(base_path)
            
        if df.empty:
            return [], [], go.Figure().update_layout(title="No experiment data found.")
            
        # Format table data
        # We might not want to show all 53 columns in the UI table by default.
        # Let's pick a sensible subset or just let the user scroll.
        display_cols = ["experiment_id", "dataset_size", "embedding_model", "projection_method", 
                        "seg_ratio_mean", "sim_pm", "silhouette_transitions", "silhouette_features"]
        
        # Keep only columns that actually exist in the df
        actual_cols = [c for c in display_cols if c in df.columns]
        
        columns = df_to_dash_columns(df[actual_cols])
        
        # Round floats for display
        display_df = df.copy()
        for col in display_df.select_dtypes(include=['float']).columns:
            display_df[col] = display_df[col].round(4)
            
        records = df_to_dash_records(display_df)
        
        # Create bar chart
        if selected_metric in df.columns and "experiment_id" in df.columns:
            # Sort by metric value for better visualization
            plot_df = df.sort_values(by=selected_metric, ascending=False)
            
            fig = px.bar(
                plot_df, 
                x="experiment_id", 
                y=selected_metric,
                color="embedding_model" if "embedding_model" in df.columns else None,
                title=f"Comparison: {selected_metric}",
                text_auto='.3f'
            )
            fig.update_layout(xaxis_tickangle=-45, margin=dict(b=100))
        else:
            fig = go.Figure().update_layout(title=f"Metric '{selected_metric}' not found in results.")
            
        return columns, records, fig

    @app.callback(
        Output("ks-heatmap-chart", "figure"),
        Input("btn-run-ks", "n_clicks"),
        State("ks-feature-select", "value"),
        prevent_initial_call=True
    )
    def run_ks_tests(n_clicks, feature_dim):
        """Run pairwise KS tests and display as a heatmap."""
        if not n_clicks or not feature_dim:
            raise PreventUpdate
            
        # We need the list of experiments from the registry to pass to compare_experiments
        registry = load_registry(base_path)
        if not registry:
            return go.Figure().update_layout(title="No experiments found in registry.")
            
        # Run KS tests
        ks_df = compare_experiments(registry, base_path=base_path, feature_dims=[feature_dim])
        
        if ks_df.empty:
            return go.Figure().update_layout(title="Not enough features available for comparison.")
            
        # Pivot to a matrix for heatmap
        # ks_df has: exp_a, exp_b, feature, ks_stat, ks_pvalue
        pivot_df = ks_df.pivot(index="exp_a", columns="exp_b", values="ks_pvalue")
        
        # Fill NA with 1.0 (or we can mirror the matrix)
        pivot_df = pivot_df.fillna(1.0)
        
        # Create a custom colorscale where small p-values (significant) are highlighted
        # e.g. < 0.05 is red, > 0.05 is blue/white
        
        fig = px.imshow(
            pivot_df,
            labels=dict(x="Experiment B", y="Experiment A", color="p-value"),
            title=f"KS Test p-values ({feature_dim})",
            color_continuous_scale="RdBu_r", # Red for small p-values (significant difference)
            zmin=0.0, zmax=1.0
        )
        
        # Add annotation for significance threshold
        fig.add_annotation(
            text="Red indicates significant difference (p < 0.05)",
            xref="paper", yref="paper",
            x=0.5, y=-0.2, showarrow=False
        )
        
        return fig
