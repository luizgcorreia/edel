import pickle
import logging
from pathlib import Path
from dash import Dash, Input, Output, State, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import plotly.graph_objects as go

from edel.experiments.runner import load_registry
from edel.dashboard.cache import get_results_df

logger = logging.getLogger(__name__)

def _load_features(exp_id: str, base_path: Path) -> dict | None:
    """Load cached features (distributions) for a given experiment ID."""
    features_path = base_path / "experiments" / exp_id / "features.pkl"
    if not features_path.exists():
        return None
    try:
        with open(features_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Error loading features for '{exp_id}': {e}")
        return None

def _get_or_compute_h3_moran_features(exp_id: str, feat: dict | None, base_path: Path) -> dict | None:
    """Retrieve h3_moran features if cached, otherwise compute them on the fly."""
    if feat and "h3_moran" in feat and "h3_gain_pvalue" in feat["h3_moran"]:
        # Convert all to numpy arrays if they are lists (from json/pickle loading)
        out = {}
        for k, v in feat["h3_moran"].items():
            if isinstance(v, list):
                out[k] = np.array(v)
            else:
                out[k] = v
        return out
        
    # On-the-fly computation fallback
    try:
        from edel.experiments.runner import load_registry
        from edel.io.artifact import load_artifact
        from sklearn.cluster import KMeans
        from sklearn.linear_model import Ridge
        from scipy.spatial.distance import cdist
        from edel.experiments.metrics.hypothesis_tests import load_embeddings_to_matrix, sk_normalize, compute_morans_i, compute_wasserstein

        registry = load_registry(base_path)
        record = None
        for rec in registry:
            if rec["experiment_id"] == exp_id:
                record = rec
                break
        if not record:
            return None
            
        # Load clustering/embeddings DataFrame
        df = load_artifact(record["artifact_refs"]["clustering"])
        
        dimensions = record["config"].get("embedding", {}).get("n_dimensions", 1536)
        
        # Norm and load matrices
        def load(aspect: str) -> np.ndarray:
            mat = load_embeddings_to_matrix(df, f"{aspect}_embedding", dimensions)
            mat -= mat.mean(axis=0)
            return sk_normalize(mat)

        emb_p = load("problem")
        emb_i = load("interpretation")
        N = emb_p.shape[0]

        # Time split logic
        years = df.get("publication_year")
        valid_years = []
        if years is not None:
            for y in years.values:
                try:
                    valid_years.append(int(float(y)))
                except:
                    valid_years.append(-1)
        valid_years = np.array(valid_years)

        unique_years = np.unique(valid_years[valid_years > 0])

        if len(unique_years) >= 2:
            split_year = np.percentile(valid_years[valid_years > 0], 70)
            hist_mask = (valid_years > 0) & (valid_years <= split_year)
            fut_mask = (valid_years > 0) & (valid_years > split_year)
        else:
            split_idx = int(0.7 * N)
            hist_mask = np.zeros(N, dtype=bool)
            hist_mask[:split_idx] = True
            fut_mask = ~hist_mask

        if hist_mask.sum() < 5 or fut_mask.sum() < 5:
            split_idx = N // 2
            hist_mask = np.zeros(N, dtype=bool)
            hist_mask[:split_idx] = True
            fut_mask = ~hist_mask

        I_hist = emb_i[hist_mask]
        P_hist = emb_p[hist_mask]
        I_fut = emb_i[fut_mask]
        P_fut = emb_p[fut_mask]

        reg = Ridge(alpha=1.0)
        reg.fit(I_hist, P_hist)
        P_pred = reg.predict(I_fut)

        n_clusters = min(10, N)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        kmeans.fit(emb_p)
        centroids = kmeans.cluster_centers_

        hist_labels = kmeans.predict(P_hist)
        fut_labels = kmeans.predict(P_fut)
        pred_labels = kmeans.predict(P_pred)

        c_hist = np.bincount(hist_labels, minlength=n_clusters)
        c_fut = np.bincount(fut_labels, minlength=n_clusters)
        c_pred = np.bincount(pred_labels, minlength=n_clusters)

        x = c_pred - c_hist
        y = c_fut - c_hist

        d_ij = cdist(centroids, centroids, metric="euclidean")
        with np.errstate(divide="ignore"):
            w = 1.0 / d_ij
        w[np.isinf(w)] = 0.0
        np.fill_diagonal(w, 0.0)

        row_sums = w.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        w = w / row_sums

        I_obs = compute_morans_i(x, y, w)

        # Centroid 2D projection
        centroids_2d = None
        proj_x_cols = [c for c in df.columns if c.startswith("proj_") and c.endswith("_x")]
        if proj_x_cols:
            col_x = proj_x_cols[0]
            col_y = col_x[:-2] + "_y"
            if col_y in df.columns:
                labels = kmeans.labels_
                centroids_2d = np.zeros((n_clusters, 2))
                for c_idx in range(n_clusters):
                    mask = (labels == c_idx)
                    if mask.any():
                        centroids_2d[c_idx, 0] = df.loc[mask, col_x].mean()
                        centroids_2d[c_idx, 1] = df.loc[mask, col_y].mean()
                    else:
                        centroids_2d[c_idx] = centroids[c_idx][:2]
        if centroids_2d is None:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            centroids_2d = pca.fit_transform(centroids)

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        z_x = x - x_mean
        z_y = y - y_mean
        x_std = np.std(x)
        y_std = np.std(y)
        z_x_std = z_x / x_std if x_std > 0 else z_x
        z_y_std = z_y / y_std if y_std > 0 else z_y
        lag_z_y = w @ z_y
        lag_z_y_std = w @ z_y_std

        # On-the-fly p-value computation
        w_edel = compute_wasserstein(P_pred, P_fut)
        w_baseline = compute_wasserstein(P_hist, P_fut)
        obs_gain = w_baseline - w_edel

        n_hist = hist_mask.sum()
        B_h3 = 100
        rng = np.random.default_rng(42)
        shuf_gains = []
        for _ in range(B_h3):
            shuf_idx = rng.permutation(N)
            hist_idx_b = shuf_idx[:n_hist]
            fut_idx_b = shuf_idx[n_hist:]
            
            I_hist_b = emb_i[hist_idx_b]
            P_hist_b = emb_p[hist_idx_b]
            I_fut_b = emb_i[fut_idx_b]
            P_fut_b = emb_p[fut_idx_b]
            
            reg_b = Ridge(alpha=1.0)
            reg_b.fit(I_hist_b, P_hist_b)
            P_pred_b = reg_b.predict(I_fut_b)
            
            w_edel_b = compute_wasserstein(P_pred_b, P_fut_b)
            w_baseline_b = compute_wasserstein(P_hist_b, P_fut_b)
            shuf_gains.append(w_baseline_b - w_edel_b)
            
        shuf_gains = np.array(shuf_gains)
        h3_gain_pvalue = float((1 + np.sum(shuf_gains >= obs_gain)) / (B_h3 + 1))

        return {
            "x_raw": x,
            "y_raw": y,
            "z_x": z_x,
            "z_y": z_y,
            "lag_z_y": lag_z_y,
            "z_x_std": z_x_std,
            "lag_z_y_std": lag_z_y_std,
            "centroids_2d": centroids_2d,
            "moran_i": float(I_obs),
            "h3_gain_pvalue": h3_gain_pvalue,
        }
    except Exception as ex:
        logger.error(f"Failed to compute Moran features on the fly: {ex}", exc_info=True)
        return None

def _build_moran_scatterplot(moran_feat: dict, title: str) -> go.Figure:
    """Build a Bivariate Moran Scatterplot (z_x vs spatial lag of z_y) with regression line."""
    fig = go.Figure()
    
    z_x_std = np.array(moran_feat["z_x_std"])
    lag_z_y_std = np.array(moran_feat["lag_z_y_std"])
    moran_i = moran_feat["moran_i"]
    
    # Grid reference lines
    fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="lightgray")
    fig.add_vline(x=0, line_width=1, line_dash="solid", line_color="lightgray")
    
    # Scatter points representing semantic regions (clusters)
    fig.add_trace(go.Scatter(
        x=z_x_std,
        y=lag_z_y_std,
        mode="markers+text",
        text=[f"C{i}" for i in range(len(z_x_std))],
        textposition="top center",
        marker=dict(
            size=12,
            color="#007bff",
            line=dict(width=1, color="white")
        ),
        name="Clusters",
        hoverinfo="text",
        hovertext=[f"Cluster {i}<br>Std Pred Change (z_x): {z:.3f}<br>Lag Std Obs Change (W z_y): {l:.3f}" for i, (z, l) in enumerate(zip(z_x_std, lag_z_y_std))]
    ))
    
    # Regression line: y = Moran's I * x
    if len(z_x_std) > 0:
        x_min, x_max = min(z_x_std), max(z_x_std)
        # Pad slightly
        x_line = np.array([x_min - 0.5, x_max + 0.5])
        y_line = moran_i * x_line
        
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color="#dc3545", dash="dash", width=2),
            name=f"Slope = {moran_i:.4f}"
        ))
        
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#333")),
        xaxis_title="Standardized Predicted Change (z_x)",
        yaxis_title="Spatial Lag of Standardized Observed Change (W z_y)",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"),
        template="plotly_white",
        margin=dict(l=60, r=40, t=50, b=45),
        height=320,
    )
    return fig

def _build_spatial_density_map(moran_feat: dict, title: str, use_predicted: bool) -> go.Figure:
    """Build a 2D projection density change map of the 10 cluster centroids."""
    coords = np.array(moran_feat["centroids_2d"])
    vals = np.array(moran_feat["x_raw"] if use_predicted else moran_feat["y_raw"])
    
    fig = go.Figure(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers+text",
        text=[f"C{i}" for i in range(len(vals))],
        textposition="top center",
        textfont=dict(color="#333", size=10, family="Outfit, sans-serif"),
        marker=dict(
            size=18,
            color=vals,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(
                title=dict(text="Change", font=dict(size=10)),
                thickness=12,
                len=0.8,
                tickfont=dict(size=8)
            ),
            cmid=0,
            showscale=True,
            line=dict(width=1.5, color="#444")
        ),
        hoverinfo="text",
        hovertext=[f"Cluster {i}<br>Density Change: {v:.1f}" for i, v in enumerate(vals)]
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#555")),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=280,
    )
    return fig

def register_hypothesis_callbacks(app: Dash, base_path: Path) -> None:
    
    # Callback to populate the run selection dropdowns from registry
    @app.callback(
        [Output("hyp-hypothesis-select", "options"),
         Output("hyp-control-select", "options")],
        [Input("artifact-update-store", "data")]
    )
    def update_run_selectors(update_val):
        try:
            registry = load_registry(base_path)
            options = [{"label": exp["experiment_id"], "value": exp["experiment_id"]} for exp in registry]
            return options, options
        except Exception as e:
            logger.error(f"Error updating run selectors: {e}")
            return [], []

    # Callback to run the hypothesis tests and generate the report
    @app.callback(
        Output("hyp-report-container", "children"),
        [Input("btn-run-hyp-tests", "n_clicks")],
        [State("hyp-hypothesis-select", "value"),
         State("hyp-control-select", "value")],
        prevent_initial_call=True
    )
    def generate_hypothesis_report(n_clicks, hyp_id, ctrl_id):
        if not n_clicks:
            return html.Div("Select a Hypothesis run and a Control run, then click 'Run Hypothesis Tests' to generate the validation report.", className="text-center text-muted p-5 my-4 border rounded bg-light")
            
        if not hyp_id or not ctrl_id:
            return dbc.Alert("Please select both a Hypothesis run and a Control run.", color="warning", className="mt-3")

        try:
            # Load results dataframe (which contains individual run metrics)
            df = get_results_df(base_path)
            if df.empty:
                return dbc.Alert("No analyzed results found. Please rebuild the results cache on the 'Metrics Analysis' tab.", color="danger", className="mt-3")

            hyp_rows = df[df["experiment_id"] == hyp_id]
            ctrl_rows = df[df["experiment_id"] == ctrl_id]

            if hyp_rows.empty or ctrl_rows.empty:
                return dbc.Alert("One or both selected runs have not been analyzed yet. Please rebuild the cache on the 'Metrics Analysis' tab first.", color="danger", className="mt-3")

            hyp_metrics = hyp_rows.iloc[0].to_dict()
            ctrl_metrics = ctrl_rows.iloc[0].to_dict()

            # Load features for direct comparison
            feat_hyp = _load_features(hyp_id, base_path)
            feat_ctrl = _load_features(ctrl_id, base_path)

            # 1. Compare feature distributions directly (cross-experiment)
            direct_h1_results = {}
            if feat_hyp and feat_ctrl:
                dims = [
                    ("norm_pm_dist", "Norm P-M"),
                    ("norm_mf_dist", "Norm M-F"),
                    ("norm_fi_dist", "Norm F-I"),
                    ("norm_pf_dist", "Norm P-F"),
                    ("norm_pi_dist", "Norm P-I"),
                    ("norm_mi_dist", "Norm M-I"),
                    ("cos_pm_mf_dist", "Cosine (PM, MF)"),
                    ("cos_pm_fi_dist", "Cosine (PM, FI)"),
                    ("cos_mf_fi_dist", "Cosine (MF, FI)"),
                ]
                for key, label in dims:
                    if key in feat_hyp and key in feat_ctrl:
                        res = ks_2samp(feat_hyp[key], feat_ctrl[key])
                        direct_h1_results[label] = {
                            "stat": float(res.statistic),
                            "pvalue": float(res.pvalue)
                        }

            # Determine if H1, H2, H3 are supported
            # H1: Supported if energy distance p-value < 0.05 (primary multivariate test)
            h1_pval = hyp_metrics.get("h1_energy_pvalue", 1.0)
            h1_supported = h1_pval < 0.05

            # H2: Supported if local neighborhoods show significant clustering
            h2_keys = ["pm", "pf", "pi", "mp", "mf", "mi", "fp", "fm", "fi", "ip", "im", "if"]
            h2_pvals = [hyp_metrics.get(f"h2_pvalue_{k}", 1.0) for k in h2_keys]
            h2_supported = sum(p < 0.05 for p in h2_pvals) >= 3

            # H3: Supported if predictive gain is positive AND the temporal permutation p-value < 0.05
            # NOTE: h3_gain_pvalue may not be in the cache for older experiments; h3_supported is
            # finalized after the on-the-fly fallback resolves hyp_gain_p (see below).
            h3_supported = False  # will be recomputed after fallback resolves hyp_gain_p

            report_children = []

            def make_badge(supported: bool):
                if supported:
                    return dbc.Badge("SUPPORTED", color="success", className="px-2 py-1")
                else:
                    return dbc.Badge("NOT SUPPORTED", color="danger", className="px-2 py-1")


            # H1 Details Section
            h1_energy_rows = [
                html.Tr([
                    html.Td("Energy Distance (6D multivariate)"),
                    html.Td(f"{hyp_metrics.get('h1_energy_stat', 0.0):.4f}"),
                    html.Td(f"{hyp_metrics.get('h1_energy_pvalue', 1.0):.4g}"),
                    html.Td(f"{ctrl_metrics.get('h1_energy_stat', 0.0):.4f}"),
                    html.Td(f"{ctrl_metrics.get('h1_energy_pvalue', 1.0):.4g}"),
                ]),
            ]

            h1_w_rows = []
            for k in ["norm_pm", "norm_mf", "norm_fi", "cos_pm_mf", "cos_pm_fi", "cos_mf_fi"]:
                h1_w_rows.append(html.Tr([
                    html.Td(k.upper().replace("_", " ")),
                    html.Td(f"{hyp_metrics.get(f'h1_w_{k}', 0.0):.4f}"),
                    html.Td(f"{ctrl_metrics.get(f'h1_w_{k}', 0.0):.4f}"),
                ]))

            h1_energy_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Test"),
                    html.Th("Hyp Stat"),
                    html.Th("Hyp p-val"),
                    html.Th("Ctrl Stat"),
                    html.Th("Ctrl p-val"),
                ])),
                html.Tbody(h1_energy_rows)
            ], className="table table-sm small border-primary")

            h1_w_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Feature Dimension"),
                    html.Th("Hyp W-dist (effect size)"),
                    html.Th("Ctrl W-dist (effect size)"),
                ])),
                html.Tbody(h1_w_rows)
            ], className="table table-striped table-sm small border mt-3")

            h1_ks_rows = []
            for k in ["norm_pm", "norm_mf", "norm_fi", "cos_pm_mf", "cos_pm_fi", "cos_mf_fi"]:
                h1_ks_rows.append(html.Tr([
                    html.Td(k.upper().replace("_", " ")),
                    html.Td(f"{hyp_metrics.get(f'h1_ks_stat_{k}', 0.0):.4f}"),
                    html.Td(f"{hyp_metrics.get(f'h1_ks_pvalue_{k}', 1.0):.4g}"),
                    html.Td(f"{ctrl_metrics.get(f'h1_ks_stat_{k}', 0.0):.4f}"),
                    html.Td(f"{ctrl_metrics.get(f'h1_ks_pvalue_{k}', 1.0):.4g}"),
                ]))

            h1_ks_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Feature Dimension"),
                    html.Th("Hyp KS Stat"),
                    html.Th("Hyp p-val"),
                    html.Th("Ctrl KS Stat"),
                    html.Th("Ctrl p-val"),
                ])),
                html.Tbody(h1_ks_rows)
            ], className="table table-sm small border-secondary mt-1")

            # Direct Comparison
            direct_h1_rows = []
            for label, val in direct_h1_results.items():
                direct_h1_rows.append(html.Tr([
                    html.Td(label),
                    html.Td(f"{val['stat']:.4f}"),
                    html.Td(f"{val['pvalue']:.4g}"),
                ]))
            direct_h1_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Feature Dimension"),
                    html.Th("KS Statistic (Hyp vs Ctrl)"),
                    html.Th("p-value"),
                ])),
                html.Tbody(direct_h1_rows)
            ], className="table table-bordered table-sm small border mt-3")

            report_children.append(html.Div([
                html.H5("H1: Structural Transition", className="mt-4 text-primary"),
                html.P("Tests whether epistemic trajectories have a structured coupling. The primary test is a multivariate energy distance on the 6D distribution of transition features (3 sequential operator norms + 3 pairwise cosines). Per-edge Wasserstein distances provide interpretable effect sizes. KS tests are secondary diagnostics.", className="small text-muted"),

                html.H6("Primary: Multivariate Energy Distance", className="mt-3"),
                html.P("Energy distance D² between observed and shuffled 6D feature distributions. Reject H0 (p < 0.05) if trajectories are structured.", className="small text-muted"),
                h1_energy_table,

                html.H6("Per-Edge Wasserstein Effect Sizes", className="mt-3"),
                html.P("1D Wasserstein distance between observed and shuffled for each sequential feature. Larger values indicate stronger structuring of that transition.", className="small text-muted"),
                h1_w_table,

                html.H6("KS Diagnostics (Secondary)", className="mt-3"),
                html.P("Two-sample KS tests for each feature individually.", className="small text-muted"),
                h1_ks_table,

                html.H6("Direct Comparison (Hypothesis vs. Control)", className="mt-3"),
                html.P("Performs a two-sample KS test directly comparing the trajectory distributions of both runs across all 9 feature dimensions (6 edge norms + 3 cosines).", className="small text-muted"),
                direct_h1_table if direct_h1_results else html.P("No feature distributions available for direct comparison.", className="text-muted small")
            ]))

            # H2 Details Section
            h2_transitions = [
                ("pm", "D(M|p)"),
                ("pf", "D(F|p)"),
                ("pi", "D(I|p)"),
                ("mp", "D(P|m)"),
                ("mf", "D(F|m)"),
                ("mi", "D(I|m)"),
                ("fp", "D(P|f)"),
                ("fm", "D(M|f)"),
                ("fi", "D(I|f)"),
                ("ip", "D(P|i)"),
                ("im", "D(M|i)"),
                ("if", "D(F|i)"),
            ]
            h2_rows = []
            for k, label in h2_transitions:
                hyp_p = hyp_metrics.get(f'h2_pvalue_{k}', 1.0)
                ctrl_p = ctrl_metrics.get(f'h2_pvalue_{k}', 1.0)
                hyp_z = hyp_metrics.get(f'h2_z_{k}', 0.0)
                ctrl_z = ctrl_metrics.get(f'h2_z_{k}', 0.0)
                h2_rows.append(html.Tr([
                    html.Td(label),
                    html.Td(f"{hyp_metrics.get(f'h2_w_dist_{k}', 0.0):.4f}"),
                    html.Td(f"{hyp_p:.4f}", className="text-success fw-bold" if hyp_p < 0.05 else ""),
                    html.Td(f"{hyp_z:+.2f}", className="text-success fw-bold" if hyp_z > 0 else "text-muted"),
                    html.Td(f"{ctrl_metrics.get(f'h2_w_dist_{k}', 0.0):.4f}"),
                    html.Td(f"{ctrl_p:.4f}", className="text-success fw-bold" if ctrl_p < 0.05 else ""),
                    html.Td(f"{ctrl_z:+.2f}", className="text-success fw-bold" if ctrl_z > 0 else "text-muted"),
                ]))
            h2_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Transition Operator"),
                    html.Th("Hyp W_obs"),
                    html.Th("Hyp p-val"),
                    html.Th("Hyp z-score"),
                    html.Th("Ctrl W_obs"),
                    html.Th("Ctrl p-val"),
                    html.Th("Ctrl z-score"),
                ])),
                html.Tbody(h2_rows)
            ], className="table table-striped table-sm small border")

            # H2b Details Section
            h2b_pairs = [
                ("pm", "P \u2194 M"),
                ("mf", "M \u2194 F"),
                ("fi", "F \u2194 I"),
                ("pf", "P \u2194 F"),
                ("pi", "P \u2194 I"),
                ("mi", "M \u2194 I"),
            ]
            h2b_rows = []
            for k, label in h2b_pairs:
                hyp_ef = hyp_metrics.get(f'h2b_entropy_forward_{k}', 0.0)
                hyp_er = hyp_metrics.get(f'h2b_entropy_reverse_{k}', 0.0)
                hyp_bf = hyp_metrics.get(f'h2b_branching_forward_{k}', 1.0)
                hyp_br = hyp_metrics.get(f'h2b_branching_reverse_{k}', 1.0)
                hyp_diff = hyp_metrics.get(f'h2b_diff_{k}', 0.0)
                hyp_p = hyp_metrics.get(f'h2b_pvalue_{k}', 1.0)

                ctrl_ef = ctrl_metrics.get(f'h2b_entropy_forward_{k}', 0.0)
                ctrl_er = ctrl_metrics.get(f'h2b_entropy_reverse_{k}', 0.0)
                ctrl_bf = ctrl_metrics.get(f'h2b_branching_forward_{k}', 1.0)
                ctrl_br = ctrl_metrics.get(f'h2b_branching_reverse_{k}', 1.0)
                ctrl_diff = ctrl_metrics.get(f'h2b_diff_{k}', 0.0)
                ctrl_p = ctrl_metrics.get(f'h2b_pvalue_{k}', 1.0)

                h2b_rows.append(html.Tr([
                    html.Td(label),
                    html.Td(f"{hyp_ef:.2f} ({hyp_bf:.1f})"),
                    html.Td(f"{hyp_er:.2f} ({hyp_br:.1f})"),
                    html.Td(f"{hyp_diff:+.3f}", className="text-success fw-bold" if hyp_p < 0.05 else ""),
                    html.Td(f"{hyp_p:.4f}", className="text-success fw-bold" if hyp_p < 0.05 else ""),
                    html.Td(f"{ctrl_ef:.2f} ({ctrl_bf:.1f})"),
                    html.Td(f"{ctrl_er:.2f} ({ctrl_br:.1f})"),
                    html.Td(f"{ctrl_diff:+.3f}", className="text-success fw-bold" if ctrl_p < 0.05 else ""),
                    html.Td(f"{ctrl_p:.4f}", className="text-success fw-bold" if ctrl_p < 0.05 else ""),
                ]))

            h2b_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Pair"),
                    html.Th("Hyp Fwd H (B)"),
                    html.Th("Hyp Rev H (B)"),
                    html.Th("Hyp Diff"),
                    html.Th("Hyp p-val"),
                    html.Th("Ctrl Fwd H (B)"),
                    html.Th("Ctrl Rev H (B)"),
                    html.Th("Ctrl Diff"),
                    html.Th("Ctrl p-val"),
                ])),
                html.Tbody(h2b_rows)
            ], className="table table-striped table-sm small border")

            report_children.append(html.Div([
                html.H5("H2: Local Transition Organization", className="mt-4 text-primary"),
                html.P("Tests if local neighborhood transitions are statistically constrained/clustered. A significant permuted p-value (< 0.05) supports localized organizational constraints. Asymmetry metrics are provided below as a secondary characterization of the transition structure.", className="small text-muted"),
                h2_table,
                html.H6("Transition Asymmetry Metrics (Secondary)", className="mt-3 text-secondary"),
                html.P("Directionality bias of local transitions. Entropy H represents average information dispersion; branching B is the average number of target locations reached.", className="small text-muted"),
                h2b_table
            ]))

            # Retrieve / compute H3 Moran's I features for both runs
            moran_hyp = _get_or_compute_h3_moran_features(hyp_id, feat_hyp, base_path)
            moran_ctrl = _get_or_compute_h3_moran_features(ctrl_id, feat_ctrl, base_path)

            # Get p-value from cached metrics, or fall back to on-the-fly computed p-value
            hyp_gain_p = hyp_metrics.get('h3_gain_pvalue')
            if hyp_gain_p is None and moran_hyp:
                hyp_gain_p = moran_hyp.get('h3_gain_pvalue')
            if hyp_gain_p is None:
                hyp_gain_p = 1.0

            ctrl_gain_p = ctrl_metrics.get('h3_gain_pvalue')
            if ctrl_gain_p is None and moran_ctrl:
                ctrl_gain_p = moran_ctrl.get('h3_gain_pvalue')
            if ctrl_gain_p is None:
                ctrl_gain_p = 1.0

            # Finalize h3_supported now that hyp_gain_p is fully resolved (incl. fallback)
            h3_supported = (hyp_metrics.get("h3_predictive_gain", -1) > 0) and (hyp_gain_p < 0.05)

            # Validation Dashboard Summary Card (built here so h3_supported is fully resolved)
            report_children.insert(0, html.Div([
                html.H4("Validation Summary Dashboard", className="border-bottom pb-2 mb-3"),
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H5("H1: Structural Shift", className="card-title"),
                            html.Div([make_badge(h1_supported)]),
                            html.P("Trajectories differ significantly from random aspect-shuffling.", className="small text-muted mt-2 mb-0")
                        ])
                    ], color="success" if h1_supported else "light", outline=True), md=4),
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H5("H2: Local Organization", className="card-title"),
                            html.Div([make_badge(h2_supported)]),
                            html.P("Transitions show neighborhood-level spatial organization and directionality bias.", className="small text-muted mt-2 mb-0")
                        ])
                    ], color="success" if h2_supported else "light", outline=True), md=4),
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H5("H3: Predictive Power", className="card-title"),
                            html.Div([make_badge(h3_supported)]),
                            html.P("Historical model predicts future locations better than persistence (gain > 0, p < 0.05).", className="small text-muted mt-2 mb-0")
                        ])
                    ], color="success" if h3_supported else "light", outline=True), md=4),
                ], className="mb-4")
            ]))
            h3_rows = [
                html.Tr([
                    html.Td("Global W_EDEL (Wasserstein prediction error)"),
                    html.Td(f"{hyp_metrics.get('h3_w_edel', 0.0):.4f}"),
                    html.Td(f"{ctrl_metrics.get('h3_w_edel', 0.0):.4f}"),
                ]),
                html.Tr([
                    html.Td("Global W_baseline (persistence prediction error)"),
                    html.Td(f"{hyp_metrics.get('h3_w_baseline', 0.0):.4f}"),
                    html.Td(f"{ctrl_metrics.get('h3_w_baseline', 0.0):.4f}"),
                ]),
                html.Tr([
                    html.Td("Predictive Gain (W_baseline - W_EDEL)"),
                    html.Td(f"{hyp_metrics.get('h3_predictive_gain', 0.0):.4f}", className="text-success fw-bold" if hyp_metrics.get('h3_predictive_gain', 0) > 0 else "text-danger"),
                    html.Td(f"{ctrl_metrics.get('h3_predictive_gain', 0.0):.4f}", className="text-success fw-bold" if ctrl_metrics.get('h3_predictive_gain', 0) > 0 else "text-danger"),
                ]),
                html.Tr([
                    html.Td("Predictive Gain Significance (p-value)"),
                    html.Td(f"{hyp_gain_p:.4f}", className="text-success fw-bold" if hyp_gain_p < 0.05 else ""),
                    html.Td(f"{ctrl_gain_p:.4f}"),
                ]),
                html.Tr([
                    html.Td(html.Span(["Bivariate Moran's I ", html.Em("(exploratory)", className="text-muted")])),
                    html.Td(f"{hyp_metrics.get('h3_moran_i', 0.0):.4f}"),
                    html.Td(f"{ctrl_metrics.get('h3_moran_i', 0.0):.4f}"),
                ]),
                html.Tr([
                    html.Td(html.Span(["Moran's I Significance ", html.Em("(exploratory)", className="text-muted")])),
                    html.Td(f"{hyp_metrics.get('h3_moran_pvalue', 1.0):.4f}"),
                    html.Td(f"{ctrl_metrics.get('h3_moran_pvalue', 1.0):.4f}"),
                ]),
            ]
            h3_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Predictive Validation Metric"),
                    html.Th("Hypothesis Run"),
                    html.Th("Control Run"),
                ])),
                html.Tbody(h3_rows)
            ], className="table table-striped table-sm small border")

            h3_visuals = []
            if moran_hyp and moran_ctrl:
                # 1. Bivariate Moran Scatterplots Row
                fig_hyp_scatter = _build_moran_scatterplot(moran_hyp, f"Hypothesis Run ({hyp_id}) - Moran Scatterplot")
                fig_ctrl_scatter = _build_moran_scatterplot(moran_ctrl, f"Control Run ({ctrl_id}) - Moran Scatterplot")
                
                scatter_row = dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_hyp_scatter), width=6),
                    dbc.Col(dcc.Graph(figure=fig_ctrl_scatter), width=6),
                ], className="mb-4")

                # 2. Spatial Density Maps Rows
                fig_hyp_pred = _build_spatial_density_map(moran_hyp, "Predicted Density Change", use_predicted=True)
                fig_hyp_obs = _build_spatial_density_map(moran_hyp, "Observed Density Change", use_predicted=False)
                fig_ctrl_pred = _build_spatial_density_map(moran_ctrl, "Predicted Density Change", use_predicted=True)
                fig_ctrl_obs = _build_spatial_density_map(moran_ctrl, "Observed Density Change", use_predicted=False)

                spatial_row = dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("Hypothesis Spatial Density Maps", className="text-center fw-bold mb-2 text-secondary", style={"fontSize": "0.9rem"}),
                            dbc.Row([
                                dbc.Col(dcc.Graph(figure=fig_hyp_pred), width=6),
                                dbc.Col(dcc.Graph(figure=fig_hyp_obs), width=6),
                            ])
                        ], className="p-2 border rounded bg-light mb-3")
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H6("Control Spatial Density Maps", className="text-center fw-bold mb-2 text-secondary", style={"fontSize": "0.9rem"}),
                            dbc.Row([
                                dbc.Col(dcc.Graph(figure=fig_ctrl_pred), width=6),
                                dbc.Col(dcc.Graph(figure=fig_ctrl_obs), width=6),
                            ])
                        ], className="p-2 border rounded bg-light mb-3")
                    ], width=6),
                ])

                h3_visuals = [
                    html.H6("Spatial Predictive Alignment & Density Maps", className="mt-4 mb-3 text-secondary"),
                    scatter_row,
                    spatial_row
                ]

            report_children.append(html.Div([
                html.H5("H3: Predictive Capacity Details", className="mt-4 text-primary"),
                html.P("Hypothesis 3 tests forecasting capacity using a 70/30 time split. H3 is supported when the predictive gain is positive (gain > 0) and statistically significant (temporal permutation p-value < 0.05). Bivariate Moran's I is shown for exploratory spatial interpretation only.", className="small text-muted"),
                h3_table,
                *h3_visuals
            ]))

            return html.Div(report_children)

        except Exception as e:
            logger.error(f"Error generating hypothesis report: {e}", exc_info=True)
            return dbc.Alert(f"An error occurred while compiling the report: {e}", color="danger", className="mt-3")
