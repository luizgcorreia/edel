"""Callbacks for the Robustness Tests Dashboard Panel."""

import logging
from pathlib import Path
from dash import Dash, Input, Output, State, html, dcc, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from edel.experiments.runner import load_registry
from edel.io.artifact import load_artifact
from edel.robustness.registry import ROBUSTNESS_REGISTRY, get_test
from edel.robustness.runner import run_robustness_sweep
from edel.robustness.cache import load_robustness_result, save_robustness_result
from edel.pipeline.projection import load_embeddings_to_matrix
from edel.io.llm import get_llm_client
from sklearn.preprocessing import normalize as sk_normalize

logger = logging.getLogger(__name__)

_DATASET_CACHE = {}


def register_robustness_callbacks(app: Dash, base_path: Path) -> None:
    
    # 1. Populate Dropdowns & Test Checklist
    @app.callback(
        [Output("rob-experiment-select", "options"),
         Output("rob-compare-experiment-select", "options"),
         Output("rob-test-select", "options")],
        [Input("artifact-update-store", "data")]
    )
    def update_selectors(update_val):
        try:
            registry = load_registry(base_path)
            exp_options = [{"label": exp["experiment_id"], "value": exp["experiment_id"]} for exp in registry]
            
            test_options = [
                {"label": f"{test.label} [{test.priority}]", "value": test.name}
                for test in ROBUSTNESS_REGISTRY
            ]
            
            return exp_options, exp_options, test_options
        except Exception as e:
            logger.error(f"Error updating robustness selectors: {e}")
            return [], [], []

    # 2. Document Selection & Autocomplete Search (Synchronized)
    @app.callback(
        [Output("rob-doc-search", "options"),
         Output("rob-doc-search", "value"),
         Output("rob-selected-count", "children"),
         Output("rob-selected-ids-store", "data")],
        [Input("rob-experiment-select", "value"),
         Input("btn-rob-sample", "n_clicks"),
         Input("rob-doc-search", "search_value"),
         Input("rob-doc-search", "value")],
        [State("rob-sample-size", "value")]
    )
    def sync_document_selection(exp_id, n_clicks, search_value, dropdown_values, sample_size):
        ctx = callback_context
        if not exp_id:
            return [], [], "No experiment selected.", []
            
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        
        # Lazy load/cache the embedding dataframe
        cache_key = f"emb_{exp_id}"
        if cache_key not in _DATASET_CACHE:
            try:
                registry = load_registry(base_path)
                record = next((r for r in registry if r["experiment_id"] == exp_id), None)
                if not record:
                    return [], [], "Experiment not found.", []
                df = load_artifact(record["artifact_refs"]["embedding"])
                _DATASET_CACHE[cache_key] = df
            except Exception as e:
                return [], [], f"Error loading experiment data: {e}", []
                
        df = _DATASET_CACHE.get(cache_key)
        if df is None or df.empty:
            return [], [], "Empty or missing embedding artifact.", []
            
        def build_options(matches_df, current_vals):
            opts = [{"label": f"{row.get('title', 'Unknown')} ({idx})", "value": str(idx)} for idx, row in matches_df.iterrows()]
            if current_vals:
                for val in current_vals:
                    val_str = str(val)
                    if val_str not in [o["value"] for o in opts]:
                        try:
                            idx_lookup = int(val) if isinstance(df.index, pd.RangeIndex) or pd.api.types.is_integer_dtype(df.index) else val
                        except ValueError:
                            idx_lookup = val
                        if idx_lookup in df.index:
                            title = df.loc[idx_lookup].get("title", "Selected Paper")
                            opts.insert(0, {"label": f"{title} ({val_str})", "value": val_str})
            return opts

        if triggered_id == "rob-experiment-select":
            # Default to sampling 10 papers when selecting an experiment
            actual_sample_size = min(10, len(df))
            sampled_df = df.sample(n=actual_sample_size, random_state=42)
            doc_ids = [str(idx) for idx in sampled_df.index.tolist()]
            default_matches = df.head(20)
            options = build_options(default_matches, doc_ids)
            return options, doc_ids, f"{len(doc_ids)} documents selected.", doc_ids
            
        elif triggered_id == "btn-rob-sample":
            actual_sample_size = sample_size if sample_size and sample_size > 0 else 10
            actual_sample_size = min(actual_sample_size, len(df))
            sampled_df = df.sample(n=actual_sample_size, random_state=42)
            doc_ids = [str(idx) for idx in sampled_df.index.tolist()]
            default_matches = df.head(20)
            options = build_options(default_matches, doc_ids)
            return options, doc_ids, f"{len(doc_ids)} documents selected.", doc_ids
            
        elif triggered_id == "rob-doc-search" and ctx.triggered[0]["prop_id"].split(".")[1] == "search_value":
            # Autocomplete search
            if search_value and len(search_value) >= 2:
                search_value_low = search_value.lower()
                mask = pd.Series(False, index=df.index)
                for col in ["title", "abstract_text", "problem", "method"]:
                    if col in df.columns:
                        mask |= df[col].astype(str).str.lower().str.contains(search_value_low, na=False)
                mask |= df.index.astype(str).str.lower().str.contains(search_value_low, na=False)
                matches = df[mask].head(50)
            else:
                matches = df.head(20)
                
            options = build_options(matches, dropdown_values)
            return options, no_update, no_update, no_update
            
        else:
            # Dropdown value manually updated
            doc_ids = [str(v) for v in dropdown_values] if dropdown_values else []
            default_matches = df.head(20)
            options = build_options(default_matches, doc_ids)
            return options, no_update, f"{len(doc_ids)} documents selected.", doc_ids

    # 3. Run Sweep & Render
    @app.callback(
        Output("rob-report-container", "children"),
        [Input("btn-run-robustness", "n_clicks")],
        [State("rob-experiment-select", "value"),
         State("rob-compare-experiment-select", "value"),
         State("rob-selected-ids-store", "data"),
         State("rob-test-select", "value"),
         State("rob-n-max", "value"),
         State("rob-n-step", "value"),
         State("rob-plot-options", "value")],
        prevent_initial_call=True
    )
    def run_and_render_robustness(n_clicks, exp_id, comp_exp_id, doc_ids, test_names, n_max, n_step, plot_options):
        if not n_clicks:
            return html.Div("Select configuration and run tests.")
        if not exp_id or not doc_ids or not test_names:
            return dbc.Alert("Please select an experiment, sample some documents, and select at least one test.", color="warning")
            
        show_std = "std" in (plot_options or [])
        show_traces = "traces" in (plot_options or [])
        
        n_values = list(range(0, n_max + 1, n_step))
        
        registry = load_registry(base_path)
        
        def extract_lexicon_from_df(df):
            import re
            words = []
            text_cols = ["abstract_text", "problem", "method", "finding", "interpretation"]
            for col in text_cols:
                if col in df.columns:
                    for text in df[col].dropna():
                        tokens = re.findall(r'\b[a-zA-Z]{4,15}\b', str(text).lower())
                        words.extend(tokens)
            stop_words = {"the", "and", "of", "to", "in", "is", "that", "it", "was", "for", "on", "are", "as", "with", "his", "they", "i", "at", "be", "this", "have", "from", "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we", "when", "your", "can", "said", "there", "use", "an", "each", "which", "she", "do", "how", "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these", "so", "some", "her", "would", "make", "like", "him", "into", "time", "has", "look", "two", "more", "write", "go", "see", "number", "no", "way", "could", "people", "my", "than", "first", "water", "been", "call", "who", "oil", "its", "now", "find"}
            unique_words = list(set(words) - stop_words)
            return unique_words

        def run_sweep_for_exp(target_exp_id, target_doc_ids):
            record = next((r for r in registry if r["experiment_id"] == target_exp_id), None)
            if not record:
                raise ValueError(f"Experiment {target_exp_id} not found.")
                
            df_full = load_artifact(record["artifact_refs"]["embedding"])
            # Ensure we only use valid indices that exist in this df
            typed_ids = []
            is_int_index = isinstance(df_full.index, pd.RangeIndex) or pd.api.types.is_integer_dtype(df_full.index)
            for did in target_doc_ids:
                try:
                    typed_ids.append(int(did) if is_int_index else str(did))
                except ValueError:
                    typed_ids.append(did)
            
            valid_ids = [did for did in typed_ids if did in df_full.index]
            df = df_full.loc[valid_ids].copy()
            
            if df.empty:
                raise ValueError(f"No valid documents found for {target_exp_id}.")
                
            config = record["config"]
            embed_cfg = config.get("embedding", {})
            dimensions = embed_cfg.get("n_dimensions", 1536)
            
            # Extract lexicons dynamically
            within_lexicon = extract_lexicon_from_df(df_full)
            out_lexicon = []
            other_records = [r for r in registry if r["experiment_id"] != target_exp_id]
            for o_rec in other_records:
                try:
                    o_df = load_artifact(o_rec["artifact_refs"]["embedding"])
                    if o_df is not None and not o_df.empty:
                        out_lexicon.extend(extract_lexicon_from_df(o_df))
                        if len(out_lexicon) >= 500:
                            break
                except Exception:
                    continue
            
            # Fallback if no other experiment or failed to load
            if not out_lexicon:
                out_lexicon = ["mitochondria", "photosynthesis", "derivative", "portfolio", "arbitrage", "cardiovascular", "pathogen", "genome", "liquidity", "antibiotic"]
            
            # Instantiate LLM client
            llm_client = get_llm_client(embed_cfg)
            
            # Create a mockable embed_fn that calls the LLMClient
            def embed_fn(texts: list[str]) -> list[list[float]]:
                return llm_client.generate_embedding(texts)
                
            results = {}
            for t_name in test_names:
                # Check cache first
                cached = load_robustness_result(target_exp_id, valid_ids, t_name, base_path)
                if cached and cached.get("n_values") == n_values:
                    results[t_name] = cached
                    continue
                    
                # Run sweep
                test_obj = get_test(t_name)
                res = run_robustness_sweep(
                    test_obj, 
                    df, 
                    embed_fn, 
                    dimensions, 
                    n_values, 
                    llm_client=llm_client,
                    within_lexicon=within_lexicon,
                    out_lexicon=out_lexicon
                )
                save_robustness_result(target_exp_id, valid_ids, t_name, res, base_path)
                results[t_name] = res
                
            return results
            
        try:
            # 1. Run for primary experiment
            primary_results = run_sweep_for_exp(exp_id, doc_ids)
            
            # 2. Run for comparison experiment (if selected)
            comp_results = None
            if comp_exp_id:
                comp_results = run_sweep_for_exp(comp_exp_id, doc_ids)
                
            # 3. Render Plots
            aspects = ["problem", "method", "finding", "interpretation"]
            colors = {"problem": "#1f77b4", "method": "#ff7f0e", "finding": "#2ca02c", "interpretation": "#d62728"}
            
            report_children = []
            
            for t_name in test_names:
                test_obj = get_test(t_name)
                p_res = primary_results[t_name]
                c_res = comp_results[t_name] if comp_results else None
                
                fig = go.Figure()
                
                # Plot Primary
                for aspect in aspects:
                    mean_disp = np.array(p_res["mean_displacement"][aspect])
                    std_disp = np.array(p_res["std_displacement"][aspect])
                    
                    fig.add_trace(go.Scatter(
                        x=n_values, y=mean_disp,
                        mode='lines+markers',
                        name=f"{exp_id} - {aspect}",
                        line=dict(color=colors[aspect], width=3),
                        legendgroup=aspect
                    ))
                    
                    if show_std:
                        fig.add_trace(go.Scatter(
                            x=n_values + n_values[::-1],
                            y=list(mean_disp + std_disp) + list(mean_disp - std_disp)[::-1],
                            fill='toself',
                            fillcolor=colors[aspect],
                            opacity=0.2,
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f"{aspect} ± 1 std",
                            legendgroup=aspect,
                            showlegend=False
                        ))
                        
                    if show_traces:
                        for doc_id, doc_vals in p_res["per_document"].items():
                            fig.add_trace(go.Scatter(
                                x=n_values, y=doc_vals[aspect],
                                mode='lines',
                                line=dict(color=colors[aspect], width=1),
                                opacity=0.1,
                                showlegend=False,
                                legendgroup=aspect
                            ))
                            
                # Plot Comparison
                if c_res:
                    for aspect in aspects:
                        mean_disp = np.array(c_res["mean_displacement"][aspect])
                        fig.add_trace(go.Scatter(
                            x=n_values, y=mean_disp,
                            mode='lines+markers',
                            name=f"{comp_exp_id} - {aspect}",
                            line=dict(color=colors[aspect], width=3, dash='dash'),
                            legendgroup=f"comp_{aspect}"
                        ))
                        
                fig.update_layout(
                    title=f"Displacement Sweep: {test_obj.label}",
                    xaxis_title="Perturbation Intensity (N)",
                    yaxis_title="Mean Cosine Displacement",
                    template="plotly_white",
                    height=400,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                
                report_children.append(html.Div([
                    dcc.Graph(figure=fig)
                ], className="mb-4 border rounded p-3 bg-white"))
                
            return report_children
            
        except Exception as e:
            logger.error(f"Error running robustness tests: {e}", exc_info=True)
            return dbc.Alert(f"An error occurred: {str(e)}", color="danger")

    # 4. Run Structural Correlations
    @app.callback(
        Output("rob-structural-container", "children"),
        [Input("btn-run-structural", "n_clicks")],
        [State("rob-experiment-select", "value"),
         State("rob-struct-aspect1", "value"),
         State("rob-struct-aspect2", "value"),
         State("rob-struct-metric", "value")],
        prevent_initial_call=True
    )
    def run_structural_correlation(n_clicks, exp_id, aspect1, aspect2, metric):
        if not n_clicks:
            return html.Div()
        if not exp_id:
            return dbc.Alert("Please select an experiment.", color="warning")
        if aspect1 == aspect2:
            return dbc.Alert("Please select two different aspects to compare.", color="warning")
            
        try:
            registry = load_registry(base_path)
            record = next((r for r in registry if r["experiment_id"] == exp_id), None)
            if not record:
                return dbc.Alert("Experiment not found.", color="danger")
                
            df = load_artifact(record["artifact_refs"]["embedding"])
            
            # Check if columns exist
            for col in [aspect1, aspect2, f"{aspect1}_embedding", f"{aspect2}_embedding"]:
                if col not in df.columns:
                    return dbc.Alert(f"Missing required column: {col}", color="danger")
                    
            if metric == "sentence_count":
                from nltk.tokenize import sent_tokenize
                from edel.robustness.nlp import init_nltk
                init_nltk()
                
                def get_sent_len(text_series):
                    counts = []
                    for val in text_series.fillna("").astype(str):
                        if not val.strip():
                            counts.append(0)
                        else:
                            try:
                                counts.append(len(sent_tokenize(val)))
                            except Exception:
                                counts.append(len(val.split('.')))
                    return pd.Series(counts, index=text_series.index)
                
                len1 = get_sent_len(df[aspect1])
                len2 = get_sent_len(df[aspect2])
                len_diff = len1 - len2
                x_title = f"Sentence Count Diff: sents({aspect1}) - sents({aspect2})"
                title_prefix = "Sentence Count"
            elif metric == "pmfi_ratio":
                # Compute length ratio: len(aspect1) / len(aspect2)
                len1 = df[aspect1].fillna("").astype(str).str.split().str.len()
                len2 = df[aspect2].fillna("").astype(str).str.split().str.len()
                len2_safe = len2.replace(0, 1.0)
                len_diff = len1 / len2_safe
                x_title = f"Length Ratio: len({aspect1}) / len({aspect2})"
                title_prefix = "Length Ratio"
            elif metric == "descriptive_noun_ratio":
                from edel.robustness.nlp import tokenize_and_tag, init_nltk
                init_nltk()
                
                def get_ratio(text_series):
                    ratios = []
                    for val in text_series.fillna("").astype(str):
                        if not val.strip():
                            ratios.append(0.0)
                        else:
                            try:
                                tagged = tokenize_and_tag(val)
                                desc_count = sum(1 for _, tag in tagged if tag.startswith('J') or tag.startswith('R'))
                                noun_count = sum(1 for _, tag in tagged if tag.startswith('N'))
                                ratios.append(desc_count / (noun_count if noun_count > 0 else 1.0))
                            except Exception:
                                ratios.append(0.0)
                    return pd.Series(ratios, index=text_series.index)
                
                len1 = get_ratio(df[aspect1])
                len2 = get_ratio(df[aspect2])
                len_diff = len1 - len2
                x_title = f"Descriptive/Noun Ratio Diff: ratio({aspect1}) - ratio({aspect2})"
                title_prefix = "Descriptive/Noun Ratio"
            else:
                # Compute Length differences: len(aspect1) - len(aspect2)
                # Tokenize by simple split to approximate length
                len1 = df[aspect1].fillna("").astype(str).str.split().str.len()
                len2 = df[aspect2].fillna("").astype(str).str.split().str.len()
                len_diff = len1 - len2
                x_title = f"Length Diff: len({aspect1}) - len({aspect2})"
                title_prefix = "Length"
            
            # Compute ||m - p|| : L2 distance or Cosine distance
            dim = record["config"].get("embedding", {}).get("n_dimensions", 1536)
            emb1 = load_embeddings_to_matrix(df, f"{aspect1}_embedding", dim)
            emb2 = load_embeddings_to_matrix(df, f"{aspect2}_embedding", dim)
            
            emb1 = sk_normalize(emb1)
            emb2 = sk_normalize(emb2)
            
            # Cosine distance
            sims = np.sum(emb1 * emb2, axis=1)
            dist = 1.0 - np.clip(sims, -1.0, 1.0)
            
            # Plot
            fig = go.Figure(data=go.Scatter(
                x=len_diff,
                y=dist,
                mode='markers',
                marker=dict(
                    size=6,
                    opacity=0.6,
                    color=dist,
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            
            # Calculate correlation
            corr = np.corrcoef(len_diff, dist)[0, 1]
            
            fig.update_layout(
                title=f"{title_prefix} vs Displacement ({aspect1.title()} vs {aspect2.title()})<br><sup>Pearson r = {corr:.3f}</sup>",
                xaxis_title=x_title,
                yaxis_title=f"Cosine Displacement: ||{aspect1} - {aspect2}||",
                template="plotly_white",
                height=500
            )
            
            return dcc.Graph(figure=fig)
            
        except Exception as e:
            logger.error(f"Error computing structural correlations: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger")
