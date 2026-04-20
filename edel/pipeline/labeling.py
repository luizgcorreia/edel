"""Stage 7: Summarization and Labeling."""

from __future__ import annotations

import json
import logging
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from edel.io.llm import LLMClient, OpenAIClient

logger = logging.getLogger(__name__)


def run_labeling_stage(
    df: pd.DataFrame, field: pd.DataFrame, config: dict, llm_client: LLMClient
) -> Dict[str, Any]:
    """Orchestrate the labeling stage."""
    label_cfg = config.get("labeling", {})
    text_col = label_cfg.get("text_column", "abstract_text")
    topic = label_cfg.get("topic", None)
    language = label_cfg.get("language", "en")

    # Configuration for different sub-tasks
    axis_cfg = label_cfg.get("axis", {})
    cluster_cfg = label_cfg.get("clusters", {})

    results = {
        "clusters": {},
        "axes": []
    }

    # 1. Label Axes (Semantic contrast between extremes)
    if axis_cfg.get("enabled", True):
        method = axis_cfg.get("projection", "umap")
        n_samples = axis_cfg.get("n_samples", 5)
        print(f"Labeling axes for projection: {method}...")
        
        for axis_idx in [0, 1]:
            pos_block, neg_block = sample_axis_extremes(
                df, method, axis_idx, text_col, n_samples
            )
            if pos_block and neg_block:
                label_json = generate_axis_label(
                    llm_client, pos_block, neg_block, topic, axis_idx, method, language
                )
                results["axes"].append(label_json)

    # 2. Label Clusters (Knowledge domains)
    if cluster_cfg.get("enabled", True):
        cluster_keys = cluster_cfg.get("cluster_keys", [])
        n_samples = cluster_cfg.get("n_samples", 5)
        
        for key in cluster_keys:
            cluster_col = f"cluster_{key}"
            if cluster_col not in df.columns and cluster_col not in field.columns:
                print(f"Warning: Cluster column {cluster_col} not found. Skipping.")
                continue
            
            print(f"Labeling clusters for: {key}...")
            results["clusters"][key] = {}
            
            # Determine if it's a document-level or field-level cluster
            if cluster_col in df.columns:
                cluster_ids = get_cluster_ids(df, cluster_col)
                previous_labels = []
                for cid in cluster_ids:
                    block = sample_cluster_texts(df, cluster_col, text_col, cid, n_samples)
                    label_json = generate_cluster_label(
                        llm_client, block, key, topic, previous_labels, language
                    )
                    results["clusters"][key][int(cid)] = label_json
                    if "proposed_label" in label_json:
                        previous_labels.append(label_json["proposed_label"])
            else:
                # Field-level cluster (requires mapping back to documents)
                cluster_ids = get_cluster_ids(field, cluster_col)
                previous_labels = []
                for cid in cluster_ids:
                    block = sample_field_cluster_texts(df, field, cluster_col, text_col, cid, n_samples)
                    label_json = generate_cluster_label(
                        llm_client, block, key, topic, previous_labels, language
                    )
                    results["clusters"][key][int(cid)] = label_json
                    if "proposed_label" in label_json:
                        previous_labels.append(label_json["proposed_label"])

    return results


def sample_axis_extremes(
    df: pd.DataFrame, method: str, axis_idx: int, text_col: str, n_samples: int
) -> Tuple[str, str]:
    """Sample texts from the positive and negative extremes of a projection axis."""
    x_col = f"proj_problem_{method}_x"
    y_col = f"proj_problem_{method}_y"
    
    # Fallback for single mode
    if x_col not in df.columns:
        x_col = f"proj_{method}_x"
        y_col = f"proj_{method}_y"
        
    col = x_col if axis_idx == 0 else y_col
    if col not in df.columns:
        return "", ""

    top_pos = df.nlargest(n_samples, col)[text_col].dropna().tolist()
    top_neg = df.nsmallest(n_samples, col)[text_col].dropna().tolist()

    return "\n\n".join(top_pos), "\n\n".join(top_neg)


def sample_cluster_texts(
    df: pd.DataFrame, cluster_col: str, text_col: str, cluster_id: Any, n_samples: int
) -> str:
    """Sample texts from a specific cluster."""
    subset = df[df[cluster_col] == cluster_id]
    if subset.empty:
        return ""
        
    samples = (
        subset[text_col]
        .dropna()
        .sample(min(n_samples, len(subset)), random_state=42)
        .tolist()
    )
    return "\n\n".join(samples)


def sample_field_cluster_texts(
    df: pd.DataFrame, field: pd.DataFrame, cluster_col: str, text_col: str, cluster_id: Any, n_samples: int
) -> str:
    """Sample texts from documents belonging to a field-level cluster."""
    # Assuming the field index or cell_id maps to documents
    # In the legacy code, this is simplified. 
    # For now, we'll try to find docs that fall into cells with this cluster_id.
    
    if "cell_id" not in df.columns:
        return ""
        
    target_cells = field[field[cluster_col] == cluster_id]["cell_id"].tolist()
    subset = df[df["cell_id"].isin(target_cells)]
    
    if subset.empty:
        return ""
        
    samples = (
        subset[text_col]
        .dropna()
        .sample(min(n_samples, len(subset)), random_state=42)
        .tolist()
    )
    return "\n\n".join(samples)


def get_cluster_ids(df: pd.DataFrame, cluster_col: str) -> List[Any]:
    """Get unique cluster IDs, excluding noise (-1)."""
    ids = df[cluster_col].dropna().unique().tolist()
    return sorted([i for i in ids if i != -1])


def generate_cluster_label(
    client: LLMClient,
    block: str,
    cluster_type: str,
    topic: Optional[str],
    previous_labels: List[str],
    language: str,
) -> Dict[str, Any]:
    """Use LLM to generate a label for a cluster."""
    
    topic_instr = f" Assume the topic is {topic}." if topic else ""
    
    role_instr = ""
    if cluster_type in ["domain", "domains"]:
        role_instr = "These abstracts belong to the same scientific domain. Identify the research area."
    elif cluster_type in ["regime", "regimes"]:
        role_instr = "These abstracts belong to papers with similar epistemic transitions. Focus on methodological similarities."
    elif cluster_type in ["style", "styles"]:
        role_instr = "These abstracts belong to papers with similar research style. Focus on how the research is conducted."
    
    prev_text = ", ".join(previous_labels) if previous_labels else "None"
    
    prompt = f"""
You are an expert research assistant.{topic_instr}

{role_instr}

I will give you a set of research abstracts that belong to the same cluster.

Abstracts:
{block}

Summarize the common themes and approaches in JSON format:

{{
  "cluster_topics": "1-2 sentences",
  "proposed_label": "3-5 words"
}}

The label must be different from: {prev_text}

{"Translate the JSON text to Portuguese" if language == "pt" else ""}

Return ONLY valid JSON.
"""
    try:
        response = client.generate(prompt)
        # Handle potential markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
             response = response.split("```")[1].split("```")[0].strip()
             
        return json.loads(response)
    except Exception as e:
        logger.error(f"Error labeling cluster: {e}")
        return {"error": str(e)}


def generate_axis_label(
    client: LLMClient,
    pos_block: str,
    neg_block: str,
    topic: Optional[str],
    axis_idx: int,
    method: str,
    language: str,
) -> Dict[str, Any]:
    """Use LLM to interpret the meaning of a projection axis."""
    
    topic_instr = f" Assume the topic is {topic}." if topic else ""
    
    method_instr = ""
    if method in ["diffusion", "umap", "isomap"]:
        method_instr = "The coordinates come from a nonlinear embedding. Focus on conceptual differences between extremes."
    elif method == "pca":
        method_instr = "The axis comes from a linear projection. Interpret as a continuum from one type of research to another."

    prompt = f"""
You are an expert research assistant.{topic_instr}

{method_instr}

I will give you two sets of research abstracts located at opposite extremes of axis {axis_idx}.

Negative pole:
{neg_block}

Positive pole:
{pos_block}

Describe the main difference between the two poles in JSON format:

{{
  "negative_pole": "Main characteristics of the negative pole",
  "positive_pole": "Main characteristics of the positive pole",
  "axis_label": "Short contrast label describing the difference"
}}

The label should describe a conceptual contrast, not just a topic name.

{"Translate the JSON text to Portuguese" if language == "pt" else ""}

Return ONLY valid JSON.
"""
    try:
        response = client.generate(prompt)
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
             response = response.split("```")[1].split("```")[0].strip()
             
        return json.loads(response)
    except Exception as e:
        logger.error(f"Error labeling axis: {e}")
        return {"error": str(e)}
