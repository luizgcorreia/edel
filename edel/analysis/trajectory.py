"""Trajectory analysis module for EDEL.

Core, Dash-free logic for analysing the epistemic trajectory of a paper
(or a set of synthetic segments) against the full dimensionality-reduction
dataset. Designed to be used both by the CLI script and the dashboard.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

ASPECTS = ["problem", "method", "finding", "interpretation"]


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def parse_embedding_vector(val: Any) -> np.ndarray | None:
    """Safely parse an embedding stored as a JSON string, list, or ndarray.

    Returns a 1-D float32 numpy array, or None if the value is missing/invalid.
    """
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return np.array(parsed, dtype=np.float32)
        except Exception:
            return None
    if isinstance(val, (list, np.ndarray)):
        arr = np.array(val, dtype=np.float32)
        if np.isnan(arr).any():
            return None
        return arr
    return None


def _is_valid(vec: np.ndarray | None) -> bool:
    return vec is not None and not np.isnan(vec).any()


# ---------------------------------------------------------------------------
# Neighbor search
# ---------------------------------------------------------------------------

def find_neighbors(
    df: pd.DataFrame,
    target_vec: np.ndarray,
    aspect: str,
    space: str,
    method: str,
    k: int,
    radius: float | None,
    exclude_id: str | None = None,
) -> list[dict]:
    """Find neighboring papers for a given target vector.

    Parameters
    ----------
    df:          Full dimensionality-reduction DataFrame.
    target_vec:  1-D or 2-D query vector.
    aspect:      One of ASPECTS — used to select columns or the embedding col.
    space:       ``"2d"`` (Euclidean on proj coords) or ``"embedding"`` (cosine).
    method:      Projection method name, e.g. ``"diffusion"`` or ``"umap"``.
    k:           Number of nearest neighbours (ignored when *radius* is set).
    radius:      Distance threshold; if set, returns all papers within it.
    exclude_id:  Work ID to exclude from results (the query paper itself).

    Returns
    -------
    List of neighbour dicts ordered by ascending distance, each containing
    all text fields from the original row.
    """
    if space == "2d":
        x_col = f"proj_{aspect}_{method}_x"
        y_col = f"proj_{aspect}_{method}_y"
        if x_col not in df.columns or y_col not in df.columns:
            logger.warning("2D projection columns for %s/%s not found.", aspect, method)
            return []

        coords = df[[x_col, y_col]].values.astype(np.float32)
        # Drop rows where either coordinate is NaN
        valid_mask = ~np.isnan(coords).any(axis=1)
        valid_df = df[valid_mask].copy()
        valid_coords = coords[valid_mask]

        distances = cdist([target_vec], valid_coords, metric="euclidean")[0]
        id_series = valid_df["id"]
    else:
        emb_col = f"{aspect}_embedding"
        if emb_col not in df.columns:
            logger.warning("Embedding column %s not found.", emb_col)
            return []

        parsed_vecs: list[np.ndarray] = []
        valid_indices: list[int] = []
        for idx, val in enumerate(df[emb_col].values):
            vec = parse_embedding_vector(val)
            if vec is not None:
                parsed_vecs.append(vec)
                valid_indices.append(idx)

        if not parsed_vecs:
            logger.warning("No valid embeddings found for aspect %s.", aspect)
            return []

        all_vecs = np.vstack(parsed_vecs)
        target_2d = target_vec.reshape(1, -1)
        distances = cdist(target_2d, all_vecs, metric="cosine")[0]
        id_series = df.iloc[valid_indices]["id"]
        valid_df = df.iloc[valid_indices]

    df_dists = pd.DataFrame({"_idx": valid_df.index, "id": id_series.values, "distance": distances})

    if exclude_id:
        df_dists = df_dists[df_dists["id"] != exclude_id]

    if radius is not None:
        df_dists = df_dists[df_dists["distance"] <= radius]

    df_dists = df_dists.sort_values("distance").head(k if radius is None else len(df_dists))

    neighbors: list[dict] = []
    for _, n_row in df_dists.iterrows():
        n_data = df[df["id"] == n_row["id"]]
        if n_data.empty:
            continue
        n_data = n_data.iloc[0]

        entry: dict[str, Any] = {
            "id": n_row["id"],
            "distance": float(n_row["distance"]),
            "title": n_data.get("title", "Unknown Title"),
            "publication_year": n_data.get("publication_year", ""),
            "cited_by_count": n_data.get("cited_by_count", ""),
            "doi": n_data.get("doi", ""),
            "abstract_text": n_data.get("abstract_text", ""),
        }
        for asp in ASPECTS:
            entry[asp] = n_data.get(asp, "")

        neighbors.append(entry)

    return neighbors


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_trajectory(
    df: pd.DataFrame,
    *,
    paper_id: str | None = None,
    segments: dict[str, str] | None = None,
    embedding_vectors: dict[str, np.ndarray] | None = None,
    space: str = "embedding",
    method: str = "diffusion",
    k: int = 5,
    radius: float | None = None,
) -> dict:
    """Analyse the epistemic trajectory of a paper or synthetic segment set.

    Exactly one of *paper_id* or (*segments* + *embedding_vectors*) must be
    provided.

    Parameters
    ----------
    df:                Full dimensionality-reduction DataFrame.
    paper_id:          OpenAlex Work ID to look up within *df*.
    segments:          Dict mapping aspect names to text strings (synthetic).
    embedding_vectors: Dict mapping aspect names to pre-computed 1-D vectors
                       (required when using synthetic segments).
    space:             Distance space: ``"2d"`` or ``"embedding"``.
    method:            Projection method name.
    k:                 Number of nearest neighbours.
    radius:            Optional radius override.

    Returns
    -------
    Structured result dict::

        {
          "target": {
            "id": ...,
            "title": ...,
            "problem": ...,      # text
            "method": ...,
            "finding": ...,
            "interpretation": ...,
            "abstract_text": ...,
            "proj_2d": {         # only for paper_id mode
              "problem": [x, y],
              "method":  [x, y],
              ...
            }
          },
          "aspects": {
            "problem": {
              "target_vec_2d": [x, y] | None,
              "neighbors": [{id, title, distance, problem, ...}, ...]
            },
            ...
          },
          "mode": "paper_id" | "synthetic",
          "space": "2d" | "embedding",
        }
    """
    if paper_id is None and (segments is None or embedding_vectors is None):
        raise ValueError("Provide either paper_id or (segments + embedding_vectors).")

    mode = "paper_id" if paper_id else "synthetic"

    # ---- Build target metadata ----
    if mode == "paper_id":
        matches = df[df["id"] == paper_id]
        if matches.empty:
            raise KeyError(f"Paper '{paper_id}' not found in dataset.")
        row = matches.iloc[0]
        target: dict[str, Any] = {
            "id": paper_id,
            "title": row.get("title", "Unknown"),
            "abstract_text": row.get("abstract_text", ""),
            "publication_year": row.get("publication_year", ""),
            "cited_by_count": row.get("cited_by_count", ""),
            "doi": row.get("doi", ""),
        }
        for asp in ASPECTS:
            target[asp] = row.get(asp, "")

        # Collect existing 2D coordinates
        proj_2d: dict[str, list[float] | None] = {}
        for asp in ASPECTS:
            x_col = f"proj_{asp}_{method}_x"
            y_col = f"proj_{asp}_{method}_y"
            if x_col in df.columns and y_col in df.columns:
                xv, yv = row.get(x_col), row.get(y_col)
                if pd.notna(xv) and pd.notna(yv):
                    proj_2d[asp] = [float(xv), float(yv)]
                else:
                    proj_2d[asp] = None
            else:
                proj_2d[asp] = None
        target["proj_2d"] = proj_2d

        # Build embedding vectors from stored embeddings
        embedding_vectors = {}
        for asp in ASPECTS:
            emb_col = f"{asp}_embedding"
            if emb_col in row.index:
                vec = parse_embedding_vector(row[emb_col])
                if vec is not None:
                    embedding_vectors[asp] = vec

    else:
        # Synthetic mode — no existing paper row
        target = {
            "id": None,
            "title": "(Synthetic Paper)",
            "abstract_text": "",
            "publication_year": "",
            "cited_by_count": "",
            "doi": "",
            "proj_2d": {asp: None for asp in ASPECTS},
        }
        for asp in ASPECTS:
            target[asp] = (segments or {}).get(asp, "")

    # ---- Per-aspect neighbour search ----
    results_by_aspect: dict[str, dict] = {}

    for asp in ASPECTS:
        target_vec: np.ndarray | None = None

        if space == "2d" and mode == "paper_id":
            coords = target["proj_2d"].get(asp)
            if coords is not None:
                target_vec = np.array(coords, dtype=np.float32)
        else:
            target_vec = (embedding_vectors or {}).get(asp)

        target_vec_2d = target["proj_2d"].get(asp) if mode == "paper_id" else None

        if not _is_valid(target_vec):
            results_by_aspect[asp] = {
                "target_vec_2d": target_vec_2d,
                "neighbors": [],
                "error": f"Target vector unavailable for '{asp}' in {space} space.",
            }
            continue

        neighbors = find_neighbors(
            df=df,
            target_vec=target_vec,
            aspect=asp,
            space=space,
            method=method,
            k=k,
            radius=radius,
            exclude_id=paper_id,
        )

        results_by_aspect[asp] = {
            "target_vec_2d": target_vec_2d,
            "neighbors": neighbors,
        }

    # Compute operator and trajectory metrics
    traj_metrics = compute_trajectory_metrics(df, embedding_vectors, target["proj_2d"])

    return {
        "target": target,
        "aspects": results_by_aspect,
        "mode": mode,
        "space": space,
        "trajectory_metrics": traj_metrics,
    }


def compute_trajectory_metrics(
    df: pd.DataFrame,
    embedding_vectors: dict[str, np.ndarray],
    proj_2d: dict[str, list[float] | None],
) -> dict:
    """Compute operator and trajectory metrics for raw embeddings, L2 normalized embeddings, and 2D projections."""
    def cosine_similarity(v1, v2):
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    metrics = {"embedding_raw": {}, "embedding_normalized": {}, "projection_2d": {}}

    p_vec = embedding_vectors.get("problem")
    m_vec = embedding_vectors.get("method")
    f_vec = embedding_vectors.get("finding")
    i_vec = embedding_vectors.get("interpretation")

    # 1. Raw Embeddings
    if all(v is not None for v in [p_vec, m_vec, f_vec, i_vec]):
        pm = m_vec - p_vec
        mf = f_vec - m_vec
        fi = i_vec - f_vec

        metrics["embedding_raw"] = {
            "novelty": float(np.linalg.norm(i_vec - p_vec)),
            "norm_pm": float(np.linalg.norm(pm)),
            "norm_mf": float(np.linalg.norm(mf)),
            "norm_fi": float(np.linalg.norm(fi)),
            "cos_pm_mf": cosine_similarity(pm, mf),
            "cos_pm_fi": cosine_similarity(pm, fi),
            "cos_mf_fi": cosine_similarity(mf, fi),
        }

        # 2. Normalized Embeddings (unit norm)
        p_norm = p_vec / np.linalg.norm(p_vec) if np.linalg.norm(p_vec) > 0 else p_vec
        m_norm = m_vec / np.linalg.norm(m_vec) if np.linalg.norm(m_vec) > 0 else m_vec
        f_norm = f_vec / np.linalg.norm(f_vec) if np.linalg.norm(f_vec) > 0 else f_vec
        i_norm = i_vec / np.linalg.norm(i_vec) if np.linalg.norm(i_vec) > 0 else i_vec

        pm_n = m_norm - p_norm
        mf_n = f_norm - m_norm
        fi_n = i_norm - f_norm

        metrics["embedding_normalized"] = {
            "novelty": float(np.linalg.norm(i_norm - p_norm)),
            "norm_pm": float(np.linalg.norm(pm_n)),
            "norm_mf": float(np.linalg.norm(mf_n)),
            "norm_fi": float(np.linalg.norm(fi_n)),
            "cos_pm_mf": cosine_similarity(pm_n, mf_n),
            "cos_pm_fi": cosine_similarity(pm_n, fi_n),
            "cos_mf_fi": cosine_similarity(mf_n, fi_n),
        }

    # 3. 2D Projections
    p_proj = proj_2d.get("problem")
    m_proj = proj_2d.get("method")
    f_proj = proj_2d.get("finding")
    i_proj = proj_2d.get("interpretation")

    if all(v is not None for v in [p_proj, m_proj, f_proj, i_proj]):
        p_p = np.array(p_proj)
        m_p = np.array(m_proj)
        f_p = np.array(f_proj)
        i_p = np.array(i_proj)

        pm_p = m_p - p_p
        mf_p = f_p - m_p
        fi_p = i_p - f_p

        metrics["projection_2d"] = {
            "novelty": float(np.linalg.norm(i_p - p_p)),
            "norm_pm": float(np.linalg.norm(pm_p)),
            "norm_mf": float(np.linalg.norm(mf_p)),
            "norm_fi": float(np.linalg.norm(fi_p)),
            "cos_pm_mf": cosine_similarity(pm_p, mf_p),
            "cos_pm_fi": cosine_similarity(pm_p, fi_p),
            "cos_mf_fi": cosine_similarity(mf_p, fi_p),
        }

    return metrics


# ---------------------------------------------------------------------------
# Report formatting (for CLI)
# ---------------------------------------------------------------------------

def format_report(result: dict, k: int, radius: float | None) -> str:
    """Format an ``analyze_trajectory`` result as a Markdown string."""
    target = result["target"]
    lines: list[str] = [
        f"# Trajectory Analysis for Paper: {target['title']}",
        f"**ID**: {target.get('id') or '(synthetic)'}",
        f"**Distance Space**: {result['space'].upper()}",
        f"**Search Criteria**: {'Radius = ' + str(radius) if radius else 'K = ' + str(k)}",
        "",
    ]

    # Add Trajectory Metrics if present
    traj_metrics = result.get("trajectory_metrics", {})
    if traj_metrics:
        lines.append("## Trajectory & Operator Metrics")
        lines.append("")
        
        emb_raw = traj_metrics.get("embedding_raw", {})
        if emb_raw:
            lines.append("### Embedding Space (Raw):")
            lines.append(f"- **Novelty (||i - p||)**: {emb_raw.get('novelty', 0.0):.4f}")
            lines.append(f"- **Operator Norms (Step Sizes)**:")
            lines.append(f"  - Problem → Method (||pm||): {emb_raw.get('norm_pm', 0.0):.4f}")
            lines.append(f"  - Method → Finding (||mf||): {emb_raw.get('norm_mf', 0.0):.4f}")
            lines.append(f"  - Finding → Interpretation (||fi||): {emb_raw.get('norm_fi', 0.0):.4f}")
            lines.append(f"- **Operator Alignments (Cosine Similarity)**:")
            lines.append(f"  - cos(pm, mf): {emb_raw.get('cos_pm_mf', 0.0):.4f}")
            lines.append(f"  - cos(pm, fi): {emb_raw.get('cos_pm_fi', 0.0):.4f}")
            lines.append(f"  - cos(mf, fi): {emb_raw.get('cos_mf_fi', 0.0):.4f}")
            lines.append("")

        emb_norm = traj_metrics.get("embedding_normalized", {})
        if emb_norm:
            lines.append("### Embedding Space (L2 Normalized):")
            lines.append(f"- **Novelty (||i - p||)**: {emb_norm.get('novelty', 0.0):.4f}")
            lines.append(f"- **Operator Norms (Step Sizes)**:")
            lines.append(f"  - Problem → Method (||pm||): {emb_norm.get('norm_pm', 0.0):.4f}")
            lines.append(f"  - Method → Finding (||mf||): {emb_norm.get('norm_mf', 0.0):.4f}")
            lines.append(f"  - Finding → Interpretation (||fi||): {emb_norm.get('norm_fi', 0.0):.4f}")
            lines.append(f"- **Operator Alignments (Cosine Similarity)**:")
            lines.append(f"  - cos(pm, mf): {emb_norm.get('cos_pm_mf', 0.0):.4f}")
            lines.append(f"  - cos(pm, fi): {emb_norm.get('cos_pm_fi', 0.0):.4f}")
            lines.append(f"  - cos(mf, fi): {emb_norm.get('cos_mf_fi', 0.0):.4f}")
            lines.append("")

        proj = traj_metrics.get("projection_2d", {})
        if proj:
            lines.append("### 2D Projection Space:")
            lines.append(f"- **Novelty (||i - p||)**: {proj.get('novelty', 0.0):.4f}")
            lines.append(f"- **Operator Norms (Step Sizes)**:")
            lines.append(f"  - Problem → Method (||pm||): {proj.get('norm_pm', 0.0):.4f}")
            lines.append(f"  - Method → Finding (||mf||): {proj.get('norm_mf', 0.0):.4f}")
            lines.append(f"  - Finding → Interpretation (||fi||): {proj.get('norm_fi', 0.0):.4f}")
            lines.append(f"- **Operator Alignments (Cosine Similarity)**:")
            lines.append(f"  - cos(pm, mf): {proj.get('cos_pm_mf', 0.0):.4f}")
            lines.append(f"  - cos(pm, fi): {proj.get('cos_pm_fi', 0.0):.4f}")
            lines.append(f"  - cos(mf, fi): {proj.get('cos_mf_fi', 0.0):.4f}")
            lines.append("")

    lines.append("---")
    lines.append("")

    for i, asp in enumerate(ASPECTS):
        asp_result = result["aspects"].get(asp, {})
        neighbors = asp_result.get("neighbors", [])
        error = asp_result.get("error")

        lines.append(f"## {i + 1}. Movement: {asp.capitalize()}")
        lines.append(f'**Target Segment:** "{target.get(asp, "")}"')
        lines.append("")

        if error:
            lines.append(f"*({error})*")
            lines.append("")
            continue

        lines.append(f"**Neighborhood Size:** {len(neighbors)} papers found")
        lines.append("")
        lines.append("### Neighbors:")
        lines.append("")

        if not neighbors:
            lines.append("No neighbors found.")
            lines.append("")
            continue

        for idx, n in enumerate(neighbors):
            lines.append(f"{idx + 1}. **{n['title']}** (Distance: {n['distance']:.4f})")
            lines.append(f'   **Segment:** "{n.get(asp, "")}"')

            if asp == "interpretation":
                n_problem = n.get("problem", "")
                if n_problem and isinstance(n_problem, str) and n_problem.strip():
                    lines.append(f'   **Opened Problem:** "{n_problem}"')

            lines.append("")

    return "\n".join(lines)
