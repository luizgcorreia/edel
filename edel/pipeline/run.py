"""Full pipeline orchestration with deterministic artifact persistence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from edel.io.artifact import (
    load_artifact,
    make_stage_artifact,
    save_artifact,
)
from edel.io.llm import get_llm_client
from edel.pipeline import (
    run_clustering_stage,
    run_data_stage,
    run_embedding_stage,
    run_labeling_stage,
    run_landscape_stage,
    run_projection_stage,
    run_structuring_stage,
    run_vector_field_stage,
)

logger = logging.getLogger(__name__)


def run_full_pipeline(
    config: dict, base_path: str | Path = "artifacts", force: bool = False
) -> Dict[str, Any]:
    """Execute all 8 stages of the EDEL pipeline using artifacts for persistence.
    
    If an artifact already exists for a stage, it will be loaded instead of re-computed,
    unless force=True.
    """
    base_path = Path(base_path)
    final_results = {}

    # --- Stage 1: Data Collection ---
    art_data = make_stage_artifact(config, base_path, "data_collection", "dataset")
    try:
        if force: raise FileNotFoundError()
        df = load_artifact(art_data)
        print("Stage 1: Loaded existing dataset artifact.")
    except (FileNotFoundError, Exception):
        print("Stage 1: Running data collection...")
        df = run_data_stage(config)
        save_artifact(art_data, df)
    final_results["data"] = df

    # --- Stage 2: Structured Abstracts ---
    art_sa = make_stage_artifact(config, base_path, "structured_abstracts", "sa")
    try:
        if force: raise FileNotFoundError()
        df = load_artifact(art_sa)
        print("Stage 2: Loaded existing structured abstracts artifact.")
    except (FileNotFoundError, Exception):
        print("Stage 2: Running structuring...")
        df = run_structuring_stage(df, config)
        save_artifact(art_sa, df)
    final_results["structuring"] = df

    # --- Stage 3: Text Embeddings ---
    art_emb = make_stage_artifact(config, base_path, "embeddings", "embeddings")
    try:
        if force: raise FileNotFoundError()
        df = load_artifact(art_emb)
        print("Stage 3: Loaded existing embeddings artifact.")
    except (FileNotFoundError, Exception):
        print("Stage 3: Running embedding...")
        df = run_embedding_stage(df, config)
        save_artifact(art_emb, df)
    final_results["embedding"] = df

    # --- Stage 4: Dimensionality Reduction ---
    art_dr = make_stage_artifact(config, base_path, "dimensionality_reduction", "dr")
    try:
        if force: raise FileNotFoundError()
        df = load_artifact(art_dr)
        print("Stage 4: Loaded existing projection artifact.")
    except (FileNotFoundError, Exception):
        print("Stage 4: Running projection...")
        df = run_projection_stage(df, config)
        save_artifact(art_dr, df)
    final_results["projection"] = df

    # --- Stage 5: Vector Field ---
    art_vf = make_stage_artifact(config, base_path, "vector_field", "vf")
    try:
        if force: raise FileNotFoundError()
        field = load_artifact(art_vf)
        print("Stage 5: Loaded existing vector field artifact.")
    except (FileNotFoundError, Exception):
        print("Stage 5: Running vector field computation...")
        field = run_vector_field_stage(df, config)
        save_artifact(art_vf, field)
    final_results["vector_field"] = field

    # --- Stage 6: Clustering ---
    art_cls = make_stage_artifact(config, base_path, "clustering", "clustering")
    art_fcls = make_stage_artifact(config, base_path, "clustering", "field_clustering")
    try:
        if force: raise FileNotFoundError()
        df = load_artifact(art_cls)
        field = load_artifact(art_fcls)
        print("Stage 6: Loaded existing clustering artifacts.")
    except (FileNotFoundError, Exception):
        print("Stage 6: Running clustering...")
        df, field = run_clustering_stage(df, field, config)
        save_artifact(art_cls, df)
        save_artifact(art_fcls, field)
    final_results["clustering_df"] = df
    final_results["clustering_field"] = field

    # --- Stage 7: Labeling ---
    art_lbl = make_stage_artifact(config, base_path, "labeling", "labeled")
    try:
        if force: raise FileNotFoundError()
        labels = load_artifact(art_lbl)
        print("Stage 7: Loaded existing labels artifact.")
    except (FileNotFoundError, Exception):
        print("Stage 7: Running labeling...")
        llm_client = get_llm_client(config.get("labeling", {}))
        labels = run_labeling_stage(df, field, config, llm_client)
        save_artifact(art_lbl, labels)
    final_results["labels"] = labels

    # --- Stage 8: Landscape ---
    art_land = make_stage_artifact(config, base_path, "output", "plot")
    try:
        if force: raise FileNotFoundError()
        landscape = load_artifact(art_land)
        print("Stage 8: Loaded existing landscape calculations.")
    except (FileNotFoundError, Exception):
        print("Stage 8: Running landscape preparation...")
        landscape = run_landscape_stage(df, field, config)
        save_artifact(art_land, landscape)
    final_results["landscape"] = landscape

    print("\n✅ Full pipeline execution finished successfully.")
    return final_results
