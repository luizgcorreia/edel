import os
import json
import pickle
import pandas as pd
from pathlib import Path

from edel.io.artifact import (
    load_artifact,
    make_stage_artifact,
    save_artifact,
)
from edel.pipeline import (
    run_projection_stage,
    run_vector_field_stage,
    run_clustering_stage,
    run_landscape_stage,
)

def main():
    # 1. Load config
    config_path = "artifacts/configs/registry.json"
    with open(config_path, "r") as f:
        registry = json.load(f)
    config = registry["afp_baseline"]
    base_path = Path("artifacts")

    print("--- Loading Stage 3 Embeddings ---")
    art_emb = make_stage_artifact(config, base_path, "embeddings", "embeddings")
    df = load_artifact(art_emb)
    print(f"Loaded embeddings artifact: {art_emb.path_prefix} (shape: {df.shape})")

    # --- Stage 4: Dimensionality Reduction ---
    print("\n--- Running Stage 4: Projection ---")
    art_dr = make_stage_artifact(config, base_path, "dimensionality_reduction", "dr")
    res = run_projection_stage(df, config)
    if isinstance(res, tuple):
        df, report = res
        save_artifact(art_dr, df)
        if report:
            art_report = make_stage_artifact(config, base_path, "dimensionality_reduction", "report")
            save_artifact(art_report, report)
    else:
        df = res
        save_artifact(art_dr, df)
    print(f"Saved Stage 4 projection artifact to: {art_dr.path_prefix}")

    # --- Stage 5: Vector Field ---
    print("\n--- Running Stage 5: Vector Field ---")
    art_vf = make_stage_artifact(config, base_path, "vector_field", "vf")
    field = run_vector_field_stage(df, config)
    save_artifact(art_vf, field)
    print(f"Saved Stage 5 vector field artifact to: {art_vf.path_prefix}")

    # --- Stage 6: Clustering ---
    print("\n--- Running Stage 6: Clustering ---")
    art_cls = make_stage_artifact(config, base_path, "clustering", "clustering")
    art_fcls = make_stage_artifact(config, base_path, "clustering", "field_clustering")
    
    res = run_clustering_stage(df, field, config)
    if isinstance(res, tuple) and len(res) == 3:
        df, field, report = res
    else:
        df, field = res
        report = None
        
    save_artifact(art_cls, df)
    save_artifact(art_fcls, field)
    if report:
        art_report = make_stage_artifact(config, base_path, "clustering", "report")
        save_artifact(art_report, report)
    print(f"Saved Stage 6 clustering artifacts to: {art_cls.path_prefix} and {art_fcls.path_prefix}")

    # --- Stage 8: Landscape ---
    print("\n--- Running Stage 8: Landscape ---")
    art_land = make_stage_artifact(config, base_path, "output", "landscape_results")
    landscape = run_landscape_stage(df, field, config)
    save_artifact(art_land, landscape)
    print(f"Saved Stage 8 landscape results to: {art_land.path_prefix}")

    print("\n✅ Recomputation of downstream stages complete!")

if __name__ == "__main__":
    main()
