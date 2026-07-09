"""
Repair script: Propagate structuring order fix to embeddings.
Rebuilds embeddings_633ff026.parquet with correct alignment.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil

from edel.io.artifact import load_artifact, make_stage_artifact, save_artifact
from edel.io.llm import get_llm_client
from edel.pipeline.embedding import process_batch


def main():
    base_path = Path("artifacts")

    # 1. Load experiment config
    with open("artifacts/configs/registry.json") as f:
        registry = json.load(f)
    config = registry["scientometrics_full_umap"]

    # Paths
    emb_dir = base_path / "embeddings" / "openalex_T10102_global"
    emb_path = emb_dir / "embeddings_633ff026.parquet"
    emb_backup_path = emb_dir / "embeddings_633ff026_shuffled.parquet"

    sa_path = base_path / "structured_abstracts" / "openalex_T10102_global" / "sa_b31b87bd.parquet"

    print("Step 1: Backing up current shuffled embeddings...")
    if not emb_backup_path.exists():
        shutil.copy2(emb_path, emb_backup_path)
        print(f"Backed up to {emb_backup_path}")
    else:
        print("Backup already exists.")

    print("\nStep 2: Loading datasets...")
    # Load repaired sa artifact
    df_sa = pd.read_parquet(sa_path)
    print(f"Loaded repaired structured abstracts: {df_sa.shape}")

    # Filter to English subset
    df_sa_en = df_sa[df_sa["language"] == "en"].copy()
    print(f"English structured abstracts: {df_sa_en.shape}")

    # Load shuffled embeddings
    df_emb_shuffled = pd.read_parquet(emb_backup_path)
    print(f"Loaded shuffled embeddings: {df_emb_shuffled.shape}")

    print("\nStep 3: Building lookup mapping from shuffled embeddings...")
    # Map segment text 4-tuple -> embeddings
    emb_lookup = {}
    for idx, row in tqdm(df_emb_shuffled.iterrows(), total=len(df_emb_shuffled), desc="Caching shuffled embeddings"):
        # Check if at least one embedding field is populated/notna
        has_emb = any(pd.notna(row[f"{field}_embedding"]) for field in ["problem", "method", "finding", "interpretation"])
        if has_emb:
            key = (row["problem"], row["method"], row["finding"], row["interpretation"])
            emb_lookup[key] = {
                f"{field}_embedding": row[f"{field}_embedding"]
                for field in ["problem", "method", "finding", "interpretation"]
            }

    print(f"Cached {len(emb_lookup)} embedding tuples.")

    # 4. Initialize embedding columns in repaired English df
    for field in ["problem", "method", "finding", "interpretation"]:
        df_sa_en[f"{field}_embedding"] = None

    # 5. Classify papers
    empty_indices = []
    found_indices = []
    missing_indices = []

    for idx, row in df_sa_en.iterrows():
        p, m, f, i = row["problem"], row["method"], row["finding"], row["interpretation"]
        if not p and not m and not f and not i:
            empty_indices.append(idx)
            continue
        
        key = (p, m, f, i)
        if key in emb_lookup:
            found_indices.append(idx)
            # Copy embeddings
            for field in ["problem", "method", "finding", "interpretation"]:
                df_sa_en.at[idx, f"{field}_embedding"] = emb_lookup[key][f"{field}_embedding"]
        else:
            missing_indices.append(idx)

    print(f"\nClassification results:")
    print(f"- Total: {len(df_sa_en)}")
    print(f"- Empty segments (no embeddings needed): {len(empty_indices)}")
    print(f"- Found and mapped from shuffled file: {len(found_indices)}")
    print(f"- Missing (require API call): {len(missing_indices)}")

    # 6. Re-embed missing ones if there are any
    if len(missing_indices) > 0:
        print(f"\nStep 4: Requesting embeddings for {len(missing_indices)} missing papers...")
        df_missing = df_sa_en.loc[missing_indices].copy()
        
        # Reset index to avoid any indexing issues during batching
        original_indices = df_missing.index.tolist()
        df_missing = df_missing.reset_index(drop=True)
        
        embed_cfg = config.get("embedding", {})
        client = get_llm_client(embed_cfg)
        
        print("Starting process_batch for missing embeddings...")
        batch_log_path = emb_dir / "repair_batch_log.json"
        
        df_missing_embedded = process_batch(
            df_missing,
            client,
            mode=embed_cfg.get("mode", "multi"),
            batch_size=embed_cfg.get("batch_size", 5000),
            provider=embed_cfg.get("provider", "openai"),
            batch_log_path=batch_log_path
        )
        
        # Map back to df_sa_en by original index using alignment
        print("Merging missing embeddings back...")
        for i, idx in enumerate(original_indices):
            for field in ["problem", "method", "finding", "interpretation"]:
                df_sa_en.at[idx, f"{field}_embedding"] = df_missing_embedded.at[i, f"{field}_embedding"]

    print("\nStep 5: Verifying repaired embeddings...")
    # Check that all non-empty papers have embeddings
    unmapped_count = 0
    for idx, row in df_sa_en.iterrows():
        p, m, f, i = row["problem"], row["method"], row["finding"], row["interpretation"]
        if not p and not m and not f and not i:
            continue
        
        # Check if at least one expected embedding is present
        # Note: if a paper has non-empty problem but empty method, the method embedding is expected to be None/null
        # So we check that the fields that are non-empty have non-null embeddings.
        for field in ["problem", "method", "finding", "interpretation"]:
            val = row[field]
            emb = row[f"{field}_embedding"]
            if val and (emb is None or pd.isna(emb)):
                unmapped_count += 1
                break

    print(f"Verification: {unmapped_count} non-empty papers are still unmapped/unembedded.")
    if unmapped_count > 0:
        raise ValueError("Error: Some papers with non-empty segments are still missing embeddings!")

    # Save final artifact
    print(f"\nStep 6: Saving repaired embeddings artifact to {emb_path}...")
    df_sa_en.to_parquet(emb_path, index=False)
    print("Repaired embeddings saved successfully!")


if __name__ == "__main__":
    main()
