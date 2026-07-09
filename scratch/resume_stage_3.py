import os
import json
import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).parent.parent))

from edel.io.artifact import (
    load_artifact,
    make_stage_artifact,
    save_artifact,
)
from edel.pipeline.embedding import run_embedding_stage

def main():
    config_path = "artifacts/configs/registry.json"
    with open(config_path, "r") as f:
        registry = json.load(f)
    config = registry["afp_syntax_null"]
    base_path = Path("artifacts")

    # Load structured abstracts (stage 2)
    print("--- Loading Stage 2 Structured Abstracts ---")
    art_sa = make_stage_artifact(config, base_path, "structured_abstracts", "sa")
    df = load_artifact(art_sa)
    print(f"Loaded structured abstracts: {art_sa.path_prefix} (shape: {df.shape})")

    # Run Stage 3: Embedding
    print("\n--- Running Stage 3: Embedding (Resuming/Polling Batch) ---")
    art_emb = make_stage_artifact(config, base_path, "embeddings", "embeddings")
    df_emb = run_embedding_stage(df, config, base_path=base_path)
    save_artifact(art_emb, df_emb)
    print(f"Saved Stage 3 embeddings artifact to: {art_emb.path_prefix}")

if __name__ == "__main__":
    main()
