import os
import json
from pathlib import Path
from edel.io.artifact import load_artifact, make_stage_artifact, save_artifact
from edel.io.llm import get_llm_client
import edel.pipeline as pipeline

def main():
    # 1. Load config
    config_path = "artifacts/configs/registry.json"
    with open(config_path, "r") as f:
        registry = json.load(f)
    config = registry["afp_baseline"]
    base_path = Path("artifacts")

    print("Loading clustering and field artifacts...")
    art_df = make_stage_artifact(config, base_path, "clustering", "clustering")
    art_field = make_stage_artifact(config, base_path, "clustering", "field_clustering")
    
    df = load_artifact(art_df)
    field = load_artifact(art_field)
    
    print(f"Loaded clustering: {df.shape}, field: {field.shape}")
    
    # Setup LLM Client
    label_cfg = config.get("labeling", {})
    print("Labeling config:", label_cfg)
    llm_client = get_llm_client(label_cfg)
    
    print("\n--- Running Stage 7: Labeling ---")
    data = pipeline.run_labeling_stage(df, field, config, llm_client)
    
    print("\nLabeling results:")
    print(json.dumps(data, indent=2))
    
    # Save labeling artifact
    art_labeled = make_stage_artifact(config, base_path, "labeling", "labeled")
    save_artifact(art_labeled, data)
    print(f"\nSaved Stage 7 labeling artifact to: {art_labeled.path_prefix}")

if __name__ == "__main__":
    main()
