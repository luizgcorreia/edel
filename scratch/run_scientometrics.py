import json
import logging
from pathlib import Path
from edel.pipeline.run import run_full_pipeline
from edel.config.defaults import RUN_CONFIG

def main():
    logging.basicConfig(level=logging.INFO)
    registry_path = Path("artifacts/configs/registry.json")
    
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry config not found at {registry_path}")
        
    with open(registry_path, "r") as f:
        registry = json.load(f)
        
    config = registry.get("scientometrics_full_umap")
    if not config:
        raise ValueError("scientometrics_full_umap config not found in registry")
        
    print("Running pipeline for scientometrics_full_umap...")
    run_full_pipeline(config, base_path="artifacts")
    print("Pipeline run completed!")

if __name__ == "__main__":
    main()
