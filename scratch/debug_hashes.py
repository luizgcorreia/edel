import os
import sys
import json
import hashlib
from pathlib import Path

# Add edel to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edel.experiments.registry import init_registry, get_experiment, list_experiments

def stable_hash_debug(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    h = hashlib.md5(payload.encode("utf-8")).hexdigest()
    print(f"Payload: {payload}")
    print(f"Hash: {h}")
    return h

def main():
    init_registry("artifacts/configs")
    config = get_experiment("scientometrics_baseline")
    
    print("\n--- GLOBAL HASH ---")
    global_keys = ("random_seed",) # Try different ones here
    global_cfg = {k: config.get(k) for k in global_keys if k in config}
    gh = stable_hash_debug(global_cfg)
    
    print("\n--- DATA STAGE HASH ---")
    stage_params = config.get("data", {})
    sph = stable_hash_debug(stage_params)
    
    # Chain
    chained = stable_hash_debug({"prev_hash": gh, "params_hash": sph})
    print(f"Final Data Collection Hash: {chained[:8]}")

if __name__ == "__main__":
    main()
