import os
import sys
import json
from pathlib import Path

# Add edel to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edel.experiments.registry import init_registry, get_experiment, list_experiments
from edel.io.artifact import stage_hash

def main():
    init_registry("artifacts/configs")
    
    experiments = list_experiments()
    print(f"Experiments: {experiments}")
    
    for name in experiments:
        config = get_experiment(name)
        try:
            h = stage_hash(config, "clustering")
            print(f"[{name}] Clustering Hash: {h}")
        except Exception as e:
            print(f"[{name}] Error: {e}")

if __name__ == "__main__":
    main()
