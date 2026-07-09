import json
from pathlib import Path
import sys

# Add edel to path
sys.path.append(str(Path(__file__).parent.parent))

from edel.config.defaults import RUN_CONFIG
from edel.io.artifact import make_stage_artifact

def main():
    # Load default config (assuming standard run)
    config = RUN_CONFIG.copy()
    
    # 1. Generate the deterministic path for the batch_log
    base_path = Path("artifacts")
    batch_log_art = make_stage_artifact(config, base_path, "structured_abstracts", "batch_log")
    batch_log_path = batch_log_art.path_prefix.with_suffix(".json")
    
    # 2. Ensure directory exists
    batch_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 3. Create the recovered active batches list
    # Format: "{batch_id}::{job_uuid}::{number_of_prompts}"
    # We use 1000 as a placeholder for prompts, it's just used for console printouts
    recovered_batches = [
        "1582015263195267072::90514891-9eaf-4f90-b29e-133429a3b7ab::1000",
        "1256770927606104064::7678dcb5-559a-43c2-a8fd-b385e869e0bc::1000",
        "5011365641949544448::e0d841d2-48fe-47eb-8c63-4b021d0cc50a::1000"
    ]
    
    # 4. Write to disk
    with open(batch_log_path, "w") as f:
        json.dump(recovered_batches, f, indent=2)
        
    print(f"✅ Successfully wrote recovered batch log to:")
    print(f"   {batch_log_path}")
    print(f"You can now rerun the structuring stage and it will pick up these jobs!")

if __name__ == "__main__":
    main()
