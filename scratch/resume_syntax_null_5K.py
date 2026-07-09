"""Resume embedding stage for syntax_null_none_global (5K).

Recovers two completed OpenAI batches, submits remaining, assembles artifact.
"""
import json
import sys
import time
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edel.io.artifact import make_stage_artifact, save_artifact, load_artifact
from edel.io.llm import get_llm_client
from edel.pipeline.embedding import filter_by_aspects

BASE = Path("artifacts")
EXP = "syntax_null_5K"

# Load config
with open(BASE / "configs" / "registry.json") as f:
    registry = json.load(f)
config = registry[EXP]

# Override for 5K mode: update data provider params
config.setdefault("data", {}).setdefault("provider", {}).setdefault("params", {})

# Load structured abstracts
art_sa = make_stage_artifact(config, BASE, "structured_abstracts", "sa")
print(f"Loading structured abstracts from {art_sa.parquet_path}...")
import pandas as pd
df = pd.read_parquet(art_sa.parquet_path)
print(f"Loaded {len(df)} rows")

# Apply aspect filtering
df_filtered, filter_report = filter_by_aspects(df)
print(f"After filtering: {len(df_filtered)} rows")

# Build the client
client = get_llm_client(config["embedding"])

# ---- PHASE 1: Poll existing completed batches ----
completed_batch_ids = [
    "batch_6a2806b957888190b62aa978245abe4b",
    "batch_6a28038e66848190a2f6bf4896601c82",
]

all_results = {}
for b_id in completed_batch_ids:
    print(f"Polling batch {b_id}...")
    status_info = client.poll_batch(b_id)
    status = status_info["status"]
    print(f"  Status: {status}")
    if status == "completed" and status_info.get("results"):
        all_results.update(status_info["results"])
        print(f"  Recovered {len(status_info['results'])} results")
    elif status != "completed":
        print(f"  WARNING: batch not completed yet, will retry later")
        completed_batch_ids.remove(b_id)

print(f"Total recovered results: {len(all_results)}")

# ---- PHASE 2: Determine remaining and submit new batches ----
fields_to_embed = ["problem", "method", "finding", "interpretation"]

all_prompts = {}
for idx in df_filtered.index:
    for field in fields_to_embed:
        text = str(df_filtered.at[idx, field]).strip()
        if text:
            all_prompts[f"emb::{idx}::{field}"] = text

print(f"Total prompts: {len(all_prompts)}")
print(f"Already embedded: {len(all_results)}")

missing_keys = [k for k in all_prompts if k not in all_results]
print(f"Missing prompts: {len(missing_keys)}")

batch_size = 5000
active_batches = list(completed_batch_ids)  # Keep tracking completed ones
batch_log_art = make_stage_artifact(config, BASE, "embeddings", "batch_log")
batch_log_path = batch_log_art.path_prefix.with_suffix(".json")
batch_log_path.parent.mkdir(parents=True, exist_ok=True)

# Submit missing prompts in chunks
total_missing = len(missing_keys)
next_chunk_idx = 0

while len(active_batches) < 1 and next_chunk_idx * batch_size < total_missing:
    start_idx = next_chunk_idx * batch_size
    end_idx = min((next_chunk_idx + 1) * batch_size, total_missing)
    chunk_keys = missing_keys[start_idx:end_idx]
    chunk_prompts = {k: all_prompts[k] for k in chunk_keys}

    print(f"Submitting batch {next_chunk_idx + 1}/{(total_missing + batch_size - 1)//batch_size} "
          f"({len(chunk_prompts)} prompts)...")
    batch_id = client.create_batch(chunk_prompts, endpoint="/v1/embeddings")
    active_batches.append(batch_id)
    print(f"  Created: {batch_id}")
    next_chunk_idx += 1

    # Save batch log
    with open(batch_log_path, "w") as f:
        json.dump(active_batches, f, indent=2)

# ---- PHASE 3: Poll loop for any still-running batches ----
while True:
    remaining = []
    finished_this_cycle = False

    for b_id in active_batches:
        try:
            status_info = client.poll_batch(b_id)
            status = status_info["status"]
            if status == "completed" and status_info.get("results"):
                all_results.update(status_info["results"])
                print(f"[{time.strftime('%H:%M:%S')}] Batch {b_id[:25]}... completed ({len(status_info['results'])} results)")
                finished_this_cycle = True
            elif status in ("failed", "cancelled", "expired"):
                print(f"[{time.strftime('%H:%M:%S')}] Batch {b_id[:25]}... {status}")
            else:
                remaining.append(b_id)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error polling {b_id[:25]}...: {e}")
            remaining.append(b_id)

    active_batches = remaining

    # Submit more chunks if space
    while len(active_batches) < 1 and next_chunk_idx * batch_size < total_missing:
        start_idx = next_chunk_idx * batch_size
        end_idx = min((next_chunk_idx + 1) * batch_size, total_missing)
        chunk_keys = missing_keys[start_idx:end_idx]
        chunk_prompts = {k: all_prompts[k] for k in chunk_keys}
        print(f"Submitting batch {next_chunk_idx + 1} ({len(chunk_prompts)} prompts)...")
        batch_id = client.create_batch(chunk_prompts, endpoint="/v1/embeddings")
        active_batches.append(batch_id)
        print(f"  Created: {batch_id}")
        next_chunk_idx += 1
        with open(batch_log_path, "w") as f:
            json.dump(active_batches, f, indent=2)

    if not active_batches and next_chunk_idx * batch_size >= total_missing:
        break

    time.sleep(60 if not finished_this_cycle else 5)

print(f"\nAll batches complete! Total results: {len(all_results)}")

# ---- PHASE 4: Assemble final artifact ----
out = df_filtered.copy()
for field in fields_to_embed:
    target_col = f"{field}_embedding"
    col_data = []
    missing_count = 0
    for idx in out.index:
        key = f"emb::{idx}::{field}"
        res = all_results.get(key)
        if res is not None:
            col_data.append(json.dumps(res))
        else:
            col_data.append(None)
            missing_count += 1
    out[target_col] = col_data
    print(f"{field}_embedding: {missing_count} missing")

# Save embeddings
art_emb = make_stage_artifact(config, BASE, "embeddings", "embeddings")
print(f"Saving embeddings to {art_emb.parquet_path}...")
save_artifact(art_emb, out)
print(f"Saved {len(out)} rows")

# Save filter report
art_report = make_stage_artifact(config, BASE, "embeddings", "filter_report")
save_artifact(art_report, filter_report)

print("Done! Embeddings artifact assembled.")
