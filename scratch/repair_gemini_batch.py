"""
Repair script: Re-process Vertex AI batch outputs for sa_b31b87bd
to fix the shuffled segmentation caused by the positional-index bug.

This script:
1. Loads the pre-structuring DataFrame (from the data collection stage)
2. Finds all GCS batch job outputs for this experiment
3. Uses content-hash matching (the fix) to correctly map responses to papers
4. Re-runs parse_and_merge_results
5. Saves the corrected parquet
"""

import json
import hashlib
import pickle
import re
from pathlib import Path

import pandas as pd
from google.cloud import storage
from tqdm import tqdm

from edel.io.artifact import load_artifact, make_stage_artifact, save_artifact
from edel.pipeline.structuring import (
    filter_abstracts,
    parse_and_merge_results,
    compute_segmentation_metrics,
)
from edel.prompts import create_structuring_prompt


def main():
    # 1. Load experiment config
    with open("artifacts/configs/registry.json") as f:
        registry = json.load(f)
    config = registry["scientometrics_full_umap"]
    base_path = Path("artifacts")
    
    print("Config:", json.dumps(config.get("structured_abstracts", {}), indent=2))
    print("Processing mode:", config.get("processing_mode"))
    
    # 2. Load the data collection artifact (pre-structuring)
    art_data = make_stage_artifact(config, base_path, "data_collection", "dataset")
    df_raw = load_artifact(art_data)
    print(f"\nRaw data: {df_raw.shape}")
    
    # 3. Apply same filtering as the structuring stage
    stage_cfg = config.get("structured_abstracts", {})
    min_sentences = stage_cfg.get("min_sentences", 4)
    min_tokens = stage_cfg.get("min_tokens", 80)
    target_lang = stage_cfg.get("language")
    
    df_filtered, filter_report = filter_abstracts(df_raw, min_sentences, min_tokens, target_lang)
    
    # Apply sampling if configured
    n_docs = stage_cfg.get("n_documents")
    if n_docs and n_docs < len(df_filtered):
        seed = config.get("random_seed", 42)
        df_filtered = df_filtered.sample(n=n_docs, random_state=seed)
    
    print(f"Filtered data: {df_filtered.shape}")
    print(f"Filter report: {filter_report}")
    
    # 4. Rebuild prompts to create content-hash -> custom_id mapping
    topic = config.get("data", {}).get("provider", {}).get("topic_name")
    definitions = stage_cfg.get("aspect_definitions")
    
    hash_to_custom_id = {}
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Building hash map"):
        prompt = create_structuring_prompt(
            title=row["title"],
            abstract_text=row["abstract_text"],
            keywords=row.get("keywords", []),
            topic=topic,
            definitions=definitions,
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        custom_id = f"request-{idx}"
        hash_to_custom_id[prompt_hash] = custom_id
    
    print(f"\nBuilt hash map with {len(hash_to_custom_id)} entries")
    
    # 5. Download all batch outputs from GCS and match by content hash
    project_id = "windy-skyline-494616-m1"
    bucket_name = f"edel-batch-staging-{project_id}".lower().replace("_", "-")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    # List all batch job UUIDs
    blobs = list(bucket.list_blobs(prefix="batch_jobs/"))
    uuids = set()
    for b in blobs:
        parts = b.name.split("/")
        if len(parts) >= 3:
            uuids.add(parts[1])
    
    print(f"\nFound {len(uuids)} batch job UUIDs in GCS")
    
    all_results = {}
    matched_count = 0
    unmatched_count = 0
    
    for uid in tqdm(sorted(uuids), desc="Processing batch outputs"):
        output_blobs = [b for b in blobs if uid in b.name and "/output/" in b.name and b.name.endswith(".jsonl")]
        
        for ob in output_blobs:
            content = ob.download_as_string().decode("utf-8")
            lines = [l for l in content.split("\n") if l.strip()]
            
            for line in lines:
                try:
                    data = json.loads(line)
                    req = data.get("request", {})
                    res = data.get("response", {})
                    
                    # Extract prompt text
                    if "contents" in req:
                        prompt_text = req["contents"][0]["parts"][0]["text"]
                    elif "content" in req:
                        prompt_text = req["content"]["parts"][0]["text"]
                    else:
                        continue
                    
                    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
                    custom_id = hash_to_custom_id.get(prompt_hash)
                    
                    if not custom_id:
                        unmatched_count += 1
                        continue
                    
                    # Extract response
                    candidates = res.get("candidates", [])
                    if candidates and candidates[0].get("content", {}).get("parts"):
                        parsed_val = candidates[0]["content"]["parts"][0].get("text", "{}")
                    else:
                        parsed_val = "{}"
                    
                    all_results[custom_id] = parsed_val
                    matched_count += 1
                    
                except Exception as e:
                    print(f"Error parsing line: {e}")
    
    print(f"\nMatched: {matched_count}, Unmatched: {unmatched_count}")
    print(f"Unique results: {len(all_results)}")
    print(f"Expected: {len(df_filtered)}")
    print(f"Coverage: {len(all_results) / len(df_filtered) * 100:.1f}%")
    
    # 6. Parse & Merge
    df_structured = parse_and_merge_results(df_filtered, all_results)
    
    # 7. Verify fix
    match_count = 0
    for _, row in df_structured.iterrows():
        seg = row["problem"]
        abstract = row["abstract_text"]
        if not seg or "abstract:" not in seg:
            continue
        abstract_part = seg.split("abstract:\n", 1)[-1].strip()
        if not abstract_part:
            continue
        words = abstract_part.split()[:6]
        snippet = " ".join(words).lower()
        if snippet in abstract.lower():
            match_count += 1
    
    total_with_problem = sum(1 for _, r in df_structured.iterrows() 
                            if r["problem"] and "abstract:" in r["problem"])
    print(f"\nVerification: {match_count}/{total_with_problem} problem segments match their abstract")
    
    # 8. Compute metrics
    seg_metrics = compute_segmentation_metrics(df_structured)
    filter_report.update(seg_metrics)
    print(f"\nSegmentation metrics: {json.dumps(seg_metrics, indent=2)}")
    
    # 9. Save
    art_sa = make_stage_artifact(config, base_path, "structured_abstracts", "sa")
    save_artifact(art_sa, df_structured)
    
    art_report = make_stage_artifact(config, base_path, "structured_abstracts", "filter_report")
    save_artifact(art_report, filter_report)
    
    print(f"\nSaved repaired artifact to: {art_sa.path_prefix}")
    print(f"Saved filter report to: {art_report.path_prefix}")


if __name__ == "__main__":
    main()
