"""Stage 3: Text Embedding."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from edel.io.llm import LLMClient, get_llm_client


def run_embedding_stage(df: pd.DataFrame, config: dict, base_path: str | Path = "artifacts") -> pd.DataFrame:
    """Orchestrate the text embedding stage."""
    embed_cfg = config.get("embedding", {})
    provider = embed_cfg.get("provider", "openai")
    model = embed_cfg.get("model", "text-embedding-ada-002")
    mode = embed_cfg.get("mode", "multi")  # "single" or "multi"
    processing_mode = config.get("processing_mode", "batch")
    batch_size = embed_cfg.get("batch_size", 5000)
    target_lang = embed_cfg.get("language")

    if target_lang and "language" in df.columns:
        print(f"Filtering embeddings to only include language: {target_lang}")
        df = df[df["language"] == target_lang].copy()

    # Pass the entire embed_cfg so that additional kwargs like 'location' reach the LLM client
    client = get_llm_client(embed_cfg)

    if processing_mode == "batch":
        from edel.io.artifact import make_stage_artifact
        batch_log_art = make_stage_artifact(config, Path(base_path), "embeddings", "batch_log")
        batch_log_path = batch_log_art.path_prefix.with_suffix(".json")
        batch_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        return process_batch(df, client, mode, batch_size, batch_log_path=batch_log_path)
    else:
        return process_simple(df, client, mode)


def process_simple(df: pd.DataFrame, client: LLMClient, mode: str) -> pd.DataFrame:
    """Generate embeddings sequentially."""
    out = df.copy()

    fields_to_embed = []
    if mode == "single":
        fields_to_embed = ["_combined_text"]
        out["_combined_text"] = out.apply(
            lambda r: f"Problem: {r['problem']}. Method: {r['method']}. Finding: {r['finding']}. Interpretation: {r['interpretation']}.",
            axis=1,
        )
    else:
        fields_to_embed = ["problem", "method", "finding", "interpretation"]

    for field in fields_to_embed:
        target_col = "embedding" if mode == "single" else f"{field}_embedding"
        embeddings = []
        print(f"Generating embeddings for {field}...")
        for _, row in tqdm(out.iterrows(), total=len(out), desc=f"Embedding {field}"):
            text = str(row[field]).strip()
            if not text:
                embeddings.append(None)
                continue
            try:
                emb = client.generate_embedding(text)
                embeddings.append(json.dumps(emb))
            except Exception as e:
                print(f"Error embedding row: {e}")
                embeddings.append(None)
        out[target_col] = embeddings

    if "_combined_text" in out.columns:
        out = out.drop(columns=["_combined_text"])

    return out


def process_batch(
    df: pd.DataFrame, client: LLMClient, mode: str, batch_size: int, batch_log_path: Path | None = None
) -> pd.DataFrame:
    """Generate embeddings using Batch API with optimal chunking and resume capabilities."""
    out = df.copy()

    # 1. Collect all texts to embed across all fields
    fields_to_embed = []
    if mode == "single":
        fields_to_embed = ["_combined_text"]
        out["_combined_text"] = out.apply(
            lambda r: f"Problem: {r['problem']}. Method: {r['method']}. Finding: {r['finding']}. Interpretation: {r['interpretation']}.",
            axis=1,
        )
    else:
        fields_to_embed = ["problem", "method", "finding", "interpretation"]

    all_prompts = {}
    for idx, row in out.iterrows():
        for field in fields_to_embed:
            text = str(row[field]).strip()
            if text:
                all_prompts[f"emb::{idx}::{field}"] = text

    if not all_prompts:
        print("No non-empty texts to embed.")
        return out

    all_results = {}
    active_batches = []
    
    if batch_log_path and batch_log_path.exists():
        try:
            with open(batch_log_path, 'r') as f:
                active_batches = json.load(f)
            print(f"Loaded {len(active_batches)} existing embedding batch jobs from log.")
        except Exception as e:
            print(f"Warning: Failed to load batch log: {e}")

    # 2. Initial poll of existing batches
    remaining_batches = []
    for b_id in active_batches:
        try:
            status_info = client.poll_batch(b_id)
            if status_info["status"] == "completed" and status_info["results"]:
                all_results.update(status_info["results"])
                print(f"Recovered completed batch {b_id[:25]}...")
            elif status_info["status"] in ["failed", "cancelled", "expired"]:
                print(f"Batch {b_id[:25]}... failed. Discarding from tracking.")
            else:
                remaining_batches.append(b_id)
        except Exception as e:
            print(f"Connection error while polling existing batch {b_id[:25]}... : {e}. Will retry.")
            remaining_batches.append(b_id)
            
    active_batches = remaining_batches

    # 3. Find missing prompts
    missing_keys = [k for k in all_prompts.keys() if k not in all_results]
    
    if missing_keys:
        total_missing = len(missing_keys)
        num_chunks = math.ceil(total_missing / batch_size)
        print(f"Submitting {total_missing} missing embeddings in {num_chunks} batch(es)...")
        
        for i in range(num_chunks):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_missing)
            chunk_keys = missing_keys[start_idx:end_idx]
            
            chunk_prompts = {k: all_prompts[k] for k in chunk_keys}
            
            try:
                batch_id = client.create_batch(chunk_prompts, endpoint="/v1/embeddings")
                active_batches.append(batch_id)
                print(f"Started new embedding batch job: {batch_id[:25]}...")
            except Exception as e:
                print(f"Failed to submit embedding batch chunk {i+1}: {e}")
                
        if batch_log_path:
            with open(batch_log_path, 'w') as f:
                json.dump(active_batches, f, indent=2)
    else:
        print("All embeddings successfully recovered from log! No new batches needed.")

    # 4. Continuous Poll loop
    while active_batches:
        print(f"Waiting for {len(active_batches)} embedding batch(es) to complete... next check in 60s")
        time.sleep(60)
        
        remaining = []
        for b_id in active_batches:
            try:
                status_info = client.poll_batch(b_id)
                status = status_info["status"]
                counts = status_info["request_counts"]
                print(
                    f"[{time.strftime('%H:%M:%S')}] Batch {b_id[:25]}... Status: {status}, "
                    f"Completed: {counts.get('completed', 0)}/{counts.get('total', 0)}"
                )
                
                if status == "completed":
                    if status_info["results"]:
                        all_results.update(status_info["results"])
                        print(f"Batch {b_id[:25]}... completed and results collected.")
                elif status in ["failed", "cancelled", "expired"]:
                    print(f"Batch {b_id[:25]}... failed ({status}).")
                    raise RuntimeError(f"Embedding batch failed: {status}")
                else:
                    remaining.append(b_id)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Warning: Connection error while polling batch {b_id[:25]}... : {e}. Will retry...")
                remaining.append(b_id)
                
        active_batches = remaining
        
        if batch_log_path:
            with open(batch_log_path, 'w') as f:
                json.dump(active_batches, f, indent=2)

    # 5. Map results back to DataFrame
    for field in fields_to_embed:
        target_col = "embedding" if mode == "single" else f"{field}_embedding"
        col_data = []
        for idx in out.index:
            key = f"emb::{idx}::{field}"
            res = all_results.get(key)
            col_data.append(json.dumps(res) if res is not None else None)
        out[target_col] = col_data

    if "_combined_text" in out.columns:
        out = out.drop(columns=["_combined_text"])

    return out
