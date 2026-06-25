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


def filter_by_aspects(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter out rows where any of the four aspects (problem, method, finding, interpretation) are empty or missing."""
    aspects = ["problem", "method", "finding", "interpretation"]
    initial_count = len(df)
    
    report = {
        "initial_count": initial_count,
        "aspect_coverage": {}
    }
    
    # 1. Calculate individual coverage
    for aspect in aspects:
        if aspect in df.columns:
            # We treat empty string, NaN, or whitespace-only as failed segmentation
            is_valid = df[aspect].notna() & (df[aspect].astype(str).str.strip() != "")
            stayed = int(is_valid.sum())
            filtered = initial_count - stayed
        else:
            filtered = initial_count
            stayed = 0
            
        report["aspect_coverage"][aspect] = {
            "filtered": filtered,
            "stayed": stayed
        }
        
    # 2. Apply combined filter
    if all(aspect in df.columns for aspect in aspects):
        is_all_valid = pd.Series(True, index=df.index)
        for aspect in aspects:
            is_all_valid &= df[aspect].notna() & (df[aspect].astype(str).str.strip() != "")
        df_filtered = df[is_all_valid].copy()
    else:
        # If any aspect column is completely missing, filter everything
        df_filtered = df.iloc[0:0].copy()
        
    report["total_filtered"] = initial_count - len(df_filtered)
    report["final_count"] = len(df_filtered)
    
    return df_filtered, report


def run_embedding_stage(
    df: pd.DataFrame, config: dict, base_path: str | Path = "artifacts", return_report: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Orchestrate the text embedding stage."""
    # Filter out entries where any of the 4 aspects are missing
    df_filtered, filter_report = filter_by_aspects(df)
    df = df_filtered

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

    if provider == "none":
        fields_to_embed = ["problem", "method", "finding", "interpretation"] if mode != "single" else ["_combined_text"]
        target_cols = [f"{f}_embedding" for f in fields_to_embed] if mode != "single" else ["embedding"]
        for col in target_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Embedding provider is 'none' but required column '{col}' is missing from the input DataFrame."
                )
        print("Stage 3: Embedding provider is 'none'. Skipping LLM API calls.")
        if return_report:
            return df, filter_report
        return df

    # Pass the entire embed_cfg so that additional kwargs like 'location' reach the LLM client
    client = get_llm_client(embed_cfg)

    if processing_mode == "batch":
        from edel.io.artifact import make_stage_artifact
        batch_log_art = make_stage_artifact(config, Path(base_path), "embeddings", "batch_log")
        batch_log_path = batch_log_art.path_prefix.with_suffix(".json")
        batch_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_out = process_batch(df, client, mode, batch_size, provider=provider, batch_log_path=batch_log_path)
    else:
        df_out = process_simple(df, client, mode, batch_size=128)

    if return_report:
        return df_out, filter_report
    return df_out


def process_simple(df: pd.DataFrame, client: LLMClient, mode: str, batch_size: int = 128) -> pd.DataFrame:
    """Generate embeddings sequentially using batching if supported."""
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
        
        # Collect all rows for this field
        rows_data = []
        for idx, row in out.iterrows():
            text = str(row[field]).strip()
            rows_data.append((idx, text))
            
        embeddings_map = {}
        non_empty_batches = []
        current_batch = []
        
        for idx, text in rows_data:
            if not text:
                embeddings_map[idx] = None
                continue
            current_batch.append((idx, text))
            if len(current_batch) == batch_size:
                non_empty_batches.append(current_batch)
                current_batch = []
        if current_batch:
            non_empty_batches.append(current_batch)
            
        print(f"Generating embeddings for {field} in chunks of {batch_size}...")
        
        with tqdm(total=len(rows_data), desc=f"Embedding {field}") as pbar:
            for batch in non_empty_batches:
                batch_indices = [idx for idx, _ in batch]
                batch_texts = [text for _, text in batch]
                
                try:
                    res = client.generate_embedding(batch_texts)
                    for idx, emb in zip(batch_indices, res):
                        embeddings_map[idx] = json.dumps(emb)
                except Exception as e:
                    print(f"Batch embedding failed: {e}. Falling back to row-by-row...")
                    for idx, text in batch:
                        try:
                            emb = client.generate_embedding(text)
                            embeddings_map[idx] = json.dumps(emb)
                        except Exception as inner_e:
                            print(f"Error embedding row: {inner_e}")
                            embeddings_map[idx] = None
                pbar.update(len(batch))
            
            # Update progress bar for any empty texts that we skipped
            empty_count = sum(1 for _, text in rows_data if not text)
            pbar.update(empty_count)
            
        embeddings_list = [embeddings_map[idx] for idx in out.index]
        out[target_col] = embeddings_list

    if "_combined_text" in out.columns:
        out = out.drop(columns=["_combined_text"])

    return out


def process_batch(
    df: pd.DataFrame, client: LLMClient, mode: str, batch_size: int, provider: str = "openai", batch_log_path: Path | None = None
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
    
    # 4. Throttled Submission & Poll Loop
    # For OpenAI, we limit to 1 active batch to avoid enqueued token limits.
    # For Vertex AI, we allow up to 5 concurrent batches.
    max_active = 1 if provider == "openai" else 5
    
    total_missing = len(missing_keys)
    next_chunk_idx = 0
    
    if total_missing > 0:
        print(f"Starting throttled submission for {total_missing} missing embeddings (max_active={max_active})...")
    
    while missing_keys[next_chunk_idx * batch_size:] or active_batches:
        # A. Fill the "active" queue if there is space
        while len(active_batches) < max_active and next_chunk_idx * batch_size < total_missing:
            start_idx = next_chunk_idx * batch_size
            end_idx = min((next_chunk_idx + 1) * batch_size, total_missing)
            chunk_keys = missing_keys[start_idx:end_idx]
            
            chunk_prompts = {k: all_prompts[k] for k in chunk_keys}
            
            try:
                batch_id = client.create_batch(chunk_prompts, endpoint="/v1/embeddings")
                active_batches.append(batch_id)
                print(f"[{time.strftime('%H:%M:%S')}] Started batch {next_chunk_idx + 1}/{math.ceil(total_missing/batch_size)}: {batch_id[:25]}...")
                next_chunk_idx += 1
                
                # Update persistent log immediately
                if batch_log_path:
                    with open(batch_log_path, 'w') as f:
                        json.dump(active_batches, f, indent=2)
            except Exception as e:
                if "limit reached" in str(e).lower() or "429" in str(e):
                    print(f"Rate limit or enqueued limit reached. Waiting for active batches to clear...")
                    break # Stop submitting for this cycle
                else:
                    print(f"Failed to submit batch chunk {next_chunk_idx + 1}: {e}")
                    # We will retry this chunk in the next iteration of the outer loop
                    time.sleep(10)
                    break

        if not active_batches:
            if next_chunk_idx * batch_size < total_missing:
                time.sleep(30)
                continue
            else:
                break

        # B. Poll active batches
        remaining = []
        finished_this_cycle = False
        
        for b_id in active_batches:
            try:
                status_info = client.poll_batch(b_id)
                status = status_info["status"]
                counts = status_info["request_counts"]
                
                if status == "completed":
                    if status_info["results"]:
                        all_results.update(status_info["results"])
                        print(f"[{time.strftime('%H:%M:%S')}] Batch {b_id[:25]}... completed.")
                        finished_this_cycle = True
                    else:
                        # Sometimes results are missing temporarily
                        remaining.append(b_id)
                elif status in ["failed", "cancelled", "expired"]:
                    print(f"[{time.strftime('%H:%M:%S')}] Batch {b_id[:25]}... failed ({status}).")
                    # We don't raise error here, we just won't have those results
                    # and they will be resubmitted if we run again.
                else:
                    remaining.append(b_id)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Warning: Connection error polling {b_id[:25]}... : {e}")
                remaining.append(b_id)
                
        if active_batches != remaining:
            active_batches = remaining
            if batch_log_path:
                with open(batch_log_path, 'w') as f:
                    json.dump(active_batches, f, indent=2)
        
        if active_batches or next_chunk_idx * batch_size < total_missing:
            # If we didn't finish any batch, wait 60s. 
            # If we did finish one, we might want to submit the next one immediately.
            if not finished_this_cycle:
                time.sleep(60)
            else:
                # Small grace period before submitting next
                time.sleep(5)

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
