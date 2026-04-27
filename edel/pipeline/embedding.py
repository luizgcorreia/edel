"""Stage 3: Text Embedding."""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from edel.io.llm import LLMClient, get_llm_client


def run_embedding_stage(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Orchestrate the text embedding stage."""
    embed_cfg = config.get("embedding", {})
    provider = embed_cfg.get("provider", "openai")
    model = embed_cfg.get("model", "text-embedding-ada-002")
    mode = embed_cfg.get("mode", "multi")  # "single" or "multi"
    processing_mode = config.get("processing_mode", "batch")
    batch_size = embed_cfg.get("batch_size", 5000)

    client = get_llm_client({"provider": provider, "model": model})

    if processing_mode == "batch":
        return process_batch(df, client, mode, batch_size)
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
    df: pd.DataFrame, client: LLMClient, mode: str, batch_size: int
) -> pd.DataFrame:
    """Generate embeddings using Batch API with optimal chunking."""
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

    # Flattened list of (row_idx, field, text)
    tasks = []
    for idx, row in out.iterrows():
        for field in fields_to_embed:
            text = str(row[field]).strip()
            if text:
                tasks.append((idx, field, text))

    if not tasks:
        print("No non-empty texts to embed.")
        return out

    # 2. Process in chunks
    num_chunks = math.ceil(len(tasks) / batch_size)
    print(f"Dividing {len(tasks)} embedding requests into {num_chunks} batch(es) (size: {batch_size}).")

    all_results = {}

    for i in range(num_chunks):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(tasks))
        chunk = tasks[start_idx:end_idx]

        print(f"\n--- Processing Embedding Batch {i+1}/{num_chunks} ({len(chunk)} items) ---")

        # Map task to custom_id: "chunkI_taskJ"
        prompts_with_ids = {f"t{j}": t[2] for j, t in enumerate(chunk)}

        batch_id = client.create_batch(prompts_with_ids, endpoint="/v1/embeddings")
        print(f"Started batch job: {batch_id}. Waiting for completion...")

        while True:
            status = client.poll_batch(batch_id)
            if status["status"] == "completed":
                print(f"Batch {i+1} completed and results collected.")
                results = status["results"]
                for j, (row_idx, field, _) in enumerate(chunk):
                    custom_id = f"t{j}"
                    if custom_id in results:
                        all_results[(row_idx, field)] = results[custom_id]
                break
            elif status["status"] in ["failed", "cancelled", "expired"]:
                print(f"Batch {i+1} failed with status: {status['status']}")
                break
            
            counts = status.get("request_counts", {})
            comp = counts.get("completed", 0)
            total = counts.get("total", 0)
            print(f"[{time.strftime('%H:%M:%S')}] Status: {status['status']}, Completed: {comp}/{total}")
            time.sleep(10)

    # 3. Map results back to DataFrame
    for field in fields_to_embed:
        target_col = "embedding" if mode == "single" else f"{field}_embedding"
        col_data = []
        for idx in out.index:
            res = all_results.get((idx, field))
            col_data.append(json.dumps(res) if res is not None else None)
        out[target_col] = col_data

    if "_combined_text" in out.columns:
        out = out.drop(columns=["_combined_text"])

    return out
