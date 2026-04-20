"""Stage 2: Structured Abstracts - Epistemic aspect extraction using LLMs."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import pandas as pd
from tqdm import tqdm

from edel.io.llm import LLMClient, get_llm_client
from edel.prompts import create_structuring_prompt


def sentence_count(text: str) -> int:
    """Count the number of sentences in a text."""
    if not text:
        return 0
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)


def filter_abstracts(
    df: pd.DataFrame, min_sentences: int = 2, min_tokens: int = 20
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter out extremely short or empty abstracts and return a report."""
    report = {"initial_count": len(df)}
    if df.empty:
        return df, report

    # 1. Missing abstract
    df_clean = df[df["abstract_text"].notna()].copy()
    report["missing_abstract_count"] = len(df) - len(df_clean)
    
    # 2. Sentence filter
    df_clean["_n_sentences"] = df_clean["abstract_text"].apply(sentence_count)
    df_sent = df_clean[df_clean["_n_sentences"] >= min_sentences].copy()
    report["insufficient_sentences_count"] = len(df_clean) - len(df_sent)

    # 3. Token filter
    df_sent["_n_tokens"] = df_sent["abstract_text"].str.split().str.len()
    df_final = df_sent[df_sent["_n_tokens"] >= min_tokens].copy()
    report["insufficient_tokens_count"] = len(df_sent) - len(df_final)
    report["final_count"] = len(df_final)

    # Drop temporary columns
    out = df_final.drop(columns=["_n_sentences", "_n_tokens"])
    return out, report


def parse_and_merge_results(df: pd.DataFrame, results: dict[str, str]) -> pd.DataFrame:
    """Parse JSON strings from LLM and merge with existing pre-filled columns."""
    out = df.copy()

    def process_row(row: pd.Series) -> pd.Series:
        custom_id = f"request-{row.name}"
        raw_json = results.get(custom_id)
        if not raw_json:
            return row

        try:
            # Clean up potential markdown code blocks
            clean_json = raw_json.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            
            for key in ["problem", "method", "finding", "interpretation"]:
                snippet = data.get(key, "")
                if snippet == "UNKNOWN":
                    snippet = ""
                
                # Merge logic: original_val \n abstract: \n snippet
                original_val = row.get(key, "")
                if pd.isna(original_val):
                    original_val = ""
                
                if snippet:
                    row[key] = f"{original_val}\nabstract:\n{snippet}".strip()
                    
        except Exception as e:
            print(f"Error parsing JSON for row {row.name}: {e}")
            
        return row

    return out.apply(process_row, axis=1)


def process_simple(
    df: pd.DataFrame, client: LLMClient, topic: str | None = None
) -> dict[str, str]:
    """Process abstracts one by one (simple mode)."""
    results = {}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Structuring Abstracts"):
        prompt = create_structuring_prompt(
            title=row["title"],
            abstract_text=row["abstract_text"],
            keywords=row.get("keywords", []),
            topic=topic,
        )
        try:
            results[f"request-{idx}"] = client.generate(prompt)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            results[f"request-{idx}"] = "{}"
    return results


def process_batch(
    df: pd.DataFrame, client: LLMClient, config: dict, topic: str | None = None
) -> dict[str, str]:
    """Process abstracts using Batch API with optimal chunking."""
    stage_cfg = config.get("structured_abstracts", {})
    # OpenAI limit is 50,000 requests per batch.
    batch_size = stage_cfg.get("batch_size", 50000)
    
    total_items = len(df)
    all_results = {}
    
    # Split dataframe into chunks of batch_size
    num_chunks = (total_items + batch_size - 1) // batch_size
    print(f"Dividing {total_items} items into {num_chunks} batch(es) (size: {batch_size}).")

    for i in range(num_chunks):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_items)
        df_chunk = df.iloc[start_idx:end_idx]
        
        print(f"\n--- Processing Batch {i+1}/{num_chunks} ({len(df_chunk)} items) ---")
        
        prompts = {}
        for idx, row in df_chunk.iterrows():
            prompts[f"request-{idx}"] = create_structuring_prompt(
                title=row["title"],
                abstract_text=row["abstract_text"],
                keywords=row.get("keywords", []),
                topic=topic,
            )

        batch_id = client.create_batch(prompts)
        print(f"Started batch job: {batch_id}. Waiting for completion...")

        while True:
            status_info = client.poll_batch(batch_id)
            status = status_info["status"]
            counts = status_info["request_counts"]
            print(
                f"[{time.strftime('%H:%M:%S')}] Status: {status}, "
                f"Completed: {counts.get('completed', 0)}/{counts.get('total', 0)}"
            )
            
            if status in ["completed", "failed", "cancelled", "expired"]:
                break
            time.sleep(60)

        if status == "completed" and status_info["results"]:
            all_results.update(status_info["results"])
            print(f"Batch {i+1} completed and results collected.")
        else:
            print(f"Batch {i+1} failed or was incomplete. Status: {status}")
            # We continue to next batch but some data will be missing aspects
            
    return all_results


def run_structuring_stage(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict[str, int]]:
    """Run the Stage 2 pipeline: Abstract Structuring."""
    stage_cfg = config.get("structured_abstracts", {})
    processing_mode = config.get("processing_mode", "simple")
    
    # 1. Filter
    min_sentences = stage_cfg.get("min_sentences", 2)
    min_tokens = stage_cfg.get("min_tokens", 20)
    df_filtered, filter_report = filter_abstracts(df, min_sentences, min_tokens)
    print(f"Filtered to {len(df_filtered)} abstracts for structuring.")
    print(f"Filter Report: {filter_report}")

    if df_filtered.empty:
        return df_filtered, filter_report

    # 2. Setup Client
    client = get_llm_client(stage_cfg)
    topic = config.get("data", {}).get("provider", {}).get("topic_name")

    # 3. Process
    if processing_mode == "batch":
        results = process_batch(df_filtered, client, config, topic)
    else:
        results = process_simple(df_filtered, client, topic)

    # 4. Parse & Merge
    df_structured = parse_and_merge_results(df_filtered, results)
    return df_structured, filter_report
