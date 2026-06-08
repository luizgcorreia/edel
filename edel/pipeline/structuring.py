"""Stage 2: Structured Abstracts - Epistemic aspect extraction using LLMs."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
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
    df: pd.DataFrame, min_sentences: int = 2, min_tokens: int = 20, target_lang: str | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter out extremely short or empty abstracts and return a report."""
    report = {"initial_count": len(df)}
    if df.empty:
        return df, report

    # 1. Target language filter
    df_lang = df.copy()
    if target_lang and "language" in df_lang.columns:
        df_lang = df_lang[df_lang["language"] == target_lang].copy()
        report["out_of_target_language_count"] = len(df) - len(df_lang)

    # 2. Missing abstract
    df_clean = df_lang[df_lang["abstract_text"].notna()].copy()
    report["missing_abstract_count"] = len(df_lang) - len(df_clean)
    
    # 3. Sentence filter
    df_clean["_n_sentences"] = df_clean["abstract_text"].apply(sentence_count)
    df_sent = df_clean[df_clean["_n_sentences"] >= min_sentences].copy()
    report["insufficient_sentences_count"] = len(df_clean) - len(df_sent)

    # 3. Token filter
    df_sent["_n_tokens"] = df_sent["abstract_text"].str.split().str.len()
    df_final = df_sent[df_sent["_n_tokens"] >= min_tokens].copy()
    report["insufficient_tokens_count"] = len(df_sent) - len(df_final)
    report["final_count"] = len(df_final)

    # 4. Language distribution (of the final filtered works)
    if "language" in df_final.columns:
        lang_counts = df_final["language"].value_counts(dropna=False).to_dict()
        # Convert any float/NaN keys to string for clean JSON serialization
        report["language_distribution"] = {str(k): int(v) for k, v in lang_counts.items()}

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


def compute_segmentation_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Calculate metrics on the segmentation quality (length, redundancy, etc)."""
    aspect_columns = ["problem", "method", "finding", "interpretation"]
    # Check if we have at least some of the columns
    present_cols = [col for col in aspect_columns if col in df.columns]
    if not present_cols:
        return {}

    results = {}

    # 1. Redundancy (Duplication between aspects)
    for i, col1 in enumerate(present_cols):
        for j, col2 in enumerate(present_cols):
            if i < j:
                # Percentage of rows where snippets are identical
                # (Ignore empty ones for duplication check to avoid inflation)
                mask = (df[col1].str.len() > 0) & (df[col2].str.len() > 0)
                if mask.any():
                    val = (df[mask][col1] == df[mask][col2]).mean()
                    results[f"dup_{col1}_{col2}"] = float(val)

    # 2. Mean segment length (tokens)
    for col in present_cols:
        results[f"len_{col}"] = float(df[col].fillna("").str.split().str.len().mean())

    # 3. Coverage (Segmentation ratio)
    if "abstract_text" in df.columns:
        df_temp = df.copy()
        for col in present_cols:
            df_temp[f"_{col}_len"] = df_temp[col].fillna("").str.split().str.len()

        df_temp["_segmented_total_len"] = df_temp[[f"_{col}_len" for col in present_cols]].sum(axis=1)
        df_temp["_abstract_len"] = df_temp["abstract_text"].fillna("").str.split().str.len()

        # Avoid division by zero
        df_temp = df_temp[df_temp["_abstract_len"] > 0]
        if not df_temp.empty:
            ratio = df_temp["_segmented_total_len"] / df_temp["_abstract_len"]
            results["seg_ratio_mean"] = float(ratio.mean())
            results["seg_ratio_std"] = float(ratio.std())
            results["abstract_len_mean"] = float(df_temp["_abstract_len"].mean())
            results["seg_total_mean"] = float(df_temp["_segmented_total_len"].mean())

    return results


def process_simple(
    df: pd.DataFrame, client: LLMClient, topic: str | None = None, definitions: dict[str, str] | None = None
) -> dict[str, str]:
    """Process abstracts one by one (simple mode)."""
    results = {}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Structuring Abstracts"):
        prompt = create_structuring_prompt(
            title=row["title"],
            abstract_text=row["abstract_text"],
            keywords=row.get("keywords", []),
            topic=topic,
            definitions=definitions,
        )
        try:
            results[f"request-{idx}"] = client.generate(prompt)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            results[f"request-{idx}"] = "{}"
    return results


def process_batch(
    df: pd.DataFrame, client: LLMClient, batch_size: int, topic: str | None = None, batch_log_path: Path | None = None, definitions: dict[str, str] | None = None
) -> dict[str, str]:
    """Process abstracts using Batch API with optimal chunking and resume capabilities."""

    total_items = len(df)
    all_results = {}
    active_batches = []

    if batch_log_path and batch_log_path.exists():
        try:
            with open(batch_log_path, 'r') as f:
                active_batches = json.load(f)
            print(f"Loaded {len(active_batches)} existing batch jobs from log.")
        except Exception as e:
            print(f"Warning: Failed to load batch log: {e}")

    # 1. Continuous Poll loop for initial active batches (if any loaded from log)
    while active_batches:
        print(f"Waiting for {len(active_batches)} existing batch(es) to complete... next check in 60s")

        remaining = []
        for b_id in active_batches:
            try:
                status_info = client.poll_batch(b_id)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Warning: Connection error while polling batch {b_id[:25]}... : {e}. Will retry...")
                remaining.append(b_id)
                continue

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
                # Remove terminally failed recovered batches and resubmit missing requests below.
            else:
                remaining.append(b_id)

        active_batches = remaining
        if batch_log_path:
            with open(batch_log_path, 'w') as f:
                json.dump(active_batches, f, indent=2)

        if active_batches:
            time.sleep(60)

    # 2. Find missing items after waiting/recovering
    missing_indices = [idx for idx in df.index if f"request-{idx}" not in all_results]
    
    if missing_indices:
        df_missing = df.loc[missing_indices]
        total_missing = len(df_missing)
        num_chunks = (total_missing + batch_size - 1) // batch_size
        print(f"Submitting {total_missing} missing items in {num_chunks} batch(es)...")
        
        for i in range(num_chunks):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_missing)
            df_chunk = df_missing.iloc[start_idx:end_idx]
            
            prompts = {}
            for idx, row in df_chunk.iterrows():
                prompts[f"request-{idx}"] = create_structuring_prompt(
                    title=row["title"],
                    abstract_text=row["abstract_text"],
                    keywords=row.get("keywords", []),
                    topic=topic,
                    definitions=definitions,
                )
            
            try:
                batch_id = client.create_batch(prompts)
                active_batches.append(batch_id)
                print(f"Started new batch job: {batch_id[:25]}...")
            except Exception as e:
                print(f"Failed to submit batch chunk {i+1}: {e}")

        # Save updated active batches
        if batch_log_path:
            with open(batch_log_path, 'w') as f:
                json.dump(active_batches, f, indent=2)

        # 3. Continuous Poll loop for the newly submitted batches
        while active_batches:
            print(f"Waiting for {len(active_batches)} new batch(es) to complete... next check in 60s")
            time.sleep(60)

            remaining = []
            for b_id in active_batches:
                try:
                    status_info = client.poll_batch(b_id)
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] Warning: Connection error while polling batch {b_id[:25]}... : {e}. Will retry...")
                    remaining.append(b_id)
                    continue

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
                    if batch_log_path:
                        with open(batch_log_path, 'w') as f:
                            json.dump(remaining, f, indent=2)
                    raise RuntimeError(f"Batch failed: {status}")
                else:
                    remaining.append(b_id)

            active_batches = remaining
            if batch_log_path:
                with open(batch_log_path, 'w') as f:
                    json.dump(active_batches, f, indent=2)
    else:
        print("All items successfully recovered! No new batches needed.")
                
    return all_results


def run_structuring_stage(df: pd.DataFrame, config: dict, base_path: str | Path = "artifacts") -> tuple[pd.DataFrame, dict[str, int]]:
    """Run the Stage 2 pipeline: Abstract Structuring."""
    stage_cfg = config.get("structured_abstracts", {})
    processing_mode = config.get("processing_mode", "simple")
    target_lang = stage_cfg.get("language")
    definitions = stage_cfg.get("aspect_definitions")

    # Check if provider is "none"
    if stage_cfg.get("provider") == "none":
        required_cols = ["problem", "method", "finding", "interpretation"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Structured abstracts provider is 'none' but required column '{col}' is missing from input DataFrame."
                )

        df_filtered = df.copy()
        filter_report = {
            "initial_count": len(df),
            "final_count": len(df),
        }

        # 1b. Sample if requested
        n_docs = stage_cfg.get("n_documents")
        if n_docs and n_docs < len(df_filtered):
            seed = config.get("random_seed", 42)
            print(f"Sampling {n_docs} abstracts from filtered set (seed: {seed})...")
            df_filtered = df_filtered.sample(n=n_docs, random_state=seed)
            filter_report["sampled_count"] = n_docs
            filter_report["final_count"] = n_docs

        print(f"Final selection: {len(df_filtered)} abstracts (provider: none).")
        print(f"Filter/Sampling Report: {filter_report}")

        # Compute Segmentation Metrics
        seg_metrics = compute_segmentation_metrics(df_filtered)
        filter_report.update(seg_metrics)

        return df_filtered, filter_report

    # 1. Filter
    min_sentences = stage_cfg.get("min_sentences", 4)
    min_tokens = stage_cfg.get("min_tokens", 80)
    df_filtered, filter_report = filter_abstracts(df, min_sentences, min_tokens, target_lang)
    
    # 1b. Sample if requested
    n_docs = stage_cfg.get("n_documents")
    if n_docs and n_docs < len(df_filtered):
        seed = config.get("random_seed", 42)
        print(f"Sampling {n_docs} abstracts from filtered set (seed: {seed})...")
        df_filtered = df_filtered.sample(n=n_docs, random_state=seed)
        filter_report["sampled_count"] = n_docs

    print(f"Final selection: {len(df_filtered)} abstracts for structuring.")
    print(f"Filter/Sampling Report: {filter_report}")

    if df_filtered.empty:
        return df_filtered, filter_report

    # 2. Setup Client
    client = get_llm_client(stage_cfg)
    topic = config.get("data", {}).get("provider", {}).get("topic_name")

    # 3. Process
    if processing_mode == "batch":
        from edel.io.artifact import make_stage_artifact
        batch_size = stage_cfg.get("batch_size", 1000)
        
        batch_log_art = make_stage_artifact(config, Path(base_path), "structured_abstracts", "batch_log")
        batch_log_path = batch_log_art.path_prefix.with_suffix(".json")
        batch_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = process_batch(df_filtered, client, batch_size, topic, batch_log_path=batch_log_path, definitions=definitions)
    else:
        results = process_simple(df_filtered, client, topic, definitions=definitions)

    # 4. Parse & Merge
    df_structured = parse_and_merge_results(df_filtered, results)

    # 5. Compute Segmentation Metrics
    seg_metrics = compute_segmentation_metrics(df_structured)
    filter_report.update(seg_metrics)

    return df_structured, filter_report
