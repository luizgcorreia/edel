#!/usr/bin/env python3
"""Automated script to build a static RAG index for all AFP entries incrementally.

Verifies and builds the recorded heap for each session on-demand, spawns the REPL,
ingests the aspects, embeds them, and incrementally updates the static index.
Tracks progress in a JSON file to support resuming interrupted jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from edel.il.ingest import ingest_session_lemmas
from edel.pipeline.embedding import run_embedding_stage
from edel.il.index import NumpyRAGIndex


def get_afp_sessions(roots_file: Path) -> list[str]:
    """Parse the AFP ROOTS file to get the list of session directory names."""
    if not roots_file.exists():
        print(f"Error: ROOTS file not found at {roots_file}")
        sys.exit(1)
        
    sessions = []
    with open(roots_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sessions.append(line)
    return sessions


def build_session_heap(isabelle_path: str, afp_thys_dir: Path, session: str) -> bool:
    """Build the recorded heap for the specific session using isabelle build."""
    print(f"\n[Isabelle] Verifying/building recorded heap for session: {session}...")
    cmd = [
        isabelle_path, "build",
        "-b",
        "-o", "record_theories=true",
        "-d", str(afp_thys_dir),
        "-j", "8",
        session
    ]
    try:
        # Run and print output live to the console
        result = subprocess.run(cmd, check=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to build session heap for {session}: {e}")
        return False


def get_isabelle_heaps_dir(isabelle_path: str) -> Path:
    """Get the ISABELLE_HEAPS directory path using isabelle getenv."""
    try:
        res = subprocess.run([isabelle_path, "getenv", "ISABELLE_HEAPS"], capture_output=True, text=True, check=True)
        for line in res.stdout.splitlines():
            if line.startswith("ISABELLE_HEAPS="):
                return Path(line.split("=", 1)[1].strip())
    except Exception as e:
        print(f"Warning: Failed to get ISABELLE_HEAPS via getenv: {e}")
    return Path(os.path.expanduser("~/.isabelle/Isabelle2025-2/heaps"))


def cleanup_session_heaps(heaps_dir: Path, session: str):
    """Deletes built heap images and logs for a session to save disk space."""
    if not heaps_dir.exists():
        return
    # Look for heap files matching the session name in subdirectories of heaps_dir
    for p in heaps_dir.glob(f"*/{session}"):
        if p.is_file():
            try:
                p.unlink()
                print(f"[Cleanup] Deleted heap image: {p}")
            except Exception as e:
                print(f"[Cleanup] Warning: Failed to delete heap image {p}: {e}")
                
    for p in heaps_dir.glob(f"*/log/{session}.*"):
        if p.is_file():
            try:
                p.unlink()
                print(f"[Cleanup] Deleted log file: {p}")
            except Exception as e:
                print(f"[Cleanup] Warning: Failed to delete log file {p}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Incremental AFP RAG Index Builder")
    parser.add_argument("--isabelle", default="/home/correia/Isabelle2025-2/bin/isabelle", help="Path to isabelle binary")
    parser.add_argument("--afp-dir", default="/home/correia/edel/external/afp-2025-2/thys", help="Path to AFP thys/ directory")
    parser.add_argument("--output", default="artifacts/rag_index", help="Output RAG index directory")
    parser.add_argument("--provider", default="voyage", choices=["openai", "voyage"], help="Embedding provider")
    parser.add_argument("--model", default="voyage-code-3", help="Embedding model name")
    parser.add_argument("--port", type=int, default=9147, help="Port to run REPL daemon on")
    parser.add_argument("--skip-embedding", "--skip-embeddings", "--skip-embeding", dest="skip_embedding", action="store_true", help="Skip embedding stage (for debugging segments/metadata)")
    parser.add_argument("--calculate-missing-embeddings", "--compute-missing-embeddings", dest="calculate_missing_embeddings", action="store_true", help="Load existing index, calculate missing embeddings, and exit.")
    parser.add_argument("--include-hol", action="store_true", help="Include parent HOL theories in addition to AFP session theories")
    parser.add_argument("--cleanup-heaps", action="store_true", help="Delete built heap images and logs after processing to save disk space")
    
    args = parser.parse_args()
    
    isabelle_path = args.isabelle
    heaps_dir = get_isabelle_heaps_dir(isabelle_path)
    afp_thys_dir = Path(args.afp_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Validate API Key (Only if we are actually embedding)
    api_key = None
    if not args.skip_embedding or args.calculate_missing_embeddings:
        api_key_env = "VOYAGE_API_KEY" if args.provider == "voyage" else "OPENAI_API_KEY"
        api_key = os.getenv(api_key_env)
        if not api_key:
            print(f"Error: {api_key_env} is not set in environment or .env file.")
            sys.exit(1)
            
    # 2. Load Existing Index (if any)
    master_index = NumpyRAGIndex()
    index_exists = (output_dir / "metadata.parquet").exists() and (output_dir / "embeddings.npz").exists()
    if index_exists:
        try:
            master_index.load(output_dir)
            print(f"Loaded existing index with {len(master_index.metadata)} lemmas.")
        except Exception as e:
            print(f"Warning: Failed to load existing index: {e}")
            if args.calculate_missing_embeddings:
                print("Error: Cannot calculate missing embeddings because loading index failed.")
                sys.exit(1)
    else:
        if args.calculate_missing_embeddings:
            print(f"Error: Existing index not found in {output_dir}. Cannot calculate missing embeddings.")
            sys.exit(1)
            
    # 3. Calculate Missing Embeddings Flow
    if args.calculate_missing_embeddings:
        N = len(master_index.metadata)
        M = N
        aspects = ["problem", "method", "finding", "interpretation"]
        for aspect in aspects:
            arr = master_index.embeddings.get(aspect)
            if arr is None:
                M = 0
            else:
                M = min(M, len(arr))
                
        if M >= N:
            print("No missing embeddings found. All lemmas have embeddings.")
        else:
            print(f"Found {N - M} lemmas missing embeddings. Calculating...")
            missing_metadata = master_index.metadata[M:]
            
            # Truncate existing embeddings to M to ensure perfect alignment
            for aspect in aspects:
                if master_index.embeddings.get(aspect) is not None:
                    master_index.embeddings[aspect] = master_index.embeddings[aspect][:M]
                    
            chunk_size = 200
            for i in range(0, len(missing_metadata), chunk_size):
                chunk_meta = missing_metadata[i : i + chunk_size]
                print(f"\n[Embedding] Processing missing chunk {i // chunk_size + 1}/{(len(missing_metadata) + chunk_size - 1) // chunk_size} (lemmas {M + i} to {M + i + len(chunk_meta) - 1})...")
                
                df_chunk = pd.DataFrame(chunk_meta)
                embed_config = {
                    "embedding": {
                        "provider": args.provider,
                        "model": args.model,
                        "api_key": api_key,
                        "required_aspects": [],  # Avoid filtering out any rows
                    },
                    "processing_mode": "simple"
                }
                try:
                    df_embedded = run_embedding_stage(df_chunk, embed_config)
                    
                    # Find embedding dimension from existing arrays or from the current df_embedded
                    embedding_dim = None
                    for aspect in aspects:
                        if master_index.embeddings[aspect] is not None and len(master_index.embeddings[aspect]) > 0:
                            embedding_dim = master_index.embeddings[aspect].shape[1]
                            break
                            
                    if embedding_dim is None:
                        for aspect in aspects:
                            col = f"{aspect}_embedding"
                            if col in df_embedded.columns:
                                for val in df_embedded[col]:
                                    if val is not None:
                                        if isinstance(val, str):
                                            try:
                                                parsed_val = json.loads(val)
                                                if parsed_val:
                                                    embedding_dim = len(parsed_val)
                                                    break
                                            except Exception:
                                                pass
                                        else:
                                            embedding_dim = len(val)
                                            break
                            if embedding_dim is not None:
                                break
                                
                    if embedding_dim is None:
                        embedding_dim = 1024  # default fallback
                        
                    for aspect in aspects:
                        col = f"{aspect}_embedding"
                        col_data = df_embedded[col].tolist() if col in df_embedded.columns else [None] * len(df_chunk)
                        
                        # Parse each entry and replace None/invalid with zero vectors
                        parsed_data = []
                        for val in col_data:
                            if val is None:
                                parsed_data.append(np.zeros(embedding_dim, dtype=np.float32))
                            elif isinstance(val, str):
                                try:
                                    parsed_val = json.loads(val)
                                    if parsed_val:
                                        parsed_data.append(np.array(parsed_val, dtype=np.float32))
                                    else:
                                        parsed_data.append(np.zeros(embedding_dim, dtype=np.float32))
                                except Exception:
                                    parsed_data.append(np.zeros(embedding_dim, dtype=np.float32))
                            else:
                                parsed_data.append(np.array(val, dtype=np.float32))
                                
                        chunk_arr = np.vstack(parsed_data)
                        
                        old_arr = master_index.embeddings.get(aspect)
                        if old_arr is None or len(old_arr) == 0:
                            master_index.embeddings[aspect] = chunk_arr
                        else:
                            master_index.embeddings[aspect] = np.concatenate([old_arr, chunk_arr], axis=0)
                            
                    # Save incremental index after each chunk
                    master_index.save(output_dir)
                    print(f"[Index] Saved progress. Unified index now has {len(master_index.embeddings['problem'])} embeddings.")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to embed/index chunk: {e}")
                    sys.exit(1)
                    
        # Run Phase 2: Landscape Height Post-Processing
        print("\n==================================================")
        print("Running Phase 2: Landscape Height Post-Processing...")
        from edel.il.compute_landscape_height import compute_and_save_landscape_height
        compute_and_save_landscape_height(output_dir)
        print("Landscape height computation complete.")
        print("==================================================")
        return

    # 4. Parse AFP Sessions
    roots_file = afp_thys_dir / "ROOTS"
    sessions = get_afp_sessions(roots_file)
    print(f"Loaded {len(sessions)} sessions from {roots_file}")
    
    # 5. Load Progress
    progress_file = output_dir / "progress.json"
    processed_sessions = set()
    if progress_file.exists():
        try:
            with open(progress_file, "r") as f:
                processed_sessions = set(json.load(f))
            print(f"Resuming job: {len(processed_sessions)} / {len(sessions)} sessions already indexed.")
        except Exception as e:
            print(f"Warning: Failed to load progress file, starting fresh: {e}")
            
    # 5. Process Loop
    for idx, session in enumerate(sessions, 1):
        print(f"\n==================================================")
        print(f"Processing session [{idx}/{len(sessions)}]: {session}")
        print(f"==================================================")
        
        if session in processed_sessions:
            print(f"[SKIP] Session '{session}' already processed.")
            continue
            
        # A. Build the session heap
        build_ok = build_session_heap(isabelle_path, afp_thys_dir, session)
        if not build_ok:
            print(f"[WARNING] Skipping session '{session}' due to heap build failure.")
            # Record it as processed to avoid blocking future resumes
            processed_sessions.add(session)
            with open(progress_file, "w") as f:
                json.dump(list(processed_sessions), f, indent=2)
            if args.cleanup_heaps:
                cleanup_session_heaps(heaps_dir, session)
            continue
            
        # B. Start REPL Daemon
        print(f"[REPL] Starting REPL daemon for session '{session}'...")
        repl_proc = subprocess.Popen([
            sys.executable, "AutoCorrode/ir/repl.py",
            "--isabelle", isabelle_path,
            "--session", session,
            "--dir", str(afp_thys_dir),
            "--port", str(args.port),
            "--mcp"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        token = None
        start_time = time.time()
        
        # Read stdout line-by-line to get the auth token
        try:
            while time.time() - start_time < 90:  # 90s timeout
                # readline can block, so we set a timeout check
                line = repl_proc.stdout.readline().decode("utf-8", errors="replace")
                if not line:
                    break
                print(f"  [REPL Out] {line.strip()}")
                
                # Check for the old token format
                m = re.search(r'Tcp_Handler: listening on 127\.0\.0\.1:(\d+)\s+\(token "([^"]+)"\)', line)
                if m:
                    token = m.group(2)
                    break
                    
                # Check for the new token format
                m2 = re.search(r'IR_Repl\.token:\s+(\S+)', line)
                if m2:
                    token = m2.group(1)
                    
                # If we have the token via the new format, wait until the REPL is ready
                if token and "REPL ready" in line:
                    break
        except Exception as e:
            print(f"Error reading REPL startup: {e}")
            
        if not token:
            print(f"[ERROR] Failed to start REPL or retrieve token for '{session}'. Exiting.")
            repl_proc.terminate()
            repl_proc.wait()
            sys.exit(1)
            
        print(f"[REPL] Successfully started REPL on port {args.port} (token retrieved).")
        
        # C. Ingest lemmas
        df_new = None
        ingest_failed = False
        try:
            if args.include_hol:
                # Match current session OR any HOL/HOL-xxx theory
                theory_filter = f"^(?:{session}|HOL|HOL-[a-zA-Z0-9_-]+)\\."
            else:
                theory_filter = f"^{session}\\."
                
            df_new = ingest_session_lemmas(
                host="127.0.0.1",
                port=args.port,
                token=token,
                theory_filter=theory_filter
            )
        except Exception as e:
            print(f"[ERROR] Ingestion failed for session '{session}': {e}")
            ingest_failed = True
        finally:
            # Shutdown REPL
            print("[REPL] Stopping REPL daemon...")
            repl_proc.terminate()
            try:
                repl_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                repl_proc.kill()
                repl_proc.wait()
                
        if ingest_failed:
            sys.exit(1)
                
        # D. Embed and index
        if df_new is not None and len(df_new) > 0:
            if len(master_index.metadata) > 0:
                existing_titles = {item["title"] for item in master_index.metadata}
                df_new = df_new[~df_new["title"].isin(existing_titles)].reset_index(drop=True)
                print(f"[Deduplication] Filtered out lemmas already in the index. Remaining: {len(df_new)} new lemmas.")
                
            df_embedded = None
            if len(df_new) > 0:
                if args.skip_embedding:
                    print(f"[Embedding] Skipping embedding stage as requested for session '{session}'.")
                    df_embedded = df_new
                else:
                    print(f"[Embedding] Generating embeddings for {len(df_new)} new lemmas...")
                    try:
                        embed_config = {
                            "embedding": {
                                "provider": args.provider,
                                "model": args.model,
                                "api_key": api_key,
                            },
                            "processing_mode": "simple"
                        }
                        df_embedded = run_embedding_stage(df_new, embed_config)
                        print(f"[Embedding] Embeddings generated successfully. Valid: {len(df_embedded)}")
                    except Exception as e:
                        print(f"[ERROR] Embedding stage failed for '{session}': {e}")
                        continue
            
            try:
                if df_embedded is not None and len(df_embedded) > 0:
                    # Build temporary index
                    session_index = NumpyRAGIndex()
                    session_index.build_from_dataframe(df_embedded)
                    
                    # Merge into master index
                    if len(master_index.metadata) > 0:
                        # Merge lemmas
                        master_index.metadata.extend(session_index.metadata)
                        for aspect in ["problem", "method", "finding", "interpretation"]:
                            old_arr = master_index.embeddings.get(aspect)
                            new_arr = session_index.embeddings.get(aspect)
                            if new_arr is not None:
                                if old_arr is not None:
                                    master_index.embeddings[aspect] = np.concatenate([old_arr, new_arr], axis=0)
                                else:
                                    master_index.embeddings[aspect] = new_arr
                    else:
                        master_index = session_index
                        
                    # Save incremental index
                    master_index.save(output_dir)
                    print(f"[Index] Unified index updated with '{session}' (Total size: {len(master_index.metadata)} lemmas).")
            except Exception as e:
                print(f"[ERROR] Indexing stage failed for '{session}': {e}")
                continue
        else:
            print(f"[Info] No new lemmas to index for session '{session}'.")
            
        # E. Record Progress
        processed_sessions.add(session)
        with open(progress_file, "w") as f:
            json.dump(list(processed_sessions), f, indent=2)
            
        # F. Cleanup Session Heaps to save space if requested
        if args.cleanup_heaps:
            cleanup_session_heaps(heaps_dir, session)
            
    print("\n==================================================")
    print(f"Job complete! All {len(sessions)} sessions processed.")
    print(f"Total size of final static RAG index: {len(master_index.metadata)} lemmas.")
    print("==================================================")

    # Run Phase 2: Landscape Height Post-Processing
    print("\n==================================================")
    print("Running Phase 2: Landscape Height Post-Processing...")
    from edel.il.compute_landscape_height import compute_and_save_landscape_height
    compute_and_save_landscape_height(output_dir)
    print("Landscape height computation complete.")
    print("==================================================")


if __name__ == "__main__":
    main()
