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


def main():
    parser = argparse.ArgumentParser(description="Incremental AFP RAG Index Builder")
    parser.add_argument("--isabelle", default="/home/correia/Isabelle2025-2/bin/isabelle", help="Path to isabelle binary")
    parser.add_argument("--afp-dir", default="/home/correia/edel/external/afp-2025-2/thys", help="Path to AFP thys/ directory")
    parser.add_argument("--output", default="artifacts/rag_index", help="Output RAG index directory")
    parser.add_argument("--provider", default="voyage", choices=["openai", "voyage"], help="Embedding provider")
    parser.add_argument("--model", default="voyage-code-3", help="Embedding model name")
    parser.add_argument("--port", type=int, default=9147, help="Port to run REPL daemon on")
    
    args = parser.parse_args()
    
    isabelle_path = args.isabelle
    afp_thys_dir = Path(args.afp_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Validate API Key
    api_key_env = "VOYAGE_API_KEY" if args.provider == "voyage" else "OPENAI_API_KEY"
    api_key = os.getenv(api_key_env)
    if not api_key:
        print(f"Error: {api_key_env} is not set in environment or .env file.")
        sys.exit(1)
        
    # 2. Parse AFP Sessions
    roots_file = afp_thys_dir / "ROOTS"
    sessions = get_afp_sessions(roots_file)
    print(f"Loaded {len(sessions)} sessions from {roots_file}")
    
    # 3. Load Progress
    progress_file = output_dir / "progress.json"
    processed_sessions = set()
    if progress_file.exists():
        try:
            with open(progress_file, "r") as f:
                processed_sessions = set(json.load(f))
            print(f"Resuming job: {len(processed_sessions)} / {len(sessions)} sessions already indexed.")
        except Exception as e:
            print(f"Warning: Failed to load progress file, starting fresh: {e}")
            
    # 4. Load Existing Index (if any)
    master_index = NumpyRAGIndex()
    if (output_dir / "metadata.parquet").exists() and (output_dir / "embeddings.npz").exists():
        try:
            master_index.load(output_dir)
            print(f"Loaded existing index with {len(master_index.metadata)} lemmas.")
        except Exception as e:
            print(f"Warning: Failed to load existing index, creating new: {e}")
            
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
                m = re.search(r'Tcp_Handler: listening on 127\.0\.0\.1:(\d+)\s+\(token "([^"]+)"\)', line)
                if m:
                    token = m.group(2)
                    break
        except Exception as e:
            print(f"Error reading REPL startup: {e}")
            
        if not token:
            print(f"[ERROR] Failed to start REPL or retrieve token for '{session}'. Skipping.")
            repl_proc.terminate()
            repl_proc.wait()
            continue
            
        print(f"[REPL] Successfully started REPL on port {args.port} (token retrieved).")
        
        # C. Ingest lemmas
        df_new = None
        try:
            df_new = ingest_session_lemmas(
                host="127.0.0.1",
                port=args.port,
                token=token,
                # Filter specifically for theories belonging to the current session (excluding HOL/Pure parents)
                theory_filter=f"^{session}\\." if idx > 1 else None
            )
        except Exception as e:
            print(f"[ERROR] Ingestion failed for session '{session}': {e}")
        finally:
            # Shutdown REPL
            print("[REPL] Stopping REPL daemon...")
            repl_proc.terminate()
            try:
                repl_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                repl_proc.kill()
                repl_proc.wait()
                
        # D. Embed and index
        if df_new is not None and len(df_new) > 0:
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
                
                if len(df_embedded) > 0:
                    # Build temporary index
                    session_index = NumpyRAGIndex()
                    session_index.build_from_dataframe(df_embedded)
                    
                    # Merge into master index
                    if len(master_index.metadata) > 0:
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
                print(f"[ERROR] Embedding/indexing stage failed for '{session}': {e}")
                continue
        else:
            print(f"[Info] No new lemmas to index for session '{session}'.")
            
        # E. Record Progress
        processed_sessions.add(session)
        with open(progress_file, "w") as f:
            json.dump(list(processed_sessions), f, indent=2)
            
    print("\n==================================================")
    print(f"Job complete! All {len(sessions)} sessions processed.")
    print(f"Total size of final static RAG index: {len(master_index.metadata)} lemmas.")
    print("==================================================")


if __name__ == "__main__":
    main()
