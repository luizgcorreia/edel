"""Build script to ingest and embed AFP theory aspects into a static RAG index."""

from __future__ import annotations

import argparse
import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from edel.isabelle.ingest import ingest_session_lemmas
from edel.pipeline.embedding import run_embedding_stage
from edel.isabelle.index import NumpyRAGIndex


def main():
    parser = argparse.ArgumentParser(description="Build EDEL RAG static index for Isabelle/AFP.")
    parser.add_argument("--host", default="127.0.0.1", help="I/R daemon host")
    parser.add_argument("--port", type=int, default=9147, help="I/R daemon port")
    parser.add_argument("--token", default="", help="I/R auth token (or set IR_AUTH_TOKEN)")
    parser.add_argument("--filter", default="", help="Regex filter for theories to ingest")
    parser.add_argument("--output", default="artifacts/rag_index", help="Output directory for RAG index")
    parser.add_argument("--provider", default="openai", choices=["openai", "voyage"], help="Embedding provider")
    parser.add_argument("--model", default="text-embedding-3-large", help="Embedding model name")
    
    args = parser.parse_args()
    
    token = args.token or os.getenv("IR_AUTH_TOKEN", "")
    if not token:
        print("Error: Authentication token not provided. Use --token or set IR_AUTH_TOKEN.")
        sys.exit(1)
        
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Ingest aspects from running I/R session
    df = ingest_session_lemmas(
        host=args.host,
        port=args.port,
        token=token,
        theory_filter=args.filter if args.filter else None
    )
    
    if len(df) == 0:
        print("No lemmas found to embed. Exiting.")
        sys.exit(0)
        
    # 2. Embed the aspects using EDEL's embedding pipeline
    print(f"Embedding aspects using provider '{args.provider}' and model '{args.model}'...")
    embed_config = {
        "embedding": {
            "provider": args.provider,
            "model": args.model,
            "api_key": os.getenv("VOYAGE_API_KEY" if args.provider == "voyage" else "OPENAI_API_KEY", ""),
        },
        "processing_mode": "simple"  # Sequential simple mode
    }
    
    # Run embedding stage
    df_embedded = run_embedding_stage(df, embed_config)
    print(f"Embeddings generated successfully. Remaining valid lemmas: {len(df_embedded)}")
    
    if len(df_embedded) == 0:
        print("Error: No lemmas remained after aspect coverage filtering.")
        sys.exit(1)
        
    # 3. Build and save index
    print("Building and saving RAG vector index...")
    index = NumpyRAGIndex()
    index.build_from_dataframe(df_embedded)
    index.save(output_dir)
    print(f"Static RAG index built successfully and saved to: {output_dir}")


if __name__ == "__main__":
    main()
