#!/usr/bin/env python3
"""Script to package the EDEL RAG MCP server and its static index.

Collects the minimal Python files, static index artifacts, generates
necessary helper/dependency files, and packs everything into a
timestamped zip archive for deployment/sharing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

# Files to copy from the local workspace
REQUIRED_CODE_FILES = [
    "edel/__init__.py",
    "edel/io/__init__.py",
    "edel/io/llm.py",
    "edel/il/__init__.py",
    "edel/il/il_server.py",
    "edel/il/index.py",
    "edel/il/aspects.py",
]

# Static index files to include
REQUIRED_INDEX_FILES = [
    "metadata.parquet",
    "embeddings.npz",
]

REQUIREMENTS_CONTENT = """mcp[cli]
fastmcp
numpy
pandas
pyarrow
python-dotenv
openai
requests
"""

RUN_SERVER_CONTENT = """#!/bin/bash
# run_server.sh - Startup script for the bundled I/L (Isabelle/Landscape) MCP server

# Determine directory where this script resides
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Set default index directory (relative to script) if not set
export IL_INDEX_DIR="${IL_INDEX_DIR:-${SCRIPT_DIR}/artifacts/rag_index}"

# Default embedding parameters (can be overridden by environment)
export IL_EMBEDDING_PROVIDER="${IL_EMBEDDING_PROVIDER:-voyage}"
export IL_EMBEDDING_MODEL="${IL_EMBEDDING_MODEL:-voyage-code-3}"

# Warn if keys aren't set
if [ -z "${VOYAGE_API_KEY}" ] && [ -z "${OPENAI_API_KEY}" ]; then
  echo "Warning: Neither VOYAGE_API_KEY nor OPENAI_API_KEY is set in environment."
  echo "Please set the appropriate key before querying the RAG server."
fi

echo "Starting I/L MCP Server..."
echo "  Index Dir: ${IL_INDEX_DIR}"
echo "  Provider:  ${IL_EMBEDDING_PROVIDER}"
echo "  Model:     ${IL_EMBEDDING_MODEL}"
echo ""

# Execute the MCP server module
python -m edel.il.il_server
"""


def package_server(index_dir: Path, output_dir: Path):
    project_root = Path(__file__).resolve().parent.parent
    
    # 1. Verify necessary files exist
    print("Verifying source code files...")
    for rel_path in REQUIRED_CODE_FILES:
        full_path = project_root / rel_path
        if not full_path.exists():
            print(f"Error: Required code file {rel_path} does not exist.")
            return False
            
    print("Verifying static index files...")
    for filename in REQUIRED_INDEX_FILES:
        full_path = index_dir / filename
        if not full_path.exists():
            print(f"Error: Static index file {filename} not found in {index_dir}.")
            print("Please run the index builder first or check the path.")
            return False
            
    # 2. Create timestamped zip filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"edel_mcp_server_{timestamp}.zip"
    zip_filepath = output_dir / zip_filename
    
    # 3. Build in temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        staging_path = Path(temp_dir)
        
        # Copy code files
        print("Copying code files...")
        for rel_path in REQUIRED_CODE_FILES:
            src = project_root / rel_path
            dst = staging_path / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            
        # Copy index files
        print("Copying index artifacts...")
        dst_index_dir = staging_path / "artifacts" / "rag_index"
        dst_index_dir.mkdir(parents=True, exist_ok=True)
        for filename in REQUIRED_INDEX_FILES:
            src = index_dir / filename
            dst = dst_index_dir / filename
            shutil.copy2(src, dst)
            
        # Write requirements.txt
        print("Writing requirements.txt...")
        with open(staging_path / "requirements.txt", "w") as f:
            f.write(REQUIREMENTS_CONTENT)
            
        # Write run_server.sh
        print("Writing run_server.sh...")
        run_script_path = staging_path / "run_server.sh"
        with open(run_script_path, "w") as f:
            f.write(RUN_SERVER_CONTENT)
        # Make the startup script executable
        os.chmod(run_script_path, 0o755)
        
        # Create Zip file
        print(f"Compacting into zip archive: {zip_filepath}...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(staging_path):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(staging_path)
                    zip_file.write(file_path, arcname)
                    
    print("\n" + "=" * 50)
    print("Packaging complete!")
    print(f"Archive file: {zip_filepath}")
    print("=" * 50)
    return True


def main():
    parser = argparse.ArgumentParser(description="Package EDEL MCP Server for deployment.")
    parser.add_argument(
        "--index-dir",
        default="artifacts/rag_index",
        help="Path to static index directory containing metadata.parquet and embeddings.npz"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save the packaged zip archive"
    )
    args = parser.parse_args()
    
    index_dir = Path(args.index_dir)
    output_dir = Path(args.output_dir)
    
    success = package_server(index_dir, output_dir)
    if not success:
        os._exit(1)


if __name__ == "__main__":
    main()
