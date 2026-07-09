#!/bin/bash
# start_il_server.sh - Launch the I/L (Isabelle/Landscape) MCP server

# Define paths
CONDA_ENV_PYTHON="/home/correia/miniforge3/envs/edel/bin/python"
PROJECT_DIR="/home/correia/edel"

# Set PYTHONPATH to project root
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Check for environment variables
if [ -z "${VOYAGE_API_KEY}" ] && [ -z "${OPENAI_API_KEY}" ]; then
  echo "Warning: Neither VOYAGE_API_KEY nor OPENAI_API_KEY is set."
  echo "Please set one of these to enable embedding queries."
fi

# Default configuration if not set
export IL_EMBEDDING_PROVIDER="${IL_EMBEDDING_PROVIDER:-openai}"
export IL_EMBEDDING_MODEL="${IL_EMBEDDING_MODEL:-text-embedding-3-large}"
export IL_INDEX_DIR="${IL_INDEX_DIR:-${PROJECT_DIR}/artifacts/rag_index}"

echo "Starting I/L MCP server using:"
echo "  Python:    ${CONDA_ENV_PYTHON}"
echo "  Provider:  ${IL_EMBEDDING_PROVIDER}"
echo "  Model:     ${IL_EMBEDDING_MODEL}"
echo "  Index Dir: ${IL_INDEX_DIR}"
echo ""

exec "${CONDA_ENV_PYTHON}" -m edel.il.il_server
