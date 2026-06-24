#!/bin/bash
# start_rag_server.sh - Launch the EDEL-RAG MCP server

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
export EDEL_EMBEDDING_PROVIDER="${EDEL_EMBEDDING_PROVIDER:-openai}"
export EDEL_EMBEDDING_MODEL="${EDEL_EMBEDDING_MODEL:-text-embedding-3-large}"
export EDEL_RAG_INDEX_DIR="${EDEL_RAG_INDEX_DIR:-${PROJECT_DIR}/artifacts/rag_index}"

echo "Starting EDEL-RAG MCP server using:"
echo "  Python:    ${CONDA_ENV_PYTHON}"
echo "  Provider:  ${EDEL_EMBEDDING_PROVIDER}"
echo "  Model:     ${EDEL_EMBEDDING_MODEL}"
echo "  Index Dir: ${EDEL_RAG_INDEX_DIR}"
echo ""

exec "${CONDA_ENV_PYTHON}" -m edel.isabelle.rag_server
