#!/bin/bash
# EDEL Dashboard Startup Script for Erdos Server
# This script starts the job queue worker in a tmux session and the Dash server.

set -e

BASE_PATH="artifacts"
PORT=8050
HOST="0.0.0.0"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting EDEL Dashboard Stack...${NC}"

# 1. Start Worker in Screen
# Create a restart wrapper script for the worker
cat << 'EOF' > run_worker_loop.sh
#!/bin/bash
while true; do
    echo "Starting worker..."
    python -m edel.dashboard.worker --base-path artifacts
    echo "Worker crashed or stopped. Restarting in 2s..."
    sleep 2
done
EOF
chmod +x run_worker_loop.sh

# Check if screen session already exists
if screen -list | grep -q "\.edel_worker"; then
    echo -e "${GREEN}✓ screen session 'edel_worker' already running.${NC}"
else
    echo "Starting new screen session 'edel_worker'..."
    screen -dmS edel_worker ./run_worker_loop.sh
    echo -e "${GREEN}✓ Worker running in background. Attach with: screen -r edel_worker${NC}"
fi

# 2. Start Dash Server
echo -e "${BLUE}Starting Dash Server on http://${HOST}:${PORT}${NC}"
echo "To access locally, create an SSH tunnel:"
echo "ssh -L ${PORT}:localhost:${PORT} <your-username>@erdos"
echo ""

# Parse command line arguments
DEBUG_FLAG=""
for arg in "$@"; do
    if [ "$arg" == "--debug" ]; then
        DEBUG_FLAG="--debug"
    fi
done

# Run dash server in the foreground
python -m edel.dashboard.app --base-path "${BASE_PATH}" --host "${HOST}" --port ${PORT} ${DEBUG_FLAG}
