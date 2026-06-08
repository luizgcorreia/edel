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

# 0. Cleanup any stale instances
echo -e "${BLUE}Cleaning up stale dashboard processes...${NC}"
fuser -k ${PORT}/tcp 2>/dev/null || true

echo -e "${BLUE}Starting EDEL Dashboard Stack...${NC}"

# 1. Start Worker in Background (Tmux or Screen)
# Determine python executable to use inside the screen/tmux shell
PYTHON_EXEC="python"
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_EXEC="$CONDA_PREFIX/bin/python"
elif [ -f "$HOME/.miniforge3/envs/edel/bin/python" ]; then
    PYTHON_EXEC="$HOME/.miniforge3/envs/edel/bin/python"
fi

# Create a restart wrapper script for the worker
cat << EOF > run_worker_loop.sh
#!/bin/bash
FAIL_COUNT=0
while true; do
    echo "Starting worker (Fail count: \$FAIL_COUNT)..."
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 $PYTHON_EXEC -m edel.dashboard.worker --base-path artifacts
    
    # If it ran for less than 10 seconds, it's a "fast crash"
    # We increase the sleep time to avoid fork-bombing the server
    FAIL_COUNT=\$((FAIL_COUNT + 1))
    SLEEP_TIME=\$((2 * FAIL_COUNT))
    if [ \$SLEEP_TIME -gt 60 ]; then SLEEP_TIME=60; fi
    
    echo "Worker crashed or stopped. Restarting in \${SLEEP_TIME}s..."
    sleep \$SLEEP_TIME
    
    # Reset fail count if we've been running successfully for a while
    # (Simplified: just reset if we manually restart after a long time)
done
EOF
chmod +x run_worker_loop.sh

# Detect multiplexer
if command -v tmux >/dev/null 2>&1; then
    USE_TMUX=true
else
    USE_TMUX=false
fi

if [ "$USE_TMUX" = true ]; then
    if tmux has-session -t edel_worker 2>/dev/null; then
        echo -e "${GREEN}✓ tmux session 'edel_worker' already running.${NC}"
    else
        echo "Starting new tmux session 'edel_worker'..."
        tmux new-session -d -s edel_worker "./run_worker_loop.sh"
        echo -e "${GREEN}✓ Worker running in background (tmux). Attach with: tmux attach -t edel_worker${NC}"
    fi
else
    # Fallback to Screen
    screen -wipe >/dev/null 2>&1 || true
    if screen -list | grep -q "\.edel_worker[[:space:]]*(Detached\|Attached)"; then
        echo -e "${GREEN}✓ screen session 'edel_worker' already running.${NC}"
    else
        echo "Starting new screen session 'edel_worker'..."
        screen -dmS edel_worker ./run_worker_loop.sh
        echo -e "${GREEN}✓ Worker running in background (screen). Attach with: screen -r edel_worker${NC}"
    fi
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
