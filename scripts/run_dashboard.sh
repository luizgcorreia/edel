#!/bin/bash
# EDEL Dashboard Startup Script for Erdos Server
# This script starts the job queue worker in a tmux session and the Dash server.

set -e

BASE_PATH="artifacts"
PORT=8050
HOST="0.0.0.0"

# Temp files
WORKER_SCRIPT="/tmp/edel_worker_loop.sh"
WORKER_PID_FILE="/tmp/edel_worker.pid"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
DEBUG_FLAG=""
RESTART_WORKER=false
for arg in "$@"; do
    case "$arg" in
        --debug)     DEBUG_FLAG="--debug" ;;
        --restart-worker) RESTART_WORKER=true ;;
    esac
done

# Cleanup handler – runs on EXIT / SIGINT / SIGTERM
cleanup() {
    rm -f "$WORKER_SCRIPT"
}
trap cleanup EXIT INT TERM

# 0. Cleanup any stale dashboard processes on the port
echo -e "${BLUE}Cleaning up stale dashboard processes...${NC}"
fuser -k "${PORT}/tcp" 2>/dev/null || true

echo -e "${BLUE}Starting EDEL Dashboard Stack...${NC}"

# 1. Start Worker in Background (Tmux or Screen)
PYTHON_EXEC="python"
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_EXEC="$CONDA_PREFIX/bin/python"
elif [ -f "$HOME/.miniforge3/envs/edel/bin/python" ]; then
    PYTHON_EXEC="$HOME/.miniforge3/envs/edel/bin/python"
fi

# Create a restart wrapper script for the worker (in /tmp)
cat << EOF > "$WORKER_SCRIPT"
#!/bin/bash
FAIL_COUNT=0
while true; do
    echo "Starting worker (Fail count: \$FAIL_COUNT)..."
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 $PYTHON_EXEC -m edel.dashboard.worker --base-path artifacts
    FAIL_COUNT=\$((FAIL_COUNT + 1))
    SLEEP_TIME=\$((2 * FAIL_COUNT))
    if [ \$SLEEP_TIME -gt 60 ]; then SLEEP_TIME=60; fi
    echo "Worker crashed or stopped. Restarting in \${SLEEP_TIME}s..."
    sleep \$SLEEP_TIME
done
EOF
chmod +x "$WORKER_SCRIPT"

# Detect multiplexer
if command -v tmux >/dev/null 2>&1; then
    USE_TMUX=true
else
    USE_TMUX=false
fi

# Optionally restart the worker session
if [ "$RESTART_WORKER" = true ]; then
    echo -e "${BLUE}Restarting worker session...${NC}"
    if [ "$USE_TMUX" = true ]; then
        tmux kill-session -t edel_worker 2>/dev/null || true
    else
        screen -S edel_worker -X quit 2>/dev/null || true
        screen -wipe >/dev/null 2>&1 || true
    fi
fi

if [ "$USE_TMUX" = true ]; then
    if tmux has-session -t edel_worker 2>/dev/null; then
        echo -e "${GREEN}✓ tmux session 'edel_worker' already running.${NC}"
    else
        echo "Starting new tmux session 'edel_worker'..."
        tmux new-session -d -s edel_worker "$WORKER_SCRIPT"
        echo -e "${GREEN}✓ Worker running in background (tmux). Attach with: tmux attach -t edel_worker${NC}"
    fi
else
    screen -wipe >/dev/null 2>&1 || true
    if screen -list | grep -q "\.edel_worker[[:space:]]*(Detached\|Attached)"; then
        echo -e "${GREEN}✓ screen session 'edel_worker' already running.${NC}"
    else
        echo "Starting new screen session 'edel_worker'..."
        screen -dmS edel_worker "$WORKER_SCRIPT"
        echo -e "${GREEN}✓ Worker running in background (screen). Attach with: screen -r edel_worker${NC}"
    fi
fi

# Record worker PID for lifecycle management
if [ "$USE_TMUX" = true ]; then
    tmux list-panes -t edel_worker -F "#{pane_pid}" > "$WORKER_PID_FILE" 2>/dev/null || true
fi

# 2. Start Dash Server
echo -e "${BLUE}Starting Dash Server on http://${HOST}:${PORT}${NC}"
echo "To access locally, create an SSH tunnel:"
echo "ssh -L ${PORT}:localhost:${PORT} <your-username>@erdos"
echo ""

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python -m edel.dashboard.app --base-path "${BASE_PATH}" --host "${HOST}" --port "${PORT}" ${DEBUG_FLAG}
