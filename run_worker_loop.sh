#!/bin/bash
FAIL_COUNT=0
while true; do
    echo "Starting worker (Fail count: $FAIL_COUNT)..."
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 /home/correia/miniforge3/envs/edel/bin/python -m edel.dashboard.worker --base-path artifacts
    
    # If it ran for less than 10 seconds, it's a "fast crash"
    # We increase the sleep time to avoid fork-bombing the server
    FAIL_COUNT=$((FAIL_COUNT + 1))
    SLEEP_TIME=$((2 * FAIL_COUNT))
    if [ $SLEEP_TIME -gt 60 ]; then SLEEP_TIME=60; fi
    
    echo "Worker crashed or stopped. Restarting in ${SLEEP_TIME}s..."
    sleep $SLEEP_TIME
    
    # Reset fail count if we've been running successfully for a while
    # (Simplified: just reset if we manually restart after a long time)
done
