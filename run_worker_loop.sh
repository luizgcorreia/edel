#!/bin/bash
while true; do
    echo "Starting worker..."
    python -m edel.dashboard.worker --base-path artifacts
    echo "Worker crashed or stopped. Restarting in 2s..."
    sleep 2
done
