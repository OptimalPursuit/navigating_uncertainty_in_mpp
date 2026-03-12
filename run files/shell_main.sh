#!/bin/bash
set -euo pipefail

# Define the path to the Python script you want to run
SCRIPT_PATH="main.py"

# Define the initial log file for output
mkdir -p output_files
LOG_FILE="output_files/output.log"

# Initialize GPU variable
GPU_NUMBER=""

# Parse arguments to check for --gpu
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_NUMBER="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# If a GPU number was provided, export it
if [ -n "$GPU_NUMBER" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_NUMBER"
    echo "Using GPU $GPU_NUMBER (CUDA_VISIBLE_DEVICES=$GPU_NUMBER)."
else
    echo "No GPU number provided. Default GPU configuration will be used."
fi

echo "Starting the script..."

# Start in background (nohup) so it survives logout, but we will WAIT so calls serialize.
nohup python3 "$SCRIPT_PATH" "$@" > "$LOG_FILE" 2>&1 &
PID=$!

# Ensure we got a PID
if ! [[ "$PID" =~ ^[0-9]+$ ]]; then
    echo "Failed to capture a valid PID. The script may not have started correctly."
    exit 1
fi

# Rename log immediately so each run gets a unique log while it's running
mv "$LOG_FILE" "output_files/output${PID}.log"
LOG_FILE="output_files/output${PID}.log"

echo "Started PID $PID. Logging to $LOG_FILE"

# Block until done (so outer loops run one-by-one)
STATUS=$?

if [ "$STATUS" -eq 0 ]; then
    echo "Finished successfully (PID $PID)."
else
    echo "Exited with code $STATUS (PID $PID). See $LOG_FILE."
fi

exit "$STATUS"