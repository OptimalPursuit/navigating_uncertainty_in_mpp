#!/bin/bash

# Script to run a Python script in the background using nohup

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
    export CUDA_VISIBLE_DEVICES=$GPU_NUMBER
    echo "Using GPU $GPU_NUMBER (CUDA_VISIBLE_DEVICES=$GPU_NUMBER)."
else
    echo "No GPU number provided. Default GPU configuration will be used."
fi

# Run the Python script in the background with nohup, logging output to the specified file
echo "Starting the script in the background..."
nohup python3 "$SCRIPT_PATH" "$@" > "$LOG_FILE" 2>&1 &

# Capture the process ID of the last background command
PID=$!

# Wait for the process to finish
wait $PID

# Check if the PID is a valid number
if [[ "$PID" =~ ^[0-9]+$ ]]; then
    # Rename the log file to include the process ID for uniqueness
    mv "$LOG_FILE" "output_files/output$PID.log"
    LOG_FILE="output_files/output$PID.log"

    # Check if the process started successfully
    if ps -p $PID > /dev/null; then
        echo "Script is running in the background with PID $PID. Check $LOG_FILE for output."
    else
        echo "Failed to start the script."
    fi
else
    echo "Failed to capture a valid PID. The script may not have started correctly."
fi
