#!/bin/bash

# Script to run a Python script in the background using nohup

# Define the path to the Python script you want to run
SCRIPT_PATH="scenario_tree_mip.py"

# Define the initial log file for output
LOG_FILE="output_files/output.log"

# Run the Python script in the background with nohup, logging output to the specified file
echo "Starting the script in the background..."
nohup python3 -u "$SCRIPT_PATH" "$@" > "$LOG_FILE" 2>&1 &

# Capture the process ID of the last background command
PID=$!

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
