#!/bin/bash

# --- Argument Parsing ---
# Ensure at least two arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <config_file_path> <base_directory> [--target-model PATH] [--target-tokenizer PATH] [--lora-path PATH]"
    exit 1
fi

# Assign required arguments
CONFIG_FILE="$1"
BASE_DIR="$2"
shift 2  # Shift the first two arguments so $@ contains only optional args

# --- Calculate Output Directory ---
config_filename_full="${CONFIG_FILE##*/}"
config_filename_base="${config_filename_full%.*}"
OUTPUT_DIR="$BASE_DIR/attack_res/$config_filename_base"

# --- Pre-execution Steps ---
cd ..
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# --- Execution ---
echo "Starting Evaluation..."
echo "----------------------------------------"
echo "Using Config File: $CONFIG_FILE"
echo "Using Base Directory: $BASE_DIR"
echo "Derived Config Base Name: $config_filename_base"
echo "Output Directory: $OUTPUT_DIR"
echo "Python Path: $PYTHONPATH"
echo "Current Working Directory: $(pwd)"
echo "Optional arguments: $@"
echo "----------------------------------------"

mkdir -p "$OUTPUT_DIR"

python run.py \
    -c "$CONFIG_FILE" \
    --output "$OUTPUT_DIR" \
    --base-dir "$BASE_DIR" \
    "$@"  # Forward any optional arguments

# Capture the exit code of the python script
PYTHON_EXIT_CODE=$?

echo "----------------------------------------"
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "Evaluation script completed successfully."
else
    echo "Error: Evaluation script failed with exit code $PYTHON_EXIT_CODE."
fi
echo "Results are in: $OUTPUT_DIR"
echo "========================================"

# Exit with the same code as the python script
exit $PYTHON_EXIT_CODE
