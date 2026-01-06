#!/bin/bash

# --- Configuration ---
# Define the list of config files to process
configs=(
    "gpt2-pretrained-cosmopedia-khanacademy.yaml"
    "gpt-j-6b-pretrained-cosmopedia-khanacademy.yaml"
    "Llama-3.2-3B-pretrained-cosmopedia-khanacademy.yaml"
    "mamba-1.4b-hf-pretrained-cosmopedia-khanacademy.yaml"
)

# Move to the training directory
cd ../trainer

# Create logs directory if it doesn't exist
mkdir -p logs

# --- Main Loop ---
# Iterate over each configuration file in the array
for config in "${configs[@]}"; do
    echo "--------------------------------------------------"
    echo "Starting job for config: $config"
    echo "Start time: $(date)"

    # Run the training script for the current config file
    python get_target.py \
        --config_path "./configs/${config}" \
        --base_path "./" \
        --train_subset_size -1 \
        --ref_subset_size -1

    echo "End time: $(date)"
    echo "Job completed for config: $config"
    echo "--------------------------------------------------"
    echo ""
done

echo "All jobs have been completed."