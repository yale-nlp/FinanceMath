#!/bin/bash

raw_dir="outputs/validation/raw_pot_outputs"
processed_dir="outputs/validation/processed_pot_outputs"

# Iterate over each file in the raw output directory
for raw_file in "$raw_dir"/*; do
    filename=$(basename "$raw_file")

    # Check if the file does not exist in the processed output directory
    if [ ! -f "$processed_dir/$filename" ]; then
        python evaluation.py \
            --prediction_path "$raw_file" \
            --evaluation_output_dir "$processed_dir" \
            --prompt_type pot
    fi
done