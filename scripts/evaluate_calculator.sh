#!/bin/bash

prompt_types=(
    "cot"
)
subsets=(
    "validation"
)

api_base="TODO"
api_key="TODO"

for prompt_type in "${prompt_types[@]}"; do
    for subset in "${subsets[@]}"; do
        echo "Evaluating $prompt_type on $subset set"
        raw_dir="outputs/$subset/processed_${prompt_type}_outputs"
        processed_dir="outputs/$subset/math_expression_extracted_${prompt_type}_outputs"
        result_file="outputs/results/${subset}_${prompt_type}_math_expression_results.json"

        # remove result file if it exists
        if [ -f "$result_file" ]; then
            rm "$result_file"
        fi

        # Iterate over each file in the raw output directory
        for raw_file in "$raw_dir"/*; do
            filename=$(basename "$raw_file")
            
            python cot_evaluation_with_calculator.py \
                --prediction_path "$raw_file" \
                --evaluation_output_dir "$processed_dir" \
                --ground_truth_file "data/$subset.json" \
                --result_file "$result_file" \
                --api_base "$api_base" \
                --api_key "$api_key"

            echo "Finished evaluating $filename"
        done

        echo "Finished evaluating $prompt_type on $subset set"
    done
done