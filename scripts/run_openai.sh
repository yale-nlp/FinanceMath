#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)
models=(
  "gpt-3.5-turbo-0613"
  "gpt-4-0613"
  "gpt-3.5-turbo-1106"
  "gpt-4-1106-preview"
)

for model in "${models[@]}"; do
    echo "Running inference for model: $model"

    python run_llm.py \
        --subset validation \
        --model_name "$model" \
        --sample_num 1 \
        --prompt_type cot \
        --output_dir outputs \
        --max_tokens 512 \
        --api_key xxxxx

done

for model in "${models[@]}"; do
    echo "Running inference for model: $model"

    export PYTHONPATH=`pwd`; 
    python run_llm.py \
        --subset validation \
        --model_name "$model" \
        --sample_num 1 \
        --prompt_type pot \
        --output_dir outputs \
        --max_tokens 512 \
        --api_key xxxxx
done