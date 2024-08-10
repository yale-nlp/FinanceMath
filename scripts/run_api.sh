#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)
models=(
    'gemini-1.5-flash'
    'gemini-1.5-pro'
    
    # Claude
    'claude-3-haiku-20240307'
    'claude-3-sonnet-20240229'
    'claude-3-opus-20240229'
    'claude-3-5-sonnet-20240620'
    
    # # GPT-3.5
    'gpt-3.5-turbo-0125'

    # # GPT-4
    'gpt-4-turbo'
    'gpt-4o'
    'gpt-4o-mini'

    # Deepseek
    'deepseek-chat'
    'deepseek-coder'

    # Llama3.1
    "meta-llama/Meta-Llama-3.1-405B-Instruct"
)

sets=(
  "validation"
  "test"
)

api_base="TODO"
api_key="TODO"

for model in "${models[@]}"; do
  for set in "${sets[@]}"; do
    echo "Running inference for model: $model on $set set"
    if [[ "$model" == *"/"* ]]; then
      requests_per_minute=40
      echo "Reducing requests per minute to 40 for togetherai models"
    else
      requests_per_minute=100
    fi

    python run_llm.py \
        --model_name "$model" \
        --prompt_type cot \
        --output_dir outputs \
        --subset "$set" \
        --max_tokens 512 \
        --api \
        --api_base "$api_base" \
        --api_key "$api_key" \
        --requests_per_minute "$requests_per_minute"
      
    python run_llm.py \
        --model_name "$model" \
        --prompt_type pot \
        --output_dir outputs \
        --subset "$set" \
        --max_tokens 512 \
        --api \
        --api_base "$api_base" \
        --api_key "$api_key" \
        --requests_per_minute "$requests_per_minute"
  done
done