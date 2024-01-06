#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PYTHONPATH:$(pwd)
models=(
  # requires 1 * 48G gpus are enough
  "meta-llama/Llama-2-7b-chat-hf"
  "mistralai/Mistral-7B-Instruct-v0.1"
  "codellama/CodeLlama-7b-Instruct-hf"
  "meta-llama/Llama-2-13b-chat-hf"
  "HuggingFaceH4/starchat-beta"
  "bigcode/starcoder"
  "baichuan-inc/Baichuan2-13B-Chat"
  "Qwen/Qwen-14B-Chat"
  "microsoft/phi-1_5"
  "microsoft/phi-2"
  # requires 2 * 48G gpus
  "TheBloke/Llama-2-70B-Chat-AWQ"
  "codellama/CodeLlama-34b-Instruct-hf"
  "lmsys/vicuna-33b-v1.3"
  "mosaicml/mpt-30b-instruct"
  "codellama/CodeLlama-34b-Instruct-hf"
  "TheBloke/Lemur-70B-Chat-v1-AWQ"
  "01-ai/Yi-34B"
  "BAAI/AquilaChat2-34B"
  "gasolsun/pixiu-v1.0"
  "TheBloke/WizardMath-70B-V1.0-AWQ"
  "TheBloke/WizardLM-70B-V1.0-AWQ"
  "WizardLM/WizardCoder-Python-34B-V1.0"
  "TheBloke/deepseek-llm-67b-chat-AWQ"
  "mistralai/Mixtral-8x7B-Instruct-v0.1"
  # requires 8 * 48G gpus
  "tiiuae/falcon-180B-chat"
)

for model in "${models[@]}"; do
    echo "Running inference for model: $model"

    python run_llm.py \
        --model_name "$model" \
        --sample_num 1 \
        --gpu_memory_utilization 0.95 \
        --prompt_type cot \
        --output_dir outputs \
        --max_tokens 512
done

for model in "${models[@]}"; do
    echo "Running inference for model: $model"

    export PYTHONPATH=`pwd`; 
    python run_llm.py \
        --subset validation \
        --model_name "$model" \
        --sample_num 1 \
        --gpu_memory_utilization 0.95 \
        --prompt_type pot \
        --output_dir outputs \
        --max_tokens 512
done