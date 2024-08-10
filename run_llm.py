from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams
import json
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse
from transformers import AutoTokenizer
import random
from typing import Union
import asyncio
import openai
from utils.openai_utils import *
from utils.model_input_utils import prepare_model_inputs

def process_single_example_raw_outputs(outputs):
    processed_outputs = []
    assert len(outputs.outputs) == 1
    processed_outputs.append(outputs.outputs[0].text)
    return processed_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    
    # dataset and output
    parser.add_argument("--dataset_name", type=str, default="yale-nlp/FinanceMath")
    parser.add_argument("--subset", type=str, required=True, choices=["validation", "test"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    
    # llm setting
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=int, default=1.0)
    parser.add_argument("--prompt_type", type=str, default="cot", choices=["pot", "cot"])
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_num_examples", type=int, default=-1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--quantization", type=str, default="")
    
    # api key
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--requests_per_minute", type=int, default=100)
    
    args = parser.parse_args()
    
    gpu_count = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    
    qa_data = load_dataset(args.dataset_name, split=args.subset)
    
    if args.max_num_examples > 0:
        random.shuffle(qa_data)
        qa_data = qa_data[:args.max_num_examples]
    
    suffix_model_name = args.model_name.split("/")[-1].replace(".", "_")
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.join(args.output_dir, args.subset, f"raw_{args.prompt_type}_outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{suffix_model_name}.json")

    if os.path.exists(output_file):
        print(f"Output file already exists: {output_file}")
        exit()

    if not args.api:
        if args.quantization:
            llm = LLM(args.model_name,
                    tensor_parallel_size=gpu_count,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    trust_remote_code=True,
                    quantization=args.quantization)
        else:
            llm = LLM(args.model_name, 
                    tensor_parallel_size=gpu_count, 
                    dtype="half" if "gemma-2" not in args.model_name else "bfloat16", # https://github.com/vllm-project/vllm/issues/6177
                    swap_space=16, 
                    gpu_memory_utilization=args.gpu_memory_utilization, 
                    trust_remote_code=True)
        
        sampling_params = SamplingParams(temperature = args.temperature, 
                                        top_p = args.top_p, 
                                        max_tokens = args.max_tokens)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, verbose=False, trust_remote_code=True)
        tokenizer.use_default_system_prompt = True
        model_inputs = prepare_model_inputs(qa_data, args.prompt_type, args.model_name, args.api, tokenizer) 
        
        outputs = llm.generate(model_inputs, sampling_params)
        raw_outputs = [process_single_example_raw_outputs(output) for output in outputs]
    
    else:
        if args.api_base:
            os.environ["OPENAI_BASE_URL"] = args.api_base
        os.environ["OPENAI_API_KEY"] = args.api_key
        client = AsyncOpenAI()
        AsyncOpenAI.api_key = os.getenv('OPENAI_API_KEY')

        model_inputs = prepare_model_inputs(qa_data, args.prompt_type, args.model_name, args.api)
        model_name = args.model_name              

        raw_outputs = asyncio.run(generate_from_openai_chat_completion( 
                                                    client = client,
                                                    messages = model_inputs,
                                                    engine_name = args.model_name, 
                                                    temperature = args.temperature, 
                                                    top_p = args.top_p, 
                                                    max_tokens = args.max_tokens,
                                                    requests_per_minute = args.requests_per_minute,))
    
        
    output_data = []
    for raw_output, qa in zip(raw_outputs, qa_data):
        if type(raw_output) != list:
            qa["output"] = [raw_output]
        else:
            qa["output"] = raw_output
        output_data.append(qa)
        
    json.dump(output_data, open(output_file, "w"), indent=4, ensure_ascii=True)