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
import google.generativeai as genai
from google.generativeai import GenerativeModel
from utils.openai_utils import *

def prepare_pot_model_input(example):
    system_input = '''You are a financial expert, you are supposed to generate a Python program to answer the given question. The returned value of the program is supposed to be the answer. Here is the example of the Python program:
```python
def solution():
    # Define variables name and value
    revenue = 600000
    avg_account_receivable = 50000
    
    # Do math calculation to get the answer
    receivables_turnover = revenue / avg_account_receivable
    answer = 365 / receivables_turnover
    
    # return answer
    return answer
```
'''
    
    if example["tables"]:
        table_input = "The following table is provided for your reference."
        for table in example["tables"]:
            table_input += table + "\n"
        question_input = f"{table_input}\nQuestion: {example['question']}\n"
    else:
        question_input = f"Question: {example['question']}\n"
    program_prefix_input = '''Please generate a Python program to answer the given question. The format of the program should be the following:
```python
def solution():
    # Define variables name and value
    
    # Do math calculation to get the answer
    
    # return answer
```

Continue your output:
```python
def solution():
    # Define variables name and value
'''
    
    user_input = "\n".join([question_input, program_prefix_input])
    return system_input, user_input

def prepare_cot_model_input(example):
    system_input = "You are a financial expert, you are supposed to answer the given question. You need to first think through the problem step by step, documenting each necessary step. Then you are required to conclude your response with the final answer in your last sentence as 'Therefore, the answer is xxx'. The final answer should be a numeric value."
    
    if example["tables"]:
        table_input = "Table:\n"
        table_input += "\n\n".join(example["tables"]) + "\n"
        question_input = f"{table_input}\nQuestion: {example['question']}\n"
    else:
        question_input = f"Question: {example['question']}\n"
    program_prefix_input = "Let\'s think step by step to answer the given question.\n"
       
    user_input = "\n".join([question_input, program_prefix_input])
    return system_input, user_input
    
def prepare_model_inputs(qa_data, prompt_type, model_name, tokenizer=None):
    model_inputs = []
    for example in tqdm(qa_data):
        if prompt_type == "pot":
            system_input, user_input = prepare_pot_model_input(example)
        else:
            system_input, user_input = prepare_cot_model_input(example)
            
        if "gpt-" in model_name.lower():
            model_input = [
                {"role": "system", "content": system_input},
                {"role": "user", "content": user_input},
            ]
        elif "Lemur-70B-Chat" in model_name:
            model_input = f"<|im_start|>system\n{system_input}\n<|im_end|>\n"
            model_input += f"<|im_start|>user\n{user_input}\n<|im_end|>\n"
            model_input += f"<|im_start|>assistant\n"
        elif "Mistral-7B-Instruct" in model_name:
            chat = [{"role": "user", "content": system_input + "\n\n" + user_input}]
            model_input = tokenizer.apply_chat_template(chat, tokenize=False)
        elif "starchat" in model_name.lower():
            model_input = f"<|system|>\n{system_input}<|end|>\n<|user|>\n{user_input}<|end|>\n<|assistant|>"
        elif "WizardMath" in model_name or "WizardCoder" in model_name:
            model_input = "Below is an instruction that describes a task."
            model_input += "Write a response that appropriately completes the request.\n\n"
            model_input += f"### Instruction:\n{user_input}\n\n### Response: Let's think step by step."
        elif "deepseek" in model_name.lower() and "chat" in model_name.lower():
            model_input = f"User: {system_input}\n{user_input}\n\nAssistant:"
        elif "chat" in model_name.lower() or model_name == "deepseek-ai/deepseek-coder-33b-instruct":
            chat = [{"role": "system", "content": system_input}]
            chat.append({"role": "user", "content": user_input})
            model_input = tokenizer.apply_chat_template(chat, tokenize=False)
        else:
            model_input = system_input + "\n\n" + user_input
        
        model_inputs.append(model_input)
    return model_inputs

def process_single_example_raw_outputs(outputs):
    processed_outputs = []
    assert len(outputs.outputs) == 1
    processed_outputs.append(outputs.outputs[0].text)
    return processed_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    
    # dataset and output
    parser.add_argument("--dataset_name", type=str, default="yale-nlp/KnowledgeMath")
    parser.add_argument("--subset", type=str, required=True, choices=["validation", "test"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    
    # llm setting
    parser.add_argument("--sample_num", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=int, default=1.0)
    parser.add_argument("--prompt_type", type=str, default="cot", choices=["pot", "cot"])
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_num_examples", type=int, default=-1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    
    # api key
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    
    args = parser.parse_args()
    
    openai.api_base = args.api_base
    openai.api_key = args.api_key
    
    gpu_count = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    
    qa_data = load_dataset(args.dataset_name, split=args.subset)
    
    if args.max_num_examples > 0:
        random.shuffle(qa_data)
        qa_data = qa_data[:args.max_num_examples]
    
    model_name = args.model_name.split("/")[-1].replace(".", "_")
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = f"{args.output_dir}/{args.subset}/raw_{args.prompt_type}_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    if "/" in args.model_name:
        quantization = "awq" if "awq" in args.model_name.lower() else None

        llm = LLM(args.model_name, 
                tensor_parallel_size=gpu_count, 
                dtype="half", 
                swap_space=16, 
                quantization=quantization,
                gpu_memory_utilization=args.gpu_memory_utilization, 
                trust_remote_code=True)
        
        sampling_params = SamplingParams(n = args.sample_num, 
                                        temperature = args.temperature, 
                                        top_p = args.top_p, 
                                        max_tokens = args.max_tokens)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, verbose=False, trust_remote_code=True)
        tokenizer.use_default_system_prompt = True
        model_inputs = prepare_model_inputs(qa_data, args.prompt_type, args.model_name, tokenizer) 
        
        outputs = llm.generate(model_inputs, sampling_params)
        raw_outputs = [process_single_example_raw_outputs(output) for output in outputs]
        
    elif "gpt" in model_name:
        model_name = args.model_name        
        if "turbo" in args.model_name.lower():
            requests_per_minute = 100
        else:
            requests_per_minute = 20
            
        model_inputs = prepare_model_inputs(qa_data, args.prompt_type, args.model_name)
        raw_outputs = asyncio.run(generate_from_openai_chat_completion( 
                                                    messages = model_inputs,
                                                    engine_name = args.model_name, 
                                                    temperature = args.temperature, 
                                                    top_p = args.top_p, 
                                                    max_tokens = args.max_tokens,
                                                    requests_per_minute = requests_per_minute,))
    
    elif "gemini" in model_name:
        model_inputs = prepare_model_inputs(qa_data, args.prompt_type, args.model_name)
        raw_outputs = []
        
        model = GenerativeModel(model_name)
        genai.configure(api_key = args.api_key)
        for model_input in tqdm(model_inputs):
            raw_outputs.append(model.generate_content(model_input).text)
    
    output_data = []
    for raw_output, qa in zip(raw_outputs, qa_data):
        if type(raw_output) != list:
            qa["output"] = [raw_output]
        else:
            qa["output"] = raw_output
        output_data.append(qa)
        
    json.dump(output_data, open(f"{output_dir}/{model_name}.json", "w"), indent=4, ensure_ascii=False)