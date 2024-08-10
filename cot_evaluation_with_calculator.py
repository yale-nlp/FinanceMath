import json
import os
import argparse
from tqdm import tqdm
from utils.evaluation_utils import *
from utils.calculator_utils import *
import signal
import asyncio
import numpy
import math
import numpy as np
from numpy import sqrt 

def get_result(processed_data):
    accs = []
    math_accs = []
    for example in processed_data:
        accs.append(example["result"]["acc"])
        math_accs.append(example["result"]["extracted_math_acc"])

    avg_acc = round(sum(accs) / len(accs) * 100, 2)
    avg_math_acc = round(sum(math_accs) / len(math_accs) * 100, 2)
    return avg_acc, avg_math_acc

def evaluate_file_with_calculator(prediction_data, ground_truth_data, client):
    output_data = []
    messages = prepare_messages_for_calculator(prediction_data)

    outputs = asyncio.run(generate_from_openai_chat_completion(
                                                    client = client, 
                                                    messages = messages,
                                                    engine_name = "gpt-4o-mini", 
                                                    max_tokens = 512,
                                                    requests_per_minute = 500,
                                                    json_mode = True))
    
    outputs = [extracted_math_expression(output) for output in outputs]

    for output, pred_example in zip(outputs, prediction_data):
        try:
            output_value = eval(output)
            gt = float(ground_truth_data[pred_example["question_id"]]["ground_truth"])
            if compare_two_numbers(output_value, gt):
                acc = 1
            else:
                acc = 0
        except:
            acc = 0

        pred_example["result"]["extracted_math_expression"] = output
        pred_example["result"]["extracted_math_acc"] = acc
        
    return prediction_data

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path", type=str, required=True)
    parser.add_argument("--evaluation_output_dir", type=str, required=True)
    parser.add_argument("--ground_truth_file", type=str, default="data/validation.json")
    parser.add_argument("--result_file", type=str, required=True)
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    args = parser.parse_args()


    if args.api_base:
        os.environ["OPENAI_BASE_URL"] = args.api_base
    os.environ["OPENAI_API_KEY"] = args.api_key
    client = AsyncOpenAI()
    AsyncOpenAI.api_key = os.getenv('OPENAI_API_KEY')


    file_name = os.path.basename(args.prediction_path)
    os.makedirs(args.evaluation_output_dir, exist_ok=True)
    output_path = os.path.join(args.evaluation_output_dir, file_name)
     
    ground_truth_raw_data = json.load(open(args.ground_truth_file, "r"))
    ground_truth_data = {}
    for example in ground_truth_raw_data:
        ground_truth_data[example["question_id"]] = example

    if os.path.exists(args.result_file):
        results = json.load(open(args.result_file, "r"))
    else:
        results = []

    if os.path.exists(output_path):
        processed_data = json.load(open(output_path, "r"))
        avg_acc, avg_math_acc = get_result(processed_data)
    else:
        prediction_data = json.load(open(args.prediction_path, "r"))
        outputs = evaluate_file_with_calculator(prediction_data, ground_truth_data, client)
        avg_acc, avg_math_acc = get_result(outputs)

        json.dump(outputs, open(output_path, "w"), indent=4, ensure_ascii=False)
        
    print(f"Accuracy: {avg_acc}, Math Expression Accuracy: {avg_math_acc}")

    model_name = file_name.split(".")[0]
    results.append(
        {
            "model_name": model_name,
            "accuracy": avg_acc,
            "math_expression_acc": avg_math_acc
        }
    )

    json.dump(results, open(args.result_file, "w"), indent=4, ensure_ascii=False)



    


        