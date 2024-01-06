from evaluate import load
import json
import os
import argparse
from tqdm import tqdm
from utils.evaluation_utils import *
import signal

class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    raise TimeoutException("Code execution took too long!")
signal.signal(signal.SIGALRM, timeout_handler)

def evaluate_cot_pred_file(prediction_data):
    output_data = []
    prediction_data = extract_cot_answers(prediction_data)
    for example in tqdm(prediction_data):
        pred = example["extracted_pred_answer"]
        try:
            gt = float(example["ground_truth"])
        except:
            print(example["ground_truth"])
            continue
        if "none" in pred.lower() in pred.lower():
            result = {
                "execution_rate": 0,
                "acc": 0
            }
            example["result"] = result
        else:
            result = {
                "execution_rate": 1,
                "acc": get_acc(pred, gt),
            }
            example["result"] = result
        output_data.append(example)
        
    avg_acc = round(sum([example["result"]["acc"] for example in output_data]) / len(output_data)*100, 2)
    avg_execution_rate = round(sum([example["result"]["execution_rate"] for example in output_data]) / len(output_data) * 100, 2)
    return output_data, avg_acc, avg_execution_rate

def dummy_print(*args, **kwargs):
    pass

def evaluate_pot_pred_file(prediction_data, timeout_duration=3):
    output_data = []

    for example in tqdm(prediction_data):
        if isinstance(example['output'], list):
            candidate_str = process_single_pot_output(example['output'][0])
        elif isinstance(example['output'], str):
            candidate_str = process_single_pot_output(example['output'])
        reference = f"{example['python_solution']}"
        namespace = {"print": dummy_print}
        exec(reference, namespace)
        ground_truth_executed_result = namespace["solution"]()
        example["result"] = {}
        
        try:
            namespace = {"print": dummy_print}
            signal.alarm(timeout_duration)  # set the alarm
            exec(candidate_str, namespace)
            prediction_executed_result = namespace["solution"]()
            signal.alarm(0)  # reset the alarm
        except (TimeoutException, Exception):
            example["result"]["acc"] = 0
            example["result"]["execution_rate"] = 0
            example["result"]["prediction_executed_result"] = None
            output_data.append(example)
            continue

        example["result"]["acc"] = get_acc(prediction_executed_result, ground_truth_executed_result)
        example["result"]["execution_rate"] = 1
        
        if type(prediction_executed_result) in [float, int]:
            example["result"]["prediction_executed_result"] = round(float(prediction_executed_result), 3)
        else:
            example["result"]["prediction_executed_result"] = str(prediction_executed_result)
            
        output_data.append(example)

    avg_acc = round(sum([example["result"]["acc"] for example in output_data]) / len(output_data) * 100, 2)
    avg_execution_rate = round(sum([example["result"]["execution_rate"] for example in output_data]) / len(output_data) * 100, 2)
    return output_data, avg_acc, avg_execution_rate
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path", type=str, required=True)
    parser.add_argument("--evaluation_output_dir", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True, choices=["pot", "cot"])
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    args = parser.parse_args()
    
    results = {}
    
    openai.api_base = args.api_base
    openai.api_key = args.api_key
    
    prediction_data = json.load(open(args.prediction_path, "r"))
    
    if args.prompt_type == "cot":
        outputs, avg_acc, avg_execution_rate = evaluate_cot_pred_file(prediction_data)
    elif args.prompt_type == "pot":
        outputs, avg_acc, avg_execution_rate = evaluate_pot_pred_file(prediction_data)
        
    print(f"Accuracy: {avg_acc}, Execution Rate: {avg_execution_rate}")
    
    file_name = os.path.basename(args.prediction_path)
    
    os.makedirs(args.evaluation_output_dir, exist_ok=True)
    
    output_path = os.path.join(args.evaluation_output_dir, file_name)
    json.dump(outputs, open(output_path, "w"), indent=4, ensure_ascii=False)
