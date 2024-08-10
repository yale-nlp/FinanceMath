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

def get_result(processed_data):
    execution_rates = []
    accs = []

    topic_execution_rates = {}
    topic_accs = {}
    for example in processed_data:
        execution_rates.append(example["result"]["execution_rate"])
        accs.append(example["result"]["acc"])
        topic = example["topic"]
        if topic not in topic_execution_rates:
            topic_execution_rates[topic] = []
            topic_accs[topic] = []
        topic_execution_rates[topic].append(example["result"]["execution_rate"])
        topic_accs[topic].append(example["result"]["acc"])

    avg_acc = round(sum(accs) / len(accs) * 100, 2)
    avg_execution_rate = round(sum(execution_rates) / len(execution_rates) * 100, 2)
    topic_execution_rates = {k: round(sum(v) / len(v) * 100, 2) for k, v in topic_execution_rates.items()}
    topic_accs = {k: round(sum(v) / len(v) * 100, 2) for k, v in topic_accs.items()}

    return avg_acc, avg_execution_rate, topic_execution_rates, topic_accs

def evaluate_cot_pred_file(prediction_data, ground_truth_data, client):
    output_data = []
    prediction_data = extract_cot_answers(prediction_data, client)
    for example in tqdm(prediction_data):
        pred = example["extracted_pred_answer"]
        qid = example["question_id"]
        gt = float(ground_truth_data[qid]["ground_truth"])

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
        
    avg_acc, avg_execution_rate, topic_execution_rates, topic_accs = get_result(output_data)
    return output_data, avg_acc, avg_execution_rate, topic_execution_rates, topic_accs

def dummy_print(*args, **kwargs):
    pass

def evaluate_pot_pred_file(prediction_data, ground_truth_data, client, timeout_duration=3):
    output_data = []

    for example in tqdm(prediction_data):
        if isinstance(example['output'], list):
            candidate_str = process_single_pot_output(example['output'][0])
        elif isinstance(example['output'], str):
            candidate_str = process_single_pot_output(example['output'])

        question_id = example["question_id"]
        reference = f"{ground_truth_data[question_id]['python_solution']}"
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
        except (TimeoutException, Exception, OverflowError):
            example["result"]["acc"] = 0
            example["result"]["execution_rate"] = 0
            example["result"]["prediction_executed_result"] = None
            output_data.append(example)
            continue

        example["result"]["acc"] = get_acc(prediction_executed_result, ground_truth_executed_result, cot=False)
        example["result"]["execution_rate"] = 1
        
        try:
            if type(prediction_executed_result) in [float, int]:
                example["result"]["prediction_executed_result"] = round(float(prediction_executed_result), 3)
            else:
                example["result"]["prediction_executed_result"] = str(prediction_executed_result)
        except:
            example["result"]["prediction_executed_result"] = 0
            
        output_data.append(example)

    avg_acc, avg_execution_rate, topic_execution_rates, topic_accs = get_result(output_data)
    return output_data, avg_acc, avg_execution_rate, topic_execution_rates, topic_accs
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path", type=str, required=True)
    parser.add_argument("--evaluation_output_dir", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True, choices=["pot", "cot"])
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
    
    if "test" in args.prediction_path and not os.path.exists(args.ground_truth_file):
        raise ValueError("The ground truth is not provided for the test set, please use the leaderboard to evaluate the test set.")
    else:   
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
            avg_acc, avg_execution_rate, topic_execution_rates, topic_accs = get_result(processed_data)
        else:
            prediction_data = json.load(open(args.prediction_path, "r"))
            
            eval_func = evaluate_cot_pred_file if args.prompt_type == "cot" else evaluate_pot_pred_file
            outputs, avg_acc, avg_execution_rate, topic_execution_rates, topic_accs = eval_func(prediction_data, ground_truth_data, client)

            json.dump(outputs, open(output_path, "w"), indent=4, ensure_ascii=False)
            
        print(f"Accuracy: {avg_acc}, Execution Rate: {avg_execution_rate}")

        model_name = file_name.split(".")[0]
        results.append(
            {
                "model_name": model_name,
                "accuracy": avg_acc,
                "execution_rate": avg_execution_rate,
                "topic_accuracy": topic_accs,
                "topic_execution_rate": topic_execution_rates
            }
        )
        results = sorted(results, key=lambda x: x["accuracy"], reverse=True)

        json.dump(results, open(args.result_file, "w"), indent=4, ensure_ascii=False)


