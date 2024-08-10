import os
from time import sleep
from tqdm import tqdm
import math
import openai
import asyncio
from utils.openai_utils import *
from tqdm import tqdm
import json


def prepare_single_message_for_calculator(example):
    model_response = example["output"]

    system_input = """Convert the textual solution into a single-line math expression in python. You should return a json dict as follows:
    {
        "math_expression": xxx // for example, '(50 - 30) / 17.3'
    }
    """
    
    message = [
        {"role": "system", "content": system_input},
        {"role": "user", "content": model_response[0]}
    ]

    return message


def prepare_messages_for_calculator(prediction_data):
    messages = []
    for example in prediction_data:
        message = prepare_single_message_for_calculator(example)
        messages.append(message)

    return messages


def extracted_math_expression(output):
    return json.loads(output)["math_expression"]