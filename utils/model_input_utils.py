from tqdm import tqdm

def prepare_pot_model_input(example):
    system_input = '''You are a financial expert, you are supposed to generate a Python program to answer the given question. The returned value of the program is supposed to be the answer. Here is an example of the Python program:
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
    system_input = "You are a financial expert, you are supposed to answer the given question. You need to first think through the problem step by step, documenting each necessary step. Then you are required to conclude your response with the final answer in your last sentence as 'Therefore, the answer is {final answer}'. The final answer should be a numeric value."
    
    if example["tables"]:
        table_input = "Table:\n"
        table_input += "\n\n".join(example["tables"]) + "\n"
        question_input = f"{table_input}\nQuestion: {example['question']}\n"
    else:
        question_input = f"Question: {example['question']}\n"
    program_prefix_input = "Let\'s think step by step to answer the given question.\n"
       
    user_input = "\n".join([question_input, program_prefix_input])
    return system_input, user_input
    
def prepare_model_inputs(qa_data, prompt_type, model_name, api_based, tokenizer=None):
    model_inputs = []
    for example in tqdm(qa_data):
        if prompt_type == "pot":
            system_input, user_input = prepare_pot_model_input(example)
        else:
            system_input, user_input = prepare_cot_model_input(example)
            
        models_without_system = ("gemma", "OLMo", "Mistral", "Mixtral", "starcoder2")
        if any(model in model_name for model in models_without_system):
            model_input = [
                {"role": "user", "content": system_input + "\n" + user_input}
            ]
        else:
            model_input = [
                {"role": "system", "content": system_input},
                {"role": "user", "content": user_input}
            ]
        if not api_based:
            model_input = tokenizer.apply_chat_template(model_input, tokenize=False)
        
        model_inputs.append(model_input)
    return model_inputs