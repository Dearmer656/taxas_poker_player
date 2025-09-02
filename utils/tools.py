

DATASET_SELECTION = {
    'muffin': ['Reza8848/MUFFIN_68k'],
    'T0-SF': ['bigscience/P3', 'adversarial_qa_dbert_answer_the_following_q']
}
GENERATED_FILE = ['eval_batch_output', 'prompt_eval_group', 'prompt_eval_dict', 'generate_batch_output']

EXAMPLE_PROMPT = """======== Case 1: 
 Input, Instruction, Output pairs: {}
 Output Dialogue: {}
 ======"""

#  ——Case 2: 
#  Input, Instruction, Output pairs: {}
#  Output Dialogue: {}
#  ——Case 3: 
#  Input, Instruction, Output pairs: {}
#  Output Dialogue: {}
single_model_prices = {
    "gpt-4": {"input": 0.03, "output": 0.06},  # 每 1000 tokens
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4o": {"input": 2.5, "output": 10},
}   
batch_model_prices = {
    "gpt-4": {"input": 0.03, "output": 0.06},  # 每 1000 tokens
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-4o-mini": {"input": 0.075, "output": 0.6},    
    "gpt-4o": {"input": 1.25, "output": 10},
}
