import logging
import openai
import tiktoken
import datetime
import json
import os
import pdb
from tqdm import tqdm
from prompts.prompt import *
from utils.tools import single_model_prices, batch_model_prices
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import time

class OpenAIClient:
    def __init__(self, **generation_kwargs) -> None:
        self.api_key = os.getenv("API_KEY_SHY")
        self.generation_kwargs = generation_kwargs
        self.args = generation_kwargs['args']
        self._set_default_generation_args()
        logging.info("Generation Arguments:\n{}".format(self._generation_arg_report()))
        if not self.api_key.strip():
            logging.warning("The API key for cohere is empty. Please set `COHERE_API_KEY` properly")
        self.client = openai.OpenAI(api_key=self.api_key)
    def _set_default_generation_args(self):
        if 'tokens_to_generate' not in self.generation_kwargs:
            self.generation_kwargs['tokens_to_generate'] = 2048
        if 'temperature' not in self.generation_kwargs:
            self.generation_kwargs['temperature'] = 0.0
        if 'top_p' not in self.generation_kwargs:
            self.generation_kwargs['top_p'] = 1.0
        if 'stop' not in self.generation_kwargs:
            self.generation_kwargs['stop'] = []
        if 'random_seed' not in self.generation_kwargs:
            self.generation_kwargs['random_seed'] = 42
        if 'use_history' not in self.generation_kwargs:
            self.generation_kwargs['use_history'] = False
    
    def _generation_arg_report(self):
        arg_list = []
        for k, v in self.generation_kwargs.items():
            arg_list.append("{}\t{}".format(k, v))
        return "\n".join(arg_list)

    @retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(3))
    def _send_request(self, request):
        response = None
        try:
            messages = request['chat_history'] + [{"role": "user", "content": request['msgs']}] if self.generation_kwargs['use_history'] else [{"role": "user", "content": request['msgs']}]
            response = self.client.chat.completions.create(
                max_tokens=request['tokens_to_generate'],
                model=request['model_name'],
                messages=messages,
                temperature=request['temperature'],
                top_p=request['top_p'],
                seed=request['random_seed'],
                stop=request['stop'],
            )
        except Exception as e:
            if "id" in request:
                logging.warning("Error happens for case {}".format(request['id']))
            logging.warning(f"Error occurred while calling OpenAI: {e}")
            # print("Error message: {}".format(request['msgs']))
        return response
    def calculate_chat_cost(self, request, output):
        """
        计算一次对话的花费
        :param request: 包含 max_tokens, model_name, messages, temperature, top_p, seed, stop
        :param model_prices: 以字典形式提供的模型价格 {'model_name': {'input': 价格, 'output': 价格}}
        :return: 该次请求的成本（美元）
        """
        encoding = tiktoken.encoding_for_model(request['model_name'])
        
        model_name = request['model_name']
        input_tokens = len(encoding.encode(request['msgs']))  # 估算输入 token 数量
        output_tokens = len(encoding.encode(output))
        model_prices = batch_model_prices if self.args.api_batch_use else single_model_prices
        if model_name not in model_prices:
            raise ValueError(f"未知模型: {model_name}")
        cost = (input_tokens * model_prices[model_name]['input'] + output_tokens * model_prices[model_name]['output']) / 1000000  # 价格以兆 tokens 为单位
        return cost        
    def __call__(
        self,
        model_name: str,
        prompt: str,
        chat_history: list = None,
        **override_generation_args,
    ):
        verbose = override_generation_args.get('verbose', False)
        chat_history = [] if chat_history is None else chat_history
        
        request = {k: v for k, v in self.generation_kwargs.items()}
        for k, v in override_generation_args.items():
            if verbose:
                if k in request:
                    logging.info("Modify generation arguments for {}: {} -> {}".format(k, request[k], v))
                else:
                    logging.info("Add new generation arguments for {}: {}".format(k, v))
            request[k] = v
        request['model_name'] = model_name
        request['chat_history'] = chat_history
        request["msgs"] = prompt
        outputs = self._send_request(request)
        response, chat_history = None, []
        if verbose:
            logging.info("Full Response:\n{}".format(outputs))
        response = outputs.choices[0].message.content
        chat_history = chat_history + [{'role': 'user', "content": prompt}, {'role': outputs.choices[0].message.role, "content": response}]
        cost = self.calculate_chat_cost(request, response)
        return response, chat_history, cost
    def file_make(self, prompt_list, template, model_name, iter, mode, n_shots, prompt_dict=None):
        request_list = []
        template_record_dict = {}
        save_file_name = f"batch_input/GPU{self.args.GPU_seperate_path}/{self.args.api_model_name}_iter{iter}_{n_shots}shots/{mode}_input_prmpt.json"
        # 遍历 prompt_list，为每个 prompt 构造一个请求字典
        n = len(prompt_list) + 1 if prompt_dict is None else len(prompt_dict) + 1
        for idx in range(1, n):
            prompt = prompt_list[idx-1] if prompt_dict is None else prompt_dict[idx-1]['body']['messages'][0]['content']
            request_item = {
                "custom_id": f"request-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.generation_kwargs['tokens_to_generate']
                }
            }
            if mode == 'dialogue_generate' and len(template) != 0:
                template_record_dict[f'request-{idx}'] = template[idx-1]
            request_list.append(request_item)
        os.makedirs(f'batch_input/GPU{self.args.GPU_seperate_path}/{self.args.api_model_name}_iter{iter}_{n_shots}shots', exist_ok=True) 
        # 将构造的请求列表写入到指定文件中，格式化为 JSON 格式
        
        with open(save_file_name, "w", encoding="utf-8") as f: 
            for item in request_list:
                f.write(json.dumps(item, ensure_ascii=False)+"\n")
        if mode == 'dialogue_generate':
            with open(f'batch_input/GPU{self.args.GPU_seperate_path}/{self.args.api_model_name}_iter{iter}_{n_shots}shots/input_template_iter{iter}.json', 'w', encoding="utf-8") as f1:
                json.dump(template_record_dict, f1)
        batch_input_file = self.client.files.create(
            file=open(save_file_name, "rb"),
            purpose="batch"
        )        
        return batch_input_file
    def interaction_api_call(
        self,
        input_prompt: list[str],
        model_name: str,
        iter: int,
        n_shots: int,
        mode: str = None,
        output_saved_path: str = '',
        prompt_dict: list[dict] = None,
        extra_info: list[str] = None,
        **override_generation_args,
    ):
        """
        按顺序调用 __call__，将 prompt 和 response 以 JSONL 格式写入 output_saved_path。
        """
        # 2. 逐条调用 __call__
        responses: list[str] = []
        chat_history: list[dict] = []
        cost = 0
            
        resp, chat_history, cur_cost = self.__call__(
            model_name,
            input_prompt,
            chat_history=chat_history,
            **override_generation_args
        )
        cost += cur_cost
        responses.append(resp)
            # Optional: 控制速率

        # 3. 将所有 prompt-response 对以 JSONL 格式写入文件
        os.makedirs(os.path.dirname(output_saved_path), exist_ok=True)
        with open(output_saved_path, 'w', encoding='utf-8') as fout:
      
            json_line = {'prompt': input_prompt, 'response': resp}
            json.dump(json_line, fout, ensure_ascii=False)
            fout.write('\n')
        print(f'Interaction output saved to {output_saved_path}')
        return cost   
    def create_batch(self, batch_input):
        batch_input_file_id = batch_input.id
        batch_response  = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "dataset generate with ver5 prompt"
            }
        )        
        return batch_response
    def batch_response_id_record(self, batch_id, save_path, mode):
        timestamp = datetime.datetime.now().strftime("%m_%d_%H:%M")
        record_content = f"{timestamp}: {batch_id}\n" + '#'*50 + '\n' if mode == 'get_candidate_w_attribute' else f"{timestamp}: {batch_id}\n"
        # pdb.set_trace()
        with open(os.path.join(*save_path.split('/')[:-1], 'id_record.txt'), 'a', encoding='utf-8') as f:
            f.write(record_content)  # 加一个空格分隔