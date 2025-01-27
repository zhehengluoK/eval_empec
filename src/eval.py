import os
import sys
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
import json
from transformers import AutoTokenizer
os.environ['VLLM_USE_MODELSCOPE'] = "True"

MODEL=sys.argv[1]
MODEL_DIR = '/home/v-zluo/blob/models/' + MODEL
#MODEL_DIR='/home/v-zluo/blob/output/llama_fact_sft/Qwen1.5-7B/train_ep3/'
#DATA_DIR = "test_data/test_8k_sim.jsonl"

DATA = sys.argv[2]
DATA_DIR = DATA + ".jsonl"

if_simple = sys.argv[3] == 'simp'
if if_simple:
    simp = "_simp"
else:
    simp = ""


OUT_DIR = "eval_results/{}-{}{}.jsonl".format(MODEL.split('/')[-1], DATA, simp)
print(OUT_DIR)

def load_test_data(data_dir=DATA_DIR, batch_size=16):
    with open(data_dir) as reader:
        data_list = [json.loads(i) for i in reader]
    if batch_size == 0:
        return data_list
    else:
        data_list = [data_list[i:i+batch_size] for i in range(0, len(data_list), batch_size)]
        return data_list

def apply_template(prompt, tokenizer):
    messages = [
    {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return text

def build_chat_input(messages, tokenizer, max_new_tokens=128):
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            # if message["role"] == "system":
            #     assert i == 0
            #     system = message["content"]
            #     continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds
    max_new_tokens = max_new_tokens
    max_input_tokens = 2048 - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    max_history_tokens = max_input_tokens
    roles = ('<问>：','<答>：')
    sep = '\n'

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            message["content"]
            if message["role"] == "user":
                round_tokens.extend(tokenizer.encode(roles[0]+message["content"]+sep))
            else:
                round_tokens.extend(tokenizer.encode(roles[1]+message["content"]+sep))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.extend(tokenizer.encode(roles[1]))
    # debug
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    # print(tokenizer.decode(input_tokens),flush=True)
    return input_tokens

def build_chat_input_bch(messages, tokenizer, max_new_tokens: int=16):
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    max_new_tokens = max_new_tokens
    max_input_tokens = 2048 - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(195)
            else:
                round_tokens.append(196)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(196)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return input_tokens

def format_medgpt(query):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:{query}\n\n### Response: """

def format_vicuna(query):
    return f"""USER: {query}
ASSISTANT:"""

def format_ziya(query):
    return '<human>:' + query + '\n<bot>:'

def format_sft(query):
    prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
    return prompt


# evaluated models:
# qwen/Qwen1.5-7B-Chat
# 01ai/Yi-6B-Chat
# moda/huatuo
# AI-ModelScope/Mistral-7B-Instruct-v0.2
# shakechen/Llama-2-7b-chat-hf
# modelscope/Llama-2-13b-chat-ms
# 01ai/Yi-34B-Chat
# moda/huatuo2_13
# baichuan-inc/Baichuan2-13B-Chat
# wslzh0015/medicineGPT
# wslzh0015/MedGLM


def main():
    llm = LLM(model=MODEL_DIR, trust_remote_code=True, tensor_parallel_size=1, gpu_memory_utilization=0.9, max_model_len=2048)  # Name or path of your model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, revision="v2.0",use_fast=False, trust_remote_code=True)

    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=32)
    test_data = load_test_data(DATA_DIR)

    with open(OUT_DIR, "w") as writer:
        for batch in tqdm(test_data):
            # messages_list = [[{"role": "user", "content": item[f"question{simp}"]+"\n答案："}] for item in batch]
            # prompt_tokens = [build_chat_input_bch(messages, tokenizer) for messages in messages_list]
            #responses = llm.generate([build_chat_input() for item in batch], sampling_params)
            # responses = llm.generate(prompt_token_ids=prompt_tokens, sampling_params=sampling_params)
            responses = llm.generate([apply_template(item[f"question{simp}"] + "\n答案：", tokenizer) for item in batch], sampling_params)
            # responses = llm.generate([format_sft(item["question"]) for item in batch], sampling_params)
            
            for i in range(len(responses)):
                #prompt = responses[i].prompt
                # assert prompt == batch[i]["question"]+"\n正確選項爲:"
                generated_text = responses[i].outputs[0].text
                print(generated_text.strip())
                output_dict = batch[i]
                output_dict["model_answer"] = generated_text
                writer.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
            # break
'''
def main():
    #from modelscope.hub.snapshot_download import snapshot_download

    #model_dir = snapshot_download('wslzh0015/MedGLM', cache_dir='/mnt/workspace/med_eval/moda')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.generation.utils import GenerationConfig
    # Note: The default behavior now has injection attack prevention off.
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, revision="v2.0",use_fast=False, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", torch_dtype="auto", trust_remote_code=True) #
    # model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, revision="v2.0", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(MODEL_DIR) # 可指定不同的生成长度、top_p等相关超参
    model.generation_config.temperature = 0.0
    model.generation_config.top_p = 1
    model.generation_config.max_new_tokens = 64
    model.generation_config.do_sample = False

    test_data = load_test_data(DATA_DIR)
    with open(OUT_DIR, "w") as writer:
        for batch in tqdm(test_data):
            for item in batch:
                prompt = item[f"question{simp}"]+"\n答案："

                messages = [
                    {"role": "user", "content": prompt}
                ]
                response = model.HuatuoChat(tokenizer, messages)
                print(response)
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([prompt], return_tensors='pt').to('cuda')

                #generated_ids = model.generate(
                #    model_inputs.input_ids,
                #    max_new_tokens=16
                #)

                #generated_ids = [
                #    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                #]

                #response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                #response = model.generate(, messages)
                output_dict = item
                output_dict["model_answer"] = response
                #print(prompt)
                #print(response)
                #print()
                writer.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
'''

if __name__ == "__main__":
    main()