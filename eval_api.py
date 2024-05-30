import os
from tqdm import tqdm
import json
from openai import OpenAI


deployment_name='gpt-35-turbo'
#DATA_DIR = "test_data/test_8k_sim.jsonl"
DATA = "113"
DATA_DIR = DATA + ".jsonl"
OUT_DIR = "eval_results/{}-{}.jsonl".format(deployment_name.split('/')[-1], DATA)

def load_test_data(data_dir=DATA_DIR, batch_size=16):
    with open(data_dir) as reader:
        data_list = [json.loads(i) for i in reader]
    if batch_size == 0:
        return data_list
    else:
        data_list = [data_list[i:i+batch_size] for i in range(0, len(data_list), batch_size)]
        return data_list


def main():
    #from modelscope.hub.snapshot_download import snapshot_download

    #model_dir = snapshot_download('wslzh0015/MedGLM', cache_dir='/mnt/workspace/med_eval/moda')

    client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    )

    test_data = load_test_data(DATA_DIR)
    with open(OUT_DIR, "w") as writer:
        for batch in tqdm(test_data):
            for item in batch:
                prompt = item["question"]+"\n答案：" 

                messages = [
                    {"role": "user", "content": prompt}
                ]
                response = client.chat.completions.create(model=deployment_name, messages=messages, max_tokens=10)
                response = response.choices[0].message.content
                print(response)

                output_dict = item
                output_dict["model_answer"] = response
                #print(prompt)
                #print(response)
                #print()
                writer.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()