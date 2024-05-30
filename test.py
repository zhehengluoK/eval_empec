from Evaluator import QwenEvaluator
import json
import sys

evaluator = QwenEvaluator(None, None,None)
#qwen1_5_7b_chat_tra
#huatuo_7b_chat_tra
#mistral_7b_v0_2_instruct_tra
#llama2_7b_chat_tra
#yi34b_chat_tra
#llama2_13b_chat_tra
#huatuo2_13b_chat_tra
#baichuan2_13b_chat_tra
#medgpt_13b_tra
#medglm_6b_tra

file_path = sys.argv[1]
with open(file_path,"r") as reader:
    a = [json.loads(i) for i in reader]

count = 0
c = 0
dict_ = {}
for d in a:
    gold = d["answer"]
    subj = d["prof"]
    if subj not in dict_.keys():
        dict_[subj] = {"all":1,"cor":0}
    else:
        dict_[subj]["all"] += 1
    if len(d["model_answer"])!=0:
        ans = evaluator.extract_ans_zh(d["model_answer"])
        if len(ans)!=0 and ans[0] == gold:
            count += 1
            dict_[subj]["cor"] += 1
        
print("overall",count/len(a))
for d in dict_.keys():
    print(d, dict_[d]["cor"]/dict_[d]["all"])
    
print()
print(count/len(a))
for d in dict_.keys():
    print(round(dict_[d]["cor"]/dict_[d]["all"],4))#, dict_[d]['all'])
