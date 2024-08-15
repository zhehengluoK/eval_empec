import os
import argparse
import re
import torch
import pandas as pd
from tqdm import tqdm
from thefuzz import process
from collections import defaultdict
import json

choices = ["A", "B", "C", "D"]

def format_example(line):
    example = (
        "The following is a multiple-choice question. Please choose the most suitable one among A, B, C and D as the answer to this question.\n\n"
        + line["question"]
        + "\n"
    )
    for choice in choices:
        example += f'{choice}. {line[f"{choice}"]}\n'
    return example


def process_before_extraction(gen, choice_dict):
    # replace the choice by letter in the generated sentence
    # from longest one to shortest one
    for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1]), reverse=True):
        pattern = re.compile(re.escape(val.rstrip(".")), re.IGNORECASE)
        gen = pattern.sub(key, gen)
    return gen


def extract_choice(gen, choice_list):
    # answer is A | choice is A | choose A
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b",
        gen,
    )

    # A is correct | A is right
    if res is None:
        res = re.search(
            r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b",
            gen,
        )

    # straight answer: A
    if res is None:
        res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)

    # simply extract the first appearred letter
    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

    if res is None:
        return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
    return res.group(1)


def extract_answer(response, row):
    gen = process_before_extraction(
        response, {choice: row[choice] for choice in choices}
    )
    pred = extract_choice(gen, [row[choice] for choice in choices])
    return pred


class EngEval:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def _process_response(self, response, choice_dict):
        #choice: {"A":"I agree", "B": "Not right", "C": "I believe", "D":"what is this"}
        pred = extract_answer(response, choice_dict)
        return pred

    def process_data(self):
        pattern = r'([A-D])\.\s*(.*?)(?=\s*[A-D]\.|$)'
        with open(self.data_path) as reader:
            data_list = [json.loads(i) for i in reader]
        answer_list = []
        issues_list = []
        for d in data_list:
            choices_string = re.findall(pattern, d["question_eng"], re.DOTALL)
            choice_dict = {}
            for k,v in choices_string:
                choice_dict[k] = v.strip()
            if len(choice_dict)==4:
                pred = self._process_response(d["bot_ans"], choice_dict)
                d["extracted_ans"] = pred
                answer_list.append(d)
            else:
                issues_list.append(d)
        self.answer_list = answer_list
        with open("weird_questions.json","w") as writer:
            json.dump(issues_list, writer)
        return answer_list
    
    def calculate_acc_and_print(self):
        total = correct = 0
        prof_stats = defaultdict(lambda: [0, 0])
        for item in self.answer_list:
            total += 1
            prof_stats[item['prof']][1] += 1
            if str(item['answer']) == str(item['extracted_ans']):
                correct += 1
                prof_stats[item['prof']][0] += 1
       
        overall_accuracy = correct / total if total > 0 else 0
        print(f"Overall accuracy: {overall_accuracy:.2%}")

        for prof, (correct, total) in prof_stats.items():
            prof_accuracy = correct / total if total > 0 else 0
            print(f"Accuracy {prof}: {prof_accuracy:.2%}")


