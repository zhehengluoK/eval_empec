
## 📖 Dataset Intro
![CMB](figs/dst.png)

EMPEC consists of 157,803 exam questions across 124 subjects and 20 healthcare professions
The test data is available for preview in the `test.jsonl` file.



### EMPEC Item 
```json
{
    "subject": "解剖學與生理學",
    "subject_en": "Anatomy and Physiology",
    "profession": "職能治療師",
    "profession_en": "Occupational Therapist",
    "question": "肺臟中進行氣體交換的主要結構稱做： 
    A.肺泡（alveolus） 
    B.氣管（trachea） 
    C.支氣管（bronchus） 
    D.橫膈（diaphragm）",
    "question_en": "The main structure in the lungs where gas exchange takes place is called: 
    A. Alveolus 
    B. Trachea 
    C. Bronchus 
    D. Diaphragm",
    "answer": "A",
},
```

### Evaluation
#### vllm:
1. `python src/eval.py MODEL_NAME DATA`
2. `python src/test.py eval_results/MODEL_NAME-DATA.jsonl`

#### Proprietary:
1. `python src/eval_api.py MODEL_NAME DATA`
2. `python src/test.py eval_results/MODEL_NAME-DATA.jsonl`

