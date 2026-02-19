from answer import answer_question_stream
import json
from tqdm.auto import tqdm


# Making the test set
"""


TEST_PATH = "test/test_set_1.json"
test_set = []

with open(TEST_PATH, 'r', encoding='utf-8') as f:
    file = json.load(f)


test_set = [e["question"] for e in file]

with open('test/questions.json', 'w', encoding='utf-8') as f:
    json.dump(test_set, f, ensure_ascii=False, indent=4)"""


# ============================

# Evaluating HooshDaroo using the test_set

QUESTIONS_PATH = "test/questions.json"

results = []

with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f:
    questions = json.load(f)

for q in tqdm(questions):
    answer, rag_context, graph_context = answer_question_stream(q, history="")
    res = {"question": q,
           "rag_context": rag_context,
           "graph_rag_context": graph_context,
           "answer": answer
           }
    
    results.append(res)

with open('test/results_1.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

