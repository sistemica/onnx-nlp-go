"""Records cross-encoder logits of a fixed, multilingual set of query and document pairs.

usage: reference.py MODEL_ID OUT_JSON
"""
import json
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
long_doc = " ".join(["Die Donaudampfschifffahrt war ein wichtiger Wirtschaftszweig im neunzehnten Jahrhundert."] * 120)
pairs = [
    ("Wie hoch ist der Eiffelturm?", "Der Eiffelturm in Paris ist 330 Meter hoch und wurde 1889 fertiggestellt."),
    ("Wie hoch ist der Eiffelturm?", "Die Katze schläft den ganzen Tag auf dem Sofa."),
    ("how tall is the Eiffel Tower", "The Eiffel Tower is 330 metres tall, about the same height as an 81-storey building."),
    ("how tall is the Eiffel Tower", "Bananas are rich in potassium and are often eaten for breakfast."),
    ("Quelle est la hauteur de la tour Eiffel ?", "La tour Eiffel mesure 330 mètres de haut."),
    ("埃菲尔铁塔有多高？", "埃菲尔铁塔高330米，位于法国巴黎。"),
    ("Какова высота Эйфелевой башни?", "Эйфелева башня имеет высоту 330 метров."),
    ("What is Python?", "Python ist eine Programmiersprache. Python is a high-level programming language."),
    ("Übergrößenträger straßenbahn", "Fußball Käse Öl Mädchen"),
    ("a", "b"),
    ("Wie lange dauert die Donaudampfschifffahrt?", long_doc),
    (" ".join(["Frage"] * 300), long_doc),
    ("Wie funktioniert ein Wärmetauscher?", "Ein Wärmetauscher überträgt Wärme von der Abluft auf die Zuluft."),
    ("", "nur ein Dokument"),
]
with torch.no_grad():
    batch = tok([q for q, _ in pairs], [d for _, d in pairs], padding=True, truncation=True, max_length=512, return_tensors="pt")
    logits = model(**batch).logits
json.dump({"pairs": pairs, "logits": logits.tolist(), "num_labels": logits.shape[1]}, open(out, "w"), ensure_ascii=False)
print(len(pairs), "pairs, logits shape", tuple(logits.shape))
