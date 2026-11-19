"""Records sentence-transformers embeddings of a fixed, multilingual set of texts.

usage: reference.py MODEL_ID OUT_JSON
"""
import json
import sys

from sentence_transformers import SentenceTransformer

model_id, out = sys.argv[1], sys.argv[2]
model = SentenceTransformer(model_id, device="cpu")
long_text = " ".join(["Die Donaudampfschifffahrt war ein wichtiger Wirtschaftszweig im neunzehnten Jahrhundert."] * 120)
texts = [
    "query: Wie hoch ist der Eiffelturm?",
    "passage: Der Eiffelturm in Paris ist 330 Meter hoch und wurde 1889 fertiggestellt.",
    "query: how tall is the Eiffel Tower",
    "passage: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
    "query: Quelle est la hauteur de la tour Eiffel ?",
    "query: 埃菲尔铁塔有多高？",
    "passage: 埃菲尔铁塔高330米，位于法国巴黎。",
    "query: Какова высота Эйфелевой башни?",
    "passage: مرحبا بالعالم، هذا اختبار للنموذج.",
    "passage: Übergrößenträger straßenbahn Mädchen Fußball Käse Öl",
    "passage: ＡＢＣ１２３ ﬁﬂ ㎏ ①②③ é",
    "passage: 😀 👨‍👩‍👧 🇩🇪",
    "query: a",
    "passage: ",
    "passage: " + long_text,
    "query: " + " ".join(["wort"] * 600),
    "passage: tab\tand\nnewline   and   spaces",
    "query: Wie funktioniert ein Wärmetauscher in einer Lüftungsanlage?",
    "passage: Ein Wärmetauscher überträgt Wärme von der Abluft auf die Zuluft, ohne dass sich beide Luftströme vermischen.",
    "passage: Python is a programming language. Python ist eine Programmiersprache.",
]
vectors = model.encode(texts, batch_size=8, normalize_embeddings=False)
json.dump({"max_seq_length": model.max_seq_length, "texts": texts, "vectors": [v.tolist() for v in vectors]}, open(out, "w"), ensure_ascii=False)
print(len(texts), "texts, dimension", vectors.shape[1], "max_seq_length", model.max_seq_length)
