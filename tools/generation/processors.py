"""Records what the Hugging Face logits processors do to fixed score vectors.

usage: processors.py OUT_JSON
Each case holds the scores, the token history, the settings and the processed scores of every step
(repetition penalty, temperature, top-k, top-p, in the order generate() applies them).
-inf is written as null.
"""
import json
import sys

import numpy as np
import torch
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
)

rng = np.random.default_rng(5)
cases = []
settings = [
    dict(penalty=1.1, temperature=0.7, top_k=20, top_p=0.8),
    dict(penalty=1.0, temperature=1.0, top_k=0, top_p=1.0),
    dict(penalty=1.3, temperature=1.0, top_k=0, top_p=1.0),
    dict(penalty=1.0, temperature=0.5, top_k=0, top_p=1.0),
    dict(penalty=1.0, temperature=1.0, top_k=5, top_p=1.0),
    dict(penalty=1.0, temperature=1.0, top_k=0, top_p=0.9),
    dict(penalty=1.0, temperature=1.0, top_k=0, top_p=0.2),
    dict(penalty=1.2, temperature=1.5, top_k=50, top_p=0.95),
    dict(penalty=1.0, temperature=1.0, top_k=1000, top_p=1.0),
]
for s in settings:
    for trial in range(4):
        vocab = 300
        scores = rng.normal(0, 3, size=vocab).astype(np.float32)
        scores[rng.integers(0, vocab, 3)] *= -1
        history = rng.integers(0, vocab, size=int(rng.integers(1, 40))).tolist()
        if trial == 3:
            scores[:10] = scores[0]  # ties
        x = torch.tensor(scores).unsqueeze(0)
        ids = torch.tensor(history).unsqueeze(0)
        steps = {}
        if s["penalty"] != 1.0:
            x = RepetitionPenaltyLogitsProcessor(s["penalty"])(ids, x)
        steps["penalty"] = x[0].tolist()
        if s["temperature"] != 1.0:
            x = TemperatureLogitsWarper(s["temperature"])(ids, x)
        steps["temperature"] = x[0].tolist()
        if s["top_k"] > 0:
            x = TopKLogitsWarper(top_k=s["top_k"])(ids, x)
        steps["top_k"] = x[0].tolist()
        if s["top_p"] < 1.0:
            x = TopPLogitsWarper(top_p=s["top_p"])(ids, x)
        steps["top_p"] = x[0].tolist()
        clean = lambda v: [None if (a == float("-inf")) else a for a in v]
        cases.append({"settings": s, "scores": scores.tolist(), "history": history,
                      "steps": {k: clean(v) for k, v in steps.items()}})
json.dump(cases, open(sys.argv[1], "w"))
print(len(cases), "cases")
