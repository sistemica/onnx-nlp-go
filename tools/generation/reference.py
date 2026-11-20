"""Records a forward pass and a greedy continuation of a causal language model.

usage: reference.py MODEL_DIR OUT_DIR
Writes prompt.json (token ids, text, the chat-formatted prompt, greedy ids and text) and logits.npy
(float32, [tokens, vocabulary]) for the prompt.
"""
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
messages = [{"role": "user", "content": "Wie viele Beine hat eine Spinne?"}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt").input_ids
with torch.no_grad():
    logits = model(ids).logits[0]
    generated = model.generate(ids, max_new_tokens=24, do_sample=False, repetition_penalty=1.0)[0][ids.shape[1]:]
np.save(out + "/logits.npy", logits.numpy().astype(np.float32))
json.dump({"prompt": prompt, "ids": ids[0].tolist(), "greedy_ids": generated.tolist(),
           "greedy_text": tok.decode(generated, skip_special_tokens=False)}, open(out + "/prompt.json", "w"), ensure_ascii=False)
print(ids.shape[1], "prompt tokens; greedy:", repr(tok.decode(generated, skip_special_tokens=True)))
