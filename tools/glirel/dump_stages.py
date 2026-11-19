import json, os, sys
import numpy as np, torch
from glirel import GLiREL
from glirel.model import load_config_as_namespace
from huggingface_hub import hf_hub_download

refpath, outdir = sys.argv[1], sys.argv[2]
cfg = load_config_as_namespace(hf_hub_download("jackboyla/glirel-large-v0", "glirel_config.json"))
model = GLiREL(cfg)
model.load_state_dict(torch.load(hf_hub_download("jackboyla/glirel-large-v0", "pytorch_model.bin"), map_location="cpu"), strict=False, assign=True)
model.eval()
layer = model.token_rep_layer
tok = layer.bert_layer.tokenizer
ref = json.load(open(refpath))
for i, case in enumerate(ref["cases"]):
    prompt = []
    for l in case["labels"]:
        prompt += ["[REL]", l]
    prompt += ["[SEP]"]
    words = prompt + case["tokens"]
    enc = tok(words, is_split_into_words=True, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    word_ids = enc.word_ids(0)
    first = []
    for w in range(len(words)):
        first.append(next(j for j, x in enumerate(word_ids) if x == w))
    with torch.no_grad():
        hidden = layer.bert_layer.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state[0]
        pooled = layer.compute_word_embedding([words])[0]
    np.save(f"{outdir}/case{i}_hidden.npy", hidden.numpy().astype(np.float32))
    np.save(f"{outdir}/case{i}_pooled.npy", pooled.numpy().astype(np.float32))
    case["input_ids"], case["first_subtoken"], case["n_prompt_words"] = ids, first, len(prompt)
    print(i, "seq", len(ids), "words", len(words), "max |hidden - pooled[first]|", float((hidden[first] - pooled).abs().max()))
json.dump(ref, open(refpath, "w"), ensure_ascii=False)
