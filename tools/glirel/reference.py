import json
import sys

import torch
from glirel import GLiREL

out = sys.argv[1]
from huggingface_hub import hf_hub_download
from glirel.model import load_config_as_namespace

cfg_path = hf_hub_download("jackboyla/glirel-large-v0", "glirel_config.json")
weights_path = hf_hub_download("jackboyla/glirel-large-v0", "pytorch_model.bin")
model = GLiREL(load_config_as_namespace(cfg_path))
state = torch.load(weights_path, map_location="cpu")
result = model.load_state_dict(state, strict=False, assign=True)
print("missing keys:", len(result.missing_keys), result.missing_keys[:6])
print("unexpected keys:", len(result.unexpected_keys), result.unexpected_keys[:6])
model.to("cpu")
model.eval()
cfg = model.config
print("add_entity_markers:", getattr(cfg, "add_entity_markers", "<absent>"), "max_entity_pair_distance:", cfg.max_entity_pair_distance)

labels = ["country of origin", "licensed to broadcast to", "father", "followed by", "located in", "works for", "founded"]
cases = [
    {
        "tokens": "Derren Nesbitt had a history of being cast in " "Hollywood films before moving to the United Kingdom .".split(),
        "ner": [[0, 1, "PERSON", "Derren Nesbitt"], [11, 12, "GPE", "United Kingdom"]],
    },
    {
        "tokens": "Tim Cook , the chief executive of Apple , visited Berlin and met Angela Merkel .".split(),
        "ner": [[0, 1, "PERSON", "Tim Cook"], [7, 7, "ORG", "Apple"], [10, 10, "GPE", "Berlin"], [13, 14, "PERSON", "Angela Merkel"]],
    },
    {
        "tokens": "Marie Curie was born in Warsaw and worked in Paris .".split(),
        "ner": [[0, 1, "PERSON", "Marie Curie"], [5, 5, "GPE", "Warsaw"], [8, 8, "GPE", "Paris"]],
    },
    {
        "tokens": "Der Film wurde von Steven Spielberg in Kalifornien gedreht .".split(),
        "ner": [[4, 5, "PERSON", "Steven Spielberg"], [7, 7, "GPE", "Kalifornien"]],
    },
]

records = []
for case in cases:
    x = model.collate_fn([{"tokenized_text": case["tokens"], "ner": case["ner"]}], relation_types=labels, device=model.device)
    with torch.no_grad():
        scores, num_classes, rel_type_mask, coref = model.compute_score(x)
    probs = torch.sigmoid(scores)[0].tolist()
    preds = model.predict_relations(case["tokens"], labels, threshold=0.5, ner=case["ner"], top_k=-1)
    low = model.predict_relations(case["tokens"], labels, threshold=0.05, ner=case["ner"], top_k=-1)
    enc = model.token_rep_layer.token_rep_layer if hasattr(model.token_rep_layer, "token_rep_layer") else None
    records.append({
        "tokens": case["tokens"],
        "ner": case["ner"],
        "labels": list(x["classes_to_id"][0].keys()),
        "span_idx": x["span_idx"][0].tolist(),
        "relations_idx": x["relations_idx"][0].tolist(),
        "model_tokens": x["tokens"][0],
        "seq_length": int(x["seq_length"][0]),
        "scores": scores[0].tolist(),
        "probs": probs,
        "pred_0.5": preds,
        "pred_0.05": low,
    })
    print(case["tokens"][:6], "->", len(preds), "relations at 0.5;", len(low), "at 0.05; pairs", len(x["relations_idx"][0]))
    for r in preds:
        print("   ", r["head_text"], "|", r["label"], "|", r["tail_text"], round(r["score"], 3))
json.dump({"labels": labels, "cases": records}, open(out, "w"), ensure_ascii=False)
print("model_tokens example:", records[0]["model_tokens"][:12])
print("subtoken pooling:", cfg.subtoken_pooling, "hidden:", cfg.hidden_size)
