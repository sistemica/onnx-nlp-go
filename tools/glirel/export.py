import json
import os
import sys

import torch
from torch import nn

from glirel import GLiREL
from glirel.model import load_config_as_namespace
from huggingface_hub import hf_hub_download

outdir, refpath = sys.argv[1], sys.argv[2]
os.makedirs(outdir + "/onnx", exist_ok=True)

cfg = load_config_as_namespace(hf_hub_download("jackboyla/glirel-large-v0", "glirel_config.json"))
model = GLiREL(cfg)
state = torch.load(hf_hub_download("jackboyla/glirel-large-v0", "pytorch_model.bin"), map_location="cpu")
res = model.load_state_dict(state, strict=False, assign=True)
assert not res.missing_keys and not res.unexpected_keys, res
model.eval()

layer = model.token_rep_layer
tok = layer.bert_layer.tokenizer
tok.save_pretrained(outdir)
ids = {n: tok.convert_tokens_to_ids(n) for n in ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[REL]", "[E]", "[/E]", "[FLERT]"]}
print("special ids:", ids, "vocab size:", len(tok))
json.dump({"architecture": "glirel", "special_tokens": ids, "hidden_size": cfg.hidden_size,
           "label_embed_strategy": cfg.label_embed_strategy, "max_len": cfg.max_len,
           "subtoken_pooling": cfg.subtoken_pooling}, open(outdir + "/glirel_export.json", "w"), indent=1)


class Encoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class Head(nn.Module):
    """Everything after first-subtoken pooling, for a single sentence."""

    def __init__(self, m):
        super().__init__()
        self.projection = m.token_rep_layer.projection
        self.lstm = m.rnn.lstm
        self.span_marker = m.span_rep_layer.rel_rep_layer.span_marker
        self.out_project = m.span_rep_layer.rel_rep_layer.out_project
        self.prompt_rep_layer = m.prompt_rep_layer

    def forward(self, prompt_rep, word_rep, span_idx, head_ids, tail_ids):
        prompt = self.projection(prompt_rep)
        words = self.projection(word_rep)
        labels = prompt.reshape(-1, 2, prompt.shape[-1]).mean(dim=1).unsqueeze(0)
        words, _ = self.lstm(words)
        entities = self.span_marker(words, span_idx.unsqueeze(0))[0]
        pairs = torch.cat([entities[head_ids], entities[tail_ids]], dim=-1).unsqueeze(0)
        pair_rep = self.out_project(pairs)
        label_rep = self.prompt_rep_layer(labels)
        return torch.einsum("bkd,bcd->bkc", pair_rep, label_rep)[0]


ref = json.load(open(refpath))
labels = ref["labels"]
head = Head(model).eval()
encoder = Encoder(layer.bert_layer.model).eval()


def prompt_tokens(labels_, words):
    prompt = []
    for l in labels_:
        prompt += ["[REL]", l]
    return prompt + ["[SEP]"], words


def pooled_reps(words, labels_):
    """First-subtoken embeddings of [REL] l ... [SEP] words, exactly as TokenRepLayer builds them."""
    prompt, text = prompt_tokens(labels_, words)
    with torch.no_grad():
        emb = layer.compute_word_embedding([prompt + text])[0]  # [P+W, 1024]
    return emb[: len(prompt) - 1], emb[len(prompt):]


worst = 0.0
for case in ref["cases"]:
    prompt_rep, word_rep = pooled_reps(case["tokens"], case["labels"])
    span_idx = torch.tensor(case["span_idx"], dtype=torch.long)
    spans = [tuple(s) for s in case["span_idx"]]
    pairs = [(tuple(p[0]), tuple(p[1])) for p in case["relations_idx"]]
    head_ids = torch.tensor([spans.index(p[0]) for p in pairs])
    tail_ids = torch.tensor([spans.index(p[1]) for p in pairs])
    with torch.no_grad():
        got = head(prompt_rep.unsqueeze(0), word_rep.unsqueeze(0), span_idx, head_ids, tail_ids)
    want = torch.tensor(case["scores"])
    d = (got - want).abs().max().item()
    worst = max(worst, d)
    print("head wrapper vs reference scores:", tuple(got.shape), "max abs diff", d)
assert worst < 1e-4, worst

case = ref["cases"][1]
prompt_rep, word_rep = pooled_reps(case["tokens"], case["labels"])
spans = [tuple(s) for s in case["span_idx"]]
pairs = [(tuple(p[0]), tuple(p[1])) for p in case["relations_idx"]]
example = (prompt_rep.unsqueeze(0), word_rep.unsqueeze(0), torch.tensor(case["span_idx"]),
           torch.tensor([spans.index(p[0]) for p in pairs]), torch.tensor([spans.index(p[1]) for p in pairs]))
torch.onnx.export(
    head, example, outdir + "/onnx/head.onnx", dynamo=False, opset_version=17,
    input_names=["prompt_rep", "word_rep", "span_idx", "head_ids", "tail_ids"], output_names=["scores"],
    dynamic_axes={"prompt_rep": {1: "prompt_len"}, "word_rep": {1: "words"}, "span_idx": {0: "entities"},
                  "head_ids": {0: "pairs"}, "tail_ids": {0: "pairs"}, "scores": {0: "pairs", 1: "labels"}},
)
print("exported head.onnx")

if os.environ.get("EXPORT_ENCODER") == "1":
    enc = tok(case["tokens"], is_split_into_words=True, return_tensors="pt")
    torch.onnx.export(
        encoder, (enc["input_ids"], enc["attention_mask"]), outdir + "/onnx/encoder.onnx", dynamo=False, opset_version=17,
        input_names=["input_ids", "attention_mask"], output_names=["hidden_states"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"},
                      "hidden_states": {0: "batch", 1: "sequence"}},
    )
    print("exported encoder.onnx")
