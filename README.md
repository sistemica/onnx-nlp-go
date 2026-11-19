# onnx-nlp-go

ONNX inference engine and NLP pipelines in pure Go. No cgo, no ONNX Runtime.

| Pipeline | Entry point | Model |
|---|---|---|
| Entity extraction | `LoadGLiNER` | GLiNER 2.5 multilingual |
| Relation extraction | `LoadGLiREL` | GLiREL large |
| Keyword extraction | `NewKeyBERT` | any sentence-transformers model |
| Topic modelling | `NewBERTopic` | any sentence-transformers model |
| Embeddings | `LoadEmbedder` | all-MiniLM-L6-v2, multilingual-e5-small, bge-m3 |
| Reranking | `LoadReranker` | bge-reranker-base, ms-marco-MiniLM-L-6-v2 |
| Raw inference | `Load` | any ONNX model with supported operators |

## Install

```
go get github.com/sistemica/onnx-nlp-go
```

Requires Go 1.26. Runs on any platform Go supports; amd64 uses assembly kernels. GPUs are used when
found at run time (see [GPU](#gpu)).

```go
import onnxnlp "github.com/sistemica/onnx-nlp-go"
```

## Models

| Pipeline | Directory contents | Source |
|---|---|---|
| GLiNER | `gliner2_config.json`, `tokenizer.json`, `onnx/encoder.onnx`, `onnx/boundary.onnx` | `DanKau/gliner2.5-multi-v1-onnx` |
| Embeddings, KeyBERT, BERTopic | `modules.json`, `1_Pooling/config.json`, `sentence_bert_config.json`, `tokenizer.json`, `onnx/model.onnx` | ONNX export of a sentence-transformers model, e.g. `all-MiniLM-L6-v2`, `intfloat/multilingual-e5-small`, `BAAI/bge-m3` |
| Reranker | `tokenizer.json`, `onnx/model.onnx` | `Xenova/bge-reranker-base`, `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| GLiREL | `glirel_export.json`, `tokenizer.json`, `onnx/encoder.onnx`, `onnx/head.onnx` | built with `tools/glirel` |

Build the GLiREL directory (needs Python with `torch`, `transformers`, `glirel`, `onnx`):

```
python tools/glirel/reference.py ref.json
EXPORT_ENCODER=1 python tools/glirel/export.py glirel_export ref.json
```

## Use

### Entities

```go
g, err := onnxnlp.LoadGLiNER("gliner")
entities, err := g.Extract("Tim Cook visited Berlin.", []string{"person", "city"}, 0.5)
// []Entity{Label, Text, Start, End, Score}; Start and End are byte offsets
```

### Relations

Entities are token ranges `[Start, End)`, found beforehand (for example with GLiNER).

```go
g, err := onnxnlp.LoadGLiREL("glirel_export")
tokens := []string{"Tim", "Cook", "works", "for", "Apple", "."}
spans := []onnxnlp.Span{{0, 2}, {4, 5}}
relations, err := g.Extract(tokens, spans, []string{"works for", "founded"}, 0.5, 0)
// []Relation{Head, Tail, HeadText, TailText, Label, Score}
```

The last argument keeps at most that many labels per entity pair (0 keeps all).

### Embeddings

```go
embedder, err := onnxnlp.LoadEmbedder("multilingual-e5-small")
vectors, err := embedder.Embed([]string{"query: Wie hoch ist der Eiffelturm?"})
```

Mean and CLS pooling, WordPiece and XLM-R tokenizers. Texts are stripped of surrounding whitespace, as
sentence-transformers does. Models that expect a prefix (`query: `, `passage: `) get it from the caller.

### Reranking

```go
r, err := onnxnlp.LoadReranker("bge-reranker-base")
ranked, err := r.Rank("Wie hoch ist der Eiffelturm?", docs)
// []Ranked{Index, Score, Probability}, best first; Score is the raw logit
scores, err := r.Score(query, docs) // same order as docs
```

`MaxLen` (default 512) limits query and document together. Models with one logit per pair, or two (score is
the second minus the first), are supported.

### Keywords

```go
embedder, err := onnxnlp.LoadEmbedder("all-MiniLM-L6-v2")
keywords, err := onnxnlp.NewKeyBERT(embedder).Extract(docs, onnxnlp.KeyBERTOptions{TopN: 5, NgramMax: 2})
// [][]Keyword{Text, Score}
```

`KeyBERTOptions` also has `MMR`, `MaxSum`, `Diversity`, `NrCandidates`, `SeedKeywords`, `Candidates`,
`MinDF`, `KeepStopWords`.

### Topics

```go
topics := onnxnlp.NewBERTopic(embedder)
topics.MinTopicSize = 10      // default
topics.RemoveStopWords = true // default false
topics.UMAP.Seed = 42         // default

model, err := topics.Fit(docs)
for _, t := range model.Topics { // topic -1 holds the outliers
	fmt.Println(t.ID, t.Size, t.Words)
}
model.Assignments // topic of each document
model.Strengths   // cluster membership, 0 to 1

ids, strengths, err := model.Transform(newDocs) // assign new documents; topics do not change
```

`FitEmbeddings(docs, vectors)` takes precomputed embeddings; use `TransformEmbeddings` on that model.

### Raw inference

```go
session, err := onnxnlp.Load("model.onnx")
out, err := session.Run(map[string]*onnxnlp.Tensor{
	"input_ids": onnxnlp.NewInt([]int{1, 3}, []int64{101, 7592, 102}),
})
```

## Command line

```
go install github.com/sistemica/onnx-nlp-go/cmd/gliner@latest
go install github.com/sistemica/onnx-nlp-go/cmd/glirel@latest
go install github.com/sistemica/onnx-nlp-go/cmd/bertopic@latest
```

```
gliner  -model DIR -labels person,city [-threshold 0.5] "TEXT"
glirel  -model DIR -labels "works for,founded" -entities 0:2,4:5 [-threshold 0.5] [-topk N] TOKEN...
bertopic -model DIR [-docs FILE] [-min-topic-size 10] [-top-words 10] [-stop-words] [-seed 42] [-assignments] [-assign FILE]
```

`bertopic` reads one document per line from `-docs` or standard input. `-assign FILE` prints the topic
of every document in `FILE` after fitting.

## GPU

Large matrix products run on an accelerator when one is found; no build-time dependency. On macOS,
Apple Accelerate (`cblas_sgemm`) is tried first. CUDA (`libcuda`, `libcublas`) comes next, then Vulkan
(`libvulkan`). Only discrete GPUs are picked automatically. The Accelerate backend has not been run on a Mac.

| Variable | Effect |
|---|---|
| `ONNXNLP_GPU=off` | disable GPUs |
| `ONNXNLP_GPU_BACKEND=cuda\|vulkan` | force a backend (also allows an integrated GPU) |
| `ONNXNLP_GPU_DEVICE=N` | device index |
| `ONNXNLP_GPU_MIN_FLOPS=N` | smallest product sent to a GPU |
| `ONNXNLP_VULKAN_ALLOW_CPU=1` | allow software Vulkan devices |
| `ONNXNLP_AVX512=on` | use the AVX-512 matrix kernel |

## Limits

- Operators: the ~60 needed by the supported models, listed in `ops_misc.go`. Unsupported operators
  return an error when the session is created. Quantized (int8, fp16) models are not supported.
- Memory: weights are held as float32, plus a packed copy for the matrix kernel. bge-m3 needs about 11 GB
  with long texts.
- GLiNER: entity extraction only (no classification, relation or structure heads).
- GLiREL: `dot` scorer without refinement or coreference.
- KeyBERT: English stop words only.
- BERTopic: no topic reduction, outlier reduction or alternative representation models. UMAP is random;
  set `UMAP.Seed` for repeatable runs.

## Test

```
go test ./...
```

Runs without model files. Tests against reference data run when the matching variable is set:
`ONNXNLP_MODELS`, `ONNXNLP_MINILM`, `ONNXNLP_REF`, `ONNXNLP_TOKGOLD`, `ONNXNLP_E2E`,
`ONNXNLP_KBREF`, `ONNXNLP_GLIREL`, `ONNXNLP_GLIREL_REF`, `ONNXNLP_BERTOPIC_REF`, `ONNXNLP_EMBED_MODEL`,
`ONNXNLP_EMBED_REF`, `ONNXNLP_RERANK_MODEL`, `ONNXNLP_RERANK_REF`, `ONNXNLP_XLMR_TOKENIZER`,
`ONNXNLP_XLMR_GOLD`, `ONNXNLP_WP_TOKENIZER`, `ONNXNLP_WP_GOLD`. The Python scripts in
`tools/` produce the reference data.

## License

MIT
