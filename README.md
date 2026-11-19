# onnxgo

A pure-Go (no cgo) ONNX inference engine with three ready-made pipelines:

| Pipeline | Entry point | Model |
|---|---|---|
| Entity extraction | `LoadGLiNER(dir).Extract(text, labels, threshold)` | GLiNER2.5 boundary export (`DanKau/gliner2.5-multi-v1-onnx`) |
| Keyword extraction | `NewKeyBERT(LoadEmbedder(dir)).Extract(docs, opts)` | any sentence-transformers directory with `onnx/model.onnx`, mean pooling, WordPiece (default `all-MiniLM-L6-v2`) |
| Relation extraction | `LoadGLiREL(dir).Extract(tokens, entities, labels, threshold, topK)` | GLiREL (`jackboyla/glirel-large-v0`), exported with `tools/glirel` |
| Topic modelling | `NewBERTopic(LoadEmbedder(dir)).Fit(docs)` | BERTopic's default pipeline: sentence embeddings, UMAP, HDBSCAN, class-based TF-IDF |
| Raw inference | `Load(path).Run(inputs)` | any ONNX model whose operators are implemented (see `ops_misc.go`) |

CLI: `go run ./cmd/gliner -model DIR -labels person,company "text"`.

### GLiREL

GLiREL scores relations between entities you have already found (for example with GLiNER or spaCy). Entities are half-open token ranges; relation labels are free text.

```go
g, _ := onnxgo.LoadGLiREL("glirel_export")
rels, _ := g.Extract(tokens, []onnxgo.Span{{0, 2}, {7, 8}}, []string{"works for", "located in"}, 0.5, 0)
```

The model has no ONNX release, so build the export directory once (needs `torch`, `transformers`, `glirel`, `onnx`, about 6 GB of downloads):

1. `python tools/glirel/reference.py ref.json` records PyTorch's scores for a few sentences.
2. `EXPORT_ENCODER=1 python tools/glirel/export.py glirel_export ref.json` writes `onnx/encoder.onnx` (1.7 GB), `onnx/head.onnx`, `tokenizer.json` and `glirel_export.json`, and fails unless the head reproduces the PyTorch scores.
3. `python tools/glirel/dump_stages.py ref.json DIR` writes per-stage reference data for the model-backed tests.

`cmd/glirel` is a small command-line front end. Not implemented: the optional `refine_*`, `coref` and non-`dot` scorer variants, which this checkpoint does not use.

### BERTopic

```go
topics := onnxgo.NewBERTopic(embedder) // any TextEmbedder, e.g. LoadEmbedder("all-MiniLM-L6-v2")
topics.RemoveStopWords = true          // BERTopic keeps them by default
model, _ := topics.Fit(docs)           // or FitEmbeddings(docs, vectors)
for _, t := range model.Topics { fmt.Println(t.ID, t.Size, t.Words) } // topic -1 holds the outliers

ids, strengths, _ := model.Transform(newDocs) // assign new documents; the topics do not change
```

`cmd/bertopic -model DIR -docs docs.txt -stop-words [-assign new.txt]` prints the topics of a text file with one document per line and, with `-assign`, the topic of every document in a second file. 3000 newsgroup posts take 1.3 s to reduce and cluster (the embedding is the rest).

All of it is Go, including UMAP and HDBSCAN, and the stages follow umap-learn 0.6 (classic layout) and hdbscan with BERTopic's defaults (15 neighbours, 5 components, minimum distance 0, minimum topic size 10, excess-of-mass selection). `tools/bertopic/reference.py` records every stage of the Python pipeline for the model-backed tests. Checked against it:

| Stage | Result |
|---|---|
| UMAP neighbour graph | all 61,178 edges present, weights within 4.5e-6 |
| UMAP spectral initialisation | eigenvalues within 1e-5, eigenvectors \|cos\| = 1 with scipy's |
| UMAP optimiser | identical to the library over the epochs that are deterministic (largest difference 7e-5); the random stream differs, so later epochs differ |
| HDBSCAN | merge distances identical; on the library's own dendrogram, labels and strengths identical; on 5 embeddings, adjusted Rand 0.985 to 1.0 |
| Topic words (c-TF-IDF) | scores identical to 1e-9 on 440 word positions, with and without stop words |
| Transform: UMAP graph of new documents | all 7,500 edges present, weights within 6e-8 |
| Transform: starting positions | within 2.5e-6 of umap-learn's |
| Transform: final positions | median distance to umap-learn's 0.122, which is what umap-learn gives against itself under another seed (0.123-0.125) |
| Transform: HDBSCAN `approximate_predict` | 0 of 500 labels differ (216 of them noise), strengths identical |
| Whole pipeline | neighbour preservation 0.362 against umap-learn's 0.363; where clustering does not collapse (below) adjusted Rand 0.67 against umap-learn, which is what umap-learn gives against itself across seeds (0.65 to 0.70) |

Things to know:

- UMAP is random, so exactly the same topics as Python are not possible. Use `UMAP.Seed` for repeatable runs.
- **On the 3000-post newsgroup corpus, about half of all seeds make UMAP followed by HDBSCAN return just two clusters and no outliers.** This happens with umap-learn too (19 of 40 runs from one fixed start), and Go's rate is the same (16 of 40), so it comes from the algorithm and not the port. If you get it, change the seed or `MinTopicSize`.
- HDBSCAN on ties: many merges happen at exactly the same distance, and the order in which they are processed changes the result. The library's order comes from an unstable sort that varies by CPU, so this port uses a fixed order (by endpoints). The library's default also builds an approximate spanning tree, which this port does not; with the library's exact mode the merge distances are identical.
- `Transform` is approximate by design, like BERTopic's: new documents get the topic of their nearest neighbour in the existing clustering, and nothing is re-clustered. After aligning topic ids, 87% of training and 85% of held-out documents land in the topic BERTopic chose. It needs a model made with `Fit` (which keeps the embedder) or `TransformEmbeddings`.
- Not implemented: topic reduction (`nr_topics`), outlier reduction, representation models other than c-TF-IDF, online and guided topic modelling.

## Speed

- Matrix products use a hand-written AVX2/FMA 6x16 kernel (Go assembly, plain-Go fallback elsewhere). Static weights are packed once; ragged edges run through padded tiles instead of a scalar loop.
- `exp`, `erf` and GELU run as AVX2 kernels (softmax and GELU are memory-bound instead of compute-bound), two-operand `Einsum` runs as batched matrix products, and `Where`, `Add`, `Mul` and friends walk strided rows in parallel instead of building index tables.
- Common operator chains are fused when a session is created (bias after a weight matrix product, GELU, and the decomposed layer normalization), which cut MiniLM time by about a quarter. Fusion only applies when no intermediate result is used elsewhere.
- Work is spread over a persistent pool of helper goroutines. Helpers park instead of spinning: on hyperthreaded CPUs spinning slowed the sibling thread doing real work.
- Constant subgraphs are folded when a session is created, so work that depends only on weights (such as DeBERTa's relative-position projections) runs once.
- Large products go to a GPU when one is found at run time. Nothing is needed at build time: no cgo, no CUDA or Vulkan SDK.
  - **CUDA** (`libcuda` + `libcublas`, loaded with `purego`): tried first.
  - **Vulkan** (`libvulkan`, a compute shader embedded from `shaders/gemm.spv`): tried next. Only discrete GPUs are picked automatically; integrated GPUs share memory with the CPU and were slower end to end in testing, so they need `ONNXGO_GPU_BACKEND=vulkan`.
  - Weights stay resident on the device; activations move per product; small products and failures fall back to the CPU kernel.
- Settings: `ONNXGO_GPU=off` disables GPUs, `ONNXGO_GPU_BACKEND=cuda|vulkan` forces a backend, `ONNXGO_GPU_DEVICE=N` picks a device (CUDA default: most free memory; note CUDA and Vulkan number devices differently), `ONNXGO_GPU_MIN_FLOPS=N` sets the smallest product sent to a GPU, `ONNXGO_VULKAN_ALLOW_CPU=1` allows software Vulkan devices, `ONNXGO_AVX512=on` selects the 12-row AVX-512 matrix kernel (off by default: slower on Zen 4, about even on Zen 5).
- `Session.EnableProfiling()` / `OpStats()` report wall time per operator type.
- To rebuild the shader: `go generate ./...` (needs `glslc`).

## Tests

`go test ./...` is self-contained: it builds tiny synthetic models and tokenizers.

Model-backed tests compare against onnxruntime, Hugging Face `tokenizers`, sentence-transformers, the official `gliner2` and `keybert` packages. They run when these variables point at reference data: `ONNXGO_MODELS`, `ONNXGO_REF`, `ONNXGO_TOKGOLD`, `ONNXGO_E2E`, `ONNXGO_MINILM`, `ONNXGO_KBREF`, `ONNXGO_GLIREL`, `ONNXGO_GLIREL_REF`, `ONNXGO_BERTOPIC_REF`.

`go test -run '^$' -bench . ` runs the benchmarks (same variables).

GPU tests that need real hardware skip without it; the offload logic and every driver error path of both backends run against fake drivers everywhere.
