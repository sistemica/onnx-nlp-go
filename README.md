# onnxgo

A pure-Go (no cgo) ONNX inference engine with three ready-made pipelines:

| Pipeline | Entry point | Model |
|---|---|---|
| Entity extraction | `LoadGLiNER(dir).Extract(text, labels, threshold)` | GLiNER2.5 boundary export (`DanKau/gliner2.5-multi-v1-onnx`) |
| Keyword extraction | `NewKeyBERT(LoadEmbedder(dir)).Extract(docs, opts)` | any sentence-transformers directory with `onnx/model.onnx`, mean pooling, WordPiece (default `all-MiniLM-L6-v2`) |
| Relation extraction | `LoadGLiREL(dir).Extract(tokens, entities, labels, threshold, topK)` | GLiREL (`jackboyla/glirel-large-v0`), exported with `tools/glirel` |
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

BERTopic is not implemented.

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

Model-backed tests compare against onnxruntime, Hugging Face `tokenizers`, sentence-transformers, the official `gliner2` and `keybert` packages. They run when these variables point at reference data: `ONNXGO_MODELS`, `ONNXGO_REF`, `ONNXGO_TOKGOLD`, `ONNXGO_E2E`, `ONNXGO_MINILM`, `ONNXGO_KBREF`.

`go test -run '^$' -bench . ` runs the benchmarks (same variables).

GPU tests that need real hardware skip without it; the offload logic and every driver error path of both backends run against fake drivers everywhere.
