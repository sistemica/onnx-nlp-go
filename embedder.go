package onnxgo

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
)

const embedBatchSize = 32

// Embedder is a sentence-transformers model exported to ONNX with a WordPiece
// tokenizer, mean pooling and optional L2 normalization.
type Embedder struct {
	tok       *WordPiece
	session   *Session
	maxLen    int
	normalize bool
}

// LoadEmbedder loads a sentence-transformers directory such as all-MiniLM-L6-v2
// that contains onnx/model.onnx next to the usual configuration files.
func LoadEmbedder(dir string) (*Embedder, error) {
	var pooling struct {
		Mean bool `json:"pooling_mode_mean_tokens"`
	}
	if err := readJSON(filepath.Join(dir, "1_Pooling", "config.json"), &pooling); err != nil {
		return nil, err
	}
	if !pooling.Mean {
		return nil, fmt.Errorf("embedder: only mean pooling is supported")
	}
	var modules []struct {
		Type string `json:"type"`
	}
	if err := readJSON(filepath.Join(dir, "modules.json"), &modules); err != nil {
		return nil, err
	}
	e := &Embedder{maxLen: 256}
	for _, m := range modules {
		if m.Type == "sentence_transformers.models.Normalize" {
			e.normalize = true
		}
	}
	var cfg struct {
		MaxSeqLength int `json:"max_seq_length"`
	}
	if err := readJSON(filepath.Join(dir, "sentence_bert_config.json"), &cfg); err == nil && cfg.MaxSeqLength > 0 {
		e.maxLen = cfg.MaxSeqLength
	}
	var err error
	if e.tok, err = LoadWordPiece(filepath.Join(dir, "tokenizer.json")); err != nil {
		return nil, err
	}
	if e.session, err = Load(filepath.Join(dir, "onnx", "model.onnx")); err != nil {
		return nil, err
	}
	return e, nil
}

func readJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

// Embed returns one embedding per text, in input order.
func (e *Embedder) Embed(texts []string) ([][]float32, error) {
	ids := make([][]int, len(texts))
	order := make([]int, len(texts))
	for i, t := range texts {
		ids[i] = e.tok.Encode(t, e.maxLen)
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return len(ids[order[a]]) > len(ids[order[b]]) })
	out := make([][]float32, len(texts))
	for lo := 0; lo < len(order); lo += embedBatchSize {
		batch := order[lo:min(lo+embedBatchSize, len(order))]
		vecs, err := e.embedBatch(ids, batch)
		if err != nil {
			return nil, err
		}
		for k, idx := range batch {
			out[idx] = vecs[k]
		}
	}
	return out, nil
}

func (e *Embedder) embedBatch(ids [][]int, batch []int) ([][]float32, error) {
	width := 0
	for _, idx := range batch {
		width = max(width, len(ids[idx]))
	}
	n := len(batch)
	inputIDs, mask := zeros(Int64, []int{n, width}), zeros(Int64, []int{n, width})
	for r, idx := range batch {
		for c, id := range ids[idx] {
			inputIDs.I[r*width+c] = int64(id)
			mask.I[r*width+c] = 1
		}
	}
	res, err := e.session.Run(map[string]*Tensor{
		"input_ids":      inputIDs,
		"attention_mask": mask,
		"token_type_ids": zeros(Int64, []int{n, width}),
	})
	if err != nil {
		return nil, err
	}
	hidden := res["last_hidden_state"]
	if hidden == nil || len(hidden.Shape) != 3 {
		return nil, fmt.Errorf("embedder: model has no last_hidden_state output")
	}
	dim := hidden.Shape[2]
	vecs := make([][]float32, n)
	for r := range batch {
		v := make([]float32, dim)
		count := 0
		for c := 0; c < width; c++ {
			if mask.I[r*width+c] == 0 {
				continue
			}
			count++
			row := hidden.F[(r*width+c)*dim : (r*width+c+1)*dim]
			for d := range v {
				v[d] += row[d]
			}
		}
		for d := range v {
			v[d] /= float32(max(count, 1))
		}
		if e.normalize {
			l2Normalize(v)
		}
		vecs[r] = v
	}
	return vecs, nil
}

func l2Normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	inv := float32(1 / math.Max(math.Sqrt(sum), 1e-12))
	for i := range v {
		v[i] *= inv
	}
}

func cosine(a, b []float32) float32 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}

func argsortAscending(scores []float32) []int {
	idx := make([]int, len(scores))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return scores[idx[a]] < scores[idx[b]] })
	return idx
}

func lastN[T any](s []T, n int) []T {
	return s[max(len(s)-n, 0):]
}
