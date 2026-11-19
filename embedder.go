package onnxnlp

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
)

const embedBatchSize = 32

// Embedder is a sentence-transformers model exported to ONNX with a WordPiece or Unigram
// tokenizer, mean or CLS pooling and optional L2 normalization.
type Embedder struct {
	enc       *sentenceEncoder
	session   *Session
	maxLen    int
	normalize bool
	cls       bool
}

// LoadEmbedder loads a sentence-transformers directory such as all-MiniLM-L6-v2
// that contains onnx/model.onnx next to the usual configuration files.
func LoadEmbedder(dir string) (*Embedder, error) {
	var pooling struct {
		Mean bool `json:"pooling_mode_mean_tokens"`
		CLS  bool `json:"pooling_mode_cls_token"`
	}
	if err := readJSON(filepath.Join(dir, "1_Pooling", "config.json"), &pooling); err != nil {
		return nil, err
	}
	if pooling.Mean == pooling.CLS {
		return nil, fmt.Errorf("embedder: only mean pooling or CLS pooling is supported")
	}
	var modules []struct {
		Type string `json:"type"`
	}
	if err := readJSON(filepath.Join(dir, "modules.json"), &modules); err != nil {
		return nil, err
	}
	e := &Embedder{maxLen: 256, cls: pooling.CLS}
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
	if e.enc, err = loadSentenceEncoder(filepath.Join(dir, "tokenizer.json")); err != nil {
		return nil, err
	}
	if e.session, err = Load(filepath.Join(dir, "onnx", "model.onnx")); err != nil {
		return nil, err
	}
	return e, nil
}

// trimLikePython removes what Python's str.strip removes. sentence-transformers applies it to
// every text before tokenizing.
func trimLikePython(s string) string {
	return strings.TrimFunc(s, func(r rune) bool { return unicode.IsSpace(r) || (r >= 0x1c && r <= 0x1f) })
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
		ids[i] = e.enc.single(trimLikePython(t), e.maxLen)
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return len(ids[order[a]]) > len(ids[order[b]]) })
	out := make([][]float32, len(texts))
	for _, batch := range planBatches(order, func(i int) int { return len(ids[i]) }, embedBatchSize) {
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
	inputs, mask, width, err := encoderInputs(e.session.InputNames(), ids, nil, batch)
	if err != nil {
		return nil, err
	}
	res, err := e.session.Run(inputs)
	if err != nil {
		return nil, err
	}
	hidden := res["last_hidden_state"]
	if hidden == nil {
		hidden = res["token_embeddings"]
	}
	if hidden == nil || len(hidden.Shape) != 3 {
		return nil, fmt.Errorf("embedder: model has no last_hidden_state or token_embeddings output")
	}
	dim := hidden.Shape[2]
	vecs := make([][]float32, len(batch))
	for r := range batch {
		v := make([]float32, dim)
		if e.cls {
			copy(v, hidden.F[r*width*dim:r*width*dim+dim])
		} else {
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
