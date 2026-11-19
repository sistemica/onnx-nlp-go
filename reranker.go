package onnxnlp

import (
	"fmt"
	"math"
	"path/filepath"
	"sort"
)

const rerankBatchSize = 16

// Reranker is a cross-encoder exported to ONNX: it reads a query and a document together and
// returns one relevance score, for example bge-reranker or ms-marco-MiniLM.
type Reranker struct {
	// MaxLen limits the ids of a query and document together; longer pairs are truncated
	// from the longer text first.
	MaxLen int

	enc     *sentenceEncoder
	session *Session
}

// Ranked is one document with its score, as returned by Rank.
type Ranked struct {
	// Index is the position of the document in the input.
	Index int
	// Score is the model's raw output; higher means more relevant.
	Score float32
	// Probability is the sigmoid of Score, between 0 and 1.
	Probability float32
}

// LoadReranker loads a directory with tokenizer.json and onnx/model.onnx. The model must output
// logits with one value per pair, or two values per pair (the score is then the logit of the
// second class minus the first).
func LoadReranker(dir string) (*Reranker, error) {
	r := &Reranker{MaxLen: 512}
	var err error
	if r.enc, err = loadSentenceEncoder(filepath.Join(dir, "tokenizer.json")); err != nil {
		return nil, err
	}
	if r.session, err = Load(filepath.Join(dir, "onnx", "model.onnx")); err != nil {
		return nil, err
	}
	return r, nil
}

// Score returns the raw relevance score of every document for the query, in input order.
func (r *Reranker) Score(query string, docs []string) ([]float32, error) {
	ids := make([][]int, len(docs))
	segments := make([][]int, len(docs))
	order := make([]int, len(docs))
	for i, doc := range docs {
		ids[i], segments[i] = r.enc.pair(query, doc, r.MaxLen)
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return len(ids[order[a]]) > len(ids[order[b]]) })
	scores := make([]float32, len(docs))
	for _, batch := range planBatches(order, func(i int) int { return len(ids[i]) }, rerankBatchSize) {
		inputs, _, _, err := encoderInputs(r.session.InputNames(), ids, segments, batch)
		if err != nil {
			return nil, err
		}
		res, err := r.session.Run(inputs)
		if err != nil {
			return nil, err
		}
		logits := res["logits"]
		if logits == nil || len(logits.Shape) != 2 || logits.Shape[0] != len(batch) || (logits.Shape[1] != 1 && logits.Shape[1] != 2) {
			return nil, fmt.Errorf("reranker: model output is not one or two logits per pair")
		}
		for i, idx := range batch {
			row := logits.F[i*logits.Shape[1] : (i+1)*logits.Shape[1]]
			scores[idx] = row[len(row)-1]
			if len(row) == 2 {
				scores[idx] -= row[0]
			}
		}
	}
	return scores, nil
}

// Rank scores the documents and returns them best first; equal scores keep their input order.
func (r *Reranker) Rank(query string, docs []string) ([]Ranked, error) {
	scores, err := r.Score(query, docs)
	if err != nil {
		return nil, err
	}
	out := make([]Ranked, len(docs))
	for i, s := range scores {
		out[i] = Ranked{Index: i, Score: s, Probability: float32(1 / (1 + math.Exp(-float64(s))))}
	}
	sort.SliceStable(out, func(a, b int) bool { return out[a].Score > out[b].Score })
	return out, nil
}
