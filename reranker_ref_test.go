package onnxnlp

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"testing"
)

func TestRerankerAgainstTransformers(t *testing.T) {
	dir, refPath := os.Getenv("ONNXNLP_RERANK_MODEL"), os.Getenv("ONNXNLP_RERANK_REF")
	if dir == "" || refPath == "" {
		t.Skip("ONNXNLP_RERANK_MODEL and ONNXNLP_RERANK_REF not set")
	}
	raw, err := os.ReadFile(refPath)
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Pairs  [][2]string `json:"pairs"`
		Logits [][]float32 `json:"logits"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	r, err := LoadReranker(dir)
	if err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for i, p := range ref.Pairs {
		got, err := r.Score(p[0], []string{p[1]})
		if err != nil {
			t.Fatal(err)
		}
		d := math.Abs(float64(got[0] - ref.Logits[i][0]))
		worst = max(worst, d)
		if d > 5e-3 {
			t.Errorf("pair %d (%.30q / %.30q): logit %.5f, transformers %.5f", i, p[0], p[1], got[0], ref.Logits[i][0])
		}
	}
	t.Logf("%d pairs: largest difference from transformers' logits %.3g", len(ref.Pairs), worst)

	// The same query against several documents in one batch must score each document as when alone.
	query := ref.Pairs[0][0]
	docs := []string{ref.Pairs[0][1], ref.Pairs[1][1], ref.Pairs[10][1]}
	batched, err := r.Score(query, docs)
	if err != nil {
		t.Fatal(err)
	}
	for i, doc := range docs {
		alone, _ := r.Score(query, []string{doc})
		if math.Abs(float64(batched[i]-alone[0])) > 1e-3 {
			t.Errorf("document %d: batched %.5f, alone %.5f", i, batched[i], alone[0])
		}
	}
	ranked, err := r.Rank(query, docs)
	if err != nil {
		t.Fatal(err)
	}
	if ranked[0].Index != 0 || !slices.IsSortedFunc(ranked, func(a, b Ranked) int {
		switch {
		case a.Score > b.Score:
			return -1
		case a.Score < b.Score:
			return 1
		}
		return 0
	}) {
		t.Fatalf("the document that answers the query must rank first: %+v", ranked)
	}
}
