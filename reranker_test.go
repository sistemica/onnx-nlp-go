package onnxnlp

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/sistemica/onnx-nlp-go/internal/fixture"
)

// expectedScore recomputes the fixture reranker: the mean embedding of the pair's ids projected
// to the logits, minus the first logit when there are two.
func expectedScore(ids []int, logits int) float32 {
	table, weights := fixture.EmbeddingTable(len(fixture.MiniLMVocab)), fixture.RerankerWeights(logits)
	pooled := make([]float64, fixture.Hidden)
	for _, id := range ids {
		for d := range pooled {
			pooled[d] += float64(table[id*fixture.Hidden+d]) / float64(len(ids))
		}
	}
	out := make([]float64, logits)
	for l := range out {
		for d := range pooled {
			out[l] += pooled[d] * float64(weights[d*logits+l])
		}
	}
	if logits == 2 {
		return float32(out[1] - out[0])
	}
	return float32(out[0])
}

func rerankerDir(t *testing.T, logits int, tokenTypes bool) string {
	t.Helper()
	dir := t.TempDir()
	fixture.Must(fixture.WriteReranker(dir, logits, tokenTypes))
	return dir
}

func TestRerankerScoresPairsWithWordPiece(t *testing.T) {
	for _, tc := range []struct {
		logits     int
		tokenTypes bool
	}{{1, true}, {1, false}, {2, true}} {
		r, err := LoadReranker(rerankerDir(t, tc.logits, tc.tokenTypes))
		if err != nil {
			t.Fatal(err)
		}
		docs := []string{"machine learning", "hello", "world model", "a cafe"}
		scores, err := r.Score("hello world", docs)
		if err != nil {
			t.Fatal(err)
		}
		wp, _ := LoadWordPiece(filepath.Join(rerankerDir(t, tc.logits, tc.tokenTypes), "tokenizer.json"))
		for i, doc := range docs {
			ids, _ := wp.EncodePair("hello world", doc, 0)
			if want := expectedScore(ids, tc.logits); math.Abs(float64(scores[i]-want)) > 1e-5 {
				t.Errorf("%d logits, token types %v, %q: score %v, want %v", tc.logits, tc.tokenTypes, doc, scores[i], want)
			}
		}
	}
}

func TestRerankerScoresPairsWithUnigram(t *testing.T) {
	dir := rerankerDir(t, 1, false)
	writeTokenizerJSON(t, filepath.Join(dir, "tokenizer.json"), unigramSpec(true, xlmrNormalizer()))
	r, err := LoadReranker(dir)
	if err != nil {
		t.Fatal(err)
	}
	scores, err := r.Score("hello", []string{"world", "a hello"})
	if err != nil {
		t.Fatal(err)
	}
	if want := expectedScore([]int{0, 4, 2, 2, 5, 2}, 1); math.Abs(float64(scores[0]-want)) > 1e-5 {
		t.Fatalf("score %v, want %v", scores[0], want)
	}
}

func TestRerankerBatchesAndRanks(t *testing.T) {
	r, err := LoadReranker(rerankerDir(t, 1, true))
	if err != nil {
		t.Fatal(err)
	}
	var docs []string
	for i := 0; i < 40; i++ {
		docs = append(docs, []string{"machine", "learning model", "hello world hello", "a"}[i%4])
	}
	batched, err := r.Score("hello", docs)
	if err != nil {
		t.Fatal(err)
	}
	for i, doc := range docs {
		alone, _ := r.Score("hello", []string{doc})
		if math.Abs(float64(batched[i]-alone[0])) > 1e-5 {
			t.Fatalf("document %d: %v in a batch of %d, %v alone", i, batched[i], len(docs), alone[0])
		}
	}
	ranked, err := r.Rank("hello", docs)
	if err != nil {
		t.Fatal(err)
	}
	if len(ranked) != len(docs) {
		t.Fatalf("%d ranked of %d", len(ranked), len(docs))
	}
	for i := 1; i < len(ranked); i++ {
		if ranked[i].Score > ranked[i-1].Score {
			t.Fatalf("not sorted at %d", i)
		}
		if ranked[i].Score == ranked[i-1].Score && ranked[i].Index < ranked[i-1].Index {
			t.Fatalf("equal scores must keep input order at %d", i)
		}
	}
	for _, rk := range ranked {
		if want := float32(1 / (1 + math.Exp(-float64(rk.Score)))); math.Abs(float64(rk.Probability-want)) > 1e-6 || batched[rk.Index] != rk.Score {
			t.Fatalf("%+v inconsistent with score %v", rk, batched[rk.Index])
		}
	}
	if none, err := r.Rank("hello", nil); err != nil || len(none) != 0 {
		t.Fatalf("no documents: %v %v", none, err)
	}
	r.MaxLen = 6
	if _, err := r.Score("hello world machine learning", []string{"hello world machine learning model"}); err != nil {
		t.Fatal(err)
	}
}

func TestRerankerErrors(t *testing.T) {
	if _, err := LoadReranker(t.TempDir()); err == nil {
		t.Fatal("an empty directory must fail")
	}
	dir := rerankerDir(t, 1, true)
	os.Remove(filepath.Join(dir, "onnx", "model.onnx"))
	if _, err := LoadReranker(dir); err == nil {
		t.Fatal("a missing model must fail")
	}
	r, _ := LoadReranker(rerankerDir(t, 1, true))
	wide, _ := LoadReranker(rerankerDir(t, 3, true))
	if _, err := wide.Score("hello", []string{"world"}); err == nil {
		t.Fatal("three logits per pair are not a relevance score")
	}
	if _, err := wide.Rank("hello", []string{"world"}); err == nil {
		t.Fatal("Rank must report scoring errors")
	}
	r.session, _ = NewSession(&Graph{Inputs: []string{"input_ids", "positions"}, Outputs: []string{"logits"}})
	if _, err := r.Score("hello", []string{"world"}); err == nil {
		t.Fatal("an unsupported input must be reported")
	}
	r.session, _ = NewSession(&Graph{Inputs: []string{"input_ids"}, Outputs: []string{"logits"}, Nodes: []*Node{{Op: "Gather", In: []string{"input_ids", "input_ids"}, Out: []string{"logits"}}}})
	if _, err := r.Score("hello", []string{"world"}); err == nil {
		t.Fatal("a failing model must surface")
	}
}

func TestEmbedderWithAUnigramTokenizer(t *testing.T) {
	dir := minilmDir(t, 0, false)
	writeTokenizerJSON(t, filepath.Join(dir, "tokenizer.json"), unigramSpec(true, xlmrNormalizer()))
	e, err := LoadEmbedder(dir)
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Embed([]string{"hello world"})
	if err != nil {
		t.Fatal(err)
	}
	table := fixture.EmbeddingTable(len(fixture.MiniLMVocab))
	want := make([]float32, fixture.Hidden)
	for _, id := range []int{0, 4, 5, 2} {
		for d := range want {
			want[d] += table[id*fixture.Hidden+d] / 4
		}
	}
	if !closeTo(got[0], want) {
		t.Fatalf("embedding %v, want the mean of the rows for <s> hello world </s>: %v", got[0], want)
	}
}
