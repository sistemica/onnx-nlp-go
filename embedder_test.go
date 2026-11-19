package onnxnlp

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/sistemica/onnx-nlp-go/internal/fixture"
)

func meanRows(ids ...int) []float32 {
	table := fixture.EmbeddingTable(len(fixture.MiniLMVocab))
	out := make([]float32, fixture.Hidden)
	for _, id := range ids {
		for d := range out {
			out[d] += table[id*fixture.Hidden+d] / float32(len(ids))
		}
	}
	return out
}

func closeTo(a, b []float32) bool {
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > 1e-5 {
			return false
		}
	}
	return len(a) == len(b)
}

func TestEmbedderPoolsAndNormalizes(t *testing.T) {
	e, err := LoadEmbedder(minilmDir(t, 0, true))
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Embed([]string{"hello world"})
	if err != nil {
		t.Fatal(err)
	}
	want := meanRows(2, 4, 5, 3)
	l2Normalize(want)
	if !closeTo(got[0], want) {
		t.Fatalf("got %v, want %v", got[0], want)
	}
	var norm float64
	for _, v := range got[0] {
		norm += float64(v) * float64(v)
	}
	if math.Abs(norm-1) > 1e-5 {
		t.Fatalf("not unit length: %v", norm)
	}
}

func TestEmbedderWithoutNormalizeAndTruncation(t *testing.T) {
	e, err := LoadEmbedder(minilmDir(t, 3, false))
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Embed([]string{"hello world machine"})
	if err != nil {
		t.Fatal(err)
	}
	if want := meanRows(2, 4, 3); !closeTo(got[0], want) {
		t.Fatalf("truncated mean: got %v, want %v", got[0], want)
	}
}

func TestEmbedderBatchesAndKeepsOrder(t *testing.T) {
	e, err := LoadEmbedder(minilmDir(t, 0, true))
	if err != nil {
		t.Fatal(err)
	}
	words := []string{"hello", "world", "machine learning", "a", "models, hello world machine learning"}
	var texts []string
	for i := 0; i < 2*embedBatchSize+5; i++ {
		texts = append(texts, words[i%len(words)]+strings.Repeat(" a", i%4))
	}
	batched, err := e.Embed(texts)
	if err != nil {
		t.Fatal(err)
	}
	for i, text := range texts {
		single, err := e.Embed([]string{text})
		if err != nil {
			t.Fatal(err)
		}
		if !closeTo(batched[i], single[0]) {
			t.Fatalf("text %d (%q): batched %v, single %v", i, text, batched[i], single[0])
		}
	}
	none, err := e.Embed(nil)
	if err != nil || len(none) != 0 {
		t.Fatalf("empty input: %v %v", none, err)
	}
}

func TestEmbedTrimsTextLikeSentenceTransformers(t *testing.T) {
	e, err := LoadEmbedder(minilmDir(t, 0, true))
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Embed([]string{"alice met bob", " \t alice met bob\u001f\n", "\u001c"})
	if err != nil {
		t.Fatal(err)
	}
	if !closeTo(got[0], got[1]) {
		t.Fatalf("surrounding whitespace must not change the embedding: %v vs %v", got[0], got[1])
	}
	if len(got[2]) == 0 {
		t.Fatal("a text that is only whitespace still embeds")
	}
}

func TestLoadEmbedderErrors(t *testing.T) {
	mutate := func(name string, edit func(dir string)) {
		dir := minilmDir(t, 0, true)
		edit(dir)
		if _, err := LoadEmbedder(dir); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	mutate("no pooling config", func(d string) { os.Remove(filepath.Join(d, "1_Pooling", "config.json")) })
	mutate("no pooling mode", func(d string) {
		os.WriteFile(filepath.Join(d, "1_Pooling", "config.json"), []byte(`{"pooling_mode_max_tokens":true}`), 0o644)
	})
	mutate("two pooling modes", func(d string) {
		os.WriteFile(filepath.Join(d, "1_Pooling", "config.json"), []byte(`{"pooling_mode_cls_token":true,"pooling_mode_mean_tokens":true}`), 0o644)
	})
	mutate("unsupported tokenizer", func(d string) {
		os.WriteFile(filepath.Join(d, "tokenizer.json"), []byte(`{"model":{"type":"BPE"}}`), 0o644)
	})
	mutate("broken tokenizer", func(d string) { os.WriteFile(filepath.Join(d, "tokenizer.json"), []byte(`{`), 0o644) })
	mutate("no modules", func(d string) { os.Remove(filepath.Join(d, "modules.json")) })
	mutate("no tokenizer", func(d string) { os.Remove(filepath.Join(d, "tokenizer.json")) })
	mutate("no model", func(d string) { os.Remove(filepath.Join(d, "onnx", "model.onnx")) })
	if _, err := LoadEmbedder(t.TempDir()); err == nil {
		t.Fatal("empty directory must fail")
	}
}

func TestEmbedderCLSPoolingTakesTheFirstToken(t *testing.T) {
	dir := minilmDir(t, 0, true)
	os.WriteFile(filepath.Join(dir, "1_Pooling", "config.json"), []byte(`{"pooling_mode_cls_token":true}`), 0o644)
	cls, err := LoadEmbedder(dir)
	if err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(dir, "1_Pooling", "config.json"), []byte(`{"pooling_mode_mean_tokens":true}`), 0o644)
	mean, err := LoadEmbedder(dir)
	if err != nil {
		t.Fatal(err)
	}
	text := []string{"alice met bob", "bob"}
	a, err := cls.Embed(text)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := mean.Embed(text)
	if closeTo(a[0], b[0]) {
		t.Fatalf("CLS and mean pooling must give different vectors for a text with several tokens: %v", a[0])
	}
	single, _ := cls.Embed(text[:1])
	if !closeTo(a[0], single[0]) {
		t.Fatalf("a text must embed the same alone and in a batch: %v vs %v", a[0], single[0])
	}
}

func TestEmbedderAcceptsTokenEmbeddingsOutput(t *testing.T) {
	e, err := LoadEmbedder(minilmDir(t, 0, false))
	if err != nil {
		t.Fatal(err)
	}
	want, _ := e.Embed([]string{"hello world"})
	g := e.session.graph
	g.Nodes[len(g.Nodes)-1].Out = []string{"token_embeddings"}
	g.Outputs = []string{"token_embeddings"}
	e.session, err = NewSession(g)
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Embed([]string{"hello world"})
	if err != nil || !closeTo(got[0], want[0]) {
		t.Fatalf("the sentence-transformers output name must work too: %v %v", got, err)
	}
}

func TestEmbedderReportsAFailingModel(t *testing.T) {
	e, err := LoadEmbedder(minilmDir(t, 0, true))
	if err != nil {
		t.Fatal(err)
	}
	e.session, _ = NewSession(&Graph{
		Inputs:  []string{"input_ids", "attention_mask", "token_type_ids"},
		Outputs: []string{"last_hidden_state"},
		Nodes:   []*Node{{Op: "Gather", In: []string{"input_ids", "input_ids"}, Out: []string{"last_hidden_state"}}},
	})
	if _, err := e.Embed([]string{"hello"}); err == nil {
		t.Fatal("a failing model must surface")
	}
}

func TestEmbedderModelProblems(t *testing.T) {
	e, err := LoadEmbedder(minilmDir(t, 0, true))
	if err != nil {
		t.Fatal(err)
	}
	e.session, _ = NewSession(&Graph{
		Inputs:  []string{"input_ids", "attention_mask", "token_type_ids"},
		Nodes:   []*Node{{Op: "Identity", In: []string{"input_ids"}, Out: []string{"last_hidden_state"}}},
		Outputs: []string{"last_hidden_state"},
	})
	if _, err := e.Embed([]string{"hello"}); err == nil || !strings.Contains(err.Error(), "last_hidden_state") {
		t.Fatalf("wrong output rank: %v", err)
	}
	e.session, _ = NewSession(&Graph{Inputs: []string{"input_ids", "attention_mask", "token_type_ids", "extra"}})
	if _, err := e.Embed([]string{"hello"}); err == nil {
		t.Fatal("model run failure must surface")
	}
}

func TestVectorMath(t *testing.T) {
	zero := []float32{0, 0}
	l2Normalize(zero)
	if zero[0] != 0 || zero[1] != 0 {
		t.Fatalf("zero vector changed: %v", zero)
	}
	v := []float32{3, 4}
	l2Normalize(v)
	if !closeTo(v, []float32{0.6, 0.8}) {
		t.Fatalf("got %v", v)
	}
	if cosine([]float32{1, 0}, []float32{0, 1}) != 0 || cosine(zero, v) != 0 || cosine(v, zero) != 0 {
		t.Fatal("orthogonal or zero vectors must have cosine 0")
	}
	if got := cosine([]float32{2, 0}, []float32{5, 0}); math.Abs(float64(got)-1) > 1e-6 {
		t.Fatalf("got %v", got)
	}
	if got := argsortAscending([]float32{3, 1, 2, 1}); !slices.Equal(got, []int{1, 3, 2, 0}) {
		t.Fatalf("got %v", got)
	}
	if got := lastN([]int{1, 2, 3}, 2); !slices.Equal(got, []int{2, 3}) {
		t.Fatalf("got %v", got)
	}
	if got := lastN([]int{1}, 5); !slices.Equal(got, []int{1}) {
		t.Fatalf("got %v", got)
	}
	if argmax([]float32{1, 3, 3, 2}) != 1 {
		t.Fatal("argmax must return the first maximum")
	}
}
