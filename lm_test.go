package onnxnlp

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/sistemica/onnx-nlp-go/internal/fixture"
	"github.com/sistemica/onnx-nlp-go/internal/pb"
)

func lmDir(t *testing.T, positionIDs bool) string {
	t.Helper()
	dir := t.TempDir()
	fixture.Must(fixture.WriteLM(dir, positionIDs))
	return dir
}

func TestLanguageModelForwardExtendsTheCache(t *testing.T) {
	for _, positions := range []bool{true, false} {
		m, err := LoadLanguageModel(lmDir(t, positions))
		if err != nil {
			t.Fatal(err)
		}
		cache := m.NewCache()
		if cache.Len() != 0 {
			t.Fatalf("a new cache is empty: %d", cache.Len())
		}
		logits, err := m.Forward([]int{1, 2, 5}, cache)
		if err != nil {
			t.Fatal(err)
		}
		if len(logits.Shape) != 3 || logits.Shape[1] != 3 || logits.Shape[2] != fixture.LMVocab || cache.Len() != 3 {
			t.Fatalf("logits %v, cache %d", logits.Shape, cache.Len())
		}
		for p, token := range []int{1, 2, 5} {
			if got := argmax(logits.F[p*fixture.LMVocab : (p+1)*fixture.LMVocab]); got != (token+1)%fixture.LMVocab {
				t.Errorf("position %d: token %d is followed by %d, want %d", p, token, got, (token+1)%fixture.LMVocab)
			}
		}
		more, err := m.Forward([]int{7}, cache)
		if err != nil || argmax(more.F) != 0 || cache.Len() != 4 {
			t.Fatalf("a single token after the cache: %v, cache %d, %v", more, cache.Len(), err)
		}
		if cache.keys[0].Shape[2] != 4 || cache.values[0].Shape[2] != 4 {
			t.Fatalf("the cache holds %v keys and %v values", cache.keys[0].Shape, cache.values[0].Shape)
		}
	}
}

func TestLanguageModelErrors(t *testing.T) {
	if _, err := LoadLanguageModel(t.TempDir()); err == nil {
		t.Fatal("an empty directory must fail")
	}
	dir := lmDir(t, true)
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"num_hidden_layers":0}`), 0o644)
	if _, err := LoadLanguageModel(dir); err == nil {
		t.Fatal("a config without transformer dimensions must fail")
	}
	dir = lmDir(t, true)
	os.Remove(filepath.Join(dir, "onnx", "model.onnx"))
	if _, err := LoadLanguageModel(dir); err == nil {
		t.Fatal("a missing model must fail")
	}
	m, err := LoadLanguageModel(lmDir(t, true))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Forward(nil, m.NewCache()); err == nil {
		t.Fatal("no tokens must be rejected")
	}
	grouped := lmDir(t, true)
	os.WriteFile(filepath.Join(grouped, "config.json"), []byte(`{"num_hidden_layers":1,"num_attention_heads":4,"hidden_size":32}`), 0o644)
	gm, err := LoadLanguageModel(grouped)
	if err != nil || gm.kvHeads != 4 || gm.headDim != 8 {
		t.Fatalf("missing head counts default to the attention heads and hidden size / heads: %+v %v", gm, err)
	}
	// A graph without the attention mask cannot be a language model.
	g, err := ParseGraphFile(filepath.Join(lmDir(t, true), "onnx", "model.onnx"))
	if err != nil {
		t.Fatal(err)
	}
	g.Inputs = []string{"table", "w", "input_ids"}
	onlyIDs, _ := NewSession(g)
	faulty := &LanguageModel{session: onlyIDs, layers: 1, kvHeads: 1, headDim: 8, inputSet: map[string]bool{"input_ids": true}}
	if _, err := faulty.Forward([]int{1}, faulty.NewCache()); err == nil {
		t.Fatal("a graph that cannot be fed its inputs must surface the error")
	}
	// Graphs that return too little.
	for name, outputs := range map[string][]string{"no logits": {"present.0.key", "present.0.value"}, "no cache": {"logits"}} {
		g, err := ParseGraphFile(filepath.Join(lmDir(t, true), "onnx", "model.onnx"))
		if err != nil {
			t.Fatal(err)
		}
		g.Outputs = outputs
		s, err := NewSession(g)
		if err != nil {
			t.Fatal(err)
		}
		bad := &LanguageModel{session: s, layers: 1, kvHeads: 1, headDim: 8, inputSet: map[string]bool{"input_ids": true, "attention_mask": true, "position_ids": true}}
		if _, err := bad.Forward([]int{1}, bad.NewCache()); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	// A graph without the attention mask input is not a language model.
	bare := lmDir(t, true)
	model := pb.Model(pb.Graph([][]byte{pb.Node("Identity", []string{"input_ids"}, []string{"logits"})}, nil, []string{"input_ids"}, []string{"logits"}))
	if err := os.WriteFile(filepath.Join(bare, "onnx", "model.onnx"), model, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadLanguageModel(bare); err == nil {
		t.Fatal("a graph without attention_mask must be rejected when loading")
	}
}
