package onnxgo

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func minilmEnv(t *testing.T) (model, ref string) {
	t.Helper()
	model, ref = os.Getenv("ONNXGO_MINILM"), os.Getenv("ONNXGO_KBREF")
	if model == "" || ref == "" {
		t.Skip("ONNXGO_MINILM and ONNXGO_KBREF not set")
	}
	return model, ref
}

func TestWordPieceAgainstReference(t *testing.T) {
	model, ref := minilmEnv(t)
	wp, err := LoadWordPiece(filepath.Join(model, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		S   string `json:"s"`
		IDs []int  `json:"ids"`
	}
	readTestJSON(t, filepath.Join(ref, "wp_gold.json"), &cases)
	bad := 0
	for _, c := range cases {
		if got := wp.Encode(c.S, 256); !slices.Equal(got, c.IDs) {
			if bad < 10 {
				t.Errorf("%q: got %v, want %v", c.S, got, c.IDs)
			}
			bad++
		}
	}
	t.Logf("%d of %d cases differ", bad, len(cases))
}

func TestEmbedderAgainstReference(t *testing.T) {
	model, ref := minilmEnv(t)
	e, err := LoadEmbedder(model)
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		S string    `json:"s"`
		V []float64 `json:"v"`
	}
	readTestJSON(t, filepath.Join(ref, "embed_gold.json"), &cases)
	texts := make([]string, len(cases))
	for i, c := range cases {
		texts[i] = c.S
	}
	got, err := e.Embed(texts)
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range cases {
		var maxDiff float64
		for d, w := range c.V {
			maxDiff = max(maxDiff, math.Abs(float64(got[i][d])-w))
		}
		t.Logf("%.30q: max abs diff %.2g", c.S, maxDiff)
		if maxDiff > 1e-5 {
			t.Errorf("%q: embedding differs by %g", c.S, maxDiff)
		}
	}
}

func readTestJSON(t *testing.T, path string, v any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatal(err)
	}
}

func TestKeyBERTAgainstReference(t *testing.T) {
	model, ref := minilmEnv(t)
	e, err := LoadEmbedder(model)
	if err != nil {
		t.Fatal(err)
	}
	kb := NewKeyBERT(e)
	var cases []struct {
		Name string   `json:"name"`
		Docs []string `json:"docs"`
		Opts struct {
			Ngram        []int    `json:"ngram"`
			TopN         int      `json:"top_n"`
			MinDF        int      `json:"min_df"`
			Stop         *bool    `json:"stop"`
			MMR          bool     `json:"mmr"`
			MaxSum       bool     `json:"maxsum"`
			Diversity    float64  `json:"diversity"`
			NrCandidates int      `json:"nr_candidates"`
			Candidates   []string `json:"candidates"`
			Seeds        []string `json:"seeds"`
		} `json:"opts"`
		Result [][][]any `json:"result"`
	}
	readTestJSON(t, filepath.Join(ref, "keybert.json"), &cases)
	matched, total := 0, 0
	for _, c := range cases {
		o := KeyBERTOptions{
			TopN: c.Opts.TopN, MinDF: c.Opts.MinDF, MMR: c.Opts.MMR, MaxSum: c.Opts.MaxSum,
			Diversity: c.Opts.Diversity, NrCandidates: c.Opts.NrCandidates,
			Candidates: c.Opts.Candidates, SeedKeywords: c.Opts.Seeds,
			KeepStopWords: c.Opts.Stop != nil && !*c.Opts.Stop,
		}
		if len(c.Opts.Ngram) == 2 {
			o.NgramMin, o.NgramMax = c.Opts.Ngram[0], c.Opts.Ngram[1]
		}
		got, err := kb.Extract(c.Docs, o)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		for d, want := range c.Result {
			total += len(want)
			if len(got[d]) != len(want) {
				t.Errorf("%s doc %d: got %v, want %v", c.Name, d, got[d], want)
				continue
			}
			for i, w := range want {
				text, score := w[0].(string), w[1].(float64)
				if got[d][i].Text != text || math.Abs(got[d][i].Score-score) > 2e-4 {
					t.Errorf("%s doc %d #%d: got %q %.4f, want %q %.4f", c.Name, d, i, got[d][i].Text, got[d][i].Score, text, score)
					continue
				}
				matched++
			}
		}
	}
	t.Logf("%d of %d reference keywords matched in %d cases", matched, total, len(cases))
}
