package onnxnlp

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestEmbedderAgainstSentenceTransformers(t *testing.T) {
	dir, refPath := os.Getenv("ONNXNLP_EMBED_MODEL"), os.Getenv("ONNXNLP_EMBED_REF")
	if dir == "" || refPath == "" {
		t.Skip("ONNXNLP_EMBED_MODEL and ONNXNLP_EMBED_REF not set")
	}
	raw, err := os.ReadFile(refPath)
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		MaxSeqLength int         `json:"max_seq_length"`
		Texts        []string    `json:"texts"`
		Vectors      [][]float32 `json:"vectors"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	e, err := LoadEmbedder(dir)
	if err != nil {
		t.Fatal(err)
	}
	if e.maxLen != ref.MaxSeqLength {
		t.Fatalf("max sequence length %d, sentence-transformers uses %d", e.maxLen, ref.MaxSeqLength)
	}
	got, err := e.Embed(ref.Texts)
	if err != nil {
		t.Fatal(err)
	}
	worstCos, worstAbs := 1.0, 0.0
	for i := range got {
		if c := cosine(got[i], ref.Vectors[i]); c < 0.99999 {
			t.Logf("text %d (%.40q): cosine %.6f", i, ref.Texts[i], c)
		}
		worstCos = math.Min(worstCos, float64(cosine(got[i], ref.Vectors[i])))
		for d := range got[i] {
			worstAbs = math.Max(worstAbs, math.Abs(float64(got[i][d]-ref.Vectors[i][d])))
		}
		if len(got[i]) != len(ref.Vectors[i]) {
			t.Fatalf("text %d: dimension %d, want %d", i, len(got[i]), len(ref.Vectors[i]))
		}
	}
	t.Logf("%d texts: lowest cosine similarity with sentence-transformers %.8f, largest absolute difference %.3g", len(got), worstCos, worstAbs)
	if worstCos < 0.99999 || worstAbs > 1e-3 {
		t.Fatalf("embeddings differ from sentence-transformers: cosine %.7f, difference %g", worstCos, worstAbs)
	}
}
