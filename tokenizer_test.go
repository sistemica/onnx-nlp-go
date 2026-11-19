package onnxgo

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestTokenizerAgainstReference(t *testing.T) {
	models, gold := os.Getenv("ONNXGO_MODELS"), os.Getenv("ONNXGO_TOKGOLD")
	if models == "" || gold == "" {
		t.Skip("ONNXGO_MODELS and ONNXGO_TOKGOLD not set")
	}
	tok, err := LoadTokenizer(filepath.Join(models, "tokenizer.json"), nil)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(gold)
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		S   string `json:"s"`
		IDs []int  `json:"ids"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	bad := 0
	for _, c := range cases {
		got := tok.Tokenize(c.S)
		if !slices.Equal(got, c.IDs) {
			if bad < 15 {
				t.Errorf("%q: got %v, want %v", c.S, got, c.IDs)
			}
			bad++
		}
	}
	t.Logf("%d of %d cases differ", bad, len(cases))
	if bad > 0 {
		t.Fail()
	}
}
