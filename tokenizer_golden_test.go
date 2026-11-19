package onnxnlp

import (
	"encoding/json"
	"os"
	"slices"
	"testing"
)

type goldenRecord struct {
	S       string `json:"s"`
	Norm    string `json:"norm"`
	IDs     []int  `json:"ids"`
	Special []int  `json:"special"`
}

type goldenPair struct {
	A     string  `json:"a"`
	B     *string `json:"b"`
	Limit int     `json:"limit"`
	IDs   []int   `json:"ids"`
	Types []int   `json:"types"`
}

func loadGolden(t *testing.T) (tok *Tokenizer, cases []goldenRecord, pairs []goldenPair) {
	t.Helper()
	path, goldPath := os.Getenv("ONNXNLP_XLMR_TOKENIZER"), os.Getenv("ONNXNLP_XLMR_GOLD")
	if path == "" || goldPath == "" {
		t.Skip("ONNXNLP_XLMR_TOKENIZER and ONNXNLP_XLMR_GOLD not set")
	}
	tok, err := LoadTokenizer(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(goldPath)
	if err != nil {
		t.Fatal(err)
	}
	var gold struct {
		Cases []goldenRecord `json:"cases"`
		Pairs []goldenPair   `json:"pairs"`
	}
	if err := json.Unmarshal(raw, &gold); err != nil {
		t.Fatal(err)
	}
	return tok, gold.Cases, gold.Pairs
}

func TestTokenizerNormalizationAgainstHuggingFace(t *testing.T) {
	tok, cases, _ := loadGolden(t)
	wrong := 0
	for _, c := range cases {
		if got := tok.normalized(c.S); got != c.Norm {
			if wrong < 8 {
				t.Errorf("normalize(%+q) = %+q, want %+q", c.S, got, c.Norm)
			}
			wrong++
		}
	}
	t.Logf("%d of %d strings normalise differently from Hugging Face", wrong, len(cases))
}

func TestTokenizerEncodingAgainstHuggingFace(t *testing.T) {
	tok, cases, _ := loadGolden(t)
	plain, special := 0, 0
	for _, c := range cases {
		if got := tok.encodeText(c.S); !slices.Equal(got, c.IDs) {
			if plain < 6 {
				t.Errorf("ids(%+q) = %v, want %v", c.S, got, c.IDs)
			}
			plain++
		}
		got, err := tok.Encode(c.S, 0)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, c.Special) {
			if special < 6 {
				t.Errorf("Encode(%+q) = %v, want %v", c.S, got, c.Special)
			}
			special++
		}
	}
	t.Logf("%d strings: %d differ without special tokens, %d with", len(cases), plain, special)
}

func TestTokenizerPairsAndTruncationAgainstHuggingFace(t *testing.T) {
	tok, _, pairs := loadGolden(t)
	wrong := 0
	for _, p := range pairs {
		var ids, types []int
		if p.B == nil {
			var err error
			ids, err = tok.Encode(p.A, p.Limit)
			if err != nil {
				t.Fatal(err)
			}
			types = p.Types
		} else {
			var err error
			ids, types, err = tok.EncodePair(p.A, *p.B, p.Limit)
			if err != nil {
				t.Fatal(err)
			}
		}
		if !slices.Equal(ids, p.IDs) || !slices.Equal(types, p.Types) {
			if wrong < 5 {
				t.Errorf("limit %d, %q / %v:\n got %v %v\nwant %v %v", p.Limit, p.A, p.B, ids, types, p.IDs, p.Types)
			}
			wrong++
		}
	}
	t.Logf("%d encodings (single and pair, limits 0/8/16/32/512): %d differ", len(pairs), wrong)
}

func TestWordPiecePairsAndTruncationAgainstHuggingFace(t *testing.T) {
	path, goldPath := os.Getenv("ONNXNLP_WP_TOKENIZER"), os.Getenv("ONNXNLP_WP_GOLD")
	if path == "" || goldPath == "" {
		t.Skip("ONNXNLP_WP_TOKENIZER and ONNXNLP_WP_GOLD not set")
	}
	w, err := LoadWordPiece(path)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(goldPath)
	if err != nil {
		t.Fatal(err)
	}
	var gold struct {
		Cases []goldenRecord `json:"cases"`
		Pairs []goldenPair   `json:"pairs"`
	}
	if err := json.Unmarshal(raw, &gold); err != nil {
		t.Fatal(err)
	}
	wrongNorm, wrong := 0, 0
	for _, c := range gold.Cases {
		if got := w.normalize(c.S); got != c.Norm {
			wrongNorm++
		}
	}
	for _, p := range gold.Pairs {
		var ids, types []int
		if p.B == nil {
			ids = w.Encode(p.A, p.Limit)
			types = p.Types
		} else {
			ids, types = w.EncodePair(p.A, *p.B, p.Limit)
		}
		if !slices.Equal(ids, p.IDs) || !slices.Equal(types, p.Types) {
			if wrong < 4 {
				t.Errorf("limit %d, %.40q / %v:\n got %v %v\nwant %v %v", p.Limit, p.A, p.B, ids, types, p.IDs, p.Types)
			}
			wrong++
		}
	}
	pairsOnly, pairsWrong := 0, 0
	for _, p := range gold.Pairs {
		if p.B == nil {
			continue
		}
		pairsOnly++
		ids, types := w.EncodePair(p.A, *p.B, p.Limit)
		if !slices.Equal(ids, p.IDs) || !slices.Equal(types, p.Types) {
			pairsWrong++
			if pairsWrong <= 3 {
				t.Logf("pair limit %d: got %d ids, want %d; a=%.30q", p.Limit, len(ids), len(p.IDs), p.A)
			}
		}
	}
	t.Logf("%d encodings: %d differ; %d of %d strings normalise differently; text pairs only: %d of %d differ",
		len(gold.Pairs), wrong, wrongNorm, len(gold.Cases), pairsWrong, pairsOnly)
}
