package onnxnlp

import (
	"encoding/json"
	"os"
	"slices"
	"strings"
	"testing"
)

type bpeGolden struct {
	S       string `json:"s"`
	IDs     []int  `json:"ids"`
	Decoded string `json:"decoded"`
}

func TestBPEAgainstHuggingFace(t *testing.T) {
	path, goldPath := os.Getenv("ONNXNLP_BPE_TOKENIZER"), os.Getenv("ONNXNLP_BPE_GOLD")
	if path == "" || goldPath == "" {
		t.Skip("ONNXNLP_BPE_TOKENIZER and ONNXNLP_BPE_GOLD not set")
	}
	tok, err := LoadBPE(path)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(goldPath)
	if err != nil {
		t.Fatal(err)
	}
	var gold struct {
		Cases []bpeGolden `json:"cases"`
	}
	if err := json.Unmarshal(raw, &gold); err != nil {
		t.Fatal(err)
	}
	encodeWrong, decodeWrong, streamWrong := 0, 0, 0
	for _, c := range gold.Cases {
		if got := tok.Encode(c.S); !slices.Equal(got, c.IDs) {
			if encodeWrong < 6 {
				t.Errorf("Encode(%+q)\n got %v\nwant %v", c.S, got, c.IDs)
			}
			encodeWrong++
		}
		if got := tok.Decode(c.IDs); got != c.Decoded {
			if decodeWrong < 4 {
				t.Errorf("Decode(%v) = %+q, want %+q", c.IDs, got, c.Decoded)
			}
			decodeWrong++
		}
		stream := tok.NewDecodeStream()
		var b strings.Builder
		for _, id := range c.IDs {
			b.WriteString(stream.Push(id))
		}
		b.WriteString(stream.Flush())
		if b.String() != c.Decoded {
			if streamWrong < 4 {
				t.Errorf("streamed decode of %v = %+q, want %+q", c.IDs, b.String(), c.Decoded)
			}
			streamWrong++
		}
	}
	t.Logf("%d strings: %d encode differently, %d decode differently, %d decode differently when streamed", len(gold.Cases), encodeWrong, decodeWrong, streamWrong)
}
