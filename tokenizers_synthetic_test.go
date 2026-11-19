package onnxgo

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"onnxgo/internal/fixture"
)

func glinerDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	fixture.Must(fixture.WriteGLiNER(dir))
	return dir
}

func minilmDir(t *testing.T, maxSeq int, normalize bool) string {
	t.Helper()
	dir := t.TempDir()
	fixture.Must(fixture.WriteMiniLM(dir, maxSeq, normalize))
	return dir
}

func TestUnigramTokenize(t *testing.T) {
	tok, err := LoadTokenizer(filepath.Join(glinerDir(t), "tokenizer.json"), map[string]int{"[E]": 24})
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		in   string
		want []int
	}{
		{"alice", []int{8}},
		{"alice bob", []int{8, 10}},
		{"alice   bob\t.", []int{8, 10, 11}},
		{"[E]", []int{24}},
		{"[PAD]", []int{0}},
		{"alice ", []int{8}},
		{"", nil},
		{"   ", nil},
		{"zzz", []int{12, 3}},
		{"alicezz", []int{8, 3}},
		{"(met)", []int{4, 3}},
	}
	for _, c := range cases {
		if got := tok.Tokenize(c.in); !slices.Equal(got, c.want) {
			t.Errorf("%q: got %v, want %v", c.in, got, c.want)
		}
	}
}

func TestUnigramNormalizesUnicode(t *testing.T) {
	tok, err := LoadTokenizer(filepath.Join(glinerDir(t), "tokenizer.json"), nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := tok.Tokenize("é"), tok.Tokenize("é"); !slices.Equal(got, want) {
		t.Fatalf("NFC not applied: %v vs %v", got, want)
	}
	if got := tok.Tokenize("　 "); got != nil {
		t.Fatalf("unicode spaces should vanish: %v", got)
	}
}

func TestSplitMergedWithNext(t *testing.T) {
	got := splitMergedWithNext("▁a▁b▁")
	if !slices.Equal(got, []string{"▁a", "▁b", "▁"}) {
		t.Fatalf("got %q", got)
	}
}

func TestLoadTokenizerErrors(t *testing.T) {
	dir := t.TempDir()
	if _, err := LoadTokenizer(filepath.Join(dir, "none.json"), nil); err == nil {
		t.Fatal("missing file must fail")
	}
	write := func(s string) string {
		p := filepath.Join(dir, "t.json")
		os.WriteFile(p, []byte(s), 0o644)
		return p
	}
	if _, err := LoadTokenizer(write("{"), nil); err == nil {
		t.Fatal("bad JSON must fail")
	}
	if _, err := LoadTokenizer(write(`{"model":{"type":"BPE"}}`), nil); err == nil {
		t.Fatal("non-Unigram must fail")
	}
	if _, err := LoadTokenizer(write(`{"model":{"type":"Unigram","vocab":[["a"]]}}`), nil); err == nil {
		t.Fatal("malformed vocab entry must fail")
	}
	if _, err := LoadTokenizer(write(`{"model":{"type":"Unigram","vocab":[[1,2]]}}`), nil); err == nil {
		t.Fatal("non-string piece must fail")
	}
	if _, err := LoadTokenizer(write(`{"model":{"type":"Unigram","vocab":[["a","b"]]}}`), nil); err == nil {
		t.Fatal("non-numeric score must fail")
	}
	var e vocabEntry
	if err := json.Unmarshal([]byte(`"x"`), &e); err == nil {
		t.Fatal("vocab entry must be an array")
	}
}

func TestWordPieceEncode(t *testing.T) {
	wp, err := LoadWordPiece(filepath.Join(minilmDir(t, 0, true), "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		in   string
		max  int
		want []int
	}{
		{"Hello, world", 0, []int{2, 4, 10, 5, 3}},
		{"machine learning models", 0, []int{2, 6, 7, 8, 9, 3}},
		{"hello world!", 0, []int{2, 4, 5, 1, 3}},
		{"machinery", 0, []int{2, 1, 3}},
		{"Café", 0, []int{2, 12, 3}},
		{"hel\x00lo​", 0, []int{2, 4, 3}},
		{"hello\tworld\n", 0, []int{2, 4, 5, 3}},
		{"東京", 0, []int{2, 1, 1, 3}},
		{"hello«world", 0, []int{2, 4, 1, 5, 3}},
		{"", 0, []int{2, 3}},
		{"hello world hello world", 4, []int{2, 4, 5, 3}},
		{"hello world", 1, []int{2, 3}},
		{string(make([]rune, 0)) + repeatRune('a', 101), 0, []int{2, 1, 3}},
	}
	for _, c := range cases {
		if got := wp.Encode(c.in, c.max); !slices.Equal(got, c.want) {
			t.Errorf("%.20q (max %d): got %v, want %v", c.in, c.max, got, c.want)
		}
	}
}

func repeatRune(r rune, n int) string {
	out := make([]rune, n)
	for i := range out {
		out[i] = r
	}
	return string(out)
}

func TestWordPieceKeepsAccentsWhenConfigured(t *testing.T) {
	dir := minilmDir(t, 0, true)
	raw, _ := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	var spec map[string]any
	json.Unmarshal(raw, &spec)
	spec["normalizer"].(map[string]any)["strip_accents"] = false
	spec["normalizer"].(map[string]any)["clean_text"] = false
	spec["normalizer"].(map[string]any)["handle_chinese_chars"] = false
	raw, _ = json.Marshal(spec)
	os.WriteFile(filepath.Join(dir, "tokenizer.json"), raw, 0o644)
	wp, err := LoadWordPiece(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	if got := wp.Encode("Café", 0); !slices.Equal(got, []int{2, 1, 3}) {
		t.Fatalf("accents must be kept: %v", got)
	}
	if got := wp.Encode("東", 0); !slices.Equal(got, []int{2, 1, 3}) {
		t.Fatalf("got %v", got)
	}
}

func TestLoadWordPieceErrors(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "t.json")
	if _, err := LoadWordPiece(p); err == nil {
		t.Fatal("missing file must fail")
	}
	for name, body := range map[string]string{
		"bad json":    "{",
		"wrong model": `{"normalizer":{"type":"BertNormalizer"},"model":{"type":"BPE"}}`,
		"wrong norm":  `{"normalizer":{"type":"Lowercase"},"model":{"type":"WordPiece"}}`,
		"no cls":      `{"normalizer":{"type":"BertNormalizer"},"model":{"type":"WordPiece","unk_token":"[UNK]","vocab":{"[SEP]":1,"[UNK]":0}}}`,
		"no sep":      `{"normalizer":{"type":"BertNormalizer"},"model":{"type":"WordPiece","unk_token":"[UNK]","vocab":{"[CLS]":1,"[UNK]":0}}}`,
		"no unk":      `{"normalizer":{"type":"BertNormalizer"},"model":{"type":"WordPiece","unk_token":"[UNK]","vocab":{"[CLS]":1,"[SEP]":2}}}`,
	} {
		os.WriteFile(p, []byte(body), 0o644)
		if _, err := LoadWordPiece(p); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}
