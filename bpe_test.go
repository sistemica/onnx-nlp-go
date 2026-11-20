package onnxnlp

import (
	"encoding/json"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// bpeSpecFor returns a tokenizer.json for a vocabulary of single characters, "ab" and "abc", with
// the given pre-tokenizer.
func bpeSpecFor(pre any) map[string]any {
	vocab := map[string]int{}
	for i, tok := range []string{"a", "b", "c", "ab", "abc", "Ġ", "Ċ", "1", "2", "Ã", "©", "x", "Ġx", "'", "s", "Ġa", "'s"} {
		vocab[tok] = i
	}
	return map[string]any{
		"added_tokens": []any{
			map[string]any{"id": 100, "content": "<|end|>", "special": true},
			map[string]any{"id": 101, "content": "<pad>", "special": true, "lstrip": true},
			map[string]any{"id": 102, "content": "<r>", "special": false, "rstrip": true},
		},
		"normalizer":    nil,
		"pre_tokenizer": pre,
		"decoder":       map[string]any{"type": "ByteLevel"},
		"model":         map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{"a b", "ab c", "Ġ x", "' s", "Ġ a"}},
	}
}

func byteLevel(regex bool) map[string]any {
	return map[string]any{"type": "ByteLevel", "add_prefix_space": false, "use_regex": regex}
}

func loadBPEFrom(t *testing.T, spec map[string]any) (*BPE, error) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	writeTokenizerJSON(t, path, spec)
	return LoadBPE(path)
}

func mustBPE(t *testing.T, spec map[string]any) *BPE {
	t.Helper()
	b, err := loadBPEFrom(t, spec)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestBPEMergesAndByteMapping(t *testing.T) {
	b := mustBPE(t, bpeSpecFor(byteLevel(true)))
	for _, tc := range []struct {
		text string
		want []int
	}{
		{"abc", []int{4}},
		{"ab", []int{3}},
		{"ba", []int{1, 0}},
		{"abcab", []int{4, 3}},
		{"aab", []int{0, 3}},
		{"a b", []int{0, 5, 1}},
		{"x x", []int{11, 12}},
		{"zzz", nil},
		{"é", []int{9, 10}},
		{"", nil},
		{"it's", []int{16}},
	} {
		if got := b.Encode(tc.text); !slices.Equal(got, tc.want) {
			t.Errorf("Encode(%q) = %v, want %v", tc.text, got, tc.want)
		}
	}
	if got := b.Encode("s's"); !slices.Equal(got, []int{14, 16}) {
		t.Errorf("a contraction becomes one piece: %v", got)
	}
	if again := b.Encode("abc"); !slices.Equal(again, []int{4}) {
		t.Errorf("a cached word encodes the same: %v", again)
	}
	if got := b.Encode("a\xffb"); len(got) == 0 {
		t.Errorf("invalid UTF-8 is replaced, not dropped: %v", got)
	}
}

func TestBPEAddedTokensAndWhitespaceStripping(t *testing.T) {
	b := mustBPE(t, bpeSpecFor(byteLevel(true)))
	for _, tc := range []struct {
		text string
		want []int
	}{
		{"ab<|end|>c", []int{3, 100, 2}},
		{"<|end|>", []int{100}},
		{"a <pad> b", []int{0, 101, 5, 1}},
		{"a <r> b", []int{0, 5, 102, 1}},
		{"<r> <pad>", []int{102, 101}},
		{"a<r>  ", []int{0, 102}},
	} {
		if got := b.Encode(tc.text); !slices.Equal(got, tc.want) {
			t.Errorf("Encode(%q) = %v, want %v", tc.text, got, tc.want)
		}
	}
	if id, ok := b.TokenID("<|end|>"); !ok || id != 100 {
		t.Fatalf("TokenID = %d %v", id, ok)
	}
	if _, ok := b.TokenID("<missing>"); ok {
		t.Fatal("unknown token")
	}
	if !b.IsSpecial(100) || b.IsSpecial(102) || b.IsSpecial(3) {
		t.Fatal("IsSpecial must follow the special flag of added tokens")
	}
}

func TestBPEPreTokenizers(t *testing.T) {
	split := func(pattern string) map[string]any {
		return map[string]any{"type": "Split", "pattern": map[string]any{"Regex": pattern}, "behavior": "Isolated", "invert": false}
	}
	qwen := `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	sequence := func(steps ...any) map[string]any { return map[string]any{"type": "Sequence", "pretokenizers": steps} }
	pieces := func(pre any, text string) []string {
		b := mustBPE(t, bpeSpecFor(pre))
		out := []string{text}
		for _, split := range b.splitters[:len(b.splitters)-1] {
			out = split(out)
		}
		return out
	}
	for _, tc := range []struct {
		name string
		pre  any
		text string
		want []string
	}{
		{"qwen digits", sequence(split(qwen), byteLevel(false)), "ab12", []string{"ab", "1", "2"}},
		{"qwen spaces before a word", sequence(split(qwen), byteLevel(false)), "a  b", []string{"a", " ", " b"}},
		{"qwen newlines", sequence(split(qwen), byteLevel(false)), "a\n\nb", []string{"a", "\n\n", "b"}},
		{"qwen trailing spaces", sequence(split(qwen), byteLevel(false)), "a  ", []string{"a", "  "}},
		{"qwen contraction", sequence(split(qwen), byteLevel(false)), "It'S", []string{"It", "'S"}},
		{"gpt2 newlines", byteLevel(true), "a\n\nb", []string{"a", "\n", "\n", "b"}},
		{"gpt2 spaces", byteLevel(true), "a  b", []string{"a", " ", " b"}},
		{"gpt2 numbers", byteLevel(true), "x 12 y", []string{"x", " 12", " y"}},
		{"digits first", sequence(map[string]any{"type": "Digits", "individual_digits": true}, byteLevel(true)), "a12b", []string{"a", "1", "2", "b"}},
		{"gap between matches", sequence(split(`\p{L}+`), byteLevel(false)), "ab 12 c", []string{"ab", " 12 ", "c"}},
		{"no match at all", sequence(split(`\p{L}+`), byteLevel(false)), "12", []string{"12"}},
		{"empty match", sequence(split(`x*`), byteLevel(false)), "ab", []string{"a", "b"}},
		{"prefix space", map[string]any{"type": "ByteLevel", "add_prefix_space": true, "use_regex": true}, "ab", []string{" ab"}},
	} {
		if tc.name == "prefix space" {
			b := mustBPE(t, bpeSpecFor(tc.pre))
			if got := b.Encode("x"); !slices.Equal(got, []int{12}) {
				t.Errorf("%s: Encode(x) = %v, want the token for a space and x", tc.name, got)
			}
			continue
		}
		got := pieces(tc.pre, tc.text)
		if !slices.Equal(got, tc.want) {
			t.Errorf("%s: %q split into %q, want %q", tc.name, tc.text, got, tc.want)
		}
	}
}

func TestBPEIgnoreMergesAndNormalizer(t *testing.T) {
	spec := bpeSpecFor(byteLevel(false))
	spec["model"].(map[string]any)["ignore_merges"] = true
	spec["model"].(map[string]any)["merges"] = []string{}
	b := mustBPE(t, spec)
	if got := b.Encode("ab"); !slices.Equal(got, []int{3}) {
		t.Errorf("a word in the vocabulary is taken whole: %v", got)
	}
	if got := b.Encode("ba"); !slices.Equal(got, []int{1, 0}) {
		t.Errorf("a word outside the vocabulary is split into characters: %v", got)
	}
	spec = bpeSpecFor(byteLevel(false))
	spec["normalizer"] = map[string]any{"type": "Lowercase"}
	if got := mustBPE(t, spec).Encode("ABC"); !slices.Equal(got, []int{4}) {
		t.Errorf("the normalizer runs before splitting: %v", got)
	}
}

func TestBPEDecodeKeepsCharactersOutsideTheByteAlphabet(t *testing.T) {
	b := mustBPE(t, bpeSpecFor(byteLevel(true)))
	id := byteToken(t, b, 0)
	b.tokens = append(b.tokens, "€")
	if got := b.Decode([]int{len(b.tokens) - 1, id}); got != "€"+b.Decode([]int{id}) {
		t.Fatalf("Decode = %q", got)
	}
}

func TestBPEPairListMerges(t *testing.T) {
	spec := bpeSpecFor(byteLevel(false))
	spec["model"].(map[string]any)["merges"] = [][2]string{{"a", "b"}, {"ab", "c"}}
	if got := mustBPE(t, spec).Encode("abc"); !slices.Equal(got, []int{4}) {
		t.Errorf("merges written as pairs: %v", got)
	}
}

func TestBPEDecodeAndStream(t *testing.T) {
	b := mustBPE(t, bpeSpecFor(byteLevel(true)))
	if got := b.Decode([]int{4, 5, 11, 100}); got != "abc x<|end|>" {
		t.Errorf("Decode = %q", got)
	}
	if got := b.Decode([]int{9, 10}); got != "é" {
		t.Errorf("two byte tokens form one character: %q", got)
	}
	if got := b.Decode([]int{9}); got != "�" {
		t.Errorf("an incomplete character is replaced: %q", got)
	}
	if got := b.Decode([]int{-1, 9999, 0}); got != "a" {
		t.Errorf("ids outside the vocabulary are ignored: %q", got)
	}
	s := b.NewDecodeStream()
	if got := s.Push(9); got != "" {
		t.Errorf("half a character is held back: %q", got)
	}
	if got := s.Push(10); got != "é" {
		t.Errorf("the second half completes it: %q", got)
	}
	if got := s.Push(0); got != "a" {
		t.Errorf("plain text is passed on at once: %q", got)
	}
	s.Push(9)
	if got := s.Flush(); got != "�" {
		t.Errorf("Flush gives up on an incomplete character: %q", got)
	}
	if got := s.Flush(); got != "" {
		t.Errorf("a second Flush has nothing left: %q", got)
	}
	// A character of three and of four bytes, one byte per token.
	for _, text := range []string{"€", "😀"} {
		stream := b.NewDecodeStream()
		var out strings.Builder
		var ids []int
		for i := 0; i < len(text); i++ {
			id := byteToken(t, b, text[i])
			ids = append(ids, id)
			out.WriteString(stream.Push(id))
		}
		out.WriteString(stream.Flush())
		if out.String() != text || b.Decode(ids) != text {
			t.Errorf("%q streamed as %q, decoded as %q", text, out.String(), b.Decode(ids))
		}
	}
}

// byteToken returns the id of a token that stands for the single byte, adding it when missing.
func byteToken(t *testing.T, b *BPE, by byte) int {
	t.Helper()
	token := string(byteChars[by])
	if id, ok := b.ids[token]; ok {
		return id
	}
	id := len(b.tokens)
	b.ids[token] = id
	b.tokens = append(b.tokens, token)
	return id
}

func TestLoadBPEErrors(t *testing.T) {
	good := func() map[string]any { return bpeSpecFor(byteLevel(true)) }
	for name, mutate := range map[string]func(map[string]any){
		"wrong model":       func(s map[string]any) { s["model"].(map[string]any)["type"] = "Unigram" },
		"wrong decoder":     func(s map[string]any) { s["decoder"] = map[string]any{"type": "Metaspace"} },
		"no decoder":        func(s map[string]any) { delete(s, "decoder") },
		"bad merges":        func(s map[string]any) { s["model"].(map[string]any)["merges"] = 3 },
		"bad normalizer":    func(s map[string]any) { s["normalizer"] = map[string]any{"type": "Prepend"} },
		"bad pre-tokenizer": func(s map[string]any) { s["pre_tokenizer"] = map[string]any{"type": "Whitespace"} },
		"non-isolated split": func(s map[string]any) {
			s["pre_tokenizer"] = map[string]any{"type": "Split", "pattern": map[string]any{"Regex": "a"}, "behavior": "Removed"}
		},
		"split without regex": func(s map[string]any) {
			s["pre_tokenizer"] = map[string]any{"type": "Split", "pattern": map[string]any{"String": "a"}, "behavior": "Isolated"}
		},
		"bad regex": func(s map[string]any) {
			s["pre_tokenizer"] = map[string]any{"type": "Split", "pattern": map[string]any{"Regex": "("}, "behavior": "Isolated"}
		},
		"other lookahead": func(s map[string]any) {
			s["pre_tokenizer"] = map[string]any{"type": "Split", "pattern": map[string]any{"Regex": `a(?=b)`}, "behavior": "Isolated"}
		},
		"digits grouped": func(s map[string]any) {
			s["pre_tokenizer"] = map[string]any{"type": "Digits", "individual_digits": false}
		},
		"broken child": func(s map[string]any) {
			s["pre_tokenizer"] = map[string]any{"type": "Sequence", "pretokenizers": []any{map[string]any{"type": "Whitespace"}}}
		},
	} {
		spec := good()
		mutate(spec)
		if _, err := loadBPEFrom(t, spec); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	if _, err := LoadBPE(filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Error("a missing file must be reported")
	}
	broken := filepath.Join(t.TempDir(), "broken.json")
	if err := writeFile(broken, "{"); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadBPE(broken); err == nil {
		t.Error("invalid JSON must be reported")
	}
	badPre := good()
	badPre["pre_tokenizer"] = json.RawMessage(`{"type": 5}`)
	if _, err := loadBPEFrom(t, badPre); err == nil {
		t.Error("a pre-tokenizer of the wrong JSON shape must be reported")
	}
	none := good()
	none["pre_tokenizer"] = nil
	none["added_tokens"] = []any{}
	b, err := loadBPEFrom(t, none)
	if err != nil {
		t.Fatal(err)
	}
	if got := b.Encode("abc"); !slices.Equal(got, []int{4}) {
		t.Errorf("without a pre-tokenizer the whole text is one piece: %v", got)
	}
}
