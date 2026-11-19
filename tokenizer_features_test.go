package onnxnlp

import (
	"encoding/json"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestUnigramEncodeWrapsTextInTheTemplate(t *testing.T) {
	tok := unigramTokenizer(t, true)
	ids, err := tok.Encode("hello  world", 0)
	if err != nil || !slices.Equal(ids, []int{0, 4, 5, 2}) {
		t.Fatalf("Encode = %v %v", ids, err)
	}
	if ids, _ := tok.Encode("ÿ", 0); !slices.Equal(ids, []int{0, 4, 2}) {
		t.Fatalf("the charsmap rewrites ÿ to hello: %v", ids)
	}
	ids, segments, err := tok.EncodePair("hello", "a", 0)
	if err != nil || !slices.Equal(ids, []int{0, 4, 2, 2, 6, 2}) || !slices.Equal(segments, []int{0, 0, 0, 0, 1, 1}) {
		t.Fatalf("EncodePair = %v %v %v", ids, segments, err)
	}
	ids, _, _ = tok.EncodePair("hello world", "a hello", 6)
	if len(ids) != 6 || ids[0] != 0 || ids[len(ids)-1] != 2 {
		t.Fatalf("a pair truncated to 6 ids keeps its special tokens: %v", ids)
	}
	if ids, _ := tok.Encode("hello world", 3); !slices.Equal(ids, []int{0, 4, 2}) {
		t.Fatalf("truncated to 3: %v", ids)
	}
	if ids, _ := tok.Encode("hello world", 1); !slices.Equal(ids, []int{0, 2}) {
		t.Fatalf("a limit below the special tokens keeps only those: %v", ids)
	}
}

func TestUnigramEncodeNeedsATemplate(t *testing.T) {
	tok := unigramTokenizer(t, false)
	if _, err := tok.Encode("hello", 0); err == nil {
		t.Fatal("Encode without a template must fail")
	}
	if _, _, err := tok.EncodePair("hello", "a", 0); err == nil {
		t.Fatal("EncodePair without a template must fail")
	}
	if got := tok.Tokenize("hello"); !slices.Equal(got, []int{4}) {
		t.Fatalf("Tokenize does not need one: %v", got)
	}
}

func TestSpecialTokensInTextAndTheirWhitespace(t *testing.T) {
	tok := unigramTokenizer(t, true)
	for _, tc := range []struct {
		text string
		want []int
	}{
		{"hello <mask> world", []int{0, 4, 9, 5, 2}},
		{"hello<mask>world", []int{0, 4, 9, 5, 2}},
		{"<mask>", []int{0, 9, 2}},
		{"hello <r> world", []int{0, 4, 7, 8, 5, 2}},
		{"<r>   world", []int{0, 8, 5, 2}},
		{"hello </s> world <s>", []int{0, 4, 7, 2, 5, 7, 0, 2}},
		{"hello <r>", []int{0, 4, 7, 8, 2}},
		{"<r> <mask>", []int{0, 8, 9, 2}},
	} {
		got, _ := tok.Encode(tc.text, 0)
		if !slices.Equal(got, tc.want) {
			t.Errorf("Encode(%q) = %v, want %v", tc.text, got, tc.want)
		}
	}
}

func TestEncodeWithoutSpecialTokensInTheTokenizer(t *testing.T) {
	spec := unigramSpec(true, xlmrNormalizer())
	spec["added_tokens"] = []any{}
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	writeTokenizerJSON(t, path, spec)
	tok, err := LoadTokenizer(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	if ids, _ := tok.Encode("hello", 0); !slices.Equal(ids, []int{0, 4, 2}) {
		t.Fatalf("Encode = %v", ids)
	}
}

func TestNormalizerStepsFromTokenizerJSON(t *testing.T) {
	load := func(normalizer any) (*Tokenizer, error) {
		path := filepath.Join(t.TempDir(), "tokenizer.json")
		writeTokenizerJSON(t, path, unigramSpec(true, normalizer))
		return LoadTokenizer(path, nil)
	}
	step := func(kind string, extra map[string]any) map[string]any {
		extra["type"] = kind
		return extra
	}
	for _, tc := range []struct {
		name       string
		normalizer any
		in, want   string
	}{
		{"nfd", step("NFD", nil2()), "é", "e\u0301"},
		{"nfkc", step("NFKC", nil2()), "ﬁ", "fi"},
		{"nfkd", step("NFKD", nil2()), "ﬁé", "fie\u0301"},
		{"nfc", step("NFC", nil2()), "e\u0301", "é"},
		{"lowercase", step("Lowercase", nil2()), "HeLLo", "hello"},
		{"strip both", step("Strip", map[string]any{"strip_left": true, "strip_right": true}), "  hi \t", "hi"},
		{"strip left", step("Strip", map[string]any{"strip_left": true, "strip_right": false}), "  hi  ", "hi  "},
		{"strip right", step("Strip", map[string]any{"strip_left": false, "strip_right": true}), "  hi  ", "  hi"},
		{"replace string", step("Replace", map[string]any{"pattern": map[string]string{"String": "ab"}, "content": "X"}), "aabb", "aXb"},
		{"replace regex with whitespace class", step("Replace", map[string]any{"pattern": map[string]string{"Regex": `\s{2,}|[\n\r\t]`}, "content": " "}), "a \u00a0\tb\nc", "a b c"},
		{"sequence", step("Sequence", map[string]any{"normalizers": []any{step("Lowercase", nil2()), step("Strip", map[string]any{"strip_left": true})}}), "  AB", "ab"},
	} {
		tok, err := load(tc.normalizer)
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		if got := tok.normalized(tc.in); got != tc.want {
			t.Errorf("%s: normalized(%+q) = %+q, want %+q", tc.name, tc.in, got, tc.want)
		}
	}
	for name, normalizer := range map[string]any{
		"unknown type":      step("Prepend", nil2()),
		"bad regex":         step("Replace", map[string]any{"pattern": map[string]string{"Regex": "("}, "content": ""}),
		"bad charsmap":      step("Precompiled", map[string]any{"precompiled_charsmap": []byte{1, 2}}),
		"unknown in a list": step("Sequence", map[string]any{"normalizers": []any{step("Prepend", nil2())}}),
	} {
		if _, err := load(normalizer); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	tok, err := load(nil)
	if err != nil || tok.normalized("  A  ") != "  A  " {
		t.Fatalf("a tokenizer without a normalizer leaves text alone: %v", err)
	}
}

func nil2() map[string]any { return map[string]any{} }

func TestLoadTokenizerRejectsBrokenSpecs(t *testing.T) {
	dir := t.TempDir()
	try := func(name string, mutate func(spec map[string]any)) {
		spec := unigramSpec(true, xlmrNormalizer())
		mutate(spec)
		path := filepath.Join(dir, name+".json")
		writeTokenizerJSON(t, path, spec)
		if _, err := LoadTokenizer(path, nil); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	try("pre-tokenizer", func(s map[string]any) { s["pre_tokenizer"] = map[string]any{"type": "Whitespace"} })
	try("post-processor", func(s map[string]any) { s["post_processor"] = map[string]any{"type": "BertProcessing"} })
	try("template token", func(s map[string]any) {
		s["post_processor"] = map[string]any{"type": "TemplateProcessing", "single": []any{map[string]any{"SpecialToken": map[string]any{"id": "<nope>"}}}}
	})
	try("template item", func(s map[string]any) {
		s["post_processor"] = map[string]any{"type": "TemplateProcessing", "single": []any{map[string]any{"Other": map[string]any{}}}}
	})
	try("template pair", func(s map[string]any) {
		s["post_processor"] = map[string]any{"type": "TemplateProcessing", "single": []any{}, "pair": []any{map[string]any{"Other": map[string]any{}}}}
	})
	if _, err := LoadTokenizer(filepath.Join(dir, "missing.json"), nil); err == nil {
		t.Error("a missing file must be reported")
	}
	badPost := filepath.Join(dir, "badpost.json")
	spec := unigramSpec(true, xlmrNormalizer())
	spec["post_processor"] = []int{1}
	writeTokenizerJSON(t, badPost, spec)
	if _, err := LoadTokenizer(badPost, nil); err == nil {
		t.Error("a post-processor of the wrong JSON type must be reported")
	}
}

func TestPrependsSpace(t *testing.T) {
	for _, tc := range []struct {
		json string
		want bool
		fail bool
	}{
		{``, false, false},
		{`null`, false, false},
		{`{"type":"Metaspace","add_prefix_space":true}`, true, false},
		{`{"type":"Metaspace","add_prefix_space":false}`, false, false},
		{`{"type":"Metaspace"}`, true, false},
		{`{"type":"Metaspace","prepend_scheme":"always"}`, true, false},
		{`{"type":"Metaspace","prepend_scheme":"first"}`, true, false},
		{`{"type":"Metaspace","prepend_scheme":"never"}`, false, false},
		{`{"type":"Sequence","pretokenizers":[{"type":"Metaspace","prepend_scheme":"never"},{"type":"Metaspace"}]}`, true, false},
		{`{"type":"Sequence","pretokenizers":[]}`, false, false},
		{`{"type":"Sequence","pretokenizers":[{"type":"Split"}]}`, false, true},
		{`{"type":"ByteLevel"}`, false, true},
		{`{`, false, true},
	} {
		got, err := prependsSpace(json.RawMessage(tc.json))
		if (err != nil) != tc.fail || got != tc.want {
			t.Errorf("prependsSpace(%s) = %v, %v", tc.json, got, err)
		}
	}
}

func TestGoPatternTranslatesRustWhitespace(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{`\s{2,}`, `[` + rustWhitespace + `]{2,}`},
		{`[\s,]`, `[` + rustWhitespace + `,]`},
		{`a\.b`, `a\.b`},
		{`\\s`, `\\s`},
		{`x\`, `x\`},
	} {
		if got := goPattern(tc.in); got != tc.want {
			t.Errorf("goPattern(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestTruncationFollowsHuggingFace(t *testing.T) {
	seq := func(n int) []int { return make([]int, n) }
	for _, tc := range []struct{ la, lb, budget, wantA, wantB int }{
		{10, 10, 20, 10, 10},
		{43, 70, 5, 2, 3},
		{70, 43, 5, 3, 2},
		{9, 9, 5, 2, 3},
		{3, 50, 10, 3, 7},
		{50, 3, 10, 7, 3},
		{0, 8, 4, 0, 4},
		{8, 0, 4, 4, 0},
		{6, 6, 0, 0, 0},
		{4, 4, 8, 4, 4},
	} {
		a, b := truncateLongestFirst(seq(tc.la), seq(tc.lb), tc.budget)
		if len(a) != tc.wantA || len(b) != tc.wantB {
			t.Errorf("truncate(%d, %d, budget %d) = %d, %d; want %d, %d", tc.la, tc.lb, tc.budget, len(a), len(b), tc.wantA, tc.wantB)
		}
	}
}

func TestWordPiecePairs(t *testing.T) {
	w, err := LoadWordPiece(filepath.Join(minilmDir(t, 0, true), "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	ids, segments := w.EncodePair("hello world", "machine learning", 0)
	want := []int{w.cls, 4, 5, w.sep, 6, 7, w.sep}
	if !slices.Equal(ids, want) || !slices.Equal(segments, []int{0, 0, 0, 0, 1, 1, 1}) {
		t.Fatalf("EncodePair = %v %v, want %v", ids, segments, want)
	}
	ids, segments = w.EncodePair("hello world", "machine learning models", 6)
	if len(ids) != 6 || len(segments) != 6 || ids[0] != w.cls || ids[5] != w.sep || segments[5] != 1 {
		t.Fatalf("a truncated pair keeps both separators: %v %v", ids, segments)
	}
}

func TestEncoderInputs(t *testing.T) {
	ids := [][]int{{1, 2, 3}, {4}}
	segments := [][]int{{0, 0, 1}, {0}}
	inputs, mask, width, err := encoderInputs([]string{"input_ids", "attention_mask", "token_type_ids"}, ids, segments, []int{1, 0})
	if err != nil || width != 3 || len(inputs) != 3 {
		t.Fatalf("%v %d %v", inputs, width, err)
	}
	if !slices.Equal(inputs["input_ids"].I, []int64{4, 0, 0, 1, 2, 3}) || !slices.Equal(mask.I, []int64{1, 0, 0, 1, 1, 1}) ||
		!slices.Equal(inputs["token_type_ids"].I, []int64{0, 0, 0, 0, 0, 1}) {
		t.Fatalf("padded inputs: %v %v %v", inputs["input_ids"].I, mask.I, inputs["token_type_ids"].I)
	}
	inputs, _, _, _ = encoderInputs([]string{"input_ids", "token_type_ids"}, ids, nil, []int{0})
	if _, has := inputs["attention_mask"]; has || !slices.Equal(inputs["token_type_ids"].I, []int64{0, 0, 0}) {
		t.Fatalf("only declared inputs are built, token types default to zero: %v", inputs)
	}
	if _, _, _, err := encoderInputs([]string{"input_ids", "position_ids"}, ids, nil, []int{0}); err == nil || !strings.Contains(err.Error(), "position_ids") {
		t.Fatalf("an unsupported input must be named: %v", err)
	}
}

func TestLoadSentenceEncoderErrors(t *testing.T) {
	dir := t.TempDir()
	for name, content := range map[string]string{"broken.json": `{`, "bpe.json": `{"model":{"type":"BPE"}}`, "wordpiece.json": `{"model":{"type":"WordPiece"}}`} {
		path := filepath.Join(dir, name)
		if err := writeFile(path, content); err != nil {
			t.Fatal(err)
		}
		if _, err := loadSentenceEncoder(path); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	if _, err := loadSentenceEncoder(filepath.Join(dir, "missing.json")); err == nil {
		t.Error("a missing file must be reported")
	}
	badUnigram := filepath.Join(dir, "badunigram.json")
	writeTokenizerJSON(t, badUnigram, unigramSpec(true, map[string]any{"type": "Prepend"}))
	if _, err := loadSentenceEncoder(badUnigram); err == nil {
		t.Error("a Unigram tokenizer that cannot be loaded must be reported")
	}
	noTemplate := filepath.Join(dir, "unigram.json")
	writeTokenizerJSON(t, noTemplate, unigramSpec(false, xlmrNormalizer()))
	if _, err := loadSentenceEncoder(noTemplate); err == nil {
		t.Error("a Unigram tokenizer without a template cannot encode sentences")
	}
	withTemplate := filepath.Join(dir, "template.json")
	writeTokenizerJSON(t, withTemplate, unigramSpec(true, xlmrNormalizer()))
	enc, err := loadSentenceEncoder(withTemplate)
	if err != nil {
		t.Fatal(err)
	}
	if ids := enc.single("hello", 0); !slices.Equal(ids, []int{0, 4, 2}) {
		t.Fatalf("single = %v", ids)
	}
	if ids, segments := enc.pair("hello", "a", 0); len(ids) != 6 || len(segments) != 6 {
		t.Fatalf("pair = %v %v", ids, segments)
	}
}
