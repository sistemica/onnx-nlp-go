package onnxgo

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"onnxgo/internal/fixture"
)

func TestGLiNERExtractWithSyntheticModel(t *testing.T) {
	g, err := LoadGLiNER(glinerDir(t))
	if err != nil {
		t.Fatal(err)
	}
	got, err := g.Extract(fixture.GLiNERText, []string{fixture.GLiNERLabel}, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Label != "person" || got[0].Text != "alice" || got[0].Start != 0 || got[0].End != 5 {
		t.Fatalf("entities %+v", got)
	}
	if want := float32(1 / (1 + math.Exp(-4))); math.Abs(float64(got[0].Score-want)) > 1e-6 {
		t.Fatalf("score %v, want %v", got[0].Score, want)
	}
	if none, err := g.Extract(fixture.GLiNERText, []string{fixture.GLiNERLabel}, 0.99); err != nil || len(none) != 0 {
		t.Fatalf("threshold 0.99: %+v %v", none, err)
	}
	if none, err := g.Extract("anything", nil, 0.5); err != nil || none != nil {
		t.Fatalf("no labels: %+v %v", none, err)
	}
}

func TestGLiNERExtractReportsModelErrors(t *testing.T) {
	g, err := LoadGLiNER(glinerDir(t))
	if err != nil {
		t.Fatal(err)
	}
	needsMore, _ := NewSession(&Graph{Inputs: []string{"input_ids", "attention_mask", "extra"}})
	good := g.encoder
	g.encoder = needsMore
	if _, err := g.Extract("x", []string{"a"}, 0.5); err == nil {
		t.Fatal("encoder failure must surface")
	}
	g.encoder, g.boundary = good, needsMore
	if _, err := g.Extract(fixture.GLiNERText, []string{fixture.GLiNERLabel}, 0.5); err == nil {
		t.Fatal("boundary failure must surface")
	}
}

func TestLoadGLiNERErrors(t *testing.T) {
	if _, err := LoadGLiNER(t.TempDir()); err == nil {
		t.Fatal("empty directory must fail")
	}
	mutate := func(name string, edit func(dir string)) {
		dir := glinerDir(t)
		edit(dir)
		if _, err := LoadGLiNER(dir); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	mutate("bad config json", func(d string) { os.WriteFile(filepath.Join(d, "gliner2_config.json"), []byte("{"), 0o644) })
	mutate("wrong architecture", func(d string) {
		os.WriteFile(filepath.Join(d, "gliner2_config.json"), []byte(`{"architecture":"span"}`), 0o644)
	})
	mutate("no tokenizer", func(d string) { os.Remove(filepath.Join(d, "tokenizer.json")) })
	mutate("no encoder", func(d string) { os.Remove(filepath.Join(d, "onnx", "encoder.onnx")) })
	mutate("no boundary", func(d string) { os.Remove(filepath.Join(d, "onnx", "boundary.onnx")) })
}

func TestSplitWords(t *testing.T) {
	text := "Visit https://x.org/a?b=1, mail Jo.Doe@mail.com or @bob; e-mail foo_bar 3.5%"
	var got []string
	for _, w := range splitWords(text) {
		got = append(got, w.text)
	}
	want := []string{"visit", "https://x.org/a?b=1,", "mail", "jo.doe@mail.com", "or", "@bob", ";", "e-mail", "foo_bar", "3", ".", "5", "%"}
	if !slices.Equal(got, want) {
		t.Fatalf("got %q\nwant %q", got, want)
	}
	for _, w := range splitWords("Ünï 東京タワー x́") {
		if text := "Ünï 東京タワー x́"; text[w.start:w.end] == "" {
			t.Fatalf("empty span for %q", w.text)
		}
	}
	if w := splitWords("Straße"); len(w) != 1 || w[0].text != "straße" || w[0].start != 0 || w[0].end != len("Straße") {
		t.Fatalf("got %+v", w)
	}
}

func TestEndWithPunctuation(t *testing.T) {
	for in, want := range map[string]string{"": ".", "a": "a.", "a.": "a.", "a!": "a!", "a?": "a?", "a,": "a,."} {
		if got := endWithPunctuation(in); got != want {
			t.Errorf("%q: got %q, want %q", in, got, want)
		}
	}
}

func TestBuildPrompt(t *testing.T) {
	g, err := LoadGLiNER(glinerDir(t))
	if err != nil {
		t.Fatal(err)
	}
	words := splitWords("alice met bob.")
	p := g.buildPrompt([]string{"person"}, words)
	want := []int64{4, 22, 6, 4, 24, 7, 5, 5, 21, 8, 9, 10, 11}
	if !slices.Equal(p.ids, want) {
		t.Fatalf("ids %v, want %v", p.ids, want)
	}
	if !slices.Equal(p.labelPos, []int{4}) || !slices.Equal(p.wordFirsts, []int{9, 10, 11, 12}) {
		t.Fatalf("positions: labels %v words %v", p.labelPos, p.wordFirsts)
	}
}

func TestBuildPromptKeepsPlaceholderForEmptyWord(t *testing.T) {
	g, err := LoadGLiNER(glinerDir(t))
	if err != nil {
		t.Fatal(err)
	}
	p := g.buildPrompt(nil, []word{{text: " "}})
	if len(p.wordFirsts) != 1 || p.wordFirsts[0] != len(p.ids)-1 {
		t.Fatalf("placeholder position %v for %d ids", p.wordFirsts, len(p.ids))
	}
}

func outputsFor(logits []float32, indices []int64, valid []bool, null float32) map[string]*Tensor {
	n := len(logits)
	return map[string]*Tensor{
		"pair_logits":       fT([]int{1, 1, n}, logits...),
		"candidate_indices": iT([]int{1, 1, n, 2}, indices...),
		"candidate_valid":   bT([]int{1, 1, n}, valid...),
		"null_logits":       fT([]int{1, 1}, null),
	}
}

func TestDecodeEntities(t *testing.T) {
	text := "aa bb cc dd."
	words := splitWords(text)
	decode := func(out map[string]*Tensor) []Entity {
		es, err := decodeEntities(text, []string{"x"}, words, out, 0.5)
		if err != nil {
			t.Fatal(err)
		}
		return es
	}
	texts := func(es []Entity) []string {
		var out []string
		for _, e := range es {
			out = append(out, e.Text)
		}
		return out
	}
	overlap := outputsFor([]float32{3, 2, 1, -1}, []int64{0, 2, 1, 3, 2, 4, 0, 1}, []bool{true, true, true, true}, -5)
	if got := texts(decode(overlap)); !slices.Equal(got, []string{"aa bb", "cc dd"}) {
		t.Fatalf("overlap resolution: %q", got)
	}
	abstain := outputsFor([]float32{3}, []int64{0, 1}, []bool{true}, 2)
	if got := decode(abstain); len(got) != 0 {
		t.Fatalf("abstention ignored: %+v", got)
	}
	invalid := outputsFor([]float32{3, 3, 3, 3, 3}, []int64{0, 1, -1, 1, 2, 2, 3, 9, 1, 2}, []bool{false, true, true, true, true}, -5)
	if got := texts(decode(invalid)); !slices.Equal(got, []string{"bb"}) {
		t.Fatalf("invalid candidates kept: %q", got)
	}
	blank, err := decodeEntities("a  b", []string{"x"}, []word{{"a", 0, 1}, {" ", 1, 3}}, outputsFor([]float32{3}, []int64{1, 2}, []bool{true}, -5), 0.5)
	if err != nil || len(blank) != 0 {
		t.Fatalf("whitespace-only surface kept: %+v", blank)
	}
}

func TestDecodeRejectsMismatchedOutputs(t *testing.T) {
	good := outputsFor([]float32{3}, []int64{0, 1}, []bool{true}, -5)
	if _, err := decodeEntities("a.", []string{"x", "y"}, nil, good, 0.5); err == nil {
		t.Fatal("two labels against one query must fail")
	}
	mutations := map[string]func(map[string]*Tensor){
		"missing pair_logits": func(o map[string]*Tensor) { delete(o, "pair_logits") },
		"missing null":        func(o map[string]*Tensor) { delete(o, "null_logits") },
		"bad logits rank":     func(o map[string]*Tensor) { o["pair_logits"] = fT([]int{1, 1}, 1) },
		"bad index width":     func(o map[string]*Tensor) { o["candidate_indices"] = iT([]int{1, 1, 1, 3}, 0, 1, 2) },
		"bad valid shape":     func(o map[string]*Tensor) { o["candidate_valid"] = bT([]int{1, 1, 2}, true, true) },
		"bad null shape":      func(o map[string]*Tensor) { o["null_logits"] = fT([]int{1, 2}, 0, 0) },
	}
	for name, mutate := range mutations {
		o := outputsFor([]float32{3}, []int64{0, 1}, []bool{true}, -5)
		mutate(o)
		if _, err := decodeEntities("a.", []string{"x"}, nil, o, 0.5); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestResolveFlat(t *testing.T) {
	spans := func(items ...[3]float32) []span {
		var out []span
		for _, it := range items {
			out = append(out, span{it[0], int(it[1]), int(it[2])})
		}
		return out
	}
	cases := []struct {
		name string
		in   []span
		want []span
	}{
		{"empty", nil, nil},
		{"disjoint", spans([3]float32{0.9, 0, 1}, [3]float32{0.8, 2, 3}), spans([3]float32{0.9, 0, 1}, [3]float32{0.8, 2, 3})},
		{"best total wins", spans([3]float32{0.9, 0, 2}, [3]float32{0.8, 1, 3}, [3]float32{0.7, 2, 4}), spans([3]float32{0.9, 0, 2}, [3]float32{0.7, 2, 4})},
		{"duplicates collapse", spans([3]float32{0.9, 0, 1}, [3]float32{0.6, 0, 1}), spans([3]float32{0.9, 0, 1})},
		{"larger set on equal total", spans([3]float32{0.5, 0, 2}, [3]float32{0.25, 0, 1}, [3]float32{0.25, 1, 2}), spans([3]float32{0.25, 0, 1}, [3]float32{0.25, 1, 2})},
		{"equal total and size prefers earlier", spans([3]float32{0.5, 1, 3}, [3]float32{0.5, 0, 2}), spans([3]float32{0.5, 0, 2})},
		{"smaller set on equal total", spans([3]float32{0.25, 0, 1}, [3]float32{0.25, 1, 2}, [3]float32{0.5, 0, 3}), spans([3]float32{0.25, 0, 1}, [3]float32{0.25, 1, 2})},
		{"equal total and size prefers earlier start when later", spans([3]float32{0.5, 1, 3}, [3]float32{0.5, 0, 4}), spans([3]float32{0.5, 0, 4})},
		{"equal score same start", spans([3]float32{0.5, 0, 3}, [3]float32{0.5, 0, 2}), spans([3]float32{0.5, 0, 2})},
	}
	for _, c := range cases {
		if got := resolveFlat(c.in); !slices.Equal(got, c.want) {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
	}
}

func TestSelectionLess(t *testing.T) {
	rows := []row{{span{0.5, 0, 1}, 0}, {span{0.5, 1, 2}, 1}, {span{0.4, 0, 1}, 2}}
	if !selectionLess(rows, []int{0}, []int{1}) || selectionLess(rows, []int{1}, []int{0}) {
		t.Fatal("the earlier span must rank first")
	}
	if selectionLess(rows, []int{0}, []int{0}) {
		t.Fatal("identical selections are not less")
	}
	if selectionLess(rows, []int{0, 2}, []int{0, 1}) || !selectionLess(rows, []int{0, 1}, []int{0, 2}) {
		t.Fatal("selections must compare by their first differing span")
	}
}

func TestRankLess(t *testing.T) {
	a, b := span{0.9, 0, 1}, span{0.8, 0, 1}
	if !rankLess(a, b, 0, 1) || rankLess(b, a, 1, 0) {
		t.Fatal("higher score must rank first")
	}
	if !rankLess(span{1, 0, 2}, span{1, 1, 2}, 1, 0) || !rankLess(span{1, 0, 1}, span{1, 0, 2}, 1, 0) || !rankLess(span{1, 0, 1}, span{1, 0, 1}, 0, 1) {
		t.Fatal("ties must break by start, end, then index")
	}
	if got := gatherRows(fT([]int{1, 3, 2}, 0, 1, 2, 3, 4, 5), []int{2, 0}); !slices.Equal(got.F, []float32{4, 5, 0, 1}) || got.Shape[2] != 2 {
		t.Fatalf("gatherRows %+v", got)
	}
}
