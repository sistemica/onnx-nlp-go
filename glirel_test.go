package onnxnlp

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/sistemica/onnx-nlp-go/internal/fixture"
)

var glirelTokens = []string{"alice", "met", "bob"}

func glirelDir(t *testing.T, scores [][]float32) string {
	t.Helper()
	dir := t.TempDir()
	fixture.Must(fixture.WriteGLiREL(dir, scores))
	return dir
}

func twoPairScores() [][]float32 { return [][]float32{{3, -3}, {-2, 2}} }

func loadGLiRELFixture(t *testing.T, scores [][]float32) *GLiREL {
	t.Helper()
	g, err := LoadGLiREL(glirelDir(t, scores))
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestGLiRELExtractWithSyntheticModel(t *testing.T) {
	g := loadGLiRELFixture(t, twoPairScores())
	entities := []Span{{0, 1}, {2, 3}}
	got, err := g.Extract(glirelTokens, entities, []string{"knows", "met"}, 0.5, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("relations %+v", got)
	}
	first, second := got[0], got[1]
	if first.Label != "knows" || first.Head != (Span{0, 1}) || first.Tail != (Span{2, 3}) || !slices.Equal(first.HeadText, []string{"alice"}) || !slices.Equal(first.TailText, []string{"bob"}) {
		t.Fatalf("first relation %+v", first)
	}
	if want := float32(1 / (1 + math.Exp(-3))); math.Abs(float64(first.Score-want)) > 1e-6 {
		t.Fatalf("score %v, want %v", first.Score, want)
	}
	if second.Label != "met" || second.Head != (Span{2, 3}) || second.Tail != (Span{0, 1}) {
		t.Fatalf("second relation %+v", second)
	}
	high, err := g.Extract(glirelTokens, entities, []string{"knows", "met"}, 0.95, 0)
	if err != nil || len(high) != 1 || high[0].Label != "knows" {
		t.Fatalf("threshold 0.95: %+v %v", high, err)
	}
	all, err := g.Extract(glirelTokens, entities, []string{"knows", "met"}, 0, 0)
	if err != nil || len(all) != 4 {
		t.Fatalf("threshold 0 should keep every pair/label: %+v %v", all, err)
	}
}

func TestGLiRELTopKRanksAndLimitsPerPair(t *testing.T) {
	g := loadGLiRELFixture(t, twoPairScores())
	got, err := g.Extract(glirelTokens, []Span{{0, 1}, {2, 3}}, []string{"knows", "met"}, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Label != "knows" || got[1].Label != "met" || got[0].Score < got[1].Score {
		t.Fatalf("top 1 per pair, best first: %+v", got)
	}
	two, _ := g.Extract(glirelTokens, []Span{{0, 1}, {2, 3}}, []string{"knows", "met"}, 0, 2)
	if len(two) != 4 || two[0].Score < two[1].Score || two[2].Score < two[3].Score {
		t.Fatalf("top 2: %+v", two)
	}
}

func TestGLiRELDeduplicatesLabels(t *testing.T) {
	g := loadGLiRELFixture(t, twoPairScores())
	got, err := g.Extract(glirelTokens, []Span{{0, 1}, {2, 3}}, []string{"knows", "knows", "met"}, 0.5, 0)
	if err != nil || len(got) != 2 {
		t.Fatalf("repeated labels collapse to one: %+v %v", got, err)
	}
	if !slices.Equal(uniqueLabels([]string{"b", "a", "b", "a"}), []string{"b", "a"}) {
		t.Fatal("uniqueLabels must keep first occurrences in order")
	}
}

func TestGLiRELReturnsNothingWithoutPairsOrLabels(t *testing.T) {
	g := loadGLiRELFixture(t, nil)
	for name, call := range map[string]func() ([]Relation, error){
		"no labels":   func() ([]Relation, error) { return g.Extract(glirelTokens, []Span{{0, 1}, {2, 3}}, nil, 0.5, 0) },
		"one entity":  func() ([]Relation, error) { return g.Extract(glirelTokens, []Span{{0, 1}}, []string{"knows"}, 0.5, 0) },
		"no entities": func() ([]Relation, error) { return g.Extract(glirelTokens, nil, []string{"knows"}, 0.5, 0) },
		"all pairs too far": func() ([]Relation, error) {
			g.MaxPairDistance = 1
			defer func() { g.MaxPairDistance = 0 }()
			return g.Extract(glirelTokens, []Span{{0, 1}, {2, 3}}, []string{"knows"}, 0.5, 0)
		},
	} {
		if got, err := call(); err != nil || got != nil {
			t.Errorf("%s: %+v %v", name, got, err)
		}
	}
}

func TestGLiRELValidatesEntitiesAndTruncates(t *testing.T) {
	g := loadGLiRELFixture(t, twoPairScores())
	for name, e := range map[string][]Span{
		"negative start": {{-1, 1}, {2, 3}}, "empty span": {{1, 1}, {2, 3}}, "reversed": {{2, 1}, {0, 1}}, "past the end": {{0, 1}, {2, 4}},
	} {
		if _, err := g.Extract(glirelTokens, e, []string{"knows", "met"}, 0.5, 0); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	g.maxLen = 2
	got, err := g.Extract(glirelTokens, []Span{{0, 1}, {2, 3}}, []string{"knows", "met"}, 0.5, 0)
	if err != nil || got != nil {
		t.Fatalf("an entity cut off by truncation leaves no pair: %+v %v", got, err)
	}
	g.maxLen = 0
	if got, err := g.Extract(glirelTokens, []Span{{0, 1}, {2, 3}}, []string{"knows", "met"}, 0.5, 0); err != nil || len(got) != 2 {
		t.Fatalf("no length limit: %+v %v", got, err)
	}
}

func TestGLiRELEntityPairsRespectDistance(t *testing.T) {
	g := &GLiREL{}
	entities := []Span{{0, 1}, {3, 4}, {10, 11}}
	heads, tails := g.entityPairs(entities)
	if !slices.Equal(heads, []int64{0, 0, 1, 1, 2, 2}) || !slices.Equal(tails, []int64{1, 2, 0, 2, 0, 1}) {
		t.Fatalf("all ordered pairs, row by row: %v %v", heads, tails)
	}
	g.MaxPairDistance = 4
	heads, tails = g.entityPairs(entities)
	if !slices.Equal(heads, []int64{0, 1}) || !slices.Equal(tails, []int64{1, 0}) {
		t.Fatalf("only the near pair: %v %v", heads, tails)
	}
	if abs(-3) != 3 || abs(3) != 3 {
		t.Fatal("abs")
	}
}

func TestGLiRELEncodeWordsAndPooling(t *testing.T) {
	g := loadGLiRELFixture(t, nil)
	ids, first := g.encodeWords([]string{"[REL]", "knows", "[SEP]", "alice", "", "zzz"})
	if !slices.Equal(first, []int{1, 2, 3, 4, -1, 5}) {
		t.Fatalf("first subtokens %v", first)
	}
	if ids[0] != 1 || ids[len(ids)-1] != 2 || ids[1] != 20 || ids[3] != 2 {
		t.Fatalf("ids %v", ids)
	}
	hidden := fT([]int{1, 3, 2}, 1, 2, 3, 4, 5, 6)
	got := gatherPooled(hidden, []int{2, -1, 0})
	wantF(t, got, []int{1, 3, 2}, 5, 6, 0, 0, 1, 2)
}

func TestGLiRELDecodeRelations(t *testing.T) {
	entities := []Span{{0, 1}, {1, 2}}
	scores := fT([]int{2, 2}, 2, 2, 2, -5)
	rels := decodeRelations([]string{"a", "b"}, entities, []string{"x", "y"}, []int64{0, 1}, []int64{1, 0}, scores, 0.5, 0)
	if len(rels) != 3 || rels[0].Label != "x" || rels[1].Label != "y" || rels[2].Head != (Span{1, 2}) {
		t.Fatalf("pair order then label order: %+v", rels)
	}
	ranked := decodeRelations([]string{"a", "b"}, entities, []string{"x", "y"}, []int64{0, 1}, []int64{1, 0}, scores, 0.5, 1)
	if len(ranked) != 2 || ranked[0].Label != "x" || ranked[0].Head != (Span{0, 1}) {
		t.Fatalf("equal scores keep their original order: %+v", ranked)
	}
}

func TestLoadGLiRELErrors(t *testing.T) {
	if _, err := LoadGLiREL(t.TempDir()); err == nil {
		t.Fatal("empty directory must fail")
	}
	mutate := func(name string, edit func(dir string)) {
		dir := glirelDir(t, twoPairScores())
		edit(dir)
		if _, err := LoadGLiREL(dir); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	mutate("wrong architecture", func(d string) {
		os.WriteFile(filepath.Join(d, "glirel_export.json"), []byte(`{"architecture":"gliner"}`), 0o644)
	})
	mutate("no special ids", func(d string) {
		os.WriteFile(filepath.Join(d, "glirel_export.json"), []byte(`{"architecture":"glirel","special_tokens":{}}`), 0o644)
	})
	mutate("no tokenizer", func(d string) { os.Remove(filepath.Join(d, "tokenizer.json")) })
	mutate("no encoder", func(d string) { os.Remove(filepath.Join(d, "onnx", "encoder.onnx")) })
	mutate("no head", func(d string) { os.Remove(filepath.Join(d, "onnx", "head.onnx")) })
}

func TestGLiRELReportsModelErrors(t *testing.T) {
	entities, labels := []Span{{0, 1}, {2, 3}}, []string{"knows", "met"}
	needsMore, _ := NewSession(&Graph{Inputs: []string{"input_ids", "attention_mask", "extra"}})
	g := loadGLiRELFixture(t, twoPairScores())
	good := g.encoder
	g.encoder = needsMore
	if _, err := g.Extract(glirelTokens, entities, labels, 0.5, 0); err == nil {
		t.Fatal("encoder failure must surface")
	}
	g.encoder, g.head = good, needsMore
	if _, err := g.Extract(glirelTokens, entities, labels, 0.5, 0); err == nil {
		t.Fatal("head failure must surface")
	}
	wrong := loadGLiRELFixture(t, [][]float32{{1, 2, 3}})
	if _, err := wrong.Extract(glirelTokens, entities, labels, 0.5, 0); err == nil {
		t.Fatal("scores of the wrong shape must be rejected")
	}
}
