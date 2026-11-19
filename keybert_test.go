package onnxgo

import (
	"errors"
	"slices"
	"strings"
	"testing"
)

type fakeEmbedder struct {
	vectors  map[string][]float32
	calls    [][]string
	failOn   string
	failCall int
}

func (f *fakeEmbedder) Embed(texts []string) ([][]float32, error) {
	f.calls = append(f.calls, slices.Clone(texts))
	if f.failCall == len(f.calls) {
		return nil, errors.New("embedding failed")
	}
	out := make([][]float32, len(texts))
	for i, t := range texts {
		if f.failOn != "" && strings.Contains(t, f.failOn) {
			return nil, errors.New("embedding failed")
		}
		v, ok := f.vectors[t]
		if !ok {
			v = []float32{0, 0, 1}
		}
		out[i] = v
	}
	return out, nil
}

func words(kws []Keyword) []string {
	var out []string
	for _, k := range kws {
		out = append(out, k.Text)
	}
	return out
}

func TestVectorizerTokens(t *testing.T) {
	got := vectorizerTokens("Hello, World! it's 2nd a_b x Ünï ǅǅ 東京")
	want := []string{"hello", "world", "it", "2nd", "a_b", "ünï", "ǆǆ", "東京"}
	if !slices.Equal(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
	if got := vectorizerTokens("a b c"); len(got) != 0 {
		t.Fatalf("single letters must be dropped: %q", got)
	}
	if got := vectorizerTokens("café ok"); !slices.Equal(got, []string{"cafe", "ok"}) {
		t.Fatalf("combining marks split words: %q", got)
	}
}

func TestAnalyzeNgramsAndStopWords(t *testing.T) {
	o := KeyBERTOptions{NgramMin: 1, NgramMax: 2}.withDefaults()
	got := analyze("The quick brown fox", o)
	if !slices.Equal(got, []string{"quick", "brown", "fox", "quick brown", "brown fox"}) {
		t.Fatalf("got %q", got)
	}
	o = KeyBERTOptions{NgramMin: 2, NgramMax: 3, KeepStopWords: true}.withDefaults()
	got = analyze("the quick fox", o)
	if !slices.Equal(got, []string{"the quick", "quick fox", "the quick fox"}) {
		t.Fatalf("got %q", got)
	}
	if got := ngrams([]string{"a"}, 1, 3); !slices.Equal(got, []string{"a"}) {
		t.Fatalf("got %q", got)
	}
}

func TestOptionDefaults(t *testing.T) {
	o := KeyBERTOptions{}.withDefaults()
	if o.NgramMin != 1 || o.NgramMax != 1 || o.TopN != 5 || o.MinDF != 1 || o.Diversity != 0.5 || o.NrCandidates != 20 {
		t.Fatalf("defaults %+v", o)
	}
	if o := (KeyBERTOptions{NgramMin: 3, NgramMax: 2}).withDefaults(); o.NgramMax != 3 {
		t.Fatalf("ngram max below min: %+v", o)
	}
}

func TestVocabulary(t *testing.T) {
	docs := []string{"apple banana apple", "banana cherry"}
	features, perDoc := vocabulary(docs, KeyBERTOptions{}.withDefaults())
	if !slices.Equal(features, []string{"apple", "banana", "cherry"}) || !slices.Equal(perDoc[0], []int{0, 1}) || !slices.Equal(perDoc[1], []int{1, 2}) {
		t.Fatalf("features %q perDoc %v", features, perDoc)
	}
	features, perDoc = vocabulary(docs, KeyBERTOptions{MinDF: 2}.withDefaults())
	if !slices.Equal(features, []string{"banana"}) || len(perDoc[0]) != 1 || len(perDoc[1]) != 1 {
		t.Fatalf("min_df: %q %v", features, perDoc)
	}
	features, perDoc = vocabulary(docs, KeyBERTOptions{Candidates: []string{"cherry", "apple", "zebra"}}.withDefaults())
	if !slices.Equal(features, []string{"cherry", "apple", "zebra"}) || !slices.Equal(perDoc[0], []int{1}) || !slices.Equal(perDoc[1], []int{0}) {
		t.Fatalf("candidates: %q %v", features, perDoc)
	}
}

func TestKeyBERTRanksByCosine(t *testing.T) {
	doc := "apple banana cherry doc"
	emb := &fakeEmbedder{vectors: map[string][]float32{
		doc:      {1, 0, 0},
		"apple":  {0.9, 0.1, 0},
		"banana": {0.2, 1, 0},
		"cherry": {1, 0, 0},
		"doc":    {0.5, 0.5, 0},
	}}
	kb := NewKeyBERT(emb)
	res, err := kb.Extract([]string{doc}, KeyBERTOptions{TopN: 2})
	if err != nil {
		t.Fatal(err)
	}
	if got := words(res[0]); !slices.Equal(got, []string{"cherry", "apple"}) || res[0][0].Score != 1 {
		t.Fatalf("top 2: %+v", res[0])
	}
	res, _ = kb.Extract([]string{doc}, KeyBERTOptions{TopN: 10})
	if got := words(res[0]); !slices.Equal(got, []string{"cherry", "apple", "doc", "banana"}) {
		t.Fatalf("all candidates: %+v", res[0])
	}
}

func TestKeyBERTMMRPrefersDiversity(t *testing.T) {
	emb := &fakeEmbedder{vectors: map[string][]float32{
		"alpha":            {1, 0, 0},
		"beta":             {0.99, 0.14, 0},
		"gamma":            {0.6, 0.8, 0},
		"alpha beta gamma": {1, 0.2, 0},
	}}
	kb := NewKeyBERT(emb)
	doc := []string{"alpha beta gamma"}
	diverse, err := kb.Extract(doc, KeyBERTOptions{TopN: 2, MMR: true, Diversity: 0.9})
	if err != nil {
		t.Fatal(err)
	}
	plain, _ := kb.Extract(doc, KeyBERTOptions{TopN: 2, MMR: true, Diversity: 0.01})
	if slices.Contains(words(diverse[0]), "alpha") || !slices.Contains(words(plain[0]), "alpha") {
		t.Fatalf("diverse %q, plain %q", words(diverse[0]), words(plain[0]))
	}
	one, _ := kb.Extract(doc, KeyBERTOptions{TopN: 1, MMR: true})
	if len(one[0]) != 1 {
		t.Fatalf("top 1: %+v", one[0])
	}
	many, _ := kb.Extract(doc, KeyBERTOptions{TopN: 10, MMR: true})
	if len(many[0]) != 3 {
		t.Fatalf("top 10 of 3: %+v", many[0])
	}
}

func TestKeyBERTMaxSum(t *testing.T) {
	emb := &fakeEmbedder{vectors: map[string][]float32{
		"one":                  {1, 0, 0},
		"two":                  {0.98, 0.2, 0},
		"three":                {0.7, 0.7, 0},
		"four":                 {0.5, 0.85, 0},
		"ruby jade onyx pearl": {1, 0.3, 0},
	}}
	kb := NewKeyBERT(emb)
	doc := []string{"ruby jade onyx pearl"}
	res, err := kb.Extract(doc, KeyBERTOptions{TopN: 2, MaxSum: true, NrCandidates: 4})
	if err != nil {
		t.Fatal(err)
	}
	if got := words(res[0]); len(got) != 2 || slices.Contains(got, "jade") && slices.Contains(got, "ruby") {
		t.Fatalf("max sum should avoid the most similar pair: %q", got)
	}
	if _, err := kb.Extract(doc, KeyBERTOptions{TopN: 5, MaxSum: true, NrCandidates: 3}); err == nil {
		t.Fatal("nr_candidates below top_n must fail")
	}
	tooMany, err := kb.Extract(doc, KeyBERTOptions{TopN: 9, MaxSum: true, NrCandidates: 20})
	if err != nil || len(tooMany[0]) != 0 {
		t.Fatalf("top_n above candidate count: %+v %v", tooMany, err)
	}
}

func TestKeyBERTSeedKeywordsShiftTheDocument(t *testing.T) {
	emb := &fakeEmbedder{vectors: map[string][]float32{
		"north": {1, 0, 0}, "east": {0, 1, 0}, "north east": {1, 1, 0}, "east side": {0, 4, 0},
	}}
	kb := NewKeyBERT(emb)
	base, _ := kb.Extract([]string{"north east"}, KeyBERTOptions{TopN: 2})
	seeded, err := kb.Extract([]string{"north east"}, KeyBERTOptions{TopN: 2, SeedKeywords: []string{"east side"}})
	if err != nil {
		t.Fatal(err)
	}
	if base[0][0].Score == seeded[0][0].Score || seeded[0][0].Text != "east" {
		t.Fatalf("seeding had no effect: %+v vs %+v", base[0], seeded[0])
	}
}

func TestKeyBERTEdgeCases(t *testing.T) {
	kb := NewKeyBERT(&fakeEmbedder{})
	res, err := kb.Extract([]string{"the and of to in"}, KeyBERTOptions{})
	if err != nil || len(res) != 1 || res[0] != nil {
		t.Fatalf("stop words only: %+v %v", res, err)
	}
	res, err = kb.Extract([]string{"alpha beta", "gamma"}, KeyBERTOptions{Candidates: []string{"gamma"}})
	if err != nil || res[0] != nil || len(res[1]) != 1 {
		t.Fatalf("doc without candidates: %+v %v", res, err)
	}
	res, err = kb.Extract(nil, KeyBERTOptions{})
	if err != nil || len(res) != 0 {
		t.Fatalf("no documents: %+v %v", res, err)
	}
	multi, err := kb.Extract([]string{"alpha beta", "beta gamma"}, KeyBERTOptions{TopN: 1})
	if err != nil || len(multi) != 2 || len(multi[0]) != 1 || len(multi[1]) != 1 {
		t.Fatalf("multi-document: %+v %v", multi, err)
	}
}

func TestKeyBERTPropagatesEmbeddingErrors(t *testing.T) {
	for _, failOn := range []string{"doc text", "seedy", "candidate"} {
		kb := NewKeyBERT(&fakeEmbedder{failOn: failOn})
		_, err := kb.Extract([]string{"doc text candidate"}, KeyBERTOptions{SeedKeywords: []string{"seedy"}})
		if err == nil {
			t.Errorf("failure on %q was swallowed", failOn)
		}
	}
}

func TestKeyBERTPropagatesCandidateEmbeddingErrors(t *testing.T) {
	kb := NewKeyBERT(&fakeEmbedder{failCall: 2})
	if _, err := kb.Extract([]string{"alpha beta"}, KeyBERTOptions{}); err == nil {
		t.Fatal("failure while embedding the candidates was swallowed")
	}
}

func TestKeyBERTWithSyntheticEmbedder(t *testing.T) {
	e, err := LoadEmbedder(minilmDir(t, 0, true))
	if err != nil {
		t.Fatal(err)
	}
	res, err := NewKeyBERT(e).Extract([]string{"hello world machine learning models"}, KeyBERTOptions{NgramMax: 2, TopN: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(res[0]) != 3 || res[0][0].Score < res[0][2].Score {
		t.Fatalf("unexpected result %+v", res[0])
	}
}
