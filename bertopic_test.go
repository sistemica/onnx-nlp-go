package onnxnlp

import (
	"errors"
	"hash/fnv"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

func TestCleanForCounting(t *testing.T) {
	for _, tc := range []struct{ in, language, want string }{
		{"Don't stop\tnow\nplease!", "english", "Dont stop now please"},
		{"café déjà", "english", "caf dj"},
		{"café déjà", "multilingual", "café déjà"},
		{"!!!", "english", "emptydoc"},
		{"!!! ???", "english", " "},
		{"", "multilingual", "emptydoc"},
		{"a\tb\nc", "multilingual", "a b c"},
	} {
		if got := cleanForCounting(tc.in, tc.language); got != tc.want {
			t.Errorf("cleanForCounting(%q, %q) = %q, want %q", tc.in, tc.language, got, tc.want)
		}
	}
}

func TestRenumberBySizeOrdersTopicsByDecreasingSize(t *testing.T) {
	got := renumberBySize([]int{5, 5, -1, 2, 2, 2, 9, -1, 5, 2})
	want := []int{1, 1, -1, 0, 0, 0, 2, -1, 1, 0}
	if !slices.Equal(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	if got := renumberBySize([]int{3, 4, 4, 3}); !slices.Equal(got, []int{0, 1, 1, 0}) {
		t.Fatalf("equal sizes keep their original order: %v", got)
	}
	if got := renumberBySize(nil); len(got) != 0 {
		t.Fatalf("no labels: %v", got)
	}
}

func TestDescribeTopicsComputesClassBasedTFIDF(t *testing.T) {
	docs := []string{"Apple apple!", "banana", "Banana, cherry cherry", "cherry"}
	topics := describeTopics(docs, []int{0, 0, 1, 1}, 3, "english", nil)
	if len(topics) != 2 || topics[0].ID != 0 || topics[0].Size != 2 || topics[1].Size != 2 {
		t.Fatalf("topics %+v", topics)
	}
	// Topic 0 counts apple 2 and banana 1; topic 1 counts banana 1 and cherry 3. Seven words over
	// two topics average 3 words; apple and banana occur 2 times overall, cherry 3 times.
	want := [][]TopicWord{
		{{"apple", 2.0 / 3 * math.Log(3.0/2+1)}, {"banana", 1.0 / 3 * math.Log(3.0/2+1)}, {"", 0.00001}},
		{{"cherry", 3.0 / 4 * math.Log(3.0/3+1)}, {"banana", 1.0 / 4 * math.Log(3.0/2+1)}, {"", 0.00001}},
	}
	for i, topic := range topics {
		for j, w := range topic.Words {
			if w.Word != want[i][j].Word || math.Abs(w.Score-want[i][j].Score) > 1e-12 {
				t.Errorf("topic %d word %d: %+v, want %+v", i, j, w, want[i][j])
			}
		}
	}
}

func TestDescribeTopicsHandlesStopWordsTiesAndEmptyTopics(t *testing.T) {
	docs := []string{"the cat sat", "the dog sat", "!!!"}
	stopped := describeTopics(docs, []int{0, 0, -1}, 5, "english", englishStopWords)
	if stopped[0].ID != -1 || stopped[0].Words[0].Word != "emptydoc" {
		t.Fatalf("an empty topic is described by the placeholder word: %+v", stopped[0])
	}
	for _, w := range stopped[1].Words {
		if w.Word == "the" {
			t.Fatalf("stop words must be dropped: %+v", stopped[1].Words)
		}
	}
	kept := describeTopics(docs, []int{0, 0, -1}, 5, "english", nil)
	found := false
	for _, w := range kept[1].Words {
		found = found || w.Word == "the"
	}
	if !found {
		t.Fatalf("without a stop list every word is counted: %+v", kept[1].Words)
	}
	if stopped[1].Words[1].Score != stopped[1].Words[2].Score || stopped[1].Words[1].Word >= stopped[1].Words[2].Word {
		t.Fatalf("equal scores are ordered alphabetically: %+v", stopped[1].Words)
	}
	blank := describeTopics([]string{"a b", "c"}, []int{0, 0}, 3, "english", nil)
	if blank[0].Words[0].Word != "" || blank[0].Words[0].Score != 0.00001 {
		t.Fatalf("a topic without countable words is padded: %+v", blank[0].Words)
	}
}

type stubEmbedder struct {
	err   error
	calls int
}

func (s *stubEmbedder) Embed(texts []string) ([][]float32, error) {
	s.calls++
	if s.err != nil {
		return nil, s.err
	}
	return stubVectors(texts), nil
}

// stubVectors places each document near the direction of its first word, so documents about
// the same subject end up close together.
func stubVectors(texts []string) [][]float32 {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		h := fnv.New64a()
		h.Write([]byte(strings.Fields(text)[0]))
		rng := rand.New(rand.NewPCG(h.Sum64(), 1))
		center := make([]float64, 16)
		for d := range center {
			center[d] = rng.NormFloat64()
		}
		noise := rand.New(rand.NewPCG(uint64(i), 99))
		out[i] = make([]float32, 16)
		for d := range center {
			out[i][d] = float32(center[d] + 0.2*noise.NormFloat64())
		}
	}
	return out
}

func subjectDocuments() ([]string, []int) {
	subjects := [][]string{
		{"astronomy", "telescope", "galaxy", "planet"},
		{"cooking", "recipe", "oven", "butter"},
		{"football", "goal", "striker", "league"},
	}
	var docs []string
	var truth []int
	for s, words := range subjects {
		for i := 0; i < 40; i++ {
			docs = append(docs, words[0]+" "+words[1+i%3]+" and the "+words[1+(i+1)%3])
			truth = append(truth, s)
		}
	}
	return docs, truth
}

func TestBERTopicFindsSubjectsAndDescribesThem(t *testing.T) {
	docs, truth := subjectDocuments()
	embedder := &stubEmbedder{}
	model := NewBERTopic(embedder)
	model.UMAP.Epochs = 150
	model.RemoveStopWords = true
	result, err := model.Fit(docs)
	if err != nil {
		t.Fatal(err)
	}
	if embedder.calls != 1 {
		t.Fatalf("documents should be embedded in one call, got %d", embedder.calls)
	}
	if r := adjustedRand(result.Assignments, truth); r < 0.95 {
		t.Fatalf("subjects recovered with adjusted Rand %.3f", r)
	}
	if len(result.Assignments) != len(docs) || len(result.Strengths) != len(docs) {
		t.Fatal("one assignment and strength per document")
	}
	subjectWords := map[string]bool{"astronomy": true, "cooking": true, "football": true}
	seen := 0
	for _, topic := range result.Topics {
		if topic.ID < 0 {
			continue
		}
		top := topic.Words[0].Word
		if !subjectWords[top] {
			t.Errorf("topic %d's best word is %q", topic.ID, top)
		}
		seen++
	}
	if seen != 3 {
		t.Fatalf("%d topics found, want 3", seen)
	}
}

func TestBERTopicFitErrors(t *testing.T) {
	if _, err := (&BERTopic{}).Fit([]string{"a b c"}); err == nil {
		t.Fatal("a missing embedder must be reported")
	}
	failing := NewBERTopic(&stubEmbedder{err: errors.New("no model")})
	if _, err := failing.Fit([]string{"a b c"}); err == nil || !strings.Contains(err.Error(), "no model") {
		t.Fatalf("embedder failures must surface: %v", err)
	}
	model := NewBERTopic(nil)
	if _, err := model.FitEmbeddings([]string{"a"}, nil); err == nil {
		t.Fatal("document and embedding counts must match")
	}
	if _, err := model.FitEmbeddings([]string{"a", "b"}, [][]float32{{1}, {2}}); err == nil {
		t.Fatal("two documents are too few for UMAP")
	}
	model.MinTopicSize = 1
	docs, _ := subjectDocuments()
	if _, err := model.FitEmbeddings(docs, stubVectors(docs)); err == nil {
		t.Fatal("a minimum topic size below 2 must be rejected")
	}
}
