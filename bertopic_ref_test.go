package onnxnlp

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

type bertopicRecord struct {
	Topics []int                  `json:"topics"`
	Words  map[string][][]any     `json:"words"`
	Stop   map[string][][]any     `json:"words_stop"`
	Sizes  map[string]int         `json:"sizes"`
	Probs  []float64              `json:"probs"`
	Extra  map[string]interface{} `json:"-"`
}

func loadBertopicRecord(t *testing.T, dir string) (docs []string, rec bertopicRecord) {
	t.Helper()
	for file, into := range map[string]any{"docs.json": &docs, "bertopic.json": &rec} {
		raw, err := os.ReadFile(filepath.Join(dir, file))
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(raw, into); err != nil {
			t.Fatal(err)
		}
	}
	return docs, rec
}

func TestTopicWordsAgainstReference(t *testing.T) {
	dir := bertopicRef(t)
	docs, rec := loadBertopicRecord(t, dir)
	for name, tc := range map[string]struct {
		stop map[string]struct{}
		want map[string][][]any
	}{"default": {nil, rec.Words}, "english stop words": {englishStopWords, rec.Stop}} {
		topics := describeTopics(docs, rec.Topics, 10, "english", tc.stop)
		if len(topics) != len(tc.want) {
			t.Fatalf("%s: %d topics, reference has %d", name, len(topics), len(tc.want))
		}
		scoreMismatch, wordMismatch, tiedPositions, compared := 0, 0, 0, 0
		for _, topic := range topics {
			want := tc.want[strconv.Itoa(topic.ID)]
			if rec.Sizes[strconv.Itoa(topic.ID)] != topic.Size {
				t.Errorf("topic %d: size %d, reference %d", topic.ID, topic.Size, rec.Sizes[strconv.Itoa(topic.ID)])
			}
			if len(topic.Words) != len(want) {
				t.Fatalf("%s topic %d: %d words, reference has %d", name, topic.ID, len(topic.Words), len(want))
			}
			for p, w := range topic.Words {
				wantScore := want[p][1].(float64)
				compared++
				if math.Abs(w.Score-wantScore) > 1e-9 {
					scoreMismatch++
					t.Errorf("%s topic %d word %d: %q %.12f, reference %q %.12f", name, topic.ID, p, w.Word, w.Score, want[p][0], wantScore)
					continue
				}
				if w.Word == want[p][0].(string) {
					continue
				}
				tied := 0
				for _, other := range want {
					if math.Abs(other[1].(float64)-wantScore) < 1e-12 {
						tied++
					}
				}
				boundaryScore := want[len(want)-1][1].(float64)
				if tied < 2 && math.Abs(wantScore-boundaryScore) > 1e-12 {
					wordMismatch++
					t.Errorf("%s topic %d word %d: got %q, reference %q with a unique score", name, topic.ID, p, w.Word, want[p][0])
				} else {
					tiedPositions++
				}
			}
		}
		t.Logf("%s: %d topics, %d word positions: %d score mismatches, %d word mismatches at unique scores, %d positions differ only among words with identical scores",
			name, len(topics), compared, scoreMismatch, wordMismatch, tiedPositions)
	}
}

func TestRenumberBySizeAgainstReference(t *testing.T) {
	dir := bertopicRef(t)
	_, rec := loadBertopicRecord(t, dir)
	got := renumberBySize(readNpyInts(t, filepath.Join(dir, "hdb_labels.npy")))
	mapping := map[int]int{}
	sizes := map[int]int{}
	for i := range got {
		if prev, seen := mapping[got[i]]; seen && prev != rec.Topics[i] {
			t.Fatalf("topic %d maps to both %d and %d in the reference", got[i], prev, rec.Topics[i])
		}
		mapping[got[i]] = rec.Topics[i]
		sizes[got[i]]++
	}
	swapped := 0
	for mine, theirs := range mapping {
		if sizes[mine] != rec.Sizes[strconv.Itoa(theirs)] {
			t.Errorf("topic %d has %d documents, its counterpart %d has %d", mine, sizes[mine], theirs, rec.Sizes[strconv.Itoa(theirs)])
		}
		if mine != theirs {
			swapped++
		}
	}
	t.Logf("%d topics, %d numbered differently from the reference (equal-size topics may swap places)", len(mapping), swapped)
}

// matchTopics pairs every topic of got with the reference topic that shares the most documents
// and returns the mean Jaccard overlap of their top words, weighted by topic size.
func matchTopics(got *TopicModel, wantTopics []int, wantWords map[string][][]any) (meanOverlap float64, matched int) {
	overlap := map[[2]int]int{}
	for i, g := range got.Assignments {
		overlap[[2]int{g, wantTopics[i]}]++
	}
	var weighted, weight float64
	for _, topic := range got.Topics {
		if topic.ID < 0 {
			continue
		}
		best, bestCount := -2, 0
		for key, n := range overlap {
			if key[0] == topic.ID && n > bestCount {
				best, bestCount = key[1], n
			}
		}
		if best < 0 {
			continue
		}
		mine := map[string]bool{}
		for _, w := range topic.Words {
			mine[w.Word] = true
		}
		shared := 0
		for _, w := range wantWords[strconv.Itoa(best)] {
			if mine[w[0].(string)] {
				shared++
			}
		}
		weighted += float64(topic.Size) * float64(shared) / float64(len(topic.Words)+len(wantWords[strconv.Itoa(best)])-shared)
		weight += float64(topic.Size)
		matched++
	}
	return weighted / weight, matched
}

func TestBERTopicEndToEndAgainstReference(t *testing.T) {
	dir := bertopicRef(t)
	modelDir := os.Getenv("ONNXNLP_MINILM")
	if modelDir == "" {
		t.Skip("ONNXNLP_MINILM not set")
	}
	docs, rec := loadBertopicRecord(t, dir)
	embedder, err := LoadEmbedder(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	model, err := NewBERTopic(embedder).Fit(docs)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("BERTopic.Fit on %d documents: %v", len(docs), time.Since(start).Round(100*time.Millisecond))
	noise, wantNoise := countValue(model.Assignments, -1), countValue(rec.Topics, -1)
	overlap, matched := matchTopics(model, rec.Topics, rec.Words)
	t.Logf("go: %d topics, %d outliers; BERTopic: %d topics, %d outliers", len(model.Topics)-1, noise, len(rec.Words)-1, wantNoise)
	t.Logf("adjusted Rand of the assignments vs BERTopic %.3f; %d topics matched to a BERTopic topic, size-weighted top-word Jaccard %.3f",
		adjustedRand(model.Assignments, rec.Topics), matched, overlap)
	for _, topic := range model.Topics[1:min(4, len(model.Topics))] {
		var words []string
		for _, w := range topic.Words[:5] {
			words = append(words, w.Word)
		}
		t.Logf("topic %d (%d docs): %v", topic.ID, topic.Size, words)
	}
	if overlap < 0.5 {
		t.Fatalf("topic words overlap only %.3f with BERTopic's", overlap)
	}

	var newRec struct {
		Topics []int `json:"topics"`
	}
	var newDocs []string
	for file, into := range map[string]any{"new_docs.json": &newDocs, "new_topics.json": &newRec} {
		raw, err := os.ReadFile(filepath.Join(dir, file))
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(raw, into); err != nil {
			t.Fatal(err)
		}
	}
	start = time.Now()
	topics, strengths, err := model.Transform(newDocs)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Transform of %d new documents: %v", len(newDocs), time.Since(start).Round(10*time.Millisecond))
	toLibrary := map[int]int{}
	for _, topic := range model.Topics {
		best, bestCount := -1, 0
		for key, n := range topicOverlap(model.Assignments, rec.Topics) {
			if key[0] == topic.ID && n > bestCount {
				best, bestCount = key[1], n
			}
		}
		toLibrary[topic.ID] = best
	}
	agree := func(mine, theirs []int) float64 {
		hits := 0
		for i := range mine {
			if toLibrary[mine[i]] == theirs[i] {
				hits++
			}
		}
		return float64(hits) / float64(len(mine))
	}
	trainAgreement, newAgreement := agree(model.Assignments, rec.Topics), agree(topics, newRec.Topics)
	t.Logf("topics agreeing with BERTopic after aligning topic ids: training documents %.3f, held-out documents %.3f; outliers among held-out: go %.3f, BERTopic %.3f",
		trainAgreement, newAgreement, float64(countValue(topics, -1))/float64(len(topics)), float64(countValue(newRec.Topics, -1))/float64(len(topics)))
	for i, s := range strengths {
		if s < 0 || s > 1 || (topics[i] < 0) != (s == 0) {
			t.Fatalf("document %d: topic %d with strength %v", i, topics[i], s)
		}
	}
	if newAgreement < trainAgreement-0.15 {
		t.Fatalf("held-out documents agree with BERTopic at %.3f, far below the %.3f of the training documents", newAgreement, trainAgreement)
	}
}

func topicOverlap(a, b []int) map[[2]int]int {
	out := map[[2]int]int{}
	for i := range a {
		out[[2]int{a[i], b[i]}]++
	}
	return out
}
