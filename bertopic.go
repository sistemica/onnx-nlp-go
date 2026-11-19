package onnxgo

import (
	"errors"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
)

// BERTopic finds topics in a document collection: it embeds the documents, reduces the
// embeddings with UMAP, clusters them with HDBSCAN and describes each cluster by the words with
// the highest class-based TF-IDF score. Settings follow BERTopic's defaults.
type BERTopic struct {
	Embedder     TextEmbedder
	UMAP         UMAP
	MinTopicSize int
	TopNWords    int
	// Language "english" strips every character except letters, digits and spaces before
	// counting words; any other value keeps the text as it is.
	Language string
	// RemoveStopWords drops scikit-learn's English stop words before counting, as
	// CountVectorizer(stop_words="english") does. BERTopic keeps them by default.
	RemoveStopWords bool
}

func NewBERTopic(embedder TextEmbedder) *BERTopic {
	return &BERTopic{Embedder: embedder, UMAP: DefaultUMAP(), MinTopicSize: 10, TopNWords: 10, Language: "english"}
}

type TopicWord struct {
	Word  string
	Score float64
}

// Topic is one cluster. ID -1 collects the documents that belong to no cluster.
type Topic struct {
	ID    int
	Size  int
	Words []TopicWord
}

type TopicModel struct {
	// Topics are ordered by ID, so the outlier topic comes first when present. Other IDs are
	// numbered by decreasing size.
	Topics []Topic
	// Assignments holds each document's topic ID, Strengths its cluster membership in [0, 1].
	Assignments []int
	Strengths   []float64
}

// Fit embeds docs and models their topics.
func (b *BERTopic) Fit(docs []string) (*TopicModel, error) {
	if b.Embedder == nil {
		return nil, errors.New("bertopic: no embedder; use FitEmbeddings")
	}
	embeddings, err := b.Embedder.Embed(docs)
	if err != nil {
		return nil, fmt.Errorf("bertopic: embedding: %w", err)
	}
	return b.FitEmbeddings(docs, embeddings)
}

// FitEmbeddings models the topics of docs given their embeddings.
func (b *BERTopic) FitEmbeddings(docs []string, embeddings [][]float32) (*TopicModel, error) {
	if len(docs) != len(embeddings) {
		return nil, fmt.Errorf("bertopic: %d documents but %d embeddings", len(docs), len(embeddings))
	}
	low, err := b.UMAP.Fit(embeddings)
	if err != nil {
		return nil, err
	}
	labels, strengths, err := HDBSCAN{MinClusterSize: b.MinTopicSize}.Fit(low)
	if err != nil {
		return nil, err
	}
	labels = renumberBySize(labels)
	var stop map[string]struct{}
	if b.RemoveStopWords {
		stop = englishStopWords
	}
	return &TopicModel{Topics: describeTopics(docs, labels, b.TopNWords, b.Language, stop), Assignments: labels, Strengths: strengths}, nil
}

// renumberBySize gives the clusters the IDs 0, 1, ... in order of decreasing size; ties keep
// their original order. Noise stays -1.
func renumberBySize(labels []int) []int {
	sizes := map[int]int{}
	for _, l := range labels {
		if l >= 0 {
			sizes[l]++
		}
	}
	ids := make([]int, 0, len(sizes))
	for id := range sizes {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(a, b int) bool {
		if sizes[ids[a]] != sizes[ids[b]] {
			return sizes[ids[a]] > sizes[ids[b]]
		}
		return ids[a] < ids[b]
	})
	rank := make(map[int]int, len(ids))
	for i, id := range ids {
		rank[id] = i
	}
	out := make([]int, len(labels))
	for i, l := range labels {
		out[i] = -1
		if l >= 0 {
			out[i] = rank[l]
		}
	}
	return out
}

var nonAlphanumeric = regexp.MustCompile(`[^A-Za-z0-9 ]+`)

// cleanForCounting mirrors BERTopic's preprocessing of the text that is joined per topic.
func cleanForCounting(text, language string) string {
	text = strings.NewReplacer("\n", " ", "\t", " ").Replace(text)
	if language == "english" {
		text = nonAlphanumeric.ReplaceAllString(text, "")
	}
	if text == "" {
		return "emptydoc"
	}
	return text
}

// describeTopics joins the documents of every topic, counts words and scores them with the
// class-based TF-IDF: the frequency of a word in a topic, relative to the topic's word count,
// times log(1 + average words per topic / the word's frequency over all topics).
func describeTopics(docs []string, labels []int, topN int, language string, stop map[string]struct{}) []Topic {
	members := map[int][]string{}
	for i, l := range labels {
		members[l] = append(members[l], docs[i])
	}
	ids := make([]int, 0, len(members))
	for id := range members {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	counts := make([]map[string]int, len(ids))
	total := map[string]int{}
	var words int
	for i, id := range ids {
		counts[i] = map[string]int{}
		for _, w := range vectorizerTokens(cleanForCounting(strings.Join(members[id], " "), language)) {
			if _, drop := stop[w]; drop {
				continue
			}
			counts[i][w]++
			total[w]++
			words++
		}
	}
	average := float64(words / len(ids))
	topics := make([]Topic, len(ids))
	for i, id := range ids {
		var rowTotal int
		for _, c := range counts[i] {
			rowTotal += c
		}
		scored := make([]TopicWord, 0, len(counts[i]))
		for w, c := range counts[i] {
			scored = append(scored, TopicWord{w, float64(c) / float64(rowTotal) * math.Log(average/float64(total[w])+1)})
		}
		sort.Slice(scored, func(a, b int) bool {
			if scored[a].Score != scored[b].Score {
				return scored[a].Score > scored[b].Score
			}
			return scored[a].Word < scored[b].Word
		})
		scored = scored[:min(topN, len(scored))]
		for len(scored) < topN {
			scored = append(scored, TopicWord{"", 0.00001})
		}
		topics[i] = Topic{ID: id, Size: len(members[id]), Words: scored}
	}
	return topics
}
