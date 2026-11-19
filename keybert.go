package onnxgo

import (
	"fmt"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

type Keyword struct {
	Text  string
	Score float64
}

// KeyBERTOptions mirrors KeyBERT.extract_keywords. Zero values select the
// KeyBERT defaults.
type KeyBERTOptions struct {
	NgramMin, NgramMax int
	TopN               int
	MinDF              int
	KeepStopWords      bool
	Candidates         []string
	SeedKeywords       []string
	MMR                bool
	Diversity          float64
	MaxSum             bool
	NrCandidates       int
}

func (o KeyBERTOptions) withDefaults() KeyBERTOptions {
	o.NgramMin, o.NgramMax = max(o.NgramMin, 1), max(o.NgramMax, 1)
	if o.NgramMax < o.NgramMin {
		o.NgramMax = o.NgramMin
	}
	if o.TopN == 0 {
		o.TopN = 5
	}
	o.MinDF = max(o.MinDF, 1)
	if o.Diversity == 0 {
		o.Diversity = 0.5
	}
	if o.NrCandidates == 0 {
		o.NrCandidates = 20
	}
	return o
}

// TextEmbedder turns texts into fixed-size vectors; *Embedder is the standard implementation.
type TextEmbedder interface {
	Embed(texts []string) ([][]float32, error)
}

type KeyBERT struct {
	embedder TextEmbedder
}

func NewKeyBERT(e TextEmbedder) *KeyBERT { return &KeyBERT{embedder: e} }

func isWordRune(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.IsNumber(r)
}

// vectorizerTokens implements scikit-learn's token pattern (?u)\b\w\w+\b on lowercased text.
func vectorizerTokens(doc string) []string {
	doc = strings.ToLower(doc)
	var tokens []string
	start := -1
	runes := 0
	flush := func(end int) {
		if start >= 0 && runes >= 2 {
			tokens = append(tokens, doc[start:end])
		}
		start, runes = -1, 0
	}
	for i, r := range doc {
		if isWordRune(r) {
			if start < 0 {
				start = i
			}
			runes++
		} else {
			flush(i)
		}
	}
	flush(len(doc))
	return tokens
}

func ngrams(tokens []string, lo, hi int) []string {
	var out []string
	if lo == 1 {
		out = append(out, tokens...)
	}
	for n := max(lo, 2); n <= hi; n++ {
		for i := 0; i+n <= len(tokens); i++ {
			out = append(out, strings.Join(tokens[i:i+n], " "))
		}
	}
	return out
}

func analyze(doc string, o KeyBERTOptions) []string {
	tokens := vectorizerTokens(doc)
	if !o.KeepStopWords {
		tokens = slices.DeleteFunc(tokens, func(t string) bool {
			_, stop := englishStopWords[t]
			return stop
		})
	}
	return ngrams(tokens, o.NgramMin, o.NgramMax)
}

// vocabulary returns the sorted feature names and, per document, the indices of the
// features that occur in it, like CountVectorizer(...).fit(docs) followed by transform.
func vocabulary(docs []string, o KeyBERTOptions) ([]string, [][]int) {
	terms := make([][]string, len(docs))
	for i, d := range docs {
		terms[i] = analyze(d, o)
	}
	var features []string
	if o.Candidates != nil {
		features = o.Candidates
	} else {
		docFreq := map[string]int{}
		for _, ts := range terms {
			seen := map[string]bool{}
			for _, t := range ts {
				if !seen[t] {
					seen[t] = true
					docFreq[t]++
				}
			}
		}
		for t, df := range docFreq {
			if df >= o.MinDF {
				features = append(features, t)
			}
		}
		sort.Strings(features)
	}
	index := make(map[string]int, len(features))
	for i, f := range features {
		index[f] = i
	}
	perDoc := make([][]int, len(docs))
	for i, ts := range terms {
		seen := map[int]bool{}
		for _, t := range ts {
			if j, ok := index[t]; ok && !seen[j] {
				seen[j] = true
				perDoc[i] = append(perDoc[i], j)
			}
		}
		sort.Ints(perDoc[i])
	}
	return features, perDoc
}

func round4(x float64) float64 {
	v, _ := strconv.ParseFloat(strconv.FormatFloat(x, 'f', 4, 64), 64)
	return v
}

// Extract returns the keywords of each document, best first.
func (k *KeyBERT) Extract(docs []string, opts KeyBERTOptions) ([][]Keyword, error) {
	o := opts.withDefaults()
	if o.MaxSum && o.NrCandidates < o.TopN {
		return nil, fmt.Errorf("keybert: nr_candidates (%d) must be at least top_n (%d)", o.NrCandidates, o.TopN)
	}
	result := make([][]Keyword, len(docs))
	features, perDoc := vocabulary(docs, o)
	if len(features) == 0 {
		return result, nil
	}
	docVecs, err := k.embedder.Embed(docs)
	if err != nil {
		return nil, err
	}
	if len(o.SeedKeywords) > 0 {
		seedVecs, err := k.embedder.Embed(o.SeedKeywords)
		if err != nil {
			return nil, err
		}
		mean := make([]float32, len(seedVecs[0]))
		for _, v := range seedVecs {
			for d := range mean {
				mean[d] += v[d] / float32(len(seedVecs))
			}
		}
		for _, v := range docVecs {
			for d := range v {
				v[d] = (v[d]*3 + mean[d]) / 4
			}
		}
	}
	needed := map[int]bool{}
	for _, idx := range perDoc {
		for _, j := range idx {
			needed[j] = true
		}
	}
	var texts []string
	var texts2feature []int
	for j := range features {
		if needed[j] {
			texts = append(texts, features[j])
			texts2feature = append(texts2feature, j)
		}
	}
	vecs, err := k.embedder.Embed(texts)
	if err != nil {
		return nil, err
	}
	featureVec := make(map[int][]float32, len(texts))
	for i, j := range texts2feature {
		featureVec[j] = vecs[i]
	}
	for d, idx := range perDoc {
		if len(idx) == 0 {
			continue
		}
		words := make([]string, len(idx))
		wordVecs := make([][]float32, len(idx))
		for i, j := range idx {
			words[i], wordVecs[i] = features[j], featureVec[j]
		}
		switch {
		case o.MMR:
			result[d] = mmr(docVecs[d], wordVecs, words, o.TopN, o.Diversity)
		case o.MaxSum:
			result[d] = maxSumDistance(docVecs[d], wordVecs, words, o.TopN, o.NrCandidates)
		default:
			result[d] = topByCosine(docVecs[d], wordVecs, words, o.TopN)
		}
	}
	return result, nil
}

func topByCosine(doc []float32, wordVecs [][]float32, words []string, topN int) []Keyword {
	sims := make([]float32, len(words))
	for i, v := range wordVecs {
		sims[i] = cosine(doc, v)
	}
	order := lastN(argsortAscending(sims), topN)
	out := make([]Keyword, 0, len(order))
	for i := len(order) - 1; i >= 0; i-- {
		out = append(out, Keyword{words[order[i]], round4(float64(sims[order[i]]))})
	}
	return out
}

func mmr(doc []float32, wordVecs [][]float32, words []string, topN int, diversity float64) []Keyword {
	docSim := make([]float32, len(words))
	for i, v := range wordVecs {
		docSim[i] = cosine(doc, v)
	}
	pairSim := func(a, b int) float32 { return cosine(wordVecs[a], wordVecs[b]) }
	selected := []int{argmax(docSim)}
	var remaining []int
	for i := range words {
		if i != selected[0] {
			remaining = append(remaining, i)
		}
	}
	for step := 0; step < min(topN-1, len(words)-1); step++ {
		bestPos, bestScore := 0, float32(0)
		for pos, c := range remaining {
			target := pairSim(c, selected[0])
			for _, s := range selected[1:] {
				target = max(target, pairSim(c, s))
			}
			score := float32(1-diversity)*docSim[c] - float32(diversity)*target
			if pos == 0 || score > bestScore {
				bestPos, bestScore = pos, score
			}
		}
		selected = append(selected, remaining[bestPos])
		remaining = slices.Delete(remaining, bestPos, bestPos+1)
	}
	out := make([]Keyword, len(selected))
	for i, s := range selected {
		out[i] = Keyword{words[s], round4(float64(docSim[s]))}
	}
	sort.SliceStable(out, func(a, b int) bool { return out[a].Score > out[b].Score })
	return out
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

func maxSumDistance(doc []float32, wordVecs [][]float32, words []string, topN, nrCandidates int) []Keyword {
	if topN > len(words) {
		return nil
	}
	sims := make([]float32, len(words))
	for i, v := range wordVecs {
		sims[i] = cosine(doc, v)
	}
	pool := lastN(argsortAscending(sims), nrCandidates)
	pair := make([][]float32, len(pool))
	for i := range pool {
		pair[i] = make([]float32, len(pool))
		for j := range pool {
			pair[i][j] = cosine(wordVecs[pool[i]], wordVecs[pool[j]])
		}
	}
	var best []int
	minSum := float32(100_000)
	combo := make([]int, topN)
	var walk func(pos, from int)
	walk = func(pos, from int) {
		if pos == topN {
			var sum float32
			for _, i := range combo {
				for _, j := range combo {
					if i != j {
						sum += pair[i][j]
					}
				}
			}
			if sum < minSum {
				minSum, best = sum, slices.Clone(combo)
			}
			return
		}
		for i := from; i <= len(pool)-(topN-pos); i++ {
			combo[pos] = i
			walk(pos+1, i+1)
		}
	}
	walk(0, 0)
	out := make([]Keyword, len(best))
	for i, b := range best {
		out[i] = Keyword{words[pool[b]], round4(float64(sims[pool[b]]))}
	}
	return out
}
