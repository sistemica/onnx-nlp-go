package onnxgo

import (
	"fmt"
	"path/filepath"
	"slices"
	"sort"
)

// Span is the half-open range [Start, End) of token indices of an entity.
type Span struct{ Start, End int }

// Relation is a predicted relation between two entities.
type Relation struct {
	Head, Tail         Span
	HeadText, TailText []string
	Label              string
	Score              float32
}

// GLiREL scores relations between entities that the caller has already found, for relation
// labels given at run time. It runs a DeBERTa encoder and a small head exported to ONNX.
type GLiREL struct {
	// MaxPairDistance limits how far apart the starts of two entities may be; zero means no limit.
	MaxPairDistance int

	tok          *Tokenizer
	encoder      *Session
	head         *Session
	cls, sep     int64
	maxLen       int
	relationMark string
}

// LoadGLiREL loads an export directory with tokenizer.json, glirel_export.json and
// onnx/encoder.onnx plus onnx/head.onnx.
func LoadGLiREL(dir string) (*GLiREL, error) {
	var cfg struct {
		Architecture  string         `json:"architecture"`
		SpecialTokens map[string]int `json:"special_tokens"`
		MaxLen        int            `json:"max_len"`
	}
	if err := readJSON(filepath.Join(dir, "glirel_export.json"), &cfg); err != nil {
		return nil, err
	}
	if cfg.Architecture != "glirel" {
		return nil, fmt.Errorf("glirel: unsupported architecture %q", cfg.Architecture)
	}
	cls, okCLS := cfg.SpecialTokens["[CLS]"]
	sep, okSEP := cfg.SpecialTokens["[SEP]"]
	if !okCLS || !okSEP {
		return nil, fmt.Errorf("glirel: export lacks [CLS] or [SEP] ids")
	}
	g := &GLiREL{cls: int64(cls), sep: int64(sep), maxLen: cfg.MaxLen, relationMark: "[REL]"}
	var err error
	if g.tok, err = LoadTokenizer(filepath.Join(dir, "tokenizer.json"), cfg.SpecialTokens); err != nil {
		return nil, err
	}
	if g.encoder, err = Load(filepath.Join(dir, "onnx", "encoder.onnx")); err != nil {
		return nil, err
	}
	if g.head, err = Load(filepath.Join(dir, "onnx", "head.onnx")); err != nil {
		return nil, err
	}
	return g, nil
}

// uniqueLabels drops repeated labels, keeping the first occurrence of each.
func uniqueLabels(labels []string) []string {
	var out []string
	for _, l := range labels {
		if !slices.Contains(out, l) {
			out = append(out, l)
		}
	}
	return out
}

// encodeWords tokenizes each word separately and returns the ids of [CLS] subwords [SEP] together
// with the index of every word's first subword, or -1 for a word without subwords.
func (g *GLiREL) encodeWords(words []string) (ids []int64, first []int) {
	ids = []int64{g.cls}
	first = make([]int, len(words))
	for i, w := range words {
		sub := g.tok.Tokenize(w)
		first[i] = -1
		if len(sub) > 0 {
			first[i] = len(ids)
		}
		for _, id := range sub {
			ids = append(ids, int64(id))
		}
	}
	return append(ids, g.sep), first
}

func gatherPooled(hidden *Tensor, rows []int) *Tensor {
	width := hidden.Shape[len(hidden.Shape)-1]
	out := zeros(Float32, []int{1, len(rows), width})
	for i, r := range rows {
		if r >= 0 {
			copy(out.F[i*width:(i+1)*width], hidden.F[r*width:(r+1)*width])
		}
	}
	return out
}

// entityPairs lists all ordered pairs of distinct entities, row by row, as indices into entities.
func (g *GLiREL) entityPairs(entities []Span) (heads, tails []int64) {
	for i, a := range entities {
		for j, b := range entities {
			if i == j {
				continue
			}
			if g.MaxPairDistance > 0 && abs(a.Start-b.Start) > g.MaxPairDistance {
				continue
			}
			heads, tails = append(heads, int64(i)), append(tails, int64(j))
		}
	}
	return heads, tails
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// validateEntities checks the spans against the tokens, dropping spans cut off by truncation.
func validateEntities(entities []Span, tokens, kept int) ([]Span, error) {
	var out []Span
	for _, e := range entities {
		if e.Start < 0 || e.End <= e.Start || e.End > tokens {
			return nil, fmt.Errorf("glirel: entity [%d,%d) is outside the %d tokens", e.Start, e.End, tokens)
		}
		if e.End <= kept {
			out = append(out, e)
		}
	}
	return out, nil
}

// Extract scores every ordered pair of entities against every label and returns the relations
// whose probability exceeds threshold, in pair order and then label order. With topK > 0 the
// result is ranked by score and each entity pair keeps at most topK labels.
func (g *GLiREL) Extract(tokens []string, entities []Span, labels []string, threshold float32, topK int) ([]Relation, error) {
	labels = uniqueLabels(labels)
	kept := len(tokens)
	if g.maxLen > 0 {
		kept = min(kept, g.maxLen)
	}
	entities, err := validateEntities(entities, len(tokens), kept)
	if err != nil {
		return nil, err
	}
	tokens = tokens[:kept]
	heads, tails := g.entityPairs(entities)
	if len(labels) == 0 || len(heads) == 0 {
		return nil, nil
	}
	words := make([]string, 0, 2*len(labels)+1+len(tokens))
	for _, l := range labels {
		words = append(words, g.relationMark, l)
	}
	words = append(words, "[SEP]")
	words = append(words, tokens...)
	ids, first := g.encodeWords(words)
	hidden, err := g.encoder.Run(map[string]*Tensor{
		"input_ids":      NewInt([]int{1, len(ids)}, ids),
		"attention_mask": NewInt([]int{1, len(ids)}, slices.Repeat([]int64{1}, len(ids))),
	})
	if err != nil {
		return nil, fmt.Errorf("encoder: %w", err)
	}
	prompt := 2*len(labels) + 1
	spans := make([]int64, 0, 2*len(entities))
	for _, e := range entities {
		spans = append(spans, int64(e.Start), int64(e.End-1))
	}
	out, err := g.head.Run(map[string]*Tensor{
		"prompt_rep": gatherPooled(hidden["hidden_states"], first[:prompt-1]),
		"word_rep":   gatherPooled(hidden["hidden_states"], first[prompt:]),
		"span_idx":   NewInt([]int{len(entities), 2}, spans),
		"head_ids":   NewInt([]int{len(heads)}, heads),
		"tail_ids":   NewInt([]int{len(tails)}, tails),
	})
	if err != nil {
		return nil, fmt.Errorf("head: %w", err)
	}
	scores := out["scores"]
	if len(scores.Shape) != 2 || scores.Shape[0] != len(heads) || scores.Shape[1] != len(labels) {
		return nil, fmt.Errorf("glirel: head returned scores of shape %v for %d pairs and %d labels", scores.Shape, len(heads), len(labels))
	}
	return decodeRelations(tokens, entities, labels, heads, tails, scores, threshold, topK), nil
}

func decodeRelations(tokens []string, entities []Span, labels []string, heads, tails []int64, scores *Tensor, threshold float32, topK int) []Relation {
	var rels []Relation
	for p := range heads {
		for l, label := range labels {
			if prob := sigmoid32(scores.F[p*len(labels)+l]); prob > threshold {
				h, t := entities[heads[p]], entities[tails[p]]
				rels = append(rels, Relation{h, t, tokens[h.Start:h.End], tokens[t.Start:t.End], label, prob})
			}
		}
	}
	if topK <= 0 {
		return rels
	}
	sort.SliceStable(rels, func(a, b int) bool { return rels[a].Score > rels[b].Score })
	kept := map[[2]Span]int{}
	var out []Relation
	for _, r := range rels {
		key := [2]Span{r.Head, r.Tail}
		if kept[key] < topK {
			kept[key]++
			out = append(out, r)
		}
	}
	return out
}
