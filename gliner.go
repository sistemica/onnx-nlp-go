package onnxnlp

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
)

const (
	tokSepText        = "[SEP_TEXT]"
	tokSchemaPrompt   = "[P]"
	tokEntityMarker   = "[E]"
	schemaEntitiesKey = "entities"
)

const wordChar = `[\p{L}\p{N}_\p{M}\x{200c}\x{200d}]`

var wordSplitter = regexp.MustCompile(`(?i)(?:https?://[^\s]+|www\.[^\s]+)` +
	`|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}` +
	`|@[a-z0-9_]+` +
	`|` + wordChar + `+(?:[-_]` + wordChar + `+)*` +
	`|\S`)

type Entity struct {
	Label string
	Text  string
	// Start and End are byte offsets into the text passed to Extract.
	Start, End int
	Score      float32
}

type GLiNER struct {
	tok      *Tokenizer
	encoder  *Session
	boundary *Session
}

// LoadGLiNER loads a GLiNER2.5 boundary export laid out as in
// DanKau/gliner2.5-multi-v1-onnx.
func LoadGLiNER(dir string) (*GLiNER, error) {
	var cfg struct {
		Architecture  string         `json:"architecture"`
		SpecialTokens map[string]int `json:"special_tokens"`
	}
	b, err := os.ReadFile(filepath.Join(dir, "gliner2_config.json"))
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, err
	}
	if cfg.Architecture != "boundary" {
		return nil, fmt.Errorf("gliner: unsupported architecture %q", cfg.Architecture)
	}
	g := &GLiNER{}
	if g.tok, err = LoadTokenizer(filepath.Join(dir, "tokenizer.json"), cfg.SpecialTokens); err != nil {
		return nil, err
	}
	if g.encoder, err = Load(filepath.Join(dir, "onnx", "encoder.onnx")); err != nil {
		return nil, err
	}
	if g.boundary, err = Load(filepath.Join(dir, "onnx", "boundary.onnx")); err != nil {
		return nil, err
	}
	return g, nil
}

type word struct {
	text       string
	start, end int
}

func splitWords(text string) []word {
	locs := wordSplitter.FindAllStringIndex(text, -1)
	words := make([]word, len(locs))
	for i, l := range locs {
		words[i] = word{strings.ToLower(text[l[0]:l[1]]), l[0], l[1]}
	}
	return words
}

func endWithPunctuation(text string) string {
	if text == "" {
		return "."
	}
	if strings.HasSuffix(text, ".") || strings.HasSuffix(text, "!") || strings.HasSuffix(text, "?") {
		return text
	}
	return text + "."
}

type prompt struct {
	ids        []int64
	labelPos   []int
	wordFirsts []int
}

func (g *GLiNER) buildPrompt(labels []string, words []word) prompt {
	var p prompt
	emit := func(s string) int {
		pos := len(p.ids)
		for _, id := range g.tok.Tokenize(s) {
			p.ids = append(p.ids, int64(id))
		}
		return pos
	}
	for _, s := range []string{"(", tokSchemaPrompt, schemaEntitiesKey, "("} {
		emit(s)
	}
	for _, label := range labels {
		p.labelPos = append(p.labelPos, emit(tokEntityMarker))
		emit(label)
	}
	for _, s := range []string{")", ")", tokSepText} {
		emit(s)
	}
	for _, w := range words {
		p.wordFirsts = append(p.wordFirsts, min(emit(w.text), len(p.ids)-1))
	}
	return p
}

func gatherRows(hidden *Tensor, positions []int) *Tensor {
	width := hidden.Shape[len(hidden.Shape)-1]
	out := zeros(Float32, []int{1, len(positions), width})
	for i, pos := range positions {
		copy(out.F[i*width:(i+1)*width], hidden.F[pos*width:(pos+1)*width])
	}
	return out
}

func onesBool(shape []int) *Tensor {
	t := zeros(Bool, shape)
	for i := range t.I {
		t.I[i] = 1
	}
	return t
}

type span struct {
	score      float32
	start, end int
}

type row struct {
	s span
	i int
}

// Extract returns the entities of the given labels found in text. Entities
// are grouped by label in the order of labels, best score first.
func (g *GLiNER) Extract(text string, labels []string, threshold float32) ([]Entity, error) {
	if len(labels) == 0 {
		return nil, nil
	}
	text = endWithPunctuation(text)
	words := splitWords(text)
	p := g.buildPrompt(labels, words)
	seq := len(p.ids)

	hidden, err := g.encoder.Run(map[string]*Tensor{
		"input_ids":      NewInt([]int{1, seq}, p.ids),
		"attention_mask": NewInt([]int{1, seq}, slices.Repeat([]int64{1}, seq)),
	})
	if err != nil {
		return nil, fmt.Errorf("encoder: %w", err)
	}
	states := hidden["hidden_states"]
	out, err := g.boundary.Run(map[string]*Tensor{
		"token_states": gatherRows(states, p.wordFirsts),
		"text_mask":    onesBool([]int{1, len(words)}),
		"query_states": gatherRows(states, p.labelPos),
		"query_mask":   onesBool([]int{1, len(labels)}),
	})
	if err != nil {
		return nil, fmt.Errorf("boundary: %w", err)
	}
	return decodeEntities(text, labels, words, out, threshold)
}

func checkBoundaryOutputs(out map[string]*Tensor, queries int) error {
	logits, idx, valid, null := out["pair_logits"], out["candidate_indices"], out["candidate_valid"], out["null_logits"]
	if logits == nil || idx == nil || valid == nil || null == nil ||
		len(logits.Shape) != 3 || len(idx.Shape) != 4 || len(valid.Shape) != 3 || len(null.Shape) != 2 ||
		logits.Shape[0] != 1 || logits.Shape[1] != queries || null.Shape[1] != queries ||
		!slices.Equal(logits.Shape, valid.Shape) || !slices.Equal(idx.Shape[:3], logits.Shape) || idx.Shape[3] != 2 {
		return fmt.Errorf("boundary outputs do not describe %d labels", queries)
	}
	return nil
}

func decodeEntities(text string, labels []string, words []word, out map[string]*Tensor, threshold float32) ([]Entity, error) {
	if err := checkBoundaryOutputs(out, len(labels)); err != nil {
		return nil, err
	}
	logits, idx, valid, null := out["pair_logits"], out["candidate_indices"], out["candidate_valid"], out["null_logits"]
	perQuery := logits.Shape[2]
	var entities []Entity
	for q, label := range labels {
		if sigmoid32(null.F[q]) > 0.5 {
			continue
		}
		var cands []span
		for c := 0; c < perQuery; c++ {
			at := q*perQuery + c
			if valid.I[at] == 0 {
				continue
			}
			if prob := sigmoid32(logits.F[at]); prob >= threshold {
				cands = append(cands, span{prob, int(idx.I[2*at]), int(idx.I[2*at+1])})
			}
		}
		for _, s := range resolveFlat(cands) {
			if s.start < 0 || s.start >= s.end || s.end > len(words) {
				continue
			}
			from, to := words[s.start].start, words[s.end-1].end
			surface := strings.TrimSpace(text[from:to])
			if surface == "" {
				continue
			}
			entities = append(entities, Entity{label, surface, from, to, s.score})
		}
	}
	return entities, nil
}

func rankLess(a, b span, ai, bi int) bool {
	if a.score != b.score {
		return a.score > b.score
	}
	if a.start != b.start {
		return a.start < b.start
	}
	if a.end != b.end {
		return a.end < b.end
	}
	return ai < bi
}

// resolveFlat returns the non-overlapping subset with the highest total score
// (weighted interval scheduling), ranked by score, then start, then end.
func resolveFlat(cands []span) []span {
	ranked := make([]row, len(cands))
	for i, c := range cands {
		ranked[i] = row{c, i}
	}
	sort.SliceStable(ranked, func(a, b int) bool { return rankLess(ranked[a].s, ranked[b].s, ranked[a].i, ranked[b].i) })
	var distinct []row
	seen := map[[2]int]bool{}
	for _, r := range ranked {
		key := [2]int{r.s.start, r.s.end}
		if !seen[key] {
			seen[key] = true
			distinct = append(distinct, r)
		}
	}
	byEnd := slices.Clone(distinct)
	sort.SliceStable(byEnd, func(a, b int) bool {
		x, y := byEnd[a], byEnd[b]
		if x.s.end != y.s.end {
			return x.s.end < y.s.end
		}
		return x.s.start < y.s.start
	})
	type best struct {
		total float64
		sel   []int
	}
	table := []best{{}}
	for i, r := range byEnd {
		pred := sort.Search(i, func(j int) bool { return byEnd[j].s.end > r.s.start }) - 1
		with := best{table[pred+1].total + float64(r.s.score), append(slices.Clone(table[pred+1].sel), i)}
		without := table[i]
		switch {
		case with.total > without.total:
			table = append(table, with)
		case with.total < without.total:
			table = append(table, without)
		case len(with.sel) > len(without.sel):
			table = append(table, with)
		case len(with.sel) < len(without.sel):
			table = append(table, without)
		case selectionLess(byEnd, with.sel, without.sel):
			table = append(table, with)
		default:
			table = append(table, without)
		}
	}
	var picked []row
	for _, i := range table[len(table)-1].sel {
		picked = append(picked, byEnd[i])
	}
	sort.SliceStable(picked, func(a, b int) bool { return rankLess(picked[a].s, picked[b].s, picked[a].i, picked[b].i) })
	out := make([]span, len(picked))
	for i, r := range picked {
		out[i] = r.s
	}
	return out
}

func selectionLess(rows []row, a, b []int) bool {
	rank := func(sel []int) []row {
		rs := make([]row, len(sel))
		for k, i := range sel {
			rs[k] = rows[i]
		}
		sort.SliceStable(rs, func(x, y int) bool { return rankLess(rs[x].s, rs[y].s, rs[x].i, rs[y].i) })
		return rs
	}
	ra, rb := rank(a), rank(b)
	for k := range ra {
		x, y := ra[k], rb[k]
		if rankLess(x.s, y.s, x.i, y.i) {
			return true
		}
		if rankLess(y.s, x.s, y.i, x.i) {
			return false
		}
	}
	return false
}
