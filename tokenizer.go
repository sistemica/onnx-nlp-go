package onnxgo

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

const metaspace = "▁"

type vocabEntry struct {
	piece string
	score float64
}

func (v *vocabEntry) UnmarshalJSON(b []byte) error {
	var raw [2]json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	if err := json.Unmarshal(raw[0], &v.piece); err != nil {
		return err
	}
	return json.Unmarshal(raw[1], &v.score)
}

type pieceInfo struct {
	id    int
	score float64
}

// Tokenizer is a Unigram tokenizer with Metaspace pre-tokenization, matching
// the tokenizer.json layout of the GLiNER2 encoder.
type Tokenizer struct {
	pieces   map[string]pieceInfo
	maxLen   int
	added    map[string]int
	unkID    int
	unkScore float64
}

var whitespaceRuns = regexp.MustCompile(`[\t\n\v\f\r \x{85}\p{Z}]{2,}|[\n\r\t]`)

func LoadTokenizer(path string, extraSpecial map[string]int) (*Tokenizer, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var spec struct {
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
		Model struct {
			Type  string       `json:"type"`
			UnkID int          `json:"unk_id"`
			Vocab []vocabEntry `json:"vocab"`
		} `json:"model"`
	}
	if err := json.Unmarshal(b, &spec); err != nil {
		return nil, err
	}
	if spec.Model.Type != "Unigram" {
		return nil, fmt.Errorf("tokenizer: unsupported model type %q", spec.Model.Type)
	}
	t := &Tokenizer{
		pieces: make(map[string]pieceInfo, len(spec.Model.Vocab)),
		added:  map[string]int{},
		unkID:  spec.Model.UnkID,
	}
	minScore := math.Inf(1)
	for id, e := range spec.Model.Vocab {
		t.pieces[e.piece] = pieceInfo{id: id, score: e.score}
		minScore = min(minScore, e.score)
		t.maxLen = max(t.maxLen, len([]rune(e.piece)))
	}
	t.unkScore = minScore - 10
	for _, a := range spec.AddedTokens {
		t.added[a.Content] = a.ID
	}
	for k, v := range extraSpecial {
		t.added[k] = v
	}
	return t, nil
}

// Tokenize returns the ids for s without any special tokens added.
func (t *Tokenizer) Tokenize(s string) []int {
	if id, ok := t.added[s]; ok {
		return []int{id}
	}
	s = whitespaceRuns.ReplaceAllString(s, " ")
	s = norm.NFC.String(s)
	s = strings.TrimRightFunc(s, unicode.IsSpace)
	if s == "" {
		return nil
	}
	s = strings.ReplaceAll(s, " ", metaspace)
	if !strings.HasPrefix(s, metaspace) {
		s = metaspace + s
	}
	var ids []int
	for _, piece := range splitMergedWithNext(s) {
		ids = append(ids, t.unigram([]rune(piece))...)
	}
	return ids
}

func splitMergedWithNext(s string) []string {
	var out []string
	start := 0
	for i, r := range s {
		if i > start && string(r) == metaspace {
			out = append(out, s[start:i])
			start = i
		}
	}
	return append(out, s[start:])
}

func (t *Tokenizer) unigram(runes []rune) []int {
	n := len(runes)
	best := make([]float64, n+1)
	from := make([]int, n+1)
	ids := make([]int, n+1)
	for i := 1; i <= n; i++ {
		best[i] = math.Inf(-1)
	}
	for i := 0; i < n; i++ {
		hasSingle := false
		for l := 1; l <= min(t.maxLen, n-i); l++ {
			p, ok := t.pieces[string(runes[i:i+l])]
			if !ok {
				continue
			}
			if l == 1 {
				hasSingle = true
			}
			if c := best[i] + p.score; c > best[i+l] {
				best[i+l], from[i+l], ids[i+l] = c, i, p.id
			}
		}
		if !hasSingle {
			if c := best[i] + t.unkScore; c > best[i+1] {
				best[i+1], from[i+1], ids[i+1] = c, i, t.unkID
			}
		}
	}
	var rev []int
	for pos := n; pos > 0; pos = from[pos] {
		if ids[pos] == t.unkID && len(rev) > 0 && rev[len(rev)-1] == t.unkID {
			continue
		}
		rev = append(rev, ids[pos])
	}
	out := make([]int, len(rev))
	for i, id := range rev {
		out[len(rev)-1-i] = id
	}
	return out
}
