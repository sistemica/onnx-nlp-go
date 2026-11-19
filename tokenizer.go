package onnxnlp

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"regexp"
	"sort"
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
	pieces    map[string]pieceInfo
	maxLen    int
	added     map[string]int
	specials  *regexp.Regexp
	stripping map[string][2]bool
	unkID     int
	unkScore  float64
	normalize []func(string) string
	prefix    bool
	post      *postProcessor
}

// rustWhitespace is what \s means in the Rust regex syntax of tokenizer.json: Unicode White_Space.
const rustWhitespace = `\t\n\v\f\r \x{85}\p{Z}`

// goPattern translates a Rust regex from tokenizer.json to Go syntax.
func goPattern(pattern string) string {
	var b strings.Builder
	inClass := false
	for i := 0; i < len(pattern); i++ {
		switch c := pattern[i]; {
		case c == '\\' && i+1 < len(pattern):
			if pattern[i+1] == 's' {
				if inClass {
					b.WriteString(rustWhitespace)
				} else {
					b.WriteString("[" + rustWhitespace + "]")
				}
			} else {
				b.WriteByte(c)
				b.WriteByte(pattern[i+1])
			}
			i++
		case c == '[':
			inClass = true
			b.WriteByte(c)
		case c == ']':
			inClass = false
			b.WriteByte(c)
		default:
			b.WriteByte(c)
		}
	}
	return b.String()
}

type normalizerSpec struct {
	Type        string           `json:"type"`
	Normalizers []normalizerSpec `json:"normalizers"`
	Pattern     struct {
		String string `json:"String"`
		Regex  string `json:"Regex"`
	} `json:"pattern"`
	Content    string `json:"content"`
	StripLeft  bool   `json:"strip_left"`
	StripRight bool   `json:"strip_right"`
	Charsmap   []byte `json:"precompiled_charsmap"`
}

func (n normalizerSpec) steps() ([]func(string) string, error) {
	switch n.Type {
	case "Sequence":
		var out []func(string) string
		for _, child := range n.Normalizers {
			steps, err := child.steps()
			if err != nil {
				return nil, err
			}
			out = append(out, steps...)
		}
		return out, nil
	case "Replace":
		if n.Pattern.Regex != "" {
			re, err := regexp.Compile(goPattern(n.Pattern.Regex))
			if err != nil {
				return nil, err
			}
			return []func(string) string{func(s string) string { return re.ReplaceAllLiteralString(s, n.Content) }}, nil
		}
		return []func(string) string{func(s string) string { return strings.ReplaceAll(s, n.Pattern.String, n.Content) }}, nil
	case "NFC":
		return []func(string) string{norm.NFC.String}, nil
	case "NFD":
		return []func(string) string{norm.NFD.String}, nil
	case "NFKC":
		return []func(string) string{norm.NFKC.String}, nil
	case "NFKD":
		return []func(string) string{norm.NFKD.String}, nil
	case "Lowercase":
		return []func(string) string{strings.ToLower}, nil
	case "Strip":
		return []func(string) string{func(s string) string {
			if n.StripLeft {
				s = strings.TrimLeftFunc(s, unicode.IsSpace)
			}
			if n.StripRight {
				s = strings.TrimRightFunc(s, unicode.IsSpace)
			}
			return s
		}}, nil
	case "Precompiled":
		m, err := parseCharsMap(n.Charsmap)
		if err != nil {
			return nil, err
		}
		return []func(string) string{m.normalize}, nil
	}
	return nil, fmt.Errorf("tokenizer: unsupported normalizer %q", n.Type)
}

func LoadTokenizer(path string, extraSpecial map[string]int) (*Tokenizer, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var spec struct {
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
			Special bool   `json:"special"`
			Lstrip  bool   `json:"lstrip"`
			Rstrip  bool   `json:"rstrip"`
		} `json:"added_tokens"`
		Normalizer    *normalizerSpec `json:"normalizer"`
		PreTokenizer  json.RawMessage `json:"pre_tokenizer"`
		PostProcessor json.RawMessage `json:"post_processor"`
		Model         struct {
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
	if spec.Normalizer != nil {
		steps, err := spec.Normalizer.steps()
		if err != nil {
			return nil, err
		}
		t.normalize = steps
	}
	if t.prefix, err = prependsSpace(spec.PreTokenizer); err != nil {
		return nil, err
	}
	if t.post, err = parsePostProcessor(spec.PostProcessor); err != nil {
		return nil, err
	}
	var specials []string
	minScore := math.Inf(1)
	for id, e := range spec.Model.Vocab {
		t.pieces[e.piece] = pieceInfo{id: id, score: e.score}
		minScore = min(minScore, e.score)
		t.maxLen = max(t.maxLen, len([]rune(e.piece)))
	}
	t.unkScore = minScore - 10
	for _, a := range spec.AddedTokens {
		t.added[a.Content] = a.ID
		if a.Special {
			specials = append(specials, regexp.QuoteMeta(a.Content))
			if a.Lstrip || a.Rstrip {
				if t.stripping == nil {
					t.stripping = map[string][2]bool{}
				}
				t.stripping[a.Content] = [2]bool{a.Lstrip, a.Rstrip}
			}
		}
	}
	if len(specials) > 0 {
		sort.Slice(specials, func(i, j int) bool { return len(specials[i]) > len(specials[j]) })
		t.specials = regexp.MustCompile(strings.Join(specials, "|"))
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
	s = t.normalized(s)
	if s == "" {
		return nil
	}
	s = strings.ReplaceAll(s, " ", metaspace)
	if t.prefix && !strings.HasPrefix(s, metaspace) {
		s = metaspace + s
	}
	var ids []int
	for _, piece := range splitMergedWithNext(s) {
		ids = append(ids, t.unigram([]rune(piece))...)
	}
	return ids
}

// normalized applies the normaliser of tokenizer.json to valid UTF-8 text.
func (t *Tokenizer) normalized(s string) string {
	s = strings.ToValidUTF8(s, "\ufffd")
	for _, step := range t.normalize {
		s = step(s)
	}
	return s
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

// prependsSpace reports whether the Metaspace pre-tokenizer in raw adds a leading space marker.
func prependsSpace(raw json.RawMessage) (bool, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return false, nil
	}
	var spec struct {
		Type          string            `json:"type"`
		Pretokenizers []json.RawMessage `json:"pretokenizers"`
		AddPrefix     *bool             `json:"add_prefix_space"`
		Scheme        string            `json:"prepend_scheme"`
	}
	if err := json.Unmarshal(raw, &spec); err != nil {
		return false, err
	}
	switch spec.Type {
	case "Sequence":
		for _, child := range spec.Pretokenizers {
			if prefix, err := prependsSpace(child); err != nil || prefix {
				return prefix, err
			}
		}
		return false, nil
	case "Metaspace":
		if spec.Scheme != "" {
			return spec.Scheme != "never", nil
		}
		return spec.AddPrefix == nil || *spec.AddPrefix, nil
	}
	return false, fmt.Errorf("tokenizer: unsupported pre-tokenizer %q", spec.Type)
}

// Encode returns the ids of one text with the special tokens of the tokenizer's template,
// truncated to maxLen ids in total when maxLen > 0. Special tokens inside the text are kept.
func (t *Tokenizer) Encode(text string, maxLen int) ([]int, error) {
	if t.post == nil {
		return nil, errNoTemplate
	}
	ids, _ := t.post.encode(t.encodeText(text), nil, false, maxLen)
	return ids, nil
}

// EncodePair returns the ids and segment ids of a text pair, truncated by removing tokens from
// the longer text first.
func (t *Tokenizer) EncodePair(a, b string, maxLen int) (ids, segments []int, err error) {
	if t.post == nil {
		return nil, nil, errNoTemplate
	}
	ids, segments = t.post.encode(t.encodeText(a), t.encodeText(b), true, maxLen)
	return ids, segments, nil
}

// encodeText tokenizes text, mapping special tokens that occur in it to their ids. A token marked
// lstrip or rstrip takes the whitespace next to it.
func (t *Tokenizer) encodeText(text string) []int {
	if t.specials == nil {
		return t.Tokenize(text)
	}
	var ids []int
	last := 0
	var trimNext bool
	for _, loc := range t.specials.FindAllStringIndex(text, -1) {
		token := text[loc[0]:loc[1]]
		before := text[last:loc[0]]
		if trimNext {
			before = strings.TrimLeftFunc(before, unicode.IsSpace)
		}
		strip := t.stripping[token]
		if strip[0] {
			before = strings.TrimRightFunc(before, unicode.IsSpace)
		}
		ids = append(ids, t.Tokenize(before)...)
		ids = append(ids, t.added[token])
		last, trimNext = loc[1], strip[1]
	}
	rest := text[last:]
	if trimNext {
		rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
	}
	return append(ids, t.Tokenize(rest)...)
}
