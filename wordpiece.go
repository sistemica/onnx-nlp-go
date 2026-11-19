package onnxnlp

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// WordPiece is a BERT-style tokenizer: BertNormalizer, BertPreTokenizer and WordPiece.
type WordPiece struct {
	vocab        map[string]int
	unk          string
	prefix       string
	maxWordChars int
	lowercase    bool
	stripAccents bool
	cleanText    bool
	chineseChars bool
	cls, sep     int
	post         *postProcessor
}

func LoadWordPiece(path string) (*WordPiece, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var spec struct {
		Normalizer struct {
			Type         string `json:"type"`
			CleanText    bool   `json:"clean_text"`
			Chinese      bool   `json:"handle_chinese_chars"`
			StripAccents *bool  `json:"strip_accents"`
			Lowercase    bool   `json:"lowercase"`
		} `json:"normalizer"`
		Model struct {
			Type     string         `json:"type"`
			Unk      string         `json:"unk_token"`
			Prefix   string         `json:"continuing_subword_prefix"`
			MaxChars int            `json:"max_input_chars_per_word"`
			Vocab    map[string]int `json:"vocab"`
		} `json:"model"`
	}
	if err := json.Unmarshal(b, &spec); err != nil {
		return nil, err
	}
	if spec.Model.Type != "WordPiece" || spec.Normalizer.Type != "BertNormalizer" {
		return nil, fmt.Errorf("wordpiece: unsupported tokenizer %q with normalizer %q", spec.Model.Type, spec.Normalizer.Type)
	}
	w := &WordPiece{
		vocab:        spec.Model.Vocab,
		unk:          spec.Model.Unk,
		prefix:       spec.Model.Prefix,
		maxWordChars: spec.Model.MaxChars,
		lowercase:    spec.Normalizer.Lowercase,
		stripAccents: spec.Normalizer.Lowercase,
		cleanText:    spec.Normalizer.CleanText,
		chineseChars: spec.Normalizer.Chinese,
	}
	if spec.Normalizer.StripAccents != nil {
		w.stripAccents = *spec.Normalizer.StripAccents
	}
	var ok bool
	for name, dst := range map[string]*int{"[CLS]": &w.cls, "[SEP]": &w.sep} {
		if *dst, ok = w.vocab[name]; !ok {
			return nil, fmt.Errorf("wordpiece: vocabulary has no %s", name)
		}
	}
	if _, ok := w.vocab[w.unk]; !ok {
		return nil, fmt.Errorf("wordpiece: vocabulary has no %s", w.unk)
	}
	cls, sep := templatePiece{special: []int{w.cls}}, templatePiece{special: []int{w.sep}}
	w.post = &postProcessor{
		single: []templatePiece{cls, {}, sep},
		pair:   []templatePiece{cls, {}, sep, {second: true, typeID: 1}, {special: []int{w.sep}, typeID: 1}},
	}
	return w, nil
}

// EncodePair returns [CLS] a [SEP] b [SEP] with segment ids 0 and 1, truncated to maxLen ids in
// total when maxLen > 0 by removing tokens from the longer text first.
func (w *WordPiece) EncodePair(a, b string, maxLen int) (ids, segments []int) {
	return w.post.encode(w.tokens(a), w.tokens(b), true, maxLen)
}

func (w *WordPiece) tokens(text string) []int {
	var ids []int
	for _, word := range w.preTokenize(w.normalize(text)) {
		ids = append(ids, w.wordPieces(word)...)
	}
	return ids
}

// Encode returns [CLS] ids [SEP], truncated to maxLen ids in total when maxLen > 0.
func (w *WordPiece) Encode(text string, maxLen int) []int {
	ids := append([]int{w.cls}, w.tokens(text)...)
	if maxLen > 0 && len(ids) > maxLen-1 {
		ids = ids[:max(maxLen-1, 1)]
	}
	return append(ids, w.sep)
}

func isChineseChar(r rune) bool {
	switch {
	case r >= 0x4E00 && r <= 0x9FFF, r >= 0x3400 && r <= 0x4DBF,
		r >= 0x20000 && r <= 0x2A6DF, r >= 0x2A700 && r <= 0x2B73F,
		r >= 0x2B740 && r <= 0x2B81F, r >= 0x2B820 && r <= 0x2CEAF,
		r >= 0xF900 && r <= 0xFAFF, r >= 0x2F800 && r <= 0x2FA1F:
		return true
	}
	return false
}

// isBertControl is what clean_text removes: assigned control, format, private-use and surrogate
// characters, except tab, newline and carriage return, which count as whitespace. Unassigned code
// points are kept, as the tokenizers library has no category for them.
func isBertControl(r rune) bool {
	return r != '\t' && r != '\n' && r != '\r' && unicode.In(r, unicode.Cc, unicode.Cf, unicode.Co, unicode.Cs)
}

func (w *WordPiece) normalize(text string) string {
	var b strings.Builder
	for _, r := range text {
		if w.cleanText {
			if r == 0 || r == 0xFFFD || isBertControl(r) {
				continue
			}
			if unicode.IsSpace(r) {
				r = ' '
			}
		}
		if w.chineseChars && isChineseChar(r) {
			b.WriteRune(' ')
			b.WriteRune(r)
			b.WriteRune(' ')
			continue
		}
		b.WriteRune(r)
	}
	s := b.String()
	if w.stripAccents {
		s = strings.Map(func(r rune) rune {
			if unicode.Is(unicode.Mn, r) {
				return -1
			}
			return r
		}, norm.NFD.String(s))
	}
	if w.lowercase {
		s = strings.ToLower(s)
	}
	return s
}

func isBertPunct(r rune) bool {
	if r < 128 {
		return r > ' ' && r < 127 && !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}
	return unicode.IsPunct(r)
}

func (w *WordPiece) preTokenize(s string) []string {
	var words []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			words = append(words, cur.String())
			cur.Reset()
		}
	}
	for _, r := range s {
		switch {
		case unicode.IsSpace(r):
			flush()
		case isBertPunct(r):
			flush()
			words = append(words, string(r))
		default:
			cur.WriteRune(r)
		}
	}
	flush()
	return words
}

func (w *WordPiece) wordPieces(word string) []int {
	runes := []rune(word)
	if len(runes) > w.maxWordChars {
		return []int{w.vocab[w.unk]}
	}
	var ids []int
	for start := 0; start < len(runes); {
		end, found := len(runes), -1
		for ; end > start; end-- {
			piece := string(runes[start:end])
			if start > 0 {
				piece = w.prefix + piece
			}
			if id, ok := w.vocab[piece]; ok {
				found = id
				break
			}
		}
		if found < 0 {
			return []int{w.vocab[w.unk]}
		}
		ids = append(ids, found)
		start = end
	}
	return ids
}
