package onnxnlp

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"slices"
	"sort"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

// BPE is a byte-level byte-pair-encoding tokenizer as used by GPT-2, Qwen and SmolLM2. Text is
// split into pieces, every byte of a piece is mapped to a printable character, and the merges of
// tokenizer.json are applied.
type BPE struct {
	ids          map[string]int
	tokens       []string
	ranks        map[string]int
	ignoreMerges bool
	normalize    []func(string) string
	splitters    []func([]string) []string
	added        map[string]int
	addedByID    map[int]string
	special      map[int]bool
	addedRE      *regexp.Regexp
	stripping    map[string][2]bool

	mu    sync.Mutex
	cache map[string][]int
}

// byteChars maps every byte to the printable character that stands for it in the vocabulary.
var byteChars, charBytes = func() ([256]rune, map[rune]byte) {
	var chars [256]rune
	back := make(map[rune]byte, 256)
	extra := rune(256)
	for b := 0; b < 256; b++ {
		printable := (b >= '!' && b <= '~') || (b >= 0xa1 && b <= 0xac) || (b >= 0xae && b <= 0xff)
		if printable {
			chars[b] = rune(b)
		} else {
			chars[b] = extra
			extra++
		}
		back[chars[b]] = byte(b)
	}
	return chars, back
}()

type bpeSpec struct {
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
		Lstrip  bool   `json:"lstrip"`
		Rstrip  bool   `json:"rstrip"`
	} `json:"added_tokens"`
	Normalizer   *normalizerSpec `json:"normalizer"`
	PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	Decoder      *struct {
		Type string `json:"type"`
	} `json:"decoder"`
	Model struct {
		Type         string          `json:"type"`
		Vocab        map[string]int  `json:"vocab"`
		Merges       json.RawMessage `json:"merges"`
		IgnoreMerges bool            `json:"ignore_merges"`
	} `json:"model"`
}

// LoadBPE reads a tokenizer.json with a BPE model and a byte-level pre-tokenizer.
func LoadBPE(path string) (*BPE, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var spec bpeSpec
	if err := json.Unmarshal(raw, &spec); err != nil {
		return nil, err
	}
	if spec.Model.Type != "BPE" {
		return nil, fmt.Errorf("bpe: unsupported tokenizer model %q", spec.Model.Type)
	}
	if spec.Decoder == nil || spec.Decoder.Type != "ByteLevel" {
		return nil, fmt.Errorf("bpe: only the ByteLevel decoder is supported")
	}
	b := &BPE{
		ids: spec.Model.Vocab, ranks: map[string]int{}, ignoreMerges: spec.Model.IgnoreMerges,
		added: map[string]int{}, addedByID: map[int]string{}, special: map[int]bool{}, cache: map[string][]int{},
	}
	if spec.Normalizer != nil {
		if b.normalize, err = spec.Normalizer.steps(); err != nil {
			return nil, err
		}
	}
	if b.splitters, err = bpeSplitters(spec.PreTokenizer); err != nil {
		return nil, err
	}
	merges, err := parseMerges(spec.Model.Merges)
	if err != nil {
		return nil, err
	}
	for rank, m := range merges {
		b.ranks[m] = rank
	}
	size := 0
	for _, id := range b.ids {
		size = max(size, id+1)
	}
	for _, a := range spec.AddedTokens {
		size = max(size, a.ID+1)
	}
	b.tokens = make([]string, size)
	for token, id := range b.ids {
		b.tokens[id] = token
	}
	var contents []string
	for _, a := range spec.AddedTokens {
		b.added[a.Content], b.addedByID[a.ID], b.special[a.ID] = a.ID, a.Content, a.Special
		b.tokens[a.ID] = a.Content
		contents = append(contents, regexp.QuoteMeta(a.Content))
		if a.Lstrip || a.Rstrip {
			if b.stripping == nil {
				b.stripping = map[string][2]bool{}
			}
			b.stripping[a.Content] = [2]bool{a.Lstrip, a.Rstrip}
		}
	}
	if len(contents) > 0 {
		sort.Slice(contents, func(i, j int) bool { return len(contents[i]) > len(contents[j]) })
		b.addedRE = regexp.MustCompile(strings.Join(contents, "|"))
	}
	return b, nil
}

// parseMerges accepts both layouts of the merge list: "a b" strings and ["a", "b"] pairs.
func parseMerges(raw json.RawMessage) ([]string, error) {
	var strs []string
	if err := json.Unmarshal(raw, &strs); err == nil {
		return strs, nil
	}
	var pairs [][2]string
	if err := json.Unmarshal(raw, &pairs); err != nil {
		return nil, fmt.Errorf("bpe: unreadable merges: %w", err)
	}
	out := make([]string, len(pairs))
	for i, p := range pairs {
		out[i] = p[0] + " " + p[1]
	}
	return out, nil
}

// gpt2Pattern is what the ByteLevel pre-tokenizer splits with when it uses its own regex.
const gpt2Pattern = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`

// bpeSplitters turns the pre-tokenizer of tokenizer.json into functions that split pieces further.
func bpeSplitters(raw json.RawMessage) ([]func([]string) []string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var spec struct {
		Type          string            `json:"type"`
		Pretokenizers []json.RawMessage `json:"pretokenizers"`
		Pattern       struct {
			Regex string `json:"Regex"`
		} `json:"pattern"`
		Behavior         string `json:"behavior"`
		Invert           bool   `json:"invert"`
		IndividualDigits bool   `json:"individual_digits"`
		UseRegex         *bool  `json:"use_regex"`
		AddPrefixSpace   bool   `json:"add_prefix_space"`
	}
	if err := json.Unmarshal(raw, &spec); err != nil {
		return nil, err
	}
	switch spec.Type {
	case "Sequence":
		var out []func([]string) []string
		for _, child := range spec.Pretokenizers {
			steps, err := bpeSplitters(child)
			if err != nil {
				return nil, err
			}
			out = append(out, steps...)
		}
		return out, nil
	case "Split":
		if spec.Behavior != "Isolated" || spec.Invert || spec.Pattern.Regex == "" {
			return nil, fmt.Errorf("bpe: only Isolated splits on a regex are supported")
		}
		s, err := newRegexSplitter(spec.Pattern.Regex)
		if err != nil {
			return nil, err
		}
		return []func([]string) []string{s.split}, nil
	case "Digits":
		if !spec.IndividualDigits {
			return nil, fmt.Errorf("bpe: only individual digits are supported")
		}
		return []func([]string) []string{splitDigits}, nil
	case "ByteLevel":
		var out []func([]string) []string
		if spec.AddPrefixSpace {
			out = append(out, func(pieces []string) []string {
				if len(pieces) > 0 && !strings.HasPrefix(pieces[0], " ") {
					pieces[0] = " " + pieces[0]
				}
				return pieces
			})
		}
		if spec.UseRegex == nil || *spec.UseRegex {
			s, err := newRegexSplitter(gpt2Pattern)
			if err != nil {
				return nil, err
			}
			out = append(out, s.split)
		}
		return append(out, mapBytes), nil
	}
	return nil, fmt.Errorf("bpe: unsupported pre-tokenizer %q", spec.Type)
}

// regexSplitter cuts text into the matches of a pattern, keeping the text between matches as
// pieces of their own. The lookahead \s+(?!\S), which Go's regular expressions lack, is emulated.
type regexSplitter struct {
	re *regexp.Regexp
	// shorten says that a whitespace-only match followed by more text gives up its last character,
	// which is what the lookahead forces. Newlines are exempt when the pattern has its own
	// alternative for them.
	shorten        bool
	newlinesExempt bool
}

func newRegexSplitter(pattern string) (*regexSplitter, error) {
	s := &regexSplitter{}
	const lookahead = `\s+(?!\S)|`
	if strings.Contains(pattern, lookahead) {
		pattern = strings.Replace(pattern, lookahead, "", 1)
		s.shorten = true
		s.newlinesExempt = strings.Contains(pattern, `\s*[\r\n]+`)
	}
	if strings.Contains(pattern, "(?!") || strings.Contains(pattern, "(?=") {
		return nil, fmt.Errorf("bpe: unsupported lookahead in %q", pattern)
	}
	re, err := regexp.Compile(goPattern(pattern))
	if err != nil {
		return nil, err
	}
	s.re = re
	return s, nil
}

func isWhitespaceOnly(s string, exemptNewlines bool) bool {
	for _, r := range s {
		if !unicode.IsSpace(r) || (exemptNewlines && (r == '\n' || r == '\r')) {
			return false
		}
	}
	return s != ""
}

func (s *regexSplitter) split(pieces []string) []string {
	var out []string
	for _, piece := range pieces {
		for text := piece; text != ""; {
			loc := s.re.FindStringIndex(text)
			if loc == nil {
				out = append(out, text)
				break
			}
			if loc[0] > 0 {
				out = append(out, text[:loc[0]])
			}
			end := loc[1]
			if match := text[loc[0]:end]; s.shorten && end < len(text) && utf8.RuneCountInString(match) >= 2 && isWhitespaceOnly(match, s.newlinesExempt) {
				_, size := utf8.DecodeLastRuneInString(match)
				end -= size
			}
			if end == loc[0] {
				_, size := utf8.DecodeRuneInString(text[loc[0]:])
				end = loc[0] + size
			}
			out = append(out, text[loc[0]:end])
			text = text[end:]
		}
	}
	return out
}

func splitDigits(pieces []string) []string {
	var out []string
	for _, piece := range pieces {
		start := 0
		for i, r := range piece {
			if unicode.IsNumber(r) {
				if i > start {
					out = append(out, piece[start:i])
				}
				out = append(out, string(r))
				start = i + utf8.RuneLen(r)
			}
		}
		if start < len(piece) {
			out = append(out, piece[start:])
		}
	}
	return out
}

// mapBytes replaces every byte of every piece by its printable stand-in.
func mapBytes(pieces []string) []string {
	out := make([]string, len(pieces))
	for i, piece := range pieces {
		var b strings.Builder
		for j := 0; j < len(piece); j++ {
			b.WriteRune(byteChars[piece[j]])
		}
		out[i] = b.String()
	}
	return out
}

// Encode returns the ids of text. Added tokens that occur in the text, such as <|im_start|>, map to
// their ids.
func (b *BPE) Encode(text string) []int {
	text = strings.ToValidUTF8(text, "�")
	if b.addedRE == nil {
		return b.encodeSegment(text)
	}
	var ids []int
	last, trimNext := 0, false
	for _, loc := range b.addedRE.FindAllStringIndex(text, -1) {
		token := text[loc[0]:loc[1]]
		before := text[last:loc[0]]
		if trimNext {
			before = strings.TrimLeftFunc(before, unicode.IsSpace)
		}
		strip := b.stripping[token]
		if strip[0] {
			before = strings.TrimRightFunc(before, unicode.IsSpace)
		}
		ids = append(ids, b.encodeSegment(before)...)
		ids = append(ids, b.added[token])
		last, trimNext = loc[1], strip[1]
	}
	rest := text[last:]
	if trimNext {
		rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
	}
	return append(ids, b.encodeSegment(rest)...)
}

func (b *BPE) encodeSegment(text string) []int {
	if text == "" {
		return nil
	}
	for _, step := range b.normalize {
		text = step(text)
	}
	pieces := []string{text}
	for _, split := range b.splitters {
		pieces = split(pieces)
	}
	var ids []int
	for _, piece := range pieces {
		ids = append(ids, b.word(piece)...)
	}
	return ids
}

// word applies the merges to one pre-tokenized piece.
func (b *BPE) word(piece string) []int {
	b.mu.Lock()
	cached, ok := b.cache[piece]
	b.mu.Unlock()
	if ok {
		return cached
	}
	var ids []int
	if id, found := b.ids[piece]; found && b.ignoreMerges {
		ids = []int{id}
	} else {
		symbols := make([]string, 0, len(piece))
		for _, r := range piece {
			symbols = append(symbols, string(r))
		}
		for len(symbols) > 1 {
			best, bestRank := -1, 0
			for i := 0; i+1 < len(symbols); i++ {
				if rank, found := b.ranks[symbols[i]+" "+symbols[i+1]]; found && (best < 0 || rank < bestRank) {
					best, bestRank = i, rank
				}
			}
			if best < 0 {
				break
			}
			left, right := symbols[best], symbols[best+1]
			merged := symbols[:0:0]
			for i := 0; i < len(symbols); i++ {
				if i+1 < len(symbols) && symbols[i] == left && symbols[i+1] == right {
					merged = append(merged, left+right)
					i++
				} else {
					merged = append(merged, symbols[i])
				}
			}
			symbols = merged
		}
		for _, s := range symbols {
			if id, found := b.ids[s]; found {
				ids = append(ids, id)
			}
		}
	}
	b.mu.Lock()
	b.cache[piece] = ids
	b.mu.Unlock()
	return ids
}

// TokenID returns the id of an added token such as <|im_end|>.
func (b *BPE) TokenID(content string) (int, bool) {
	id, ok := b.added[content]
	return id, ok
}

// IsSpecial reports whether id is a special token.
func (b *BPE) IsSpecial(id int) bool { return b.special[id] }

// appendBytes adds the bytes that token id stands for.
func (b *BPE) appendBytes(dst []byte, id int) []byte {
	if id < 0 || id >= len(b.tokens) {
		return dst
	}
	if content, ok := b.addedByID[id]; ok {
		return append(dst, content...)
	}
	for _, r := range b.tokens[id] {
		if by, ok := charBytes[r]; ok {
			dst = append(dst, by)
		} else {
			dst = utf8.AppendRune(dst, r)
		}
	}
	return dst
}

// Decode returns the text of ids. Special tokens are written out; bytes that do not form valid
// UTF-8 become the replacement character.
func (b *BPE) Decode(ids []int) string {
	var buf []byte
	for _, id := range ids {
		buf = b.appendBytes(buf, id)
	}
	return strings.ToValidUTF8(string(buf), "�")
}

// DecodeStream turns a stream of ids into text, holding back the bytes of a character that is not
// complete yet.
type DecodeStream struct {
	bpe     *BPE
	pending []byte
}

func (b *BPE) NewDecodeStream() *DecodeStream { return &DecodeStream{bpe: b} }

// Push adds one id and returns the text that is now complete.
func (d *DecodeStream) Push(id int) string {
	d.pending = d.bpe.appendBytes(d.pending, id)
	cut := len(d.pending)
	for tail := 1; tail <= min(3, len(d.pending)); tail++ {
		c := d.pending[len(d.pending)-tail]
		if utf8.RuneStart(c) {
			if need := runeLength(c); need > tail {
				cut = len(d.pending) - tail
			}
			break
		}
	}
	out := strings.ToValidUTF8(string(d.pending[:cut]), "�")
	d.pending = slices.Clone(d.pending[cut:])
	return out
}

// Flush returns whatever is left, as replacement characters if it is incomplete.
func (d *DecodeStream) Flush() string {
	out := strings.ToValidUTF8(string(d.pending), "�")
	d.pending = nil
	return out
}

func runeLength(first byte) int {
	switch {
	case first >= 0xf0:
		return 4
	case first >= 0xe0:
		return 3
	case first >= 0xc0:
		return 2
	}
	return 1
}
