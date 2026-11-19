package onnxnlp

import (
	"encoding/binary"
	"errors"
	"unicode"
	"unicode/utf8"
)

// charsMap is SentencePiece's precompiled normalisation map: a double-array trie from source
// strings to offsets into a pool of zero-terminated replacement strings.
type charsMap struct {
	trie []uint32
	pool []byte
}

func parseCharsMap(blob []byte) (*charsMap, error) {
	if len(blob) < 4 {
		return nil, errors.New("charsmap: truncated")
	}
	size := int(binary.LittleEndian.Uint32(blob))
	if size%4 != 0 || 4+size > len(blob) {
		return nil, errors.New("charsmap: bad trie size")
	}
	trie := make([]uint32, size/4)
	for i := range trie {
		trie[i] = binary.LittleEndian.Uint32(blob[4+4*i:])
	}
	return &charsMap{trie: trie, pool: blob[4+size:]}, nil
}

// firstMatch returns the replacement for the shortest key that is a prefix of chunk.
func (m *charsMap) firstMatch(chunk string) (string, bool) {
	unit := m.trie[0]
	node := uint32(0)
	node ^= (unit >> 10) << ((unit & (1 << 9)) >> 6)
	for i := 0; i < len(chunk); i++ {
		node ^= uint32(chunk[i])
		if int(node) >= len(m.trie) {
			return "", false
		}
		unit = m.trie[node]
		if unit&(1<<31|0xff) != uint32(chunk[i]) {
			return "", false
		}
		node ^= (unit >> 10) << ((unit & (1 << 9)) >> 6)
		if unit>>8&1 == 1 {
			if int(node) >= len(m.trie) {
				return "", false
			}
			start := int(m.trie[node] & (1<<31 - 1))
			end := start
			for end < len(m.pool) && m.pool[end] != 0 {
				end++
			}
			return string(m.pool[start:end]), true
		}
	}
	return "", false
}

// normalize rewrites valid UTF-8 text the way the Hugging Face Precompiled normaliser does: each grapheme
// shorter than six bytes is looked up whole, and anything not found is looked up character by
// character; characters without an entry stay as they are.
func (m *charsMap) normalize(text string) string {
	var out []byte
	for len(text) > 0 {
		cluster := graphemePrefix(text)
		if len(cluster) < 6 {
			if replacement, ok := m.firstMatch(cluster); ok {
				out = append(out, replacement...)
				text = text[len(cluster):]
				continue
			}
		}
		for _, r := range cluster {
			ch := string(r)
			if replacement, ok := m.firstMatch(ch); ok {
				out = append(out, replacement...)
			} else {
				out = append(out, ch...)
			}
		}
		text = text[len(cluster):]
	}
	return string(out)
}

// graphemePrefix returns the first user-perceived character of s, approximated as a base
// character followed by combining marks and joiners; every other grapheme rule only joins
// clusters of six or more bytes, which are looked up character by character anyway. Control and
// format characters never take a mark and never receive one.
func graphemePrefix(s string) string {
	first, size := utf8.DecodeRuneInString(s)
	if first == '\r' && len(s) > 1 && s[1] == '\n' {
		return s[:2]
	}
	if isGraphemeControl(first) {
		return s[:size]
	}
	for size < len(s) {
		r, n := utf8.DecodeRuneInString(s[size:])
		if isGraphemeControl(r) || (!unicode.In(r, unicode.M) && r != 0x200c && r != 0x200d) {
			break
		}
		size += n
	}
	return s[:size]
}

func isGraphemeControl(r rune) bool {
	switch {
	case r == 0x200c || r == 0x200d:
		return false
	case r >= 0x600 && r <= 0x605, r == 0x6dd, r == 0x70f, r == 0x8e2, r == 0x890, r == 0x891, r == 0x110bd, r == 0x110cd:
		return false
	}
	return unicode.In(r, unicode.Cc, unicode.Cf, unicode.Zl, unicode.Zp)
}
