package onnxnlp

import (
	"encoding/binary"
	"testing"
)

func TestCharsMapNormalize(t *testing.T) {
	m, err := parseCharsMap(buildCharsMap(testCharsMap))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ in, want string }{
		{"xäy", "xay"},
		{"a!b!", "ab"},
		{"e\u0301", "e"},
		{"e\u0301x", "ex"},
		{"a\u0301", "a\u0301"},
		{"a\u0301\u0302\u0303", "a\u0301\u0302\u0303"},
		{"\u200e\u0304", "\u200e\u0304"},
		{"\r\n", "\r\n"},
		{"ÿ", "hello"},
		{"", ""},
		{"\u200dä", "\u200da"},
		{"日本", "日本"},
		{"e\u200d", "e\u200d"},
		{"ä\u0301\u0302", "a\u0301\u0302"},
	} {
		if got := m.normalize(tc.in); got != tc.want {
			t.Errorf("normalize(%+q) = %+q, want %+q", tc.in, got, tc.want)
		}
	}
	if _, ok := m.firstMatch("\xff\xff"); ok {
		t.Error("a byte pointing outside the trie must not match")
	}
	if _, ok := m.firstMatch("q"); ok {
		t.Error("q has no entry")
	}
	if _, ok := m.firstMatch("\xc3"); ok {
		t.Error("half a key is not a match")
	}
}

func TestParseCharsMapRejectsBrokenInput(t *testing.T) {
	good := buildCharsMap(testCharsMap)
	if _, err := parseCharsMap(good[:3]); err == nil {
		t.Error("a truncated header must be rejected")
	}
	if _, err := parseCharsMap(binary.LittleEndian.AppendUint32(nil, 6)); err == nil {
		t.Error("a trie size that is not a multiple of 4 must be rejected")
	}
	if _, err := parseCharsMap(binary.LittleEndian.AppendUint32(nil, 4000)); err == nil {
		t.Error("a trie larger than the data must be rejected")
	}
	units := []uint32{2 << 10, 0, 0, 1 | 1<<8 | 1000<<10}
	blob := binary.LittleEndian.AppendUint32(nil, uint32(4*len(units)))
	for _, u := range units {
		blob = binary.LittleEndian.AppendUint32(blob, u)
	}
	m, err := parseCharsMap(blob)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := m.firstMatch("\x01"); ok {
		t.Error("a leaf pointing outside the trie must not match")
	}
}

func TestGraphemePrefix(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"a", "a"},
		{"a\u0301b", "a\u0301"},
		{"\r\nx", "\r\n"},
		{"\ra", "\r"},
		{"\u200e\u0304", "\u200e"},
		{"a\u200e", "a"},
		{"a\u200d", "a\u200d"},
		{"\u0600\u0301", "\u0600\u0301"},
		{"é", "é"},
	} {
		if got := graphemePrefix(tc.in); got != tc.want {
			t.Errorf("graphemePrefix(%+q) = %+q, want %+q", tc.in, got, tc.want)
		}
	}
}
