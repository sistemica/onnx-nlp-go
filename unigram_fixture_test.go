package onnxnlp

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type trieNode struct {
	children map[byte]*trieNode
	terminal bool
	value    uint32
}

// buildCharsMap builds a SentencePiece precompiled charsmap (double-array trie plus replacement
// pool) for the given replacements.
func buildCharsMap(entries map[string]string) []byte {
	root := &trieNode{children: map[byte]*trieNode{}}
	var pool []byte
	for key, replacement := range entries {
		node := root
		for i := 0; i < len(key); i++ {
			next := node.children[key[i]]
			if next == nil {
				next = &trieNode{children: map[byte]*trieNode{}}
				node.children[key[i]] = next
			}
			node = next
		}
		node.terminal, node.value = true, uint32(len(pool))
		pool = append(append(pool, replacement...), 0)
	}
	units := map[uint32]uint32{}
	used := map[uint32]bool{0: true}
	bases := map[uint32]bool{}
	var place func(n *trieNode, index uint32)
	place = func(n *trieNode, index uint32) {
		base := uint32(1)
		for ; ; base++ {
			free := !bases[base] && !(n.terminal && used[base])
			for c := range n.children {
				free = free && !used[base^uint32(c)]
			}
			if free {
				break
			}
		}
		bases[base] = true
		units[index] |= (index ^ base) << 10
		if n.terminal {
			used[base], units[base] = true, n.value
		}
		for c, child := range n.children {
			at := base ^ uint32(c)
			used[at] = true
			units[at] = uint32(c)
			if child.terminal {
				units[at] |= 1 << 8
			}
			place(child, at)
		}
	}
	place(root, 0)
	var size uint32
	for index := range units {
		size = max(size, index+1)
	}
	blob := binary.LittleEndian.AppendUint32(nil, size*4)
	for i := uint32(0); i < size; i++ {
		blob = binary.LittleEndian.AppendUint32(blob, units[i])
	}
	return append(blob, pool...)
}

var testCharsMap = map[string]string{"ä": "a", "!": "", "e\u0301": "e", "ÿ": "hello"}

// unigramVocab fits the 13-row embedding table of the fixture models.
var unigramVocab = []string{"<s>", "<pad>", "</s>", "<unk>", "▁hello", "▁world", "▁a", "▁", "<r>", "<mask>", "h", "e", "l"}

func unigramSpec(withTemplate bool, normalizer any) map[string]any {
	vocab := make([][2]any, len(unigramVocab))
	for i, p := range unigramVocab {
		vocab[i] = [2]any{p, -float64(i) / 10}
	}
	added := []map[string]any{}
	for id, tok := range unigramVocab {
		if tok[0] == '<' {
			added = append(added, map[string]any{"id": id, "content": tok, "special": true, "normalized": false, "lstrip": tok == "<mask>", "rstrip": tok == "<r>"})
		}
	}
	spec := map[string]any{
		"added_tokens":  added,
		"normalizer":    normalizer,
		"pre_tokenizer": map[string]any{"type": "Metaspace", "replacement": "▁", "add_prefix_space": true},
		"model":         map[string]any{"type": "Unigram", "unk_id": 3, "vocab": vocab},
	}
	if withTemplate {
		special := func(name string, id int) map[string]any {
			return map[string]any{"id": name, "ids": []int{id}, "tokens": []string{name}}
		}
		piece := func(kind, id string, typeID int) map[string]any {
			return map[string]any{kind: map[string]any{"id": id, "type_id": typeID}}
		}
		spec["post_processor"] = map[string]any{
			"type":   "TemplateProcessing",
			"single": []any{piece("SpecialToken", "<s>", 0), piece("Sequence", "A", 0), piece("SpecialToken", "</s>", 0)},
			"pair": []any{piece("SpecialToken", "<s>", 0), piece("Sequence", "A", 0), piece("SpecialToken", "</s>", 0),
				piece("SpecialToken", "</s>", 0), piece("Sequence", "B", 1), piece("SpecialToken", "</s>", 1)},
			"special_tokens": map[string]any{"<s>": special("<s>", 0), "</s>": special("</s>", 2)},
		}
	}
	return spec
}

func xlmrNormalizer() map[string]any {
	return map[string]any{"type": "Sequence", "normalizers": []any{
		map[string]any{"type": "Precompiled", "precompiled_charsmap": buildCharsMap(testCharsMap)},
		map[string]any{"type": "Replace", "pattern": map[string]string{"Regex": " {2,}"}, "content": " "},
	}}
}

func writeTokenizerJSON(t testing.TB, path string, spec map[string]any) {
	t.Helper()
	b, err := json.Marshal(spec)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func unigramTokenizer(t *testing.T, withTemplate bool) *Tokenizer {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	writeTokenizerJSON(t, path, unigramSpec(withTemplate, xlmrNormalizer()))
	tok, err := LoadTokenizer(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}
