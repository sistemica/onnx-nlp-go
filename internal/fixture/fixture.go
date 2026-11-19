// Package fixture writes tiny GLiNER and sentence-transformer model directories for tests.
package fixture

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"onnxgo/internal/pb"
)

const Hidden = 4

func write(dir, name string, data []byte) error {
	path := filepath.Join(dir, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func writeJSON(dir, name string, v any) error {
	b, _ := json.Marshal(v)
	return write(dir, name, b)
}

// EmbeddingTable returns the deterministic [vocab, Hidden] table used by the fixture models.
func EmbeddingTable(vocab int) []float32 {
	t := make([]float32, vocab*Hidden)
	for i := range t {
		t[i] = float32((i*7+3)%11-5) / 5
	}
	return t
}

func lookupModel(vocab int, output string, inputs []string) []byte {
	g := pb.Graph(
		[][]byte{pb.Node("Gather", []string{"table", "input_ids"}, []string{output}, pb.AttrInt("axis", 0))},
		[][]byte{pb.TensorFloats("table", []int64{int64(vocab), Hidden}, EmbeddingTable(vocab))},
		append([]string{"table"}, inputs...),
		[]string{output},
	)
	return pb.Model(g)
}

// GLiNERText and GLiNERLabel are the input for which the tiny boundary graph is written.
const (
	GLiNERText  = "alice met bob"
	GLiNERLabel = "person"
)

// WriteGLiNER writes a GLiNER2.5-shaped directory whose boundary graph always reports
// one entity, the first word, for GLiNERText with the single label GLiNERLabel.
func WriteGLiNER(dir string) error {
	pieces := []string{"[PAD]", "[CLS]", "[SEP]", "[UNK]", "▁(", "▁)", "▁entities", "▁person", "▁alice", "▁met", "▁bob", "▁.", "▁"}
	vocab := make([][2]any, len(pieces))
	for i, p := range pieces {
		vocab[i] = [2]any{p, -float64(i) / 10}
	}
	specials := map[string]int{"[SEP_STRUCT]": 20, "[SEP_TEXT]": 21, "[P]": 22, "[C]": 23, "[E]": 24, "[R]": 25, "[L]": 26, "[EXAMPLE]": 27, "[OUTPUT]": 28, "[DESCRIPTION]": 29}
	if err := writeJSON(dir, "tokenizer.json", map[string]any{
		"added_tokens": []map[string]any{{"id": 0, "content": "[PAD]"}},
		"model":        map[string]any{"type": "Unigram", "unk_id": 3, "vocab": vocab},
	}); err != nil {
		return err
	}
	if err := writeJSON(dir, "gliner2_config.json", map[string]any{"architecture": "boundary", "special_tokens": specials}); err != nil {
		return err
	}
	if err := write(dir, "onnx/encoder.onnx", lookupModel(30, "hidden_states", []string{"input_ids", "attention_mask"})); err != nil {
		return err
	}
	consts := [][]byte{
		pb.Node("Constant", nil, []string{"pair_logits"}, pb.AttrTensor("value", pb.TensorFloats("", []int64{1, 1, 2}, []float32{4, -4}))),
		pb.Node("Constant", nil, []string{"candidate_indices"}, pb.AttrTensor("value", pb.TensorInt64s("", []int64{1, 1, 2, 2}, []int64{0, 1, 1, 3}))),
		pb.Node("Constant", nil, []string{"candidate_valid"}, pb.AttrTensor("value", pb.TensorBools("", []int64{1, 1, 2}, []bool{true, true}))),
		pb.Node("Constant", nil, []string{"null_logits"}, pb.AttrTensor("value", pb.TensorFloats("", []int64{1, 1}, []float32{-3}))),
	}
	boundary := pb.Model(pb.Graph(consts, nil,
		[]string{"token_states", "text_mask", "query_states", "query_mask"},
		[]string{"pair_logits", "candidate_indices", "candidate_valid", "null_logits"}))
	return write(dir, "onnx/boundary.onnx", boundary)
}

// MiniLMVocab lists the WordPiece vocabulary written by WriteMiniLM.
var MiniLMVocab = []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "machine", "learning", "model", "##s", ",", "a", "cafe"}

// WriteMiniLM writes a sentence-transformers directory with mean pooling and normalization.
func WriteMiniLM(dir string, maxSeqLength int, normalize bool) error {
	vocab := map[string]int{}
	for i, w := range MiniLMVocab {
		vocab[w] = i
	}
	if err := writeJSON(dir, "tokenizer.json", map[string]any{
		"normalizer": map[string]any{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": nil, "lowercase": true},
		"model":      map[string]any{"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": vocab},
	}); err != nil {
		return err
	}
	modules := []map[string]any{{"type": "sentence_transformers.models.Transformer"}, {"type": "sentence_transformers.models.Pooling"}}
	if normalize {
		modules = append(modules, map[string]any{"type": "sentence_transformers.models.Normalize"})
	}
	if err := writeJSON(dir, "modules.json", modules); err != nil {
		return err
	}
	if err := writeJSON(dir, "1_Pooling/config.json", map[string]any{"pooling_mode_mean_tokens": true}); err != nil {
		return err
	}
	if maxSeqLength > 0 {
		if err := writeJSON(dir, "sentence_bert_config.json", map[string]any{"max_seq_length": maxSeqLength}); err != nil {
			return err
		}
	}
	return write(dir, "onnx/model.onnx", lookupModel(len(MiniLMVocab), "last_hidden_state", []string{"input_ids", "attention_mask", "token_type_ids"}))
}

// Must panics on error; it keeps test setup short.
func Must(err error) {
	if err != nil {
		panic(fmt.Sprint("fixture: ", err))
	}
}
