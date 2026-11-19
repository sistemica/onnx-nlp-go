// Package fixture writes tiny GLiNER and sentence-transformer model directories for tests.
package fixture

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/sistemica/onnx-nlp-go/internal/pb"
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
		"added_tokens":  []map[string]any{{"id": 0, "content": "[PAD]"}},
		"model":         map[string]any{"type": "Unigram", "unk_id": 3, "vocab": vocab},
		"normalizer":    unigramNormalizer,
		"pre_tokenizer": map[string]any{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
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

// GLiRELVocab lists the Unigram pieces written by WriteGLiREL.
var GLiRELVocab = []string{"[PAD]", "[CLS]", "[SEP]", "[UNK]", "▁alice", "▁met", "▁bob", "▁carol", "▁knows", "▁works", "▁.", "▁", "▁for"}

// WriteGLiREL writes a GLiREL-shaped export directory. Its head ignores the representations
// and always returns scores, which must be [pairs][labels] for the call under test.
func WriteGLiREL(dir string, scores [][]float32) error {
	vocab := make([][2]any, len(GLiRELVocab))
	for i, p := range GLiRELVocab {
		vocab[i] = [2]any{p, -float64(i) / 10}
	}
	special := map[string]int{"[CLS]": 1, "[SEP]": 2, "[REL]": 20}
	if err := writeJSON(dir, "tokenizer.json", map[string]any{
		"added_tokens":  []map[string]any{{"id": 2, "content": "[SEP]"}, {"id": 20, "content": "[REL]"}},
		"model":         map[string]any{"type": "Unigram", "unk_id": 3, "vocab": vocab},
		"normalizer":    unigramNormalizer,
		"pre_tokenizer": map[string]any{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
	}); err != nil {
		return err
	}
	if err := writeJSON(dir, "glirel_export.json", map[string]any{"architecture": "glirel", "special_tokens": special, "max_len": 512}); err != nil {
		return err
	}
	if err := write(dir, "onnx/encoder.onnx", lookupModel(32, "hidden_states", []string{"input_ids", "attention_mask"})); err != nil {
		return err
	}
	var flat []float32
	for _, row := range scores {
		flat = append(flat, row...)
	}
	cols := 0
	if len(scores) > 0 {
		cols = len(scores[0])
	}
	head := pb.Model(pb.Graph(
		[][]byte{pb.Node("Constant", nil, []string{"scores"}, pb.AttrTensor("value", pb.TensorFloats("", []int64{int64(len(scores)), int64(cols)}, flat)))},
		nil, []string{"prompt_rep", "word_rep", "span_idx", "head_ids", "tail_ids"}, []string{"scores"}))
	return write(dir, "onnx/head.onnx", head)
}

var unigramNormalizer = map[string]any{"type": "Sequence", "normalizers": []map[string]any{
	{"type": "Replace", "pattern": map[string]string{"Regex": `\s{2,}|[\n\r\t]`}, "content": " "},
	{"type": "NFC"},
	{"type": "Strip", "strip_left": false, "strip_right": true},
}}

// RerankerWeights returns the deterministic [Hidden, logits] projection of the reranker fixture.
func RerankerWeights(logits int) []float32 {
	w := make([]float32, Hidden*logits)
	for i := range w {
		w[i] = float32((i*5+1)%7-3) / 3
	}
	return w
}

// WriteReranker writes a cross-encoder directory whose model averages the token embeddings of a
// pair, ignoring padding, and projects them to the given number of logits. The WordPiece tokenizer is that of
// WriteMiniLM.
func WriteReranker(dir string, logits int, tokenTypes bool) error {
	if err := WriteMiniLM(dir, 0, false); err != nil {
		return err
	}
	inputs := []string{"input_ids", "attention_mask"}
	if tokenTypes {
		inputs = append(inputs, "token_type_ids")
	}
	g := pb.Graph(
		[][]byte{
			pb.Node("Gather", []string{"table", "input_ids"}, []string{"embedded"}, pb.AttrInt("axis", 0)),
			pb.Node("Cast", []string{"attention_mask"}, []string{"mask"}, pb.AttrInt("to", 1)),
			pb.Node("Unsqueeze", []string{"mask"}, []string{"mask3"}, pb.AttrInts("axes", 2)),
			pb.Node("Mul", []string{"embedded", "mask3"}, []string{"masked"}),
			pb.Node("ReduceSum", []string{"masked"}, []string{"sums"}, pb.AttrInts("axes", 1), pb.AttrInt("keepdims", 0)),
			pb.Node("ReduceSum", []string{"mask"}, []string{"count"}, pb.AttrInts("axes", 1), pb.AttrInt("keepdims", 1)),
			pb.Node("Div", []string{"sums", "count"}, []string{"pooled"}),
			pb.Node("MatMul", []string{"pooled", "w"}, []string{"logits"}),
		},
		[][]byte{
			pb.TensorFloats("table", []int64{int64(len(MiniLMVocab)), Hidden}, EmbeddingTable(len(MiniLMVocab))),
			pb.TensorFloats("w", []int64{Hidden, int64(logits)}, RerankerWeights(logits)),
		},
		append([]string{"table", "w"}, inputs...),
		[]string{"logits"},
	)
	return write(dir, "onnx/model.onnx", pb.Model(g))
}

// LMVocab is the vocabulary size of the language-model fixture. Its next token after t is
// (t+1) mod LMVocab.
const LMVocab = 8

// WriteLM writes a decoder directory with one layer and a key/value cache whose logits depend on the
// current token only: token t is followed by t+1. The cache stores each token's embedding.
func WriteLM(dir string, positionIDs bool) error {
	if err := writeJSON(dir, "config.json", map[string]any{
		"num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1, "hidden_size": LMVocab, "vocab_size": LMVocab,
	}); err != nil {
		return err
	}
	identity := make([]float32, LMVocab*LMVocab)
	successor := make([]float32, LMVocab*LMVocab)
	for t := 0; t < LMVocab; t++ {
		identity[t*LMVocab+t] = 1
		successor[t*LMVocab+(t+1)%LMVocab] = 1
	}
	inputs := []string{"table", "w", "input_ids", "attention_mask"}
	if positionIDs {
		inputs = append(inputs, "position_ids")
	}
	inputs = append(inputs, "past_key_values.0.key", "past_key_values.0.value")
	g := pb.Graph(
		[][]byte{
			pb.Node("Gather", []string{"table", "input_ids"}, []string{"x"}, pb.AttrInt("axis", 0)),
			pb.Node("MatMul", []string{"x", "w"}, []string{"logits"}),
			pb.Node("Unsqueeze", []string{"x"}, []string{"x4"}, pb.AttrInts("axes", 1)),
			pb.Node("Concat", []string{"past_key_values.0.key", "x4"}, []string{"present.0.key"}, pb.AttrInt("axis", 2)),
			pb.Node("Concat", []string{"past_key_values.0.value", "x4"}, []string{"present.0.value"}, pb.AttrInt("axis", 2)),
		},
		[][]byte{
			pb.TensorFloats("table", []int64{LMVocab, LMVocab}, identity),
			pb.TensorFloats("w", []int64{LMVocab, LMVocab}, successor),
		},
		inputs,
		[]string{"logits", "present.0.key", "present.0.value"},
	)
	return write(dir, "onnx/model.onnx", pb.Model(g))
}
