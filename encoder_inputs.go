package onnxnlp

import (
	"encoding/json"
	"fmt"
	"os"
)

// sentenceEncoder turns text into the ids a transformer encoder takes, with the special tokens of
// its tokenizer.
type sentenceEncoder struct {
	single func(text string, maxLen int) []int
	pair   func(a, b string, maxLen int) (ids, segments []int)
}

// loadSentenceEncoder reads a tokenizer.json with a WordPiece or Unigram model.
func loadSentenceEncoder(path string) (*sentenceEncoder, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var kind struct {
		Model struct {
			Type string `json:"type"`
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &kind); err != nil {
		return nil, err
	}
	switch kind.Model.Type {
	case "WordPiece":
		w, err := LoadWordPiece(path)
		if err != nil {
			return nil, err
		}
		return &sentenceEncoder{single: w.Encode, pair: w.EncodePair}, nil
	case "Unigram":
		t, err := LoadTokenizer(path, nil)
		if err != nil {
			return nil, err
		}
		if t.post == nil {
			return nil, errNoTemplate
		}
		return &sentenceEncoder{
			single: func(text string, maxLen int) []int { ids, _ := t.Encode(text, maxLen); return ids },
			pair: func(a, b string, maxLen int) ([]int, []int) {
				ids, segments, _ := t.EncodePair(a, b, maxLen)
				return ids, segments
			},
		}, nil
	}
	return nil, fmt.Errorf("tokenizer: unsupported model type %q", kind.Model.Type)
}

// maxBatchCost bounds rows times the square of the padded width, which is what the attention
// scores of a batch grow with: 32 rows of 512 tokens, or 7 rows of 1500, or 1 row of 4096.
const maxBatchCost = 1 << 24

// planBatches groups positions of order, which must be sorted by decreasing length, into batches of
// at most maxRows whose padded size stays within maxBatchCost. A batch always has one row.
func planBatches(order []int, length func(int) int, maxRows int) [][]int {
	var batches [][]int
	for lo := 0; lo < len(order); {
		width := max(length(order[lo]), 1)
		rows := max(1, min(maxRows, maxBatchCost/(width*width), len(order)-lo))
		batches = append(batches, order[lo:lo+rows])
		lo += rows
	}
	return batches
}

// encoderInputs pads the selected rows of ids to a common width and builds the inputs the
// session declares: input_ids, attention_mask and, when asked for, token_type_ids (zeros
// unless segments are given). The attention mask is returned as well.
func encoderInputs(names []string, ids, segments [][]int, rows []int) (inputs map[string]*Tensor, mask *Tensor, width int, err error) {
	for _, r := range rows {
		width = max(width, len(ids[r]))
	}
	n := len(rows)
	inputIDs, mask, types := zeros(Int64, []int{n, width}), zeros(Int64, []int{n, width}), zeros(Int64, []int{n, width})
	for i, r := range rows {
		for c, id := range ids[r] {
			inputIDs.I[i*width+c] = int64(id)
			mask.I[i*width+c] = 1
		}
		if segments != nil {
			for c, id := range segments[r] {
				types.I[i*width+c] = int64(id)
			}
		}
	}
	inputs = map[string]*Tensor{}
	for _, name := range names {
		switch name {
		case "input_ids":
			inputs[name] = inputIDs
		case "attention_mask":
			inputs[name] = mask
		case "token_type_ids":
			inputs[name] = types
		default:
			return nil, nil, 0, fmt.Errorf("encoder: the model needs an input %q that is not supported", name)
		}
	}
	return inputs, mask, width, nil
}
