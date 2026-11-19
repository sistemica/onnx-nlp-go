package onnxnlp

import (
	"encoding/json"
	"errors"
	"fmt"
)

// postProcessor is a TemplateProcessing post-processor: it wraps one or two token sequences in
// special tokens and assigns segment ids.
type postProcessor struct {
	single, pair []templatePiece
}

type templatePiece struct {
	special []int
	second  bool
	typeID  int
}

func parsePostProcessor(raw json.RawMessage) (*postProcessor, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var spec struct {
		Type          string                         `json:"type"`
		Single        []map[string]map[string]any    `json:"single"`
		Pair          []map[string]map[string]any    `json:"pair"`
		SpecialTokens map[string]struct{ IDs []int } `json:"special_tokens"`
	}
	if err := json.Unmarshal(raw, &spec); err != nil {
		return nil, err
	}
	if spec.Type != "TemplateProcessing" {
		return nil, fmt.Errorf("tokenizer: unsupported post-processor %q", spec.Type)
	}
	convert := func(items []map[string]map[string]any) ([]templatePiece, error) {
		var out []templatePiece
		for _, item := range items {
			for kind, body := range item {
				typeID, _ := body["type_id"].(float64)
				name, _ := body["id"].(string)
				switch kind {
				case "SpecialToken":
					tok, ok := spec.SpecialTokens[name]
					if !ok {
						return nil, fmt.Errorf("tokenizer: template uses unknown special token %q", name)
					}
					out = append(out, templatePiece{special: tok.IDs, typeID: int(typeID)})
				case "Sequence":
					out = append(out, templatePiece{second: name == "B", typeID: int(typeID)})
				default:
					return nil, fmt.Errorf("tokenizer: unsupported template item %q", kind)
				}
			}
		}
		return out, nil
	}
	var err error
	p := &postProcessor{}
	if p.single, err = convert(spec.Single); err != nil {
		return nil, err
	}
	if p.pair, err = convert(spec.Pair); err != nil {
		return nil, err
	}
	return p, nil
}

func templateWidth(template []templatePiece) int {
	n := 0
	for _, piece := range template {
		n += len(piece.special)
	}
	return n
}

// encode wraps a (and b when pair is set) and returns the ids with their segment ids, truncated
// to maxLen in total when maxLen > 0.
func (p *postProcessor) encode(a, b []int, pair bool, maxLen int) (ids, types []int) {
	template := p.single
	if pair {
		template = p.pair
	}
	if maxLen > 0 {
		a, b = truncateLongestFirst(a, b, max(maxLen-templateWidth(template), 0))
	}
	for _, piece := range template {
		var body []int
		switch {
		case piece.special != nil:
			body = piece.special
		case piece.second:
			body = b
		default:
			body = a
		}
		for _, id := range body {
			ids, types = append(ids, id), append(types, piece.typeID)
		}
	}
	return ids, types
}

// truncateLongestFirst shortens a and b to fit the budget the way the Hugging Face tokenizers
// library does for its longest_first strategy: the shorter text keeps at most half of the budget
// and the longer one keeps the rest, so with an odd budget the longer text keeps the extra
// token. Of two equally long texts the second counts as the longer.
func truncateLongestFirst(a, b []int, budget int) ([]int, []int) {
	if len(a)+len(b) <= budget {
		return a, b
	}
	if len(a) <= len(b) {
		keep := min(len(a), budget/2)
		return a[:keep], b[:budget-keep]
	}
	keep := min(len(b), budget/2)
	return a[:budget-keep], b[:keep]
}

var errNoTemplate = errors.New("tokenizer: tokenizer.json has no post-processor template")
