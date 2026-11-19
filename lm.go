package onnxnlp

import (
	"errors"
	"fmt"
	"path/filepath"
)

// LanguageModel is a decoder-only transformer exported to ONNX with a key/value cache, such as
// Qwen2.5 or SmolLM2.
type LanguageModel struct {
	session  *Session
	layers   int
	kvHeads  int
	headDim  int
	inputSet map[string]bool
}

// Cache holds the keys and values of the tokens processed so far, one pair per layer.
type Cache struct {
	keys, values []*Tensor
	length       int
}

// Len is the number of tokens the cache holds.
func (c *Cache) Len() int { return c.length }

// LoadLanguageModel loads a directory with config.json and onnx/model.onnx.
func LoadLanguageModel(dir string) (*LanguageModel, error) {
	var cfg struct {
		Layers  int `json:"num_hidden_layers"`
		Heads   int `json:"num_attention_heads"`
		KVHeads int `json:"num_key_value_heads"`
		Hidden  int `json:"hidden_size"`
		HeadDim int `json:"head_dim"`
	}
	if err := readJSON(filepath.Join(dir, "config.json"), &cfg); err != nil {
		return nil, err
	}
	if cfg.Layers == 0 || cfg.Heads == 0 || cfg.Hidden == 0 {
		return nil, errors.New("language model: config.json lacks the transformer dimensions")
	}
	m := &LanguageModel{layers: cfg.Layers, kvHeads: cfg.KVHeads, headDim: cfg.HeadDim}
	if m.kvHeads == 0 {
		m.kvHeads = cfg.Heads
	}
	if m.headDim == 0 {
		m.headDim = cfg.Hidden / cfg.Heads
	}
	var err error
	if m.session, err = Load(filepath.Join(dir, "onnx", "model.onnx")); err != nil {
		return nil, err
	}
	m.inputSet = map[string]bool{}
	for _, name := range m.session.InputNames() {
		m.inputSet[name] = true
	}
	for _, name := range []string{"input_ids", "attention_mask"} {
		if !m.inputSet[name] {
			return nil, fmt.Errorf("language model: the graph has no input %q", name)
		}
	}
	return m, nil
}

// NewCache returns an empty cache.
func (m *LanguageModel) NewCache() *Cache {
	c := &Cache{keys: make([]*Tensor, m.layers), values: make([]*Tensor, m.layers)}
	for i := range c.keys {
		c.keys[i] = zeros(Float32, []int{1, m.kvHeads, 0, m.headDim})
		c.values[i] = zeros(Float32, []int{1, m.kvHeads, 0, m.headDim})
	}
	return c
}

// Forward processes the tokens after those already in the cache and returns the logits of every
// new position, [len(ids), vocabulary], flattened. The cache is extended in place.
func (m *LanguageModel) Forward(ids []int, cache *Cache) (*Tensor, error) {
	if len(ids) == 0 {
		return nil, errors.New("language model: no tokens to process")
	}
	n, past := len(ids), cache.length
	inputIDs, mask, positions := zeros(Int64, []int{1, n}), zeros(Int64, []int{1, past + n}), zeros(Int64, []int{1, n})
	for i, id := range ids {
		inputIDs.I[i], positions.I[i] = int64(id), int64(past+i)
	}
	for i := range mask.I {
		mask.I[i] = 1
	}
	inputs := map[string]*Tensor{"input_ids": inputIDs, "attention_mask": mask}
	if m.inputSet["position_ids"] {
		inputs["position_ids"] = positions
	}
	for l := 0; l < m.layers; l++ {
		inputs[fmt.Sprintf("past_key_values.%d.key", l)] = cache.keys[l]
		inputs[fmt.Sprintf("past_key_values.%d.value", l)] = cache.values[l]
	}
	out, err := m.session.Run(inputs)
	if err != nil {
		return nil, err
	}
	logits := out["logits"]
	if logits == nil || len(logits.Shape) != 3 || logits.Shape[1] != n {
		return nil, errors.New("language model: the graph returned no logits for the new tokens")
	}
	for l := 0; l < m.layers; l++ {
		k, v := out[fmt.Sprintf("present.%d.key", l)], out[fmt.Sprintf("present.%d.value", l)]
		if k == nil || v == nil {
			return nil, fmt.Errorf("language model: the graph returned no cache for layer %d", l)
		}
		cache.keys[l], cache.values[l] = k, v
	}
	cache.length += n
	return logits, nil
}
