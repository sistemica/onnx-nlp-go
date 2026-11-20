package onnxnlp

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"sort"
)

// GenerateOptions controls how tokens are chosen. The zero value decodes greedily, up to 256 tokens.
type GenerateOptions struct {
	MaxNewTokens int
	// Temperature below or at zero picks the most likely token every time.
	Temperature float64
	// TopK keeps the k most likely tokens; zero keeps all.
	TopK int
	// TopP keeps the smallest set of tokens whose probabilities add up to p; zero or one keeps all.
	TopP float64
	// RepetitionPenalty above one makes tokens already in the text less likely.
	RepetitionPenalty float64
	Seed              uint64
	// StopTokens end the generation; the token itself is not returned.
	StopTokens []int
}

// generationConfig is the part of generation_config.json and config.json that sets defaults.
type generationConfig struct {
	DoSample          bool            `json:"do_sample"`
	Temperature       float64         `json:"temperature"`
	TopK              int             `json:"top_k"`
	TopP              float64         `json:"top_p"`
	RepetitionPenalty float64         `json:"repetition_penalty"`
	EOS               json.RawMessage `json:"eos_token_id"`
	MaxPositions      int             `json:"max_position_embeddings"`
}

// intList reads a JSON number or list of numbers.
func jsonInts(raw json.RawMessage) []int {
	var one int
	if json.Unmarshal(raw, &one) == nil {
		return []int{one}
	}
	var many []int
	if json.Unmarshal(raw, &many) == nil {
		return many
	}
	return nil
}

// loadGenerationDefaults reads the sampling settings a model asks for. Without do_sample the
// defaults are greedy, as in Hugging Face.
func loadGenerationDefaults(dir string) (GenerateOptions, int) {
	var model, gen generationConfig
	readJSON(filepath.Join(dir, "config.json"), &model)
	opts := GenerateOptions{StopTokens: jsonInts(model.EOS)}
	if err := readJSON(filepath.Join(dir, "generation_config.json"), &gen); err == nil {
		if stops := jsonInts(gen.EOS); len(stops) > 0 {
			opts.StopTokens = stops
		}
		opts.RepetitionPenalty = gen.RepetitionPenalty
		if gen.DoSample {
			opts.Temperature, opts.TopK, opts.TopP = gen.Temperature, gen.TopK, gen.TopP
		}
	}
	return opts, model.MaxPositions
}

// applyRepetitionPenalty makes every token of history less likely: positive scores are divided
// by the penalty, negative ones multiplied.
func applyRepetitionPenalty(logits []float32, history []int, penalty float64) {
	if penalty == 0 || penalty == 1 {
		return
	}
	seen := make(map[int]bool, len(history))
	for _, id := range history {
		if id < 0 || id >= len(logits) || seen[id] {
			continue
		}
		seen[id] = true
		if logits[id] < 0 {
			logits[id] *= float32(penalty)
		} else {
			logits[id] /= float32(penalty)
		}
	}
}

func applyTemperature(logits []float32, temperature float64) {
	if temperature <= 0 || temperature == 1 {
		return
	}
	for i := range logits {
		logits[i] /= float32(temperature)
	}
}

// applyTopK removes every token less likely than the k-th most likely; ties at that score stay.
func applyTopK(logits []float32, k int) {
	if k <= 0 || k >= len(logits) {
		return
	}
	best := make([]float32, 0, k)
	for _, v := range logits {
		if len(best) == k && v <= best[0] {
			continue
		}
		at := sort.Search(len(best), func(i int) bool { return best[i] >= v })
		if len(best) < k {
			best = append(best, 0)
		} else {
			copy(best, best[1:])
			at--
		}
		copy(best[at+1:], best[at:])
		best[at] = v
	}
	threshold := best[0]
	for i, v := range logits {
		if v < threshold {
			logits[i] = float32(math.Inf(-1))
		}
	}
}

// applyTopP keeps the most likely tokens whose probabilities reach p together, always at least one.
func applyTopP(logits []float32, p float64) {
	if p <= 0 || p >= 1 {
		return
	}
	order := make([]int, 0, len(logits))
	for i, v := range logits {
		if !math.IsInf(float64(v), -1) {
			order = append(order, i)
		}
	}
	sort.SliceStable(order, func(a, b int) bool { return logits[order[a]] < logits[order[b]] })
	if len(order) == 0 {
		return
	}
	top := float64(logits[order[len(order)-1]])
	var total float64
	for _, i := range order {
		total += math.Exp(float64(logits[i]) - top)
	}
	var cumulative float64
	for pos, i := range order {
		cumulative += math.Exp(float64(logits[i])-top) / total
		if pos < len(order)-1 && cumulative <= 1-p {
			logits[i] = float32(math.Inf(-1))
		}
	}
}

// sampleFrom draws a token with probability proportional to exp(logit).
func sampleFrom(logits []float32, rng *rand.Rand) int {
	top := argmax(logits)
	var total float64
	for _, v := range logits {
		total += math.Exp(float64(v) - float64(logits[top]))
	}
	target := rng.Float64() * total
	var cumulative float64
	for i, v := range logits {
		cumulative += math.Exp(float64(v) - float64(logits[top]))
		if cumulative > target {
			return i
		}
	}
	return top
}

// chooseToken applies the processors in the order Hugging Face does and picks a token.
func chooseToken(logits []float32, history []int, opts GenerateOptions, rng *rand.Rand) int {
	applyRepetitionPenalty(logits, history, opts.RepetitionPenalty)
	if opts.Temperature <= 0 {
		return argmax(logits)
	}
	applyTemperature(logits, opts.Temperature)
	applyTopK(logits, opts.TopK)
	applyTopP(logits, opts.TopP)
	return sampleFrom(logits, rng)
}

// Defaults returns the generation settings the model asks for in generation_config.json.
func (m *LanguageModel) Defaults() GenerateOptions { return m.defaults }

// Generate continues the prompt and returns the new tokens. emit, when set, is called with every
// new token as soon as it is chosen and can end the generation by returning false.
func (m *LanguageModel) Generate(prompt []int, opts GenerateOptions, emit func(token int) bool) ([]int, error) {
	if len(prompt) == 0 {
		return nil, errors.New("language model: the prompt is empty")
	}
	limit := opts.MaxNewTokens
	if limit <= 0 {
		limit = 256
	}
	if m.maxPositions > 0 && len(prompt)+limit > m.maxPositions {
		limit = m.maxPositions - len(prompt)
		if limit <= 0 {
			return nil, fmt.Errorf("language model: a prompt of %d tokens does not fit in %d positions", len(prompt), m.maxPositions)
		}
	}
	rng := rand.New(rand.NewPCG(opts.Seed, opts.Seed^0x9e3779b97f4a7c15))
	cache := m.NewCache()
	logits, err := m.Forward(prompt, cache)
	if err != nil {
		return nil, err
	}
	vocab := logits.Shape[2]
	row := logits.F[(len(prompt)-1)*vocab:]
	history := slices.Clone(prompt)
	var out []int
	for len(out) < limit {
		next := chooseToken(row, history, opts, rng)
		if slices.Contains(opts.StopTokens, next) {
			break
		}
		out, history = append(out, next), append(history, next)
		if emit != nil && !emit(next) {
			break
		}
		if len(out) == limit {
			break
		}
		if logits, err = m.Forward([]int{next}, cache); err != nil {
			return out, err
		}
		row = logits.F
	}
	return out, nil
}
