package onnxnlp

import (
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/sistemica/onnx-nlp-go/internal/fixture"
)

var inf = float32(math.Inf(-1))

func TestRepetitionPenalty(t *testing.T) {
	logits := []float32{2, -2, 4, 1}
	applyRepetitionPenalty(logits, []int{0, 1, 1, 7, -1}, 2)
	if !slices.Equal(logits, []float32{1, -4, 4, 1}) {
		t.Fatalf("positive scores are divided, negative multiplied, each token once: %v", logits)
	}
	for _, penalty := range []float64{0, 1} {
		same := []float32{1, 2}
		applyRepetitionPenalty(same, []int{0}, penalty)
		if !slices.Equal(same, []float32{1, 2}) {
			t.Fatalf("a penalty of %v changes nothing: %v", penalty, same)
		}
	}
}

func TestTemperatureScaling(t *testing.T) {
	logits := []float32{2, 4}
	applyTemperature(logits, 0.5)
	if !slices.Equal(logits, []float32{4, 8}) {
		t.Fatalf("scores are divided by the temperature: %v", logits)
	}
	for _, temperature := range []float64{0, 1, -1} {
		same := []float32{1, 2}
		applyTemperature(same, temperature)
		if !slices.Equal(same, []float32{1, 2}) {
			t.Fatalf("temperature %v changes nothing: %v", temperature, same)
		}
	}
}

func TestTopKSampling(t *testing.T) {
	logits := []float32{1, 5, 3, 5, 2, 4}
	applyTopK(logits, 3)
	if !slices.Equal(logits, []float32{inf, 5, inf, 5, inf, 4}) {
		t.Fatalf("the three best are kept: %v", logits)
	}
	tied := []float32{3, 3, 3, 1}
	applyTopK(tied, 2)
	if !slices.Equal(tied, []float32{3, 3, 3, inf}) {
		t.Fatalf("ties at the threshold all stay: %v", tied)
	}
	rising := []float32{1, 2, 3, 4, 5}
	applyTopK(rising, 2)
	if !slices.Equal(rising, []float32{inf, inf, inf, 4, 5}) {
		t.Fatalf("a rising sequence: %v", rising)
	}
	for _, k := range []int{0, -1, 9} {
		same := []float32{1, 2, 3}
		applyTopK(same, k)
		if !slices.Equal(same, []float32{1, 2, 3}) {
			t.Fatalf("k = %d keeps everything: %v", k, same)
		}
	}
}

func TestTopPSampling(t *testing.T) {
	// Probabilities 0.1, 0.2, 0.7: p = 0.8 needs the two best (0.7 + 0.2 = 0.9 >= 0.8).
	logits := []float32{float32(math.Log(0.1)), float32(math.Log(0.7)), float32(math.Log(0.2))}
	applyTopP(logits, 0.8)
	if !math.IsInf(float64(logits[0]), -1) || math.IsInf(float64(logits[1]), -1) || math.IsInf(float64(logits[2]), -1) {
		t.Fatalf("the least likely token goes: %v", logits)
	}
	best := []float32{0, 1, 2}
	applyTopP(best, 0.01)
	if !math.IsInf(float64(best[0]), -1) || !math.IsInf(float64(best[1]), -1) || math.IsInf(float64(best[2]), -1) {
		t.Fatalf("at least the most likely token stays: %v", best)
	}
	for _, p := range []float64{0, 1, 1.5} {
		same := []float32{1, 2}
		applyTopP(same, p)
		if !slices.Equal(same, []float32{1, 2}) {
			t.Fatalf("p = %v keeps everything: %v", p, same)
		}
	}
	empty := []float32{inf, inf}
	applyTopP(empty, 0.5)
	if !slices.Equal(empty, []float32{inf, inf}) {
		t.Fatalf("nothing to keep: %v", empty)
	}
}

func TestSampleFrom(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	logits := []float32{0, float32(math.Log(3)), inf}
	counts := make([]int, 3)
	const draws = 40000
	for i := 0; i < draws; i++ {
		counts[sampleFrom(logits, rng)]++
	}
	if counts[2] != 0 || math.Abs(float64(counts[1])/draws-0.75) > 0.01 {
		t.Fatalf("draws %v: token 1 should come up three times in four and token 2 never", counts)
	}
	if got := sampleFrom([]float32{inf, 3, inf}, rng); got != 1 {
		t.Fatalf("a single candidate is always chosen: %d", got)
	}
}

func TestChooseToken(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	if got := chooseToken([]float32{1, 9, 2}, nil, GenerateOptions{}, rng); got != 1 {
		t.Fatalf("without a temperature the best token wins: %d", got)
	}
	if got := chooseToken([]float32{4, 3, 1}, []int{0}, GenerateOptions{RepetitionPenalty: 2}, rng); got != 1 {
		t.Fatalf("a repeated token is demoted even when decoding greedily: %d", got)
	}
	opts := GenerateOptions{Temperature: 0.8, TopK: 1}
	for i := 0; i < 50; i++ {
		if got := chooseToken([]float32{1, 9, 2}, nil, opts, rng); got != 1 {
			t.Fatalf("top-k 1 always picks the best: %d", got)
		}
	}
}

func lmWithConfig(t *testing.T, config, generation string) *LanguageModel {
	t.Helper()
	dir := lmDir(t, true)
	if config != "" {
		fixture.Must(os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644))
	}
	if generation != "" {
		fixture.Must(os.WriteFile(filepath.Join(dir, "generation_config.json"), []byte(generation), 0o644))
	}
	m, err := LoadLanguageModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestGenerationDefaults(t *testing.T) {
	const base = `{"num_hidden_layers":1,"num_attention_heads":1,"hidden_size":8,"max_position_embeddings":64,`
	d := lmWithConfig(t, base+`"eos_token_id":7}`, "").Defaults()
	if !slices.Equal(d.StopTokens, []int{7}) || d.Temperature != 0 {
		t.Fatalf("without a generation config: %+v", d)
	}
	d = lmWithConfig(t, base+`"eos_token_id":7}`, `{"do_sample":true,"temperature":0.7,"top_k":20,"top_p":0.8,"repetition_penalty":1.1,"eos_token_id":[5,6]}`).Defaults()
	if !slices.Equal(d.StopTokens, []int{5, 6}) || d.Temperature != 0.7 || d.TopK != 20 || d.TopP != 0.8 || d.RepetitionPenalty != 1.1 {
		t.Fatalf("sampling settings: %+v", d)
	}
	d = lmWithConfig(t, "", `{"do_sample":false,"temperature":0.7,"top_k":20,"repetition_penalty":1.1}`).Defaults()
	if d.Temperature != 0 || d.TopK != 0 || d.RepetitionPenalty != 1.1 {
		t.Fatalf("without do_sample only the repetition penalty applies: %+v", d)
	}
	if got := jsonInts([]byte(`"x"`)); got != nil {
		t.Fatalf("not numbers: %v", got)
	}
}

func TestGenerateFollowsTheModelAndStops(t *testing.T) {
	m := lmWithConfig(t, "", "")
	for _, tc := range []struct {
		name   string
		opts   GenerateOptions
		prompt []int
		want   []int
	}{
		{"until the stop token", GenerateOptions{StopTokens: []int{7}}, []int{0}, []int{1, 2, 3, 4, 5, 6}},
		{"at the length limit", GenerateOptions{MaxNewTokens: 3}, []int{0}, []int{1, 2, 3}},
		{"around the chain", GenerateOptions{MaxNewTokens: 10}, []int{6}, []int{7, 0, 1, 2, 3, 4, 5, 6, 7, 0}},
		{"sampling with one candidate", GenerateOptions{Temperature: 1, TopK: 1, MaxNewTokens: 3}, []int{2}, []int{3, 4, 5}},
		{"a stop token right away", GenerateOptions{StopTokens: []int{1}}, []int{0}, nil},
	} {
		got, err := m.Generate(tc.prompt, tc.opts, nil)
		if err != nil || !slices.Equal(got, tc.want) {
			t.Errorf("%s: %v %v, want %v", tc.name, got, err, tc.want)
		}
	}
	var seen []int
	got, err := m.Generate([]int{0}, GenerateOptions{MaxNewTokens: 6}, func(token int) bool {
		seen = append(seen, token)
		return len(seen) < 2
	})
	if err != nil || !slices.Equal(got, []int{1, 2}) || !slices.Equal(seen, []int{1, 2}) {
		t.Fatalf("the callback ends the generation: %v %v %v", got, seen, err)
	}
	a, _ := m.Generate([]int{0}, GenerateOptions{Temperature: 5, MaxNewTokens: 8, Seed: 3}, nil)
	b, _ := m.Generate([]int{0}, GenerateOptions{Temperature: 5, MaxNewTokens: 8, Seed: 3}, nil)
	if !slices.Equal(a, b) {
		t.Fatalf("the same seed gives the same text: %v %v", a, b)
	}
}

func TestGenerateErrorsAndLimits(t *testing.T) {
	m := lmWithConfig(t, "", "")
	if _, err := m.Generate(nil, GenerateOptions{}, nil); err == nil {
		t.Fatal("an empty prompt must be rejected")
	}
	m.maxPositions = 5
	got, err := m.Generate([]int{0, 1, 2}, GenerateOptions{MaxNewTokens: 10}, nil)
	if err != nil || len(got) != 2 {
		t.Fatalf("the context window limits the reply to what fits: %v %v", got, err)
	}
	if _, err := m.Generate([]int{0, 1, 2, 3, 4}, GenerateOptions{}, nil); err == nil {
		t.Fatal("a prompt that fills the window leaves no room to answer")
	}
	m.maxPositions = 0
	broken, _ := NewSession(&Graph{Inputs: []string{"input_ids"}, Outputs: []string{"logits"}})
	saved := m.session
	m.session = broken
	if _, err := m.Generate([]int{0}, GenerateOptions{}, nil); err == nil {
		t.Fatal("a failing prompt pass must be reported")
	}
	m.session = saved
	got, err = m.Generate([]int{0}, GenerateOptions{MaxNewTokens: 5}, func(int) bool {
		m.session = broken
		return true
	})
	m.session = saved
	if err == nil || len(got) != 1 {
		t.Fatalf("a failing step reports the error and returns what was generated: %v %v", got, err)
	}
}
