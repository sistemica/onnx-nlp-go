package onnxnlp

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestLogitsProcessorsAgainstTransformers(t *testing.T) {
	path := os.Getenv("ONNXNLP_PROCESSORS_REF")
	if path == "" {
		t.Skip("ONNXNLP_PROCESSORS_REF not set")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Scores  []float32
		History []int
		Steps   map[string][]*float32
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	var keyed []struct {
		Settings map[string]float64 `json:"settings"`
	}
	json.Unmarshal(raw, &keyed)
	wrong := map[string]int{}
	for n, c := range cases {
		s := keyed[n].Settings
		logits := append([]float32(nil), c.Scores...)
		stages := []struct {
			name  string
			apply func()
		}{
			{"penalty", func() { applyRepetitionPenalty(logits, c.History, s["penalty"]) }},
			{"temperature", func() { applyTemperature(logits, s["temperature"]) }},
			{"top_k", func() { applyTopK(logits, int(s["top_k"])) }},
			{"top_p", func() { applyTopP(logits, s["top_p"]) }},
		}
		for _, stage := range stages {
			stage.apply()
			for i, want := range c.Steps[stage.name] {
				got := float64(logits[i])
				switch {
				case want == nil && !math.IsInf(got, -1), want != nil && (math.IsInf(got, 0) || math.Abs(got-float64(*want)) > 1e-5*(1+math.Abs(float64(*want)))):
					wrong[stage.name]++
				}
			}
		}
	}
	t.Logf("%d cases; scores that differ from transformers after each stage: %v", len(cases), wrong)
	if len(wrong) != 0 {
		t.Fatalf("processors disagree with transformers: %v", wrong)
	}
}
