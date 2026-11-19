package onnxnlp

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLanguageModelForwardAgainstPyTorch(t *testing.T) {
	dir, ref := os.Getenv("ONNXNLP_LM_MODEL"), os.Getenv("ONNXNLP_LM_REF")
	if dir == "" || ref == "" {
		t.Skip("ONNXNLP_LM_MODEL and ONNXNLP_LM_REF not set")
	}
	raw, err := os.ReadFile(filepath.Join(ref, "prompt.json"))
	if err != nil {
		t.Fatal(err)
	}
	var prompt struct {
		IDs []int `json:"ids"`
	}
	if err := json.Unmarshal(raw, &prompt); err != nil {
		t.Fatal(err)
	}
	want, shape := readNpy(t, filepath.Join(ref, "logits.npy"))
	m, err := LoadLanguageModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	cache := m.NewCache()
	start := time.Now()
	logits, err := m.Forward(prompt.IDs, cache)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("prefill of %d tokens: %v", len(prompt.IDs), time.Since(start).Round(time.Millisecond))
	if len(logits.F) != len(want) || cache.Len() != len(prompt.IDs) {
		t.Fatalf("logits %v, reference %v; cache holds %d", logits.Shape, shape, cache.Len())
	}
	worst, wrongTop := 0.0, 0
	vocab := shape[1]
	for p := 0; p < shape[0]; p++ {
		got, ref := logits.F[p*vocab:(p+1)*vocab], want[p*vocab:(p+1)*vocab]
		bestG, bestR := 0, 0
		for v := range got {
			worst = math.Max(worst, math.Abs(float64(got[v]-ref[v])))
			if got[v] > got[bestG] {
				bestG = v
			}
			if ref[v] > ref[bestR] {
				bestR = v
			}
		}
		if bestG != bestR {
			wrongTop++
		}
	}
	t.Logf("%d positions: largest logit difference %.3g; %d positions where the top token differs", shape[0], worst, wrongTop)
	if wrongTop != 0 || worst > 5e-2 {
		t.Fatalf("logits differ from PyTorch: %g, %d top tokens", worst, wrongTop)
	}
}

func TestLanguageModelGreedyDecodingAgainstPyTorch(t *testing.T) {
	dir, ref := os.Getenv("ONNXNLP_LM_MODEL"), os.Getenv("ONNXNLP_LM_REF")
	if dir == "" || ref == "" {
		t.Skip("ONNXNLP_LM_MODEL and ONNXNLP_LM_REF not set")
	}
	raw, err := os.ReadFile(filepath.Join(ref, "prompt.json"))
	if err != nil {
		t.Fatal(err)
	}
	var prompt struct {
		IDs    []int `json:"ids"`
		Greedy []int `json:"greedy_ids"`
	}
	if err := json.Unmarshal(raw, &prompt); err != nil {
		t.Fatal(err)
	}
	m, err := LoadLanguageModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	cache := m.NewCache()
	logits, err := m.Forward(prompt.IDs, cache)
	if err != nil {
		t.Fatal(err)
	}
	vocab := logits.Shape[2]
	next := argmax(logits.F[(len(prompt.IDs)-1)*vocab:])
	var got []int
	var stepTimes []time.Duration
	for len(got) < len(prompt.Greedy) {
		got = append(got, next)
		start := time.Now()
		logits, err = m.Forward([]int{next}, cache)
		if err != nil {
			t.Fatal(err)
		}
		stepTimes = append(stepTimes, time.Since(start))
		next = argmax(logits.F)
	}
	var total time.Duration
	for _, d := range stepTimes[2:] {
		total += d
	}
	per := total / time.Duration(len(stepTimes)-2)
	t.Logf("generated %d tokens; %v per token after warm-up (%.1f tokens/s); first steps %v", len(got), per.Round(time.Millisecond), 1/per.Seconds(), stepTimes[:2])
	firstDifference := -1
	for i := range got {
		if got[i] != prompt.Greedy[i] {
			firstDifference = i
			break
		}
	}
	// Feed PyTorch's own tokens: where the choice still differs, the margin shows whether the two
	// candidates were a near-tie that float rounding can flip.
	cache = m.NewCache()
	logits, _ = m.Forward(prompt.IDs, cache)
	row := logits.F[(len(prompt.IDs)-1)*vocab:]
	widest := 0.0
	for i, want := range prompt.Greedy {
		if top := argmax(row); top != want {
			margin := float64(row[top] - row[want])
			widest = math.Max(widest, margin)
			t.Logf("step %d with PyTorch's history: we pick %d, PyTorch picked %d; our logit margin between them is %.5f", i, top, want, margin)
		}
		if logits, err = m.Forward([]int{want}, cache); err != nil {
			t.Fatal(err)
		}
		row = logits.F
	}
	if widest > 0.02 {
		t.Fatalf("a disagreement with a margin of %.4f is more than float rounding", widest)
	}
	if firstDifference >= 0 {
		t.Logf("free-running greedy decoding first leaves PyTorch's path at token %d (see the margin above)", firstDifference)
	}
}
