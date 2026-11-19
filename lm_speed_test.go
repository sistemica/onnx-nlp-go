package onnxnlp

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
	"time"
)

// decodeSteps times n single-token steps.
func decodeSteps(t *testing.T, m *LanguageModel, cache *Cache, n int) time.Duration {
	t.Helper()
	start := time.Now()
	for i := 0; i < n; i++ {
		if _, err := m.Forward([]int{271}, cache); err != nil {
			t.Fatal(err)
		}
	}
	return time.Since(start) / time.Duration(n)
}

func medianOf(d []time.Duration) time.Duration {
	slices.Sort(d)
	return d[len(d)/2]
}

// TestLanguageModelDecodeSpeed alternates variants inside one process, so that interference from
// other work on the machine hits all of them alike, and reports the median time per token.
func TestLanguageModelDecodeSpeed(t *testing.T) {
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
	m, err := LoadLanguageModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	cache := m.NewCache()
	if _, err := m.Forward(prompt.IDs, cache); err != nil {
		t.Fatal(err)
	}
	decodeSteps(t, m, cache, 4)
	variants := []struct {
		name string
		set  func()
	}{
		{"matrix kernel (before)", func() { singleRowKernel, bf16Weights = false, false }},
		{"vector kernel, float32", func() { singleRowKernel, bf16Weights = true, false }},
		{"vector kernel, bfloat16", func() { singleRowKernel, bf16Weights = true, true }},
	}
	for _, v := range variants {
		v.set()
		decodeSteps(t, m, cache, 2)
	}
	times := make([][]time.Duration, len(variants))
	for round := 0; round < 7; round++ {
		for i, v := range variants {
			v.set()
			times[i] = append(times[i], decodeSteps(t, m, cache, 6))
		}
	}
	singleRowKernel, bf16Weights = true, true
	cpuBefore, wallBefore := processCPUTime(), time.Now()
	decodeSteps(t, m, cache, 60)
	t.Logf("cores kept busy during decoding: %.1f of %d logical CPUs", float64(processCPUTime()-cpuBefore)/float64(time.Since(wallBefore)), runtime.NumCPU())
	base := medianOf(append([]time.Duration(nil), times[0]...))
	for i, v := range variants {
		med := medianOf(times[i])
		t.Logf("%-26s %7.1f ms per token (%.1fx)", v.name, float64(med.Microseconds())/1000, float64(base)/float64(med))
	}
}

// TestLanguageModelPrefillSpeed times processing a prompt once the weights are packed.
func TestLanguageModelPrefillSpeed(t *testing.T) {
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
	m, err := LoadLanguageModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	first := time.Now()
	if _, err := m.Forward(prompt.IDs, m.NewCache()); err != nil {
		t.Fatal(err)
	}
	t.Logf("first prompt, which also packs the weights: %v", time.Since(first).Round(time.Millisecond))
	for _, repeat := range []int{1, 4} {
		var ids []int
		for i := 0; i < repeat; i++ {
			ids = append(ids, prompt.IDs...)
		}
		var times []time.Duration
		cpuBefore, wallBefore := processCPUTime(), time.Now()
		for round := 0; round < 5; round++ {
			start := time.Now()
			if _, err := m.Forward(ids, m.NewCache()); err != nil {
				t.Fatal(err)
			}
			times = append(times, time.Since(start))
		}
		med := medianOf(times)
		t.Logf("prompt of %3d tokens: %6.0f ms (%.0f tokens/s); %.1f cores busy of %d", len(ids), float64(med.Microseconds())/1000,
			float64(len(ids))/med.Seconds(), float64(processCPUTime()-cpuBefore)/float64(time.Since(wallBefore)), runtime.NumCPU())
	}
}
