package onnxgo

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type manifestEntry struct {
	Name  string `json:"name"`
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
	File  string `json:"file"`
}

type manifest struct {
	Inputs  []manifestEntry `json:"inputs"`
	Outputs []manifestEntry `json:"outputs"`
}

func loadRef(t *testing.T, dir string, e manifestEntry) *Tensor {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, e.File))
	if err != nil {
		t.Fatal(err)
	}
	n := numel(e.Shape)
	switch e.DType {
	case "float32":
		data := make([]float32, n)
		for i := range data {
			data[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		return NewFloat(e.Shape, data)
	case "int64":
		data := make([]int64, n)
		for i := range data {
			data[i] = int64(binary.LittleEndian.Uint64(b[8*i:]))
		}
		return NewInt(e.Shape, data)
	}
	out := zeros(Bool, e.Shape)
	for i := range out.I {
		out.I[i] = int64(b[i])
	}
	return out
}

func compareTensors(t *testing.T, name string, got, want *Tensor, tol float64) {
	t.Helper()
	if got.DType != want.DType {
		t.Errorf("%s: dtype %s, want %s", name, got.DType, want.DType)
		return
	}
	if len(got.Shape) != len(want.Shape) {
		t.Errorf("%s: shape %v, want %v", name, got.Shape, want.Shape)
		return
	}
	for i := range got.Shape {
		if got.Shape[i] != want.Shape[i] {
			t.Errorf("%s: shape %v, want %v", name, got.Shape, want.Shape)
			return
		}
	}
	if want.isFloat() {
		var worst, scale, maxDiff float64
		worstAt := 0
		for i := range want.F {
			d := math.Abs(float64(got.F[i]) - float64(want.F[i]))
			scale = max(scale, math.Abs(float64(want.F[i])))
			maxDiff = max(maxDiff, d)
			if d > tol*(1+math.Abs(float64(want.F[i]))) && d > worst {
				worst, worstAt = d, i
			}
		}
		t.Logf("%s %v: max abs diff %.3g, max |ref| %.3g", name, want.Shape, maxDiff, scale)
		if worst > 0 {
			t.Errorf("%s: max abs diff %g at %d (got %g, want %g)", name, worst, worstAt, got.F[worstAt], want.F[worstAt])
		}
		return
	}
	bad, ones := 0, 0
	for i := range want.I {
		ones += int(min(want.I[i], 1))
		if got.I[i] != want.I[i] {
			if bad == 0 {
				t.Errorf("%s: first mismatch at %d: got %d, want %d", name, i, got.I[i], want.I[i])
			}
			bad++
		}
	}
	t.Logf("%s %v %s: %d of %d differ, %d elements >= 1", name, want.Shape, want.DType, bad, len(want.I), ones)
	if bad > 0 {
		t.Errorf("%s: %d of %d elements differ", name, bad, len(want.I))
	}
}

func TestAgainstReference(t *testing.T) {
	models, ref := os.Getenv("ONNXGO_MODELS"), os.Getenv("ONNXGO_REF")
	if models == "" || ref == "" {
		t.Skip("ONNXGO_MODELS and ONNXGO_REF not set")
	}
	cases, _ := filepath.Glob(filepath.Join(ref, "*", "manifest.json"))
	sessions := map[string]*Session{}
	for _, mf := range cases {
		dir := filepath.Dir(mf)
		name := filepath.Base(dir)
		model, _, _ := strings.Cut(name, "_")
		t.Run(name, func(t *testing.T) {
			sess, ok := sessions[model]
			if !ok {
				var err error
				sess, err = Load(filepath.Join(models, "onnx", model+".onnx"))
				if err != nil {
					t.Fatal(err)
				}
				sessions[model] = sess
			}
			raw, err := os.ReadFile(mf)
			if err != nil {
				t.Fatal(err)
			}
			var m manifest
			if err := json.Unmarshal(raw, &m); err != nil {
				t.Fatal(err)
			}
			in := map[string]*Tensor{}
			for _, e := range m.Inputs {
				in[e.Name] = loadRef(t, dir, e)
			}
			got, err := sess.Run(in)
			if err != nil {
				t.Fatal(err)
			}
			for _, e := range m.Outputs {
				compareTensors(t, e.Name, got[e.Name], loadRef(t, dir, e), 1e-3)
			}
		})
	}
}
