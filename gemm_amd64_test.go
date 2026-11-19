//go:build amd64

package onnxnlp

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestGemmScalarFallback(t *testing.T) {
	saved := useAVX2
	useAVX2 = false
	defer func() { useAVX2 = saved }()
	checkGemmAgainstNaive(t)
}

func TestGemmAVX2Kernel(t *testing.T) {
	if !useAVX2 {
		t.Skip("CPU has no AVX2 and FMA")
	}
	checkGemmAgainstNaive(t)
}

func TestGemmWithEachKernel(t *testing.T) {
	defer selectKernel()
	for _, mode := range []string{"", "off", "on"} {
		t.Setenv("ONNXNLP_AVX512", mode)
		selectKernel()
		want := 6
		if mode == "on" && useAVX512 && useAVX2 {
			want = 12
		}
		if tileRows != want {
			t.Fatalf("ONNXNLP_AVX512=%q: tile height %d, want %d", mode, tileRows, want)
		}
		if rowBlock%tileRows != 0 {
			t.Fatalf("row block %d is not a multiple of the tile height %d", rowBlock, tileRows)
		}
		checkGemmAgainstNaive(t)
	}
}

func TestAVX512KernelOnWideTiles(t *testing.T) {
	if !useAVX512 || !useAVX2 {
		t.Skip("CPU has no AVX-512")
	}
	defer selectKernel()
	t.Setenv("ONNXNLP_AVX512", "on")
	selectKernel()
	rng := rand.New(rand.NewPCG(121, 122))
	for _, sz := range [][3]int{{12, 7, 16}, {13, 33, 17}, {24, 64, 32}, {25, 5, 48}, {1, 9, 16}, {11, 100, 15}} {
		m, k, n := sz[0], sz[1], sz[2]
		a, b := randomSlice(rng, m*k), randomSlice(rng, k*n)
		want := naiveGemm(a, b, m, k, n)
		out := make([]float32, m*n)
		runGemms([]gemmJob{{out: out, a: a, b: b, m: m, k: k, n: n}})
		for i := range want {
			if math.Abs(float64(out[i]-want[i])) > 1e-4*(1+math.Abs(float64(want[i]))) {
				t.Fatalf("%v element %d: got %v, want %v", sz, i, out[i], want[i])
			}
		}
	}
}
