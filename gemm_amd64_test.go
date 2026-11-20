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

func TestGemv16MatchesReference(t *testing.T) {
	if !useAVX2 {
		t.Skip("needs AVX2")
	}
	rng := rand.New(rand.NewPCG(8, 9))
	for _, k := range []int{1, 2, 3, 4, 5, 7, 8, 9, 31, 64, 100, 897} {
		x, panel := randomSlice(rng, k), randomSlice(rng, k*16)
		y := randomSlice(rng, 16)
		want := naiveGemm(x, panel, 1, k, 16)
		for i := range want {
			want[i] += y[i]
		}
		gemv16(&x[0], &panel[0], &y[0], k)
		closeSlices(t, y, want)
	}
}

func TestSingleRowProductsMatchTheReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(10, 11))
	for _, shape := range [][2]int{{1, 1}, {7, 15}, {16, 16}, {33, 17}, {896, 100}, {4864, 896}} {
		k, n := shape[0], shape[1]
		a, b := randomSlice(rng, k), randomSlice(rng, k*n)
		want := naiveGemm(a, b, 1, k, n)
		for _, static := range []bool{false, true} {
			out := make([]float32, n)
			job := gemmJob{out: out, a: a, b: b, m: 1, k: k, n: n, bStatic: static}
			if static {
				job.pack = &packCache{}
			}
			runGemmsCPU([]gemmJob{job})
			closeSlices(t, out, want)
		}
	}
	// A product that already holds a partial result is accumulated into, not overwritten.
	a, b := randomSlice(rng, 5), randomSlice(rng, 5*20)
	out := make([]float32, 20)
	for i := range out {
		out[i] = 3
	}
	runGemmsCPU([]gemmJob{{out: out, a: a, b: b, m: 1, k: 5, n: 20}})
	want := naiveGemm(a, b, 1, 5, 20)
	for i := range want {
		want[i] += 3
	}
	closeSlices(t, out, want)
}

// bf16Values returns n values that are exactly representable in bfloat16.
func bf16Values(rng *rand.Rand, n int) []float32 {
	v := randomSlice(rng, n)
	for i := range v {
		v[i] = math.Float32frombits(math.Float32bits(v[i]) &^ 0xffff)
	}
	return v
}

func TestGemv16bf16MatchesReference(t *testing.T) {
	if !useAVX2 {
		t.Skip("needs AVX2")
	}
	rng := rand.New(rand.NewPCG(12, 13))
	for _, k := range []int{1, 2, 3, 4, 5, 8, 9, 31, 100, 897} {
		x, panel := randomSlice(rng, k), bf16Values(rng, k*16)
		bits := make([]uint16, len(panel))
		for i, v := range panel {
			bits[i] = uint16(math.Float32bits(v) >> 16)
		}
		y := randomSlice(rng, 16)
		want := naiveGemm(x, panel, 1, k, 16)
		for i := range want {
			want[i] += y[i]
		}
		gemv16bf16(&x[0], &bits[0], &y[0], k)
		closeSlices(t, y, want)
	}
}

func TestSingleRowProductsUseBFloat16WhenExact(t *testing.T) {
	rng := rand.New(rand.NewPCG(14, 15))
	for _, shape := range [][2]int{{1, 1}, {7, 15}, {33, 17}, {896, 100}, {1000, 4864}} {
		k, n := shape[0], shape[1]
		a, b := randomSlice(rng, k), bf16Values(rng, k*n)
		want := naiveGemm(a, b, 1, k, n)
		pack := &packCache{}
		out := make([]float32, n)
		runGemmsCPU([]gemmJob{{out: out, a: a, b: b, m: 1, k: k, n: n, bStatic: true, pack: pack}})
		closeSlices(t, out, want)
		if pack.data16 == nil || pack.data != nil {
			t.Fatalf("%dx%d: an exact matrix must be read as bfloat16 without building float32 panels", k, n)
		}
	}
	// Weights with bits below the top sixteen are never narrowed: the result must not change.
	k, n := 64, 40
	a, b := randomSlice(rng, k), randomSlice(rng, k*n)
	pack := &packCache{}
	out := make([]float32, n)
	runGemmsCPU([]gemmJob{{out: out, a: a, b: b, m: 1, k: k, n: n, bStatic: true, pack: pack}})
	closeSlices(t, out, naiveGemm(a, b, 1, k, n))
	if pack.data16 != nil || pack.data == nil {
		t.Fatal("a matrix that is not exactly bfloat16 must stay float32")
	}
	saved := bf16Weights
	bf16Weights = false
	defer func() { bf16Weights = saved }()
	exact := bf16Values(rng, k*n)
	pack = &packCache{}
	runGemmsCPU([]gemmJob{{out: make([]float32, n), a: a, b: exact, m: 1, k: k, n: n, bStatic: true, pack: pack}})
	if pack.data16 != nil {
		t.Fatal("the switch must turn the bfloat16 path off")
	}
}
