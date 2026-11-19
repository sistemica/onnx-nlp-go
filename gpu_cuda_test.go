//go:build linux && (amd64 || arm64)

package onnxgo

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

func requireCUDA(t testing.TB) *cudaDevice {
	t.Helper()
	dev, _ := activeDevice()
	cd, ok := dev.(*cudaDevice)
	if !ok {
		t.Skip("no CUDA device available")
	}
	return cd
}

func TestCUDAGemmMatchesReference(t *testing.T) {
	dev := requireCUDA(t)
	t.Logf("device: %s", dev.Name())
	rng := rand.New(rand.NewPCG(11, 12))
	for _, sz := range [][3]int{{1, 1, 1}, {5, 7, 3}, {37, 768, 3072}, {128, 768, 768}, {120, 64, 120}, {513, 300, 257}} {
		m, k, n := sz[0], sz[1], sz[2]
		a, b := randomSlice(rng, m*k), randomSlice(rng, k*n)
		want := naiveGemm(a, b, m, k, n)
		for _, static := range []bool{false, true, true} {
			out := make([]float32, m*n)
			if err := dev.Gemm(gemmJob{out: out, a: a, b: b, m: m, k: k, n: n, bStatic: static}); err != nil {
				t.Fatalf("%v: %v", sz, err)
			}
			var worst float64
			for i := range want {
				worst = max(worst, math.Abs(float64(out[i]-want[i]))/(1+math.Abs(float64(want[i]))))
			}
			if worst > 1e-4 {
				t.Errorf("%v static=%v: relative error %g", sz, static, worst)
			}
		}
	}
	if len(dev.weights) == 0 {
		t.Error("static weights were not cached on the device")
	}
}

func TestCUDAOffloadThroughMatMul(t *testing.T) {
	requireCUDA(t)
	opened, err := openCUDA()
	if err != nil {
		t.Skipf("cannot open a second CUDA device: %v", err)
	}
	dev := opened.(*cudaDevice)
	t.Cleanup(dev.Release)
	installDevice(t, dev, 1)
	rng := rand.New(rand.NewPCG(13, 14))
	m, k, n := 64, 512, 512
	x := NewFloat([]int{2, m, k}, randomSlice(rng, 2*m*k))
	w := NewFloat([]int{k, n}, randomSlice(rng, k*n))
	w.static = true
	cached := len(dev.weights)
	got, err := opMatMul(&opCtx{in: []*Tensor{x, w}})
	if err != nil {
		t.Fatal(err)
	}
	if len(dev.weights) != cached+1 {
		t.Errorf("weight cache grew by %d, want 1", len(dev.weights)-cached)
	}
	for bi := 0; bi < 2; bi++ {
		want := naiveGemm(x.F[bi*m*k:(bi+1)*m*k], w.F, m, k, n)
		for i, v := range want {
			if math.Abs(float64(got[0].F[bi*m*n+i]-v)) > 1e-3*(1+math.Abs(float64(v))) {
				t.Fatalf("batch %d element %d: got %g, want %g", bi, i, got[0].F[bi*m*n+i], v)
			}
		}
	}
}

func BenchmarkGemmGPUvsCPU(b *testing.B) {
	dev := requireCUDA(b)
	rng := rand.New(rand.NewPCG(15, 16))
	for _, sz := range [][3]int{{37, 768, 768}, {120, 768, 768}, {120, 768, 3072}, {512, 768, 768}, {1024, 1024, 1024}, {4096, 1024, 1024}} {
		m, k, n := sz[0], sz[1], sz[2]
		a, w := randomSlice(rng, m*k), randomSlice(rng, k*n)
		flops := 2 * float64(m) * float64(k) * float64(n)
		b.Run(fmt.Sprintf("gpu/%dx%dx%d", m, k, n), func(b *testing.B) {
			out := make([]float32, m*n)
			job := gemmJob{out: out, a: a, b: w, m: m, k: k, n: n, bStatic: true}
			for i := 0; i < b.N; i++ {
				if err := dev.Gemm(job); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
		b.Run(fmt.Sprintf("cpu/%dx%dx%d", m, k, n), func(b *testing.B) {
			out := make([]float32, m*n)
			for i := 0; i < b.N; i++ {
				clear(out)
				runGemmsCPU([]gemmJob{{out: out, a: a, b: w, m: m, k: k, n: n}})
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

func TestDlopenAndRegisterFunc(t *testing.T) {
	if _, _, err := dlopenFirst([]string{"libdoesnotexist.so.99"}); err == nil {
		t.Fatal("a missing library must fail")
	}
	lib, name, err := dlopenFirst([]string{"libdoesnotexist.so.99", "libc.so.6"})
	if err != nil || name != "libc.so.6" {
		t.Fatalf("fallback to the second candidate: %q %v", name, err)
	}
	var getpid func() int32
	if err := registerFunc(&getpid, lib, "getpid"); err != nil || getpid() <= 0 {
		t.Fatalf("register getpid: %v", err)
	}
	var missing func() int32
	if err := registerFunc(&missing, lib, "onnxgo_no_such_symbol"); err == nil {
		t.Fatal("a missing symbol must fail")
	}
}

func TestLoadCUDAReportsMissingLibraries(t *testing.T) {
	if _, err := loadCUDAFrom([]string{"libdoesnotexist.so.99"}, []string{"libc.so.6"}); err == nil || !strings.Contains(err.Error(), "cuda driver") {
		t.Fatalf("missing driver: %v", err)
	}
	if _, err := loadCUDAFrom([]string{"libc.so.6"}, []string{"libdoesnotexist.so.99"}); err == nil || !strings.Contains(err.Error(), "cublas") {
		t.Fatalf("missing cublas: %v", err)
	}
	if _, err := loadCUDAFrom([]string{"libc.so.6"}, []string{"libc.so.6"}); err == nil || !strings.Contains(err.Error(), "symbol ") {
		t.Fatalf("libraries without the CUDA symbols must fail naming a symbol: %v", err)
	}
}

func TestCUDAOpenReleaseAndSelection(t *testing.T) {
	requireCUDA(t)
	t.Setenv("ONNXGO_GPU_DEVICE", "9999")
	if _, err := openCUDA(); err == nil {
		t.Fatal("a nonexistent device index must fail")
	}
	t.Setenv("ONNXGO_GPU_DEVICE", "0")
	dev, err := openCUDA()
	if err != nil {
		t.Skipf("device 0 is not usable here: %v", err)
	}
	if dev.Name() == "" {
		t.Error("device has no name")
	}
	rng := rand.New(rand.NewPCG(41, 42))
	a, b := randomSlice(rng, 20*30), randomSlice(rng, 30*40)
	out := make([]float32, 20*40)
	job := gemmJob{out: out, a: a, b: b, m: 20, k: 30, n: 40, bStatic: true}
	if err := dev.Gemm(job); err != nil {
		t.Fatal(err)
	}
	bigger := gemmJob{out: make([]float32, 40*40), a: randomSlice(rng, 40*30), b: b, m: 40, k: 30, n: 40}
	if err := dev.Gemm(bigger); err != nil {
		t.Fatalf("growing the scratch buffers: %v", err)
	}
	dev.Release()
}

func TestCUDAAllocationFailureIsReported(t *testing.T) {
	dev := requireCUDA(t)
	if _, err := dev.scratchBuffer(0, 1<<60); err == nil {
		t.Fatal("an impossible allocation must fail")
	}
}
