//go:build linux && (amd64 || arm64)

package onnxnlp

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

func openRealVulkan(t testing.TB) *vulkanDevice {
	t.Helper()
	t.Setenv("ONNXNLP_GPU_BACKEND", "vulkan")
	dev, err := openVulkan()
	if err != nil {
		t.Skipf("no Vulkan device available: %v", err)
	}
	t.Cleanup(dev.Release)
	return dev.(*vulkanDevice)
}

func TestVulkanGemmMatchesReference(t *testing.T) {
	dev := openRealVulkan(t)
	t.Logf("device: %s", dev.Name())
	rng := rand.New(rand.NewPCG(61, 62))
	for _, sz := range [][3]int{{1, 1, 1}, {5, 7, 3}, {37, 768, 3072}, {128, 768, 768}, {120, 64, 120}, {513, 300, 257}, {64, 16, 64}, {65, 17, 65}} {
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
		t.Error("static weights were not cached")
	}
}

func BenchmarkGemmVulkanVsCPU(b *testing.B) {
	dev := openRealVulkan(b)
	rng := rand.New(rand.NewPCG(63, 64))
	for _, sz := range [][3]int{{37, 768, 768}, {120, 768, 768}, {120, 768, 3072}, {512, 768, 768}, {1024, 1024, 1024}, {4096, 1024, 1024}} {
		m, k, n := sz[0], sz[1], sz[2]
		a, w := randomSlice(rng, m*k), randomSlice(rng, k*n)
		flops := 2 * float64(m) * float64(k) * float64(n)
		b.Run(fmt.Sprintf("vulkan/%dx%dx%d", m, k, n), func(b *testing.B) {
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
