package onnxnlp

import (
	"math/rand/v2"
	"testing"
	"unsafe"
)

// fakeBLAS implements cblas_sgemm for row-major, untransposed matrices and records how it was called.
type fakeBLAS struct {
	calls          int
	order, trans   int32
	lda, ldb, ldc  int32
	alpha, beta    float32
	sawUnsupported bool
}

func (f *fakeBLAS) api() *accelerateAPI {
	return &accelerateAPI{sgemm: func(order, transA, transB, m, n, k int32, alpha float32, a unsafe.Pointer, lda int32, b unsafe.Pointer, ldb int32, beta float32, c unsafe.Pointer, ldc int32) {
		f.calls++
		f.order, f.trans, f.lda, f.ldb, f.ldc, f.alpha, f.beta = order, transA, lda, ldb, ldc, alpha, beta
		if order != cblasRowMajor || transA != cblasNoTrans || transB != cblasNoTrans {
			f.sawUnsupported = true
			return
		}
		A := unsafe.Slice((*float32)(a), int(m)*int(lda))
		B := unsafe.Slice((*float32)(b), int(k)*int(ldb))
		C := unsafe.Slice((*float32)(c), int(m)*int(ldc))
		for i := 0; i < int(m); i++ {
			for j := 0; j < int(n); j++ {
				var s float32
				for p := 0; p < int(k); p++ {
					s += A[i*int(lda)+p] * B[p*int(ldb)+j]
				}
				C[i*int(ldc)+j] = alpha*s + beta*C[i*int(ldc)+j]
				if beta == 0 {
					C[i*int(ldc)+j] = alpha * s
				}
			}
		}
	}}
}

func TestAccelerateGemmCallsCBLASRowMajor(t *testing.T) {
	blas := &fakeBLAS{}
	dev, err := newAccelerateDevice(blas.api())
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	for _, shape := range [][3]int{{1, 1, 1}, {3, 5, 7}, {16, 4, 1}, {2, 33, 9}} {
		m, k, n := shape[0], shape[1], shape[2]
		a, b := make([]float32, m*k), make([]float32, k*n)
		for i := range a {
			a[i] = rng.Float32() - 0.5
		}
		for i := range b {
			b[i] = rng.Float32() - 0.5
		}
		out := make([]float32, m*n)
		for i := range out {
			out[i] = 1e9
		}
		if err := dev.Gemm(gemmJob{out: out, a: a, b: b, m: m, k: k, n: n}); err != nil {
			t.Fatal(err)
		}
		if blas.sawUnsupported || blas.lda != int32(k) || blas.ldb != int32(n) || blas.ldc != int32(n) || blas.alpha != 1 || blas.beta != 0 {
			t.Fatalf("%v: wrong call: %+v", shape, blas)
		}
		closeSlices(t, out, naiveGemm(a, b, m, k, n))
	}
	if blas.calls != 4 {
		t.Fatalf("%d calls", blas.calls)
	}
}

func TestAccelerateDeviceRejectsBadInput(t *testing.T) {
	if _, err := newAccelerateDevice(nil); err == nil {
		t.Fatal("a missing API must be reported")
	}
	if _, err := newAccelerateDevice(&accelerateAPI{}); err == nil {
		t.Fatal("a missing cblas_sgemm must be reported")
	}
	blas := &fakeBLAS{}
	dev, _ := newAccelerateDevice(blas.api())
	for name, j := range map[string]gemmJob{
		"short a":   {out: make([]float32, 4), a: make([]float32, 3), b: make([]float32, 4), m: 2, k: 2, n: 2},
		"short b":   {out: make([]float32, 4), a: make([]float32, 4), b: make([]float32, 3), m: 2, k: 2, n: 2},
		"short out": {out: make([]float32, 3), a: make([]float32, 4), b: make([]float32, 4), m: 2, k: 2, n: 2},
	} {
		if err := dev.Gemm(j); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	if blas.calls != 0 {
		t.Fatal("BLAS must not be called with undersized buffers")
	}
	dev.Release()
	if dev.Name() == "" || dev.MinFlops() <= 0 || dev.MinFlops() >= defaultGPUMinFlops {
		t.Fatalf("name %q, minimum %d", dev.Name(), dev.MinFlops())
	}
}

func TestAccelerateIsUsedThroughTheOffloadPath(t *testing.T) {
	isolateGPUEnv(t)
	blas := &fakeBLAS{}
	dev, _ := newAccelerateDevice(blas.api())
	installDevice(t, dev, dev.MinFlops())
	m, k, n := 64, 32, 16
	a, b := make([]float32, m*k), make([]float32, k*n)
	for i := range a {
		a[i] = float32(i%7) - 3
	}
	for i := range b {
		b[i] = float32(i%5) - 2
	}
	out := make([]float32, m*n)
	runGemms([]gemmJob{{out: out, a: a, b: b, m: m, k: k, n: n}})
	if blas.calls != 1 {
		t.Fatalf("a %dx%dx%d product is above the threshold and must go to Accelerate: %d calls", m, k, n, blas.calls)
	}
	closeSlices(t, out, naiveGemm(a, b, m, k, n))
	small := make([]float32, 4)
	runGemms([]gemmJob{{out: small, a: []float32{1, 2, 3, 4}, b: []float32{1, 0, 0, 1}, m: 2, k: 2, n: 2}})
	if blas.calls != 1 {
		t.Fatal("a tiny product stays on the CPU kernel")
	}
}

func TestPreferredBackendsComeFirstAndSetTheirOwnThreshold(t *testing.T) {
	isolateGPUEnv(t)
	saved, savedAccel := backends, struct {
		probed   bool
		dev      gemmDevice
		minFlops int
	}{accel.probed, accel.dev, accel.minFlops}
	t.Cleanup(func() {
		backends = saved
		accel.mu.Lock()
		accel.probed, accel.dev, accel.minFlops = savedAccel.probed, savedAccel.dev, savedAccel.minFlops
		accel.mu.Unlock()
	})
	backends = nil
	registerBackend("other", func() (gemmDevice, error) { return nil, errNoTemplate })
	blas := &fakeBLAS{}
	registerPreferredBackend("accelerate", func() (gemmDevice, error) { return newAccelerateDevice(blas.api()) })
	if len(backends) != 2 || backends[0].name != "accelerate" {
		t.Fatalf("the preferred backend is tried first: %+v", backends)
	}
	accel.mu.Lock()
	accel.probed, accel.dev = false, nil
	accel.mu.Unlock()
	dev, minFlops := activeDevice()
	if dev == nil || minFlops != accelerateMinFlops {
		t.Fatalf("device %v, threshold %d: the device's own threshold applies", dev, minFlops)
	}
	t.Setenv("ONNXNLP_GPU_MIN_FLOPS", "123")
	accel.mu.Lock()
	accel.probed, accel.dev = false, nil
	accel.mu.Unlock()
	if _, minFlops := activeDevice(); minFlops != 123 {
		t.Fatalf("an explicit setting wins over the device's: %d", minFlops)
	}
}
