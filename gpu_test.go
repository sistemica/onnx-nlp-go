package onnxnlp

import (
	"errors"
	"math/rand/v2"
	"testing"
)

type fakeDevice struct {
	calls    []gemmJob
	fail     bool
	released int
}

func (f *fakeDevice) Name() string { return "fake" }

func (f *fakeDevice) Gemm(j gemmJob) error {
	f.calls = append(f.calls, j)
	if f.fail {
		return errors.New("device lost")
	}
	copy(j.out, naiveGemm(j.a, j.b, j.m, j.k, j.n))
	return nil
}

func (f *fakeDevice) Release() { f.released++ }

// isolateGPUEnv hides the ambient GPU settings from tests that use fake devices.
func isolateGPUEnv(t *testing.T) {
	t.Helper()
	for _, name := range []string{"ONNXNLP_GPU", "ONNXNLP_GPU_BACKEND", "ONNXNLP_GPU_DEVICE", "ONNXNLP_GPU_MIN_FLOPS", "ONNXNLP_VULKAN_ALLOW_CPU"} {
		t.Setenv(name, "")
	}
}

func installDevice(t *testing.T, dev gemmDevice, minFlops int) {
	t.Helper()
	accel.mu.Lock()
	saved := struct {
		probed   bool
		dev      gemmDevice
		minFlops int
	}{accel.probed, accel.dev, accel.minFlops}
	accel.probed, accel.dev, accel.minFlops, accel.failures = true, dev, minFlops, 0
	accel.mu.Unlock()
	t.Cleanup(func() {
		accel.mu.Lock()
		accel.probed, accel.dev, accel.minFlops = saved.probed, saved.dev, saved.minFlops
		accel.mu.Unlock()
	})
}

func matmulTensors(m, k, n int) (*Tensor, *Tensor) {
	rng := rand.New(rand.NewPCG(31, 32))
	return NewFloat([]int{m, k}, randomSlice(rng, m*k)), NewFloat([]int{k, n}, randomSlice(rng, k*n))
}

func TestOffloadSendsLargeProductsToTheDevice(t *testing.T) {
	isolateGPUEnv(t)
	dev := &fakeDevice{}
	installDevice(t, dev, 1000)
	x, w := matmulTensors(8, 16, 8)
	w.static = true
	got := run1(t, "MatMul", nil, x, w)
	if len(dev.calls) != 1 || !dev.calls[0].bStatic {
		t.Fatalf("device calls %+v", dev.calls)
	}
	want := naiveGemm(x.F, w.F, 8, 16, 8)
	for i := range want {
		if d := got.F[i] - want[i]; d > 1e-4 || d < -1e-4 {
			t.Fatalf("element %d: %v vs %v", i, got.F[i], want[i])
		}
	}
}

func TestOffloadKeepsSmallProductsOnTheCPU(t *testing.T) {
	isolateGPUEnv(t)
	dev := &fakeDevice{}
	installDevice(t, dev, 1<<30)
	x, w := matmulTensors(8, 16, 8)
	run1(t, "MatMul", nil, x, w)
	if len(dev.calls) != 0 {
		t.Fatalf("small product reached the device: %+v", dev.calls)
	}
	installDevice(t, dev, 1)
	if rest := offloadGemms([]gemmJob{{m: 0, k: 1, n: 1}, {m: 1, k: 0, n: 1}, {m: 1, k: 1, n: 0}}); len(rest) != 3 {
		t.Fatalf("empty products must stay on the CPU: %d", len(rest))
	}
}

func TestOffloadFallsBackWhenTheDeviceFails(t *testing.T) {
	isolateGPUEnv(t)
	dev := &fakeDevice{fail: true}
	installDevice(t, dev, 1)
	x, w := matmulTensors(8, 16, 8)
	want := naiveGemm(x.F, w.F, 8, 16, 8)
	for round := 1; round <= maxDeviceFailures; round++ {
		got := run1(t, "MatMul", nil, x, w)
		for i := range want {
			if d := got.F[i] - want[i]; d > 1e-4 || d < -1e-4 {
				t.Fatalf("round %d: fallback result wrong at %d", round, i)
			}
		}
		if round < maxDeviceFailures && (dev.released != 0 || GPUName() == "") {
			t.Fatalf("device retired after only %d failures", round)
		}
	}
	if dev.released != 1 || GPUName() != "" {
		t.Fatalf("repeated failures must retire the device: released=%d name=%q", dev.released, GPUName())
	}
	run1(t, "MatMul", nil, x, w)
	if len(dev.calls) != maxDeviceFailures {
		t.Fatalf("a released device was used again: %d calls", len(dev.calls))
	}
}

func TestOccasionalFailureDoesNotRetireTheDevice(t *testing.T) {
	isolateGPUEnv(t)
	dev := &fakeDevice{}
	installDevice(t, dev, 1)
	x, w := matmulTensors(8, 16, 8)
	for round := 0; round < 2*maxDeviceFailures; round++ {
		dev.fail = round%2 == 0
		run1(t, "MatMul", nil, x, w)
	}
	if dev.released != 0 || GPUName() != "fake" {
		t.Fatalf("intermittent failures retired the device: released=%d", dev.released)
	}
}

func TestFailureOfAnotherDeviceIsIgnored(t *testing.T) {
	dev := &fakeDevice{}
	installDevice(t, dev, 1)
	for i := 0; i < 2*maxDeviceFailures; i++ {
		noteDeviceFailure(&fakeDevice{})
	}
	if dev.released != 0 || GPUName() != "fake" {
		t.Fatal("failures reported for a different device must not retire the active one")
	}
}

func TestGPUNameAndRelease(t *testing.T) {
	dev := &fakeDevice{}
	installDevice(t, dev, 1)
	if GPUName() != "fake" {
		t.Fatalf("name %q", GPUName())
	}
	ReleaseGPU()
	ReleaseGPU()
	if dev.released != 1 || GPUName() != "" {
		t.Fatalf("released=%d name=%q", dev.released, GPUName())
	}
}

func TestDeviceProbing(t *testing.T) {
	saved := backends
	t.Cleanup(func() { backends = saved })
	reset := func() {
		installDevice(t, nil, 0)
		accel.mu.Lock()
		accel.probed = false
		accel.mu.Unlock()
	}

	dev, other := &fakeDevice{}, &fakeDevice{}
	backends = []backend{
		{"broken", func() (gemmDevice, error) { return nil, errors.New("no gpu") }},
		{"first", func() (gemmDevice, error) { return dev, nil }},
		{"second", func() (gemmDevice, error) { return other, nil }},
	}

	reset()
	t.Setenv("ONNXNLP_GPU", "")
	t.Setenv("ONNXNLP_GPU_BACKEND", "")
	t.Setenv("ONNXNLP_GPU_MIN_FLOPS", "77")
	if d, min := activeDevice(); d != dev || min != 77 {
		t.Fatalf("probe should skip the broken backend and take the first working one: %v %d", d, min)
	}

	reset()
	t.Setenv("ONNXNLP_GPU_BACKEND", "second")
	if d, _ := activeDevice(); d != other {
		t.Fatalf("backend selection ignored: %v", d)
	}

	reset()
	t.Setenv("ONNXNLP_GPU_BACKEND", "missing")
	if d, _ := activeDevice(); d != nil {
		t.Fatalf("unknown backend must leave the CPU path: %v", d)
	}

	t.Setenv("ONNXNLP_GPU_BACKEND", "")
	for _, off := range []string{"off", "0", "false", "none"} {
		reset()
		t.Setenv("ONNXNLP_GPU", off)
		if d, min := activeDevice(); d != nil || min != 77 {
			t.Fatalf("ONNXNLP_GPU=%s: %v %d", off, d, min)
		}
	}

	reset()
	t.Setenv("ONNXNLP_GPU", "")
	t.Setenv("ONNXNLP_GPU_MIN_FLOPS", "junk")
	backends = backends[:1]
	if d, min := activeDevice(); d != nil || min != defaultGPUMinFlops {
		t.Fatalf("failing probe: %v %d", d, min)
	}
}
