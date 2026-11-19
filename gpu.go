package onnxnlp

import (
	"os"
	"strconv"
	"sync"
)

// gemmDevice runs row-major matrix products out = a*b on an accelerator.
type gemmDevice interface {
	Name() string
	// Gemm overwrites j.out with j.a*j.b. j.b is cached on the device when j.bStatic is set.
	Gemm(j gemmJob) error
	Release()
}

const (
	defaultGPUMinFlops = 1 << 24
	maxDeviceFailures  = 3
)

type backend struct {
	name string
	open func() (gemmDevice, error)
}

// backends lists the accelerator backends in order of preference.
var backends []backend

func registerBackend(name string, open func() (gemmDevice, error)) {
	backends = append(backends, backend{name, open})
}

// registerPreferredBackend adds a backend that is tried before all others.
func registerPreferredBackend(name string, open func() (gemmDevice, error)) {
	backends = append([]backend{{name, open}}, backends...)
}

var accel struct {
	mu       sync.Mutex
	probed   bool
	dev      gemmDevice
	minFlops int
	failures int
}

func gpuDisabledByEnv() bool {
	switch os.Getenv("ONNXNLP_GPU") {
	case "off", "0", "false", "none":
		return true
	}
	return false
}

func activeDevice() (gemmDevice, int) {
	accel.mu.Lock()
	defer accel.mu.Unlock()
	if !accel.probed {
		accel.probed = true
		accel.minFlops = defaultGPUMinFlops
		if v, err := strconv.Atoi(os.Getenv("ONNXNLP_GPU_MIN_FLOPS")); err == nil {
			accel.minFlops = v
		}
		if !gpuDisabledByEnv() {
			accel.dev = openFirstBackend(os.Getenv("ONNXNLP_GPU_BACKEND"))
			if own, ok := accel.dev.(interface{ MinFlops() int }); ok && os.Getenv("ONNXNLP_GPU_MIN_FLOPS") == "" {
				accel.minFlops = own.MinFlops()
			}
		}
	}
	return accel.dev, accel.minFlops
}

func openFirstBackend(only string) gemmDevice {
	for _, b := range backends {
		if only != "" && only != b.name {
			continue
		}
		if dev, err := b.open(); err == nil {
			return dev
		}
	}
	return nil
}

// GPUName returns the name of the accelerator used for large matrix products, or "" for CPU only.
func GPUName() string {
	if dev, _ := activeDevice(); dev != nil {
		return dev.Name()
	}
	return ""
}

// ReleaseGPU frees all device memory and stops using the accelerator.
func ReleaseGPU() {
	accel.mu.Lock()
	defer accel.mu.Unlock()
	releaseLocked()
}

func releaseLocked() {
	if accel.dev != nil {
		accel.dev.Release()
		accel.dev = nil
	}
	accel.probed = true
	accel.failures = 0
}

// offloadGemms runs the jobs that are worth sending to the accelerator and returns the rest.
// A job the device cannot run falls back to the CPU; repeated failures retire the device.
func offloadGemms(jobs []gemmJob) []gemmJob {
	dev, minFlops := activeDevice()
	if dev == nil {
		return jobs
	}
	var rest []gemmJob
	for _, j := range jobs {
		if 2*j.m*j.k*j.n < minFlops || j.m == 0 || j.k == 0 || j.n == 0 {
			rest = append(rest, j)
			continue
		}
		if err := dev.Gemm(j); err != nil {
			rest = append(rest, j)
			noteDeviceFailure(dev)
			continue
		}
		accel.mu.Lock()
		accel.failures = 0
		accel.mu.Unlock()
	}
	return rest
}

func noteDeviceFailure(dev gemmDevice) {
	accel.mu.Lock()
	defer accel.mu.Unlock()
	if accel.dev != dev {
		return
	}
	if accel.failures++; accel.failures >= maxDeviceFailures {
		releaseLocked()
	}
}
