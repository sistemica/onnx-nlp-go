//go:build linux && (amd64 || arm64)

package onnxnlp

import (
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
	"unsafe"
)

type fakeDriver struct {
	devices    []uint64 // free memory per device
	failures   map[string]int32
	mem        map[uint64][]byte
	next       uint64
	allocs     int
	freed      []uint64
	retained   []int32
	released   []int32
	cublasMade bool
	cublasGone bool
	current    uintptr
}

func newFakeDriver(freeMem ...uint64) *fakeDriver {
	return &fakeDriver{devices: freeMem, failures: map[string]int32{}, mem: map[uint64][]byte{}, next: 0x1000}
}

func (f *fakeDriver) fail(name string) int32 { return f.failures[name] }

func floatsOf(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

func (f *fakeDriver) api() *cudaAPI {
	return &cudaAPI{
		cuInit:           func(uint32) int32 { return f.fail("cuInit") },
		cuDeviceGetCount: func(c *int32) int32 { *c = int32(len(f.devices)); return f.fail("cuDeviceGetCount") },
		cuDeviceGet:      func(d *int32, i int32) int32 { *d = i; return f.fail("cuDeviceGet") },
		cuDeviceGetName: func(p unsafe.Pointer, n int32, dev int32) int32 {
			copy(unsafe.Slice((*byte)(p), n), "Fake GPU "+string(rune('0'+dev))+"\x00")
			return 0
		},
		cuDevicePrimaryCtxRet: func(ctx *uintptr, dev int32) int32 {
			*ctx = uintptr(100 + dev)
			f.retained = append(f.retained, dev)
			return f.fail("cuDevicePrimaryCtxRetain")
		},
		cuDevicePrimaryCtxRel: func(dev int32) int32 { f.released = append(f.released, dev); return 0 },
		cuCtxSetCurrent:       func(ctx uintptr) int32 { f.current = ctx; return 0 },
		cuMemGetInfo: func(free, total *uint64) int32 {
			*free, *total = f.devices[f.current-100], 1<<40
			return f.fail("cuMemGetInfo")
		},
		cuMemAlloc: func(ptr *uint64, n uint64) int32 {
			if code := f.fail("cuMemAlloc"); code != 0 {
				return code
			}
			f.allocs++
			*ptr = f.next
			f.mem[f.next] = make([]byte, n)
			f.next += n + 256
			return 0
		},
		cuMemFree: func(ptr uint64) int32 { f.freed = append(f.freed, ptr); delete(f.mem, ptr); return 0 },
		cuMemcpyHtoD: func(dst uint64, src unsafe.Pointer, n uint64) int32 {
			if code := f.fail("cuMemcpyHtoD"); code != 0 {
				return code
			}
			copy(f.mem[dst], unsafe.Slice((*byte)(src), n))
			return 0
		},
		cuMemcpyDtoH: func(dst unsafe.Pointer, src uint64, n uint64) int32 {
			if code := f.fail("cuMemcpyDtoH"); code != 0 {
				return code
			}
			copy(unsafe.Slice((*byte)(dst), n), f.mem[src])
			return 0
		},
		cublasCreate:  func(h *uintptr) int32 { *h = 7; f.cublasMade = true; return f.fail("cublasCreate") },
		cublasDestroy: func(uintptr) int32 { f.cublasGone = true; return 0 },
		cublasSgemm: func(h uintptr, ta, tb int32, m, n, k int32, alpha *float32, a uint64, lda int32, b uint64, ldb int32, beta *float32, c uint64, ldc int32) int32 {
			if code := f.fail("cublasSgemm"); code != 0 {
				return code
			}
			if ta != cublasOpN || tb != cublasOpN || *alpha != 1 || *beta != 0 || h != 7 {
				return 99
			}
			A, B, C := floatsOf(f.mem[a]), floatsOf(f.mem[b]), floatsOf(f.mem[c])
			for j := int32(0); j < n; j++ {
				for i := int32(0); i < m; i++ {
					var sum float32
					for p := int32(0); p < k; p++ {
						sum += A[i+p*lda] * B[p+j*ldb]
					}
					C[i+j*ldc] = sum
				}
			}
			return 0
		},
	}
}

func fakeJob(rng *rand.Rand, m, k, n int, static bool) (gemmJob, []float32) {
	a, b := randomSlice(rng, m*k), randomSlice(rng, k*n)
	return gemmJob{out: make([]float32, m*n), a: a, b: b, m: m, k: k, n: n, bStatic: static}, naiveGemm(a, b, m, k, n)
}

func TestFakeDriverGemmUsesColumnMajorConvention(t *testing.T) {
	isolateGPUEnv(t)
	drv := newFakeDriver(1 << 30)
	dev, err := newCUDADevice(drv.api())
	if err != nil {
		t.Fatal(err)
	}
	if dev.Name() != "Fake GPU 0" {
		t.Fatalf("name %q", dev.Name())
	}
	rng := rand.New(rand.NewPCG(51, 52))
	for _, sz := range [][3]int{{1, 1, 1}, {3, 5, 2}, {17, 9, 33}} {
		job, want := fakeJob(rng, sz[0], sz[1], sz[2], false)
		if err := dev.Gemm(job); err != nil {
			t.Fatal(err)
		}
		closeSlices(t, job.out, want)
	}
	dev.Release()
	if !drv.cublasMade || !drv.cublasGone || len(drv.mem) != 0 || !slices.Equal(drv.released, []int32{0}) {
		t.Fatalf("release left state behind: %+v", drv)
	}
}

func TestFakeDriverCachesStaticWeightsAndGrowsScratch(t *testing.T) {
	isolateGPUEnv(t)
	drv := newFakeDriver(1 << 30)
	dev, _ := newCUDADevice(drv.api())
	rng := rand.New(rand.NewPCG(53, 54))
	job, want := fakeJob(rng, 4, 6, 5, true)
	for i := 0; i < 3; i++ {
		clear(job.out)
		if err := dev.Gemm(job); err != nil {
			t.Fatal(err)
		}
		closeSlices(t, job.out, want)
	}
	if drv.allocs != 3 || len(dev.weights) != 1 {
		t.Fatalf("expected one weight upload plus two scratch buffers, got %d allocations and %d cached weights", drv.allocs, len(dev.weights))
	}
	bigger, _ := fakeJob(rng, 40, 6, 5, true)
	bigger.b = job.b
	if err := dev.Gemm(bigger); err != nil {
		t.Fatal(err)
	}
	closeSlices(t, bigger.out, naiveGemm(bigger.a, job.b, 40, 6, 5))
	if len(drv.freed) == 0 {
		t.Fatal("growing a scratch buffer must free the old one")
	}
	dev.Release()
	if len(drv.mem) != 0 {
		t.Fatalf("%d device buffers leaked", len(drv.mem))
	}
}

func TestFakeDriverCacheBudgetLeavesRoomForScratch(t *testing.T) {
	isolateGPUEnv(t)
	drv := newFakeDriver(400)
	dev, err := newCUDADevice(drv.api())
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(133, 134))
	for i := 0; i < 3; i++ {
		job, want := fakeJob(rng, 3, 5, 6, true)
		for round := 0; round < 2; round++ {
			clear(job.out)
			if err := dev.Gemm(job); err != nil {
				t.Fatal(err)
			}
			closeSlices(t, job.out, want)
		}
	}
	if len(dev.weights) != 2 || dev.cacheUsed != 240 {
		t.Fatalf("a 240-byte budget holds two 120-byte weight matrices: %d cached, %d bytes used", len(dev.weights), dev.cacheUsed)
	}
	dev.Release()
	if dev.cacheUsed != 0 {
		t.Fatal("releasing the device must reset the cache accounting")
	}
}

func TestFakeDriverDeviceSelection(t *testing.T) {
	isolateGPUEnv(t)
	drv := newFakeDriver(100, 900, 500)
	dev, err := newCUDADevice(drv.api())
	if err != nil {
		t.Fatal(err)
	}
	if dev.Name() != "Fake GPU 1" || dev.ordinal != 1 {
		t.Fatalf("picked %q (ordinal %d)", dev.Name(), dev.ordinal)
	}
	if !slices.Contains(drv.released, 0) || !slices.Contains(drv.released, 2) || slices.Contains(drv.released, 1) {
		t.Fatalf("contexts of unused devices must be released: %v", drv.released)
	}
	dev.Release()

	t.Setenv("ONNXNLP_GPU_DEVICE", "2")
	drv = newFakeDriver(100, 900, 500)
	dev, err = newCUDADevice(drv.api())
	if err != nil || dev.ordinal != 2 {
		t.Fatalf("explicit device: %v %v", dev, err)
	}
	dev.Release()

	t.Setenv("ONNXNLP_GPU_DEVICE", "5")
	if _, err := newCUDADevice(newFakeDriver(100).api()); err == nil || !strings.Contains(err.Error(), "no usable device") {
		t.Fatalf("out-of-range device: %v", err)
	}
}

func TestFakeDriverInitializationFailures(t *testing.T) {
	isolateGPUEnv(t)
	for _, name := range []string{"cuInit", "cuDeviceGetCount", "cuDeviceGet", "cuDevicePrimaryCtxRetain", "cuMemGetInfo", "cublasCreate"} {
		drv := newFakeDriver(1 << 20)
		drv.failures[name] = 3
		if _, err := newCUDADevice(drv.api()); err == nil {
			t.Errorf("%s failure was ignored", name)
		}
	}
	if _, err := newCUDADevice(newFakeDriver().api()); err == nil || !strings.Contains(err.Error(), "no devices") {
		t.Fatalf("zero devices: %v", err)
	}
	if err := cudaError("cuInit", 3); !strings.Contains(err.Error(), "cuInit") || !strings.Contains(err.Error(), "3") {
		t.Fatalf("error text %q", err)
	}
}

func TestFakeDriverGemmFailures(t *testing.T) {
	isolateGPUEnv(t)
	rng := rand.New(rand.NewPCG(55, 56))
	for _, tc := range []struct {
		fail   string
		static bool
	}{
		{"cuMemAlloc", false}, {"cuMemAlloc", true}, {"cuMemcpyHtoD", false}, {"cuMemcpyHtoD", true},
		{"cublasSgemm", false}, {"cuMemcpyDtoH", false},
	} {
		drv := newFakeDriver(1 << 30)
		dev, err := newCUDADevice(drv.api())
		if err != nil {
			t.Fatal(err)
		}
		drv.failures[tc.fail] = 2
		job, _ := fakeJob(rng, 3, 4, 5, tc.static)
		if err := dev.Gemm(job); err == nil {
			t.Errorf("%s (static=%v) failure was ignored", tc.fail, tc.static)
		}
		delete(drv.failures, tc.fail)
		dev.Release()
	}
}

func TestFakeDriverFailsOnEachSuccessiveStep(t *testing.T) {
	isolateGPUEnv(t)
	rng := rand.New(rand.NewPCG(57, 58))
	for _, step := range []string{"cuMemAlloc", "cuMemcpyHtoD"} {
		for failAt := 1; failAt <= 3; failAt++ {
			if step == "cuMemcpyHtoD" && failAt == 3 {
				continue
			}
			drv := newFakeDriver(1 << 30)
			dev, _ := newCUDADevice(drv.api())
			api := dev.api
			count := 0
			if step == "cuMemAlloc" {
				inner := api.cuMemAlloc
				api.cuMemAlloc = func(ptr *uint64, n uint64) int32 {
					if count++; count == failAt {
						return 2
					}
					return inner(ptr, n)
				}
			} else {
				inner := api.cuMemcpyHtoD
				api.cuMemcpyHtoD = func(dst uint64, src unsafe.Pointer, n uint64) int32 {
					if count++; count == failAt {
						return 2
					}
					return inner(dst, src, n)
				}
			}
			job, _ := fakeJob(rng, 3, 4, 5, false)
			if err := dev.Gemm(job); err == nil {
				t.Errorf("%s call %d: failure was ignored", step, failAt)
			}
			dev.Release()
		}
	}
}

func TestFakeDriverStaticAllocationFallsBackToScratch(t *testing.T) {
	isolateGPUEnv(t)
	rng := rand.New(rand.NewPCG(59, 60))
	drv := newFakeDriver(1 << 30)
	dev, _ := newCUDADevice(drv.api())
	api := dev.api
	first := true
	inner := api.cuMemAlloc
	api.cuMemAlloc = func(ptr *uint64, n uint64) int32 {
		if first {
			first = false
			return 2
		}
		return inner(ptr, n)
	}
	job, want := fakeJob(rng, 3, 4, 5, true)
	if err := dev.Gemm(job); err != nil {
		t.Fatal(err)
	}
	closeSlices(t, job.out, want)
	if len(dev.weights) != 0 {
		t.Fatal("a failed weight allocation must not be cached")
	}
	dev.Release()
}
