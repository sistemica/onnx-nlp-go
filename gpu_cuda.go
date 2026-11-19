//go:build linux && (amd64 || arm64)

package onnxgo

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"unsafe"

	"github.com/ebitengine/purego"
)

const cublasOpN = 0

type cudaAPI struct {
	cuInit                func(flags uint32) int32
	cuDeviceGetCount      func(count *int32) int32
	cuDeviceGet           func(dev *int32, ordinal int32) int32
	cuDeviceGetName       func(name unsafe.Pointer, length int32, dev int32) int32
	cuDevicePrimaryCtxRet func(ctx *uintptr, dev int32) int32
	cuDevicePrimaryCtxRel func(dev int32) int32
	cuCtxSetCurrent       func(ctx uintptr) int32
	cuMemGetInfo          func(free, total *uint64) int32
	cuMemAlloc            func(ptr *uint64, bytes uint64) int32
	cuMemFree             func(ptr uint64) int32
	cuMemcpyHtoD          func(dst uint64, src unsafe.Pointer, bytes uint64) int32
	cuMemcpyDtoH          func(dst unsafe.Pointer, src uint64, bytes uint64) int32
	cublasCreate          func(handle *uintptr) int32
	cublasDestroy         func(handle uintptr) int32
	cublasSgemm           func(handle uintptr, ta, tb int32, m, n, k int32, alpha *float32, a uint64, lda int32, b uint64, ldb int32, beta *float32, c uint64, ldc int32) int32
}

func dlopenFirst(names []string) (uintptr, string, error) {
	var last error
	for _, n := range names {
		h, err := purego.Dlopen(n, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err == nil {
			return h, n, nil
		}
		last = err
	}
	return 0, "", last
}

var (
	cudaLibraries   = []string{"libcuda.so.1", "libcuda.so"}
	cublasLibraries = []string{
		"libcublas.so.13", "libcublas.so.12", "libcublas.so",
		"/usr/local/cuda/lib64/libcublas.so", "/usr/local/cuda-13.2/lib64/libcublas.so.13",
	}
)

func loadCUDA() (*cudaAPI, error) { return loadCUDAFrom(cudaLibraries, cublasLibraries) }

func loadCUDAFrom(driverNames, blasNames []string) (*cudaAPI, error) {
	cuda, _, err := dlopenFirst(driverNames)
	if err != nil {
		return nil, fmt.Errorf("cuda driver: %w", err)
	}
	blas, _, err := dlopenFirst(blasNames)
	if err != nil {
		return nil, fmt.Errorf("cublas: %w", err)
	}
	api := &cudaAPI{}
	for lib, fns := range map[uintptr]map[string]any{
		cuda: {
			"cuInit": &api.cuInit, "cuDeviceGetCount": &api.cuDeviceGetCount, "cuDeviceGet": &api.cuDeviceGet,
			"cuDeviceGetName": &api.cuDeviceGetName, "cuDevicePrimaryCtxRetain": &api.cuDevicePrimaryCtxRet,
			"cuDevicePrimaryCtxRelease_v2": &api.cuDevicePrimaryCtxRel, "cuCtxSetCurrent": &api.cuCtxSetCurrent,
			"cuMemGetInfo_v2": &api.cuMemGetInfo, "cuMemAlloc_v2": &api.cuMemAlloc, "cuMemFree_v2": &api.cuMemFree,
			"cuMemcpyHtoD_v2": &api.cuMemcpyHtoD, "cuMemcpyDtoH_v2": &api.cuMemcpyDtoH,
		},
		blas: {
			"cublasCreate_v2": &api.cublasCreate, "cublasDestroy_v2": &api.cublasDestroy, "cublasSgemm_v2": &api.cublasSgemm,
		},
	} {
		for name, fn := range fns {
			if err := registerFunc(fn, lib, name); err != nil {
				return nil, err
			}
		}
	}
	return api, nil
}

func registerFunc(fn any, lib uintptr, name string) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("symbol %s: %v", name, r)
		}
	}()
	purego.RegisterLibFunc(fn, lib, name)
	return nil
}

type deviceBuffer struct {
	ptr  uint64
	size uint64
}

type weightKey struct {
	p *float32
	n int
}

type cudaRequest struct {
	job  gemmJob
	done chan error
	stop bool
}

type cudaDevice struct {
	api      *cudaAPI
	name     string
	ordinal  int32
	reqs     chan cudaRequest
	finished chan struct{}

	handle  uintptr
	weights map[weightKey]deviceBuffer
	scratch [3]deviceBuffer
}

func init() {
	registerBackend("cuda", openCUDA)
}

func cudaError(op string, code int32) error {
	return fmt.Errorf("cuda: %s failed with code %d", op, code)
}

func openCUDA() (gemmDevice, error) {
	api, err := loadCUDA()
	if err != nil {
		return nil, err
	}
	return newCUDADevice(api)
}

func newCUDADevice(api *cudaAPI) (*cudaDevice, error) {
	d := &cudaDevice{api: api, reqs: make(chan cudaRequest), finished: make(chan struct{}), weights: map[weightKey]deviceBuffer{}}
	ready := make(chan error, 1)
	go d.run(ready)
	if err := <-ready; err != nil {
		return nil, err
	}
	return d, nil
}

func (d *cudaDevice) Name() string { return d.name }

func (d *cudaDevice) run(ready chan<- error) {
	runtime.LockOSThread()
	defer close(d.finished)
	if err := d.init(); err != nil {
		ready <- err
		return
	}
	ready <- nil
	for req := range d.reqs {
		if req.stop {
			d.free()
			req.done <- nil
			return
		}
		req.done <- d.gemm(req.job)
	}
}

func (d *cudaDevice) init() error {
	api := d.api
	if code := api.cuInit(0); code != 0 {
		return cudaError("cuInit", code)
	}
	var count int32
	if code := api.cuDeviceGetCount(&count); code != 0 || count == 0 {
		return errors.New("cuda: no devices")
	}
	want := int32(-1)
	if v, err := strconv.Atoi(os.Getenv("ONNXGO_GPU_DEVICE")); err == nil {
		want = int32(v)
	}
	best, bestFree := int32(-1), uint64(0)
	var bestCtx uintptr
	for i := int32(0); i < count; i++ {
		if want >= 0 && i != want {
			continue
		}
		var dev int32
		var ctx uintptr
		if api.cuDeviceGet(&dev, i) != 0 || api.cuDevicePrimaryCtxRet(&ctx, dev) != 0 {
			continue
		}
		api.cuCtxSetCurrent(ctx)
		var free, total uint64
		if api.cuMemGetInfo(&free, &total) != 0 || free <= bestFree {
			api.cuDevicePrimaryCtxRel(dev)
			continue
		}
		if best >= 0 {
			api.cuDevicePrimaryCtxRel(best)
		}
		best, bestFree, bestCtx = dev, free, ctx
	}
	if best < 0 {
		return errors.New("cuda: no usable device")
	}
	api.cuCtxSetCurrent(bestCtx)
	d.ordinal = best
	nameBuf := make([]byte, 128)
	api.cuDeviceGetName(unsafe.Pointer(&nameBuf[0]), int32(len(nameBuf)), best)
	for i, b := range nameBuf {
		if b == 0 {
			d.name = string(nameBuf[:i])
			break
		}
	}
	if code := api.cublasCreate(&d.handle); code != 0 {
		return cudaError("cublasCreate", code)
	}
	return nil
}

func (d *cudaDevice) free() {
	for _, b := range d.weights {
		d.api.cuMemFree(b.ptr)
	}
	for _, b := range d.scratch {
		if b.ptr != 0 {
			d.api.cuMemFree(b.ptr)
		}
	}
	d.weights, d.scratch = map[weightKey]deviceBuffer{}, [3]deviceBuffer{}
	d.api.cublasDestroy(d.handle)
	d.api.cuDevicePrimaryCtxRel(d.ordinal)
}

func (d *cudaDevice) Release() {
	done := make(chan error, 1)
	d.reqs <- cudaRequest{stop: true, done: done}
	<-done
	<-d.finished
}

func (d *cudaDevice) Gemm(j gemmJob) error {
	done := make(chan error, 1)
	d.reqs <- cudaRequest{job: j, done: done}
	return <-done
}

func (d *cudaDevice) scratchBuffer(slot int, size uint64) (uint64, error) {
	if d.scratch[slot].size >= size {
		return d.scratch[slot].ptr, nil
	}
	if d.scratch[slot].ptr != 0 {
		d.api.cuMemFree(d.scratch[slot].ptr)
		d.scratch[slot] = deviceBuffer{}
	}
	var ptr uint64
	if code := d.api.cuMemAlloc(&ptr, size); code != 0 {
		return 0, cudaError("cuMemAlloc", code)
	}
	d.scratch[slot] = deviceBuffer{ptr, size}
	return ptr, nil
}

func (d *cudaDevice) upload(b []float32, j gemmJob) (uint64, error) {
	bytes := uint64(len(b)) * 4
	key := weightKey{&b[0], len(b)}
	if j.bStatic {
		if buf, ok := d.weights[key]; ok {
			return buf.ptr, nil
		}
		var ptr uint64
		if d.api.cuMemAlloc(&ptr, bytes) == 0 {
			if code := d.api.cuMemcpyHtoD(ptr, unsafe.Pointer(&b[0]), bytes); code != 0 {
				d.api.cuMemFree(ptr)
				return 0, cudaError("cuMemcpyHtoD", code)
			}
			d.weights[key] = deviceBuffer{ptr, bytes}
			return ptr, nil
		}
	}
	ptr, err := d.scratchBuffer(1, bytes)
	if err != nil {
		return 0, err
	}
	if code := d.api.cuMemcpyHtoD(ptr, unsafe.Pointer(&b[0]), bytes); code != 0 {
		return 0, cudaError("cuMemcpyHtoD", code)
	}
	return ptr, nil
}

func (d *cudaDevice) gemm(j gemmJob) error {
	api := d.api
	dB, err := d.upload(j.b, j)
	if err != nil {
		return err
	}
	dA, err := d.scratchBuffer(0, uint64(len(j.a))*4)
	if err != nil {
		return err
	}
	if code := api.cuMemcpyHtoD(dA, unsafe.Pointer(&j.a[0]), uint64(len(j.a))*4); code != 0 {
		return cudaError("cuMemcpyHtoD", code)
	}
	dC, err := d.scratchBuffer(2, uint64(len(j.out))*4)
	if err != nil {
		return err
	}
	alpha, beta := float32(1), float32(0)
	// Row-major C = A*B is column-major C^T = B^T * A^T.
	if code := api.cublasSgemm(d.handle, cublasOpN, cublasOpN, int32(j.n), int32(j.m), int32(j.k),
		&alpha, dB, int32(j.n), dA, int32(j.k), &beta, dC, int32(j.n)); code != 0 {
		return cudaError("cublasSgemm", code)
	}
	if code := api.cuMemcpyDtoH(unsafe.Pointer(&j.out[0]), dC, uint64(len(j.out))*4); code != 0 {
		return cudaError("cuMemcpyDtoH", code)
	}
	return nil
}
