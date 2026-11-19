package onnxnlp

import (
	"errors"
	"fmt"
	"unsafe"
)

const (
	cblasRowMajor = 101
	cblasNoTrans  = 111
	// accelerateMinFlops is the smallest product sent to Accelerate. There is no transfer to
	// pay for, so it is far below the threshold of the GPU backends.
	accelerateMinFlops = 1 << 16
)

// accelerateAPI is the part of Apple's Accelerate framework that is used: single-precision
// matrix multiplication through the CBLAS interface.
type accelerateAPI struct {
	sgemm func(order, transA, transB, m, n, k int32, alpha float32, a unsafe.Pointer, lda int32, b unsafe.Pointer, ldb int32, beta float32, c unsafe.Pointer, ldc int32)
}

// accelerateDevice runs matrix products on the CPU through Accelerate, which uses the matrix
// units of Apple silicon.
type accelerateDevice struct{ api *accelerateAPI }

func newAccelerateDevice(api *accelerateAPI) (*accelerateDevice, error) {
	if api == nil || api.sgemm == nil {
		return nil, errors.New("accelerate: cblas_sgemm is not available")
	}
	return &accelerateDevice{api: api}, nil
}

func (d *accelerateDevice) Name() string { return "Apple Accelerate" }

func (d *accelerateDevice) MinFlops() int { return accelerateMinFlops }

func (d *accelerateDevice) Release() {}

func (d *accelerateDevice) Gemm(j gemmJob) error {
	if len(j.a) < j.m*j.k || len(j.b) < j.k*j.n || len(j.out) < j.m*j.n {
		return fmt.Errorf("accelerate: buffers too small for a %dx%dx%d product", j.m, j.k, j.n)
	}
	d.api.sgemm(cblasRowMajor, cblasNoTrans, cblasNoTrans, int32(j.m), int32(j.n), int32(j.k),
		1, unsafe.Pointer(&j.a[0]), int32(j.k), unsafe.Pointer(&j.b[0]), int32(j.n), 0, unsafe.Pointer(&j.out[0]), int32(j.n))
	return nil
}
