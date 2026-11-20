package onnxnlp

import "golang.org/x/sys/cpu"

var (
	useAVX2   = cpu.X86.HasAVX2 && cpu.X86.HasFMA
	useAVX512 = cpu.X86.HasAVX512F
)

// kernel6x16 computes C[6x16] += A[6xk] * B[kx16] with row strides lda, ldb and ldc (in elements).
// It requires k > 0.
//
//go:noescape
func kernel6x16(a *float32, lda int, b *float32, ldb int, c *float32, ldc int, k int)

// kernel12x16 is the AVX-512 variant of kernel6x16 for twelve rows.
//
//go:noescape
func kernel12x16(a *float32, lda int, b *float32, ldb int, c *float32, ldc int, k int)

// gemv16 computes y[0:16] += x[0:k] * panel[k x 16], where the panel rows are contiguous. It
// requires k > 0.
//
//go:noescape
func gemv16(x *float32, panel *float32, y *float32, k int)

// gemv16bf16 is gemv16 for a panel of bfloat16 values.
//
//go:noescape
func gemv16bf16(x *float32, panel *uint16, y *float32, k int)
