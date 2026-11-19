//go:build amd64

package onnxnlp

// The AVX2 kernels below process the leading multiple of eight elements and require useAVX2.

// expSumAVX2 stores exp(src[i]-shift) in dst[i] for i < n and returns their sum.
//
//go:noescape
func expSumAVX2(dst, src *float32, n int, shift float32) float32

//go:noescape
func erfAVX2(dst, src *float32, n int)

// geluAVX2 stores src[i]*(erf(src[i]/divisor)+offset)*scale in dst[i] for i < n.
//
//go:noescape
func geluAVX2(dst, src *float32, n int, divisor, offset, scale float32)

//go:noescape
func scaleAVX2(dst *float32, n int, factor float32)
