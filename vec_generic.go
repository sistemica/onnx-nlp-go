//go:build !amd64

package onnxgo

func expSumAVX2(dst, src *float32, n int, shift float32) float32 {
	panic("expSumAVX2 is only available on amd64")
}

func erfAVX2(dst, src *float32, n int) { panic("erfAVX2 is only available on amd64") }

func geluAVX2(dst, src *float32, n int, divisor, offset, scale float32) {
	panic("geluAVX2 is only available on amd64")
}

func scaleAVX2(dst *float32, n int, factor float32) { panic("scaleAVX2 is only available on amd64") }
