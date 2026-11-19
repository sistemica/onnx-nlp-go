package onnxgo

import "math"

// exp32 is a float32 exponential with a relative error near 1e-7, matching the AVX2 kernel.
func exp32(x float32) float32 {
	x = min(max(x, -87.3365), 88.3762626647949)
	t := x*1.44269504088896341 + 0.5
	n := int32(t)
	if float32(n) > t {
		n--
	}
	fx := float32(n)
	x -= fx * 0.693359375
	x -= fx * -2.12194440e-4
	z := x * x
	y := (((((1.9875691500e-4*x+1.3981999507e-3)*x+8.3334519073e-3)*x+4.1665795894e-2)*x+1.6666665459e-1)*x + 5.0000001201e-1) * z
	return math.Float32frombits(uint32(n+127)<<23) * (y + x + 1)
}

// expSumShift stores exp(src[i]-shift) in dst[i] and returns the sum.
func expSumShift(dst, src []float32, shift float32) float32 {
	var sum float32
	done := 0
	if useAVX2 && len(src) >= 8 {
		done = len(src) &^ 7
		sum = expSumAVX2(&dst[0], &src[0], done, shift)
	}
	for i := done; i < len(src); i++ {
		e := exp32(src[i] - shift)
		dst[i] = e
		sum += e
	}
	return sum
}

func erfSlice(dst, src []float32) {
	done := 0
	if useAVX2 && len(src) >= 8 {
		done = len(src) &^ 7
		erfAVX2(&dst[0], &src[0], done)
	}
	for i := done; i < len(src); i++ {
		dst[i] = erf32(src[i])
	}
}

func geluSlice(dst, src []float32, divisor, offset, scale float32) {
	done := 0
	if useAVX2 && len(src) >= 8 {
		done = len(src) &^ 7
		geluAVX2(&dst[0], &src[0], done, divisor, offset, scale)
	}
	for i := done; i < len(src); i++ {
		v := src[i]
		dst[i] = v * (erf32(v/divisor) + offset) * scale
	}
}

func scaleSlice(dst []float32, factor float32) {
	done := 0
	if useAVX2 && len(dst) >= 8 {
		done = len(dst) &^ 7
		scaleAVX2(&dst[0], done, factor)
	}
	for i := done; i < len(dst); i++ {
		dst[i] *= factor
	}
}
