package onnxgo

import "math/rand/v2"

func naiveGemm(a, b []float32, m, k, n int) []float32 {
	out := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float64
			for p := 0; p < k; p++ {
				sum += float64(a[i*k+p]) * float64(b[p*n+j])
			}
			out[i*n+j] = float32(sum)
		}
	}
	return out
}

func randomSlice(rng *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(rng.NormFloat64())
	}
	return s
}
