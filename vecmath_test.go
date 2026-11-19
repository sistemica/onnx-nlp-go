package onnxgo

import (
	"math"
	"math/rand/v2"
	"testing"
)

func eachKernelMode(t *testing.T, check func(t *testing.T)) {
	t.Run("default", check)
	t.Run("scalar", func(t *testing.T) {
		saved := useAVX2
		useAVX2 = false
		defer func() { useAVX2 = saved }()
		check(t)
	})
}

func relErr(got, want float64) float64 { return math.Abs(got-want) / math.Max(math.Abs(want), 1e-30) }

func TestExp32Accuracy(t *testing.T) {
	var worst float64
	for x := -87.0; x <= 88.0; x += 0.0137 {
		worst = max(worst, relErr(float64(exp32(float32(x))), math.Exp(float64(float32(x)))))
	}
	if worst > 3e-7 {
		t.Fatalf("exp32 relative error %g", worst)
	}
	if exp32(-1000) <= 0 || math.IsInf(float64(exp32(1000)), 0) {
		t.Fatal("exp32 must clamp instead of overflowing or underflowing to zero")
	}
}

func TestExpSumShift(t *testing.T) {
	eachKernelMode(t, func(t *testing.T) {
		rng := rand.New(rand.NewPCG(101, 102))
		for _, n := range []int{0, 1, 7, 8, 9, 15, 16, 17, 100, 256} {
			src := make([]float32, n)
			for i := range src {
				src[i] = float32(rng.NormFloat64()) * 5
			}
			shift := float32(2.5)
			dst := make([]float32, n)
			var want float64
			sum := float64(expSumShift(dst, src, shift))
			for i, v := range src {
				e := math.Exp(float64(v - shift))
				want += e
				if relErr(float64(dst[i]), e) > 3e-7 {
					t.Fatalf("n=%d element %d: got %v, want %v", n, i, dst[i], e)
				}
			}
			if n > 0 && relErr(sum, want) > 1e-5 {
				t.Fatalf("n=%d: sum %v, want %v", n, sum, want)
			}
		}
	})
}

func TestErfSliceAndGeluSlice(t *testing.T) {
	eachKernelMode(t, func(t *testing.T) {
		for _, n := range []int{0, 1, 7, 8, 9, 31, 64, 129} {
			src := make([]float32, n)
			for i := range src {
				src[i] = float32(i)/float32(max(n, 1))*12 - 6
			}
			erf, gelu := make([]float32, n), make([]float32, n)
			erfSlice(erf, src)
			geluSlice(gelu, src, float32(math.Sqrt2), 1, 0.5)
			for i, v := range src {
				if d := math.Abs(float64(erf[i]) - math.Erf(float64(v))); d > 6e-7 {
					t.Fatalf("n=%d erf(%v): off by %g", n, v, d)
				}
				want := float64(v) * (math.Erf(float64(v)/math.Sqrt2) + 1) * 0.5
				if d := math.Abs(float64(gelu[i]) - want); d > 1e-6*(1+math.Abs(want)) {
					t.Fatalf("n=%d gelu(%v): got %v, want %v", n, v, gelu[i], want)
				}
			}
		}
	})
}

func TestScaleSlice(t *testing.T) {
	eachKernelMode(t, func(t *testing.T) {
		for _, n := range []int{0, 3, 8, 13, 40} {
			v := make([]float32, n)
			for i := range v {
				v[i] = float32(i + 1)
			}
			scaleSlice(v, 0.5)
			for i := range v {
				if v[i] != float32(i+1)/2 {
					t.Fatalf("n=%d element %d: %v", n, i, v[i])
				}
			}
		}
	})
}
