package onnxnlp

import "math"

type f32Kernels struct {
	vv func(dst, a, b []float32)
	vs func(dst, a []float32, s float32)
	sv func(dst []float32, s float32, b []float32)
}

func powScalar(d, a []float32, s float32) {
	switch s {
	case 2:
		for i := range d {
			d[i] = a[i] * a[i]
		}
	case 0.5:
		for i := range d {
			d[i] = float32(math.Sqrt(float64(a[i])))
		}
	default:
		for i := range d {
			d[i] = float32(math.Pow(float64(a[i]), float64(s)))
		}
	}
}

var f32Arith = map[string]f32Kernels{
	"Pow": {
		func(d, a, b []float32) {
			for i := range d {
				d[i] = float32(math.Pow(float64(a[i]), float64(b[i])))
			}
		},
		powScalar,
		func(d []float32, s float32, b []float32) {
			for i := range d {
				d[i] = float32(math.Pow(float64(s), float64(b[i])))
			}
		},
	},
	"Add": {
		func(d, a, b []float32) {
			for i := range d {
				d[i] = a[i] + b[i]
			}
		},
		func(d, a []float32, s float32) {
			for i := range d {
				d[i] = a[i] + s
			}
		},
		func(d []float32, s float32, b []float32) {
			for i := range d {
				d[i] = s + b[i]
			}
		},
	},
	"Sub": {
		func(d, a, b []float32) {
			for i := range d {
				d[i] = a[i] - b[i]
			}
		},
		func(d, a []float32, s float32) {
			for i := range d {
				d[i] = a[i] - s
			}
		},
		func(d []float32, s float32, b []float32) {
			for i := range d {
				d[i] = s - b[i]
			}
		},
	},
	"Mul": {
		func(d, a, b []float32) {
			for i := range d {
				d[i] = a[i] * b[i]
			}
		},
		func(d, a []float32, s float32) {
			for i := range d {
				d[i] = a[i] * s
			}
		},
		func(d []float32, s float32, b []float32) {
			for i := range d {
				d[i] = s * b[i]
			}
		},
	},
	"Div": {
		func(d, a, b []float32) {
			for i := range d {
				d[i] = a[i] / b[i]
			}
		},
		func(d, a []float32, s float32) {
			for i := range d {
				d[i] = a[i] / s
			}
		},
		func(d []float32, s float32, b []float32) {
			for i := range d {
				d[i] = s / b[i]
			}
		},
	},
}

// withFastFloat32 routes float32 inputs of a binary arithmetic operator through
// kernels that avoid per-element index tables; other dtypes use generic.
func withFastFloat32(generic opFunc, k f32Kernels) opFunc {
	return func(c *opCtx) ([]*Tensor, error) {
		if err := c.need(2); err != nil {
			return nil, err
		}
		a, b := c.in[0], c.in[1]
		if a.DType != Float32 || b.DType != Float32 {
			return generic(c)
		}
		shape, err := broadcastShape(a.Shape, b.Shape)
		if err != nil {
			return nil, err
		}
		return []*Tensor{binaryFloat32(k, a, b, shape)}, nil
	}
}

func binaryFloat32(k f32Kernels, a, b *Tensor, shape []int) *Tensor {
	out := zeros(Float32, shape)
	if len(out.F) == 0 {
		return out
	}
	if len(shape) == 0 {
		k.vv(out.F, a.F, b.F)
		return out
	}
	last := len(shape) - 1
	n, rows := shape[last], numel(shape[:last])
	sa, sb := broadcastStrides(a.Shape, shape), broadcastStrides(b.Shape, shape)
	parallelFor(rows, n, func(lo, hi int) {
		walkRows(shape, [][]int{sa, sb}, lo, hi, func(r int, off []int) {
			oa, ob := off[0], off[1]
			dst := out.F[r*n : (r+1)*n]
			switch {
			case sa[last] != 0 && sb[last] != 0:
				k.vv(dst, a.F[oa:oa+n], b.F[ob:ob+n])
			case sa[last] != 0:
				k.vs(dst, a.F[oa:oa+n], b.F[ob])
			case sb[last] != 0:
				k.sv(dst, a.F[oa], b.F[ob:ob+n])
			default:
				var one [1]float32
				k.vv(one[:], a.F[oa:oa+1], b.F[ob:ob+1])
				for i := range dst {
					dst[i] = one[0]
				}
			}
		})
	})
	return out
}
