package onnxnlp

import (
	"fmt"
	"math"
)

func at(offs []int, i int) int {
	if offs == nil {
		return i
	}
	return offs[i]
}

type pairLayout struct {
	shape  []int
	oa, ob []int
}

func pairOf(c *opCtx) (a, b *Tensor, l pairLayout, err error) {
	if err = c.need(2); err != nil {
		return
	}
	a, b = c.in[0], c.in[1]
	if a.DType != b.DType {
		err = fmt.Errorf("dtype mismatch: %s vs %s", a.DType, b.DType)
		return
	}
	l.shape, err = broadcastShape(a.Shape, b.Shape)
	if err != nil {
		return
	}
	l.oa = broadcastOffsets(a.Shape, l.shape)
	l.ob = broadcastOffsets(b.Shape, l.shape)
	return
}

func arith(fF func(a, b float32) float32, fI func(a, b int64) int64) opFunc {
	return func(c *opCtx) ([]*Tensor, error) {
		a, b, l, err := pairOf(c)
		if err != nil {
			return nil, err
		}
		out := zeros(a.DType, l.shape)
		if a.isFloat() {
			for i := range out.F {
				out.F[i] = fF(a.F[at(l.oa, i)], b.F[at(l.ob, i)])
			}
		} else {
			for i := range out.I {
				out.I[i] = fI(a.I[at(l.oa, i)], b.I[at(l.ob, i)])
			}
		}
		return []*Tensor{out}, nil
	}
}

func compare(fF func(a, b float32) bool, fI func(a, b int64) bool) opFunc {
	return func(c *opCtx) ([]*Tensor, error) {
		a, b, l, err := pairOf(c)
		if err != nil {
			return nil, err
		}
		out := zeros(Bool, l.shape)
		for i := range out.I {
			var r bool
			if a.isFloat() {
				r = fF(a.F[at(l.oa, i)], b.F[at(l.ob, i)])
			} else {
				r = fI(a.I[at(l.oa, i)], b.I[at(l.ob, i)])
			}
			if r {
				out.I[i] = 1
			}
		}
		return []*Tensor{out}, nil
	}
}

func unary(fF func(float32) float32, fI func(int64) int64) opFunc {
	return func(c *opCtx) ([]*Tensor, error) {
		if err := c.need(1); err != nil {
			return nil, err
		}
		x := c.in[0]
		out := zeros(x.DType, x.Shape)
		if x.isFloat() {
			if fF == nil {
				return nil, fmt.Errorf("unsupported for float32")
			}
			parallelFor(len(x.F), 16, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					out.F[i] = fF(x.F[i])
				}
			})
		} else {
			if fI == nil {
				return nil, fmt.Errorf("unsupported for %s", x.DType)
			}
			for i, v := range x.I {
				out.I[i] = fI(v)
			}
		}
		return []*Tensor{out}, nil
	}
}

func viaF64(f func(float64) float64) func(float32) float32 {
	return func(x float32) float32 { return float32(f(float64(x))) }
}

func boolInt(b bool) int64 {
	if b {
		return 1
	}
	return 0
}

func sign[T int64 | float32](x T) T {
	switch {
	case x > 0:
		return 1
	case x < 0:
		return -1
	}
	return 0
}

func floorMod(a, b int64) int64 {
	if b == 0 {
		return 0
	}
	r := a % b
	if r != 0 && (r < 0) != (b < 0) {
		r += b
	}
	return r
}

func opMod(c *opCtx) ([]*Tensor, error) {
	if c.node.Int("fmod", 0) == 1 {
		return arith(func(a, b float32) float32 { return float32(math.Mod(float64(a), float64(b))) },
			func(a, b int64) int64 {
				if b == 0 {
					return 0
				}
				return a % b
			})(c)
	}
	return arith(func(a, b float32) float32 {
		r := float32(math.Mod(float64(a), float64(b)))
		if r != 0 && (r < 0) != (b < 0) {
			r += b
		}
		return r
	}, floorMod)(c)
}

func opIsNaN(c *opCtx) ([]*Tensor, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	x := c.in[0]
	out := zeros(Bool, x.Shape)
	for i, v := range x.F {
		out.I[i] = boolInt(v != v)
	}
	return []*Tensor{out}, nil
}

func opCast(c *opCtx) ([]*Tensor, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	var to DType
	switch c.node.Int("to", 0) {
	case onnxFloat:
		to = Float32
	case onnxInt64, onnxInt32:
		to = Int64
	case onnxBool:
		to = Bool
	default:
		return nil, fmt.Errorf("unsupported cast target %d", c.node.Int("to", 0))
	}
	x := c.in[0]
	if x.DType == to {
		return []*Tensor{x}, nil
	}
	out := zeros(to, x.Shape)
	switch {
	case x.isFloat() && to == Int64:
		for i, v := range x.F {
			out.I[i] = int64(v)
		}
	case x.isFloat():
		for i, v := range x.F {
			out.I[i] = boolInt(v != 0)
		}
	case to == Float32:
		for i, v := range x.I {
			out.F[i] = float32(v)
		}
	case to == Bool:
		for i, v := range x.I {
			out.I[i] = boolInt(v != 0)
		}
	default:
		copy(out.I, x.I)
	}
	return []*Tensor{out}, nil
}

func opWhere(c *opCtx) ([]*Tensor, error) {
	if err := c.need(3); err != nil {
		return nil, err
	}
	cond, x, y := c.in[0], c.in[1], c.in[2]
	if x.DType != y.DType {
		return nil, fmt.Errorf("dtype mismatch: %s vs %s", x.DType, y.DType)
	}
	shape, err := broadcastShape(cond.Shape, x.Shape)
	if err != nil {
		return nil, err
	}
	if shape, err = broadcastShape(shape, y.Shape); err != nil {
		return nil, err
	}
	out := zeros(x.DType, shape)
	if out.Len() == 0 {
		return []*Tensor{out}, nil
	}
	if len(shape) == 0 {
		src := y
		if cond.I[0] != 0 {
			src = x
		}
		copyElem(out, 0, src, 0)
		return []*Tensor{out}, nil
	}
	last := len(shape) - 1
	n := shape[last]
	floats := x.isFloat()
	strides := [][]int{broadcastStrides(cond.Shape, shape), broadcastStrides(x.Shape, shape), broadcastStrides(y.Shape, shape)}
	parallelFor(out.Len()/n, n, func(lo, hi int) {
		walkRows(shape, strides, lo, hi, func(row int, off []int) {
			dst := row * n
			for j := 0; j < n; j++ {
				src, at := y, off[2]+j*strides[2][last]
				if cond.I[off[0]+j*strides[0][last]] != 0 {
					src, at = x, off[1]+j*strides[1][last]
				}
				if floats {
					out.F[dst+j] = src.F[at]
				} else {
					out.I[dst+j] = src.I[at]
				}
			}
		})
	})
	return []*Tensor{out}, nil
}

func opClip(c *opCtx) ([]*Tensor, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	x := c.in[0]
	out := zeros(x.DType, x.Shape)
	var lo, hi *Tensor
	if len(c.in) > 1 {
		lo = c.in[1]
	}
	if len(c.in) > 2 {
		hi = c.in[2]
	}
	if x.isFloat() {
		minV, maxV := float32(math.Inf(-1)), float32(math.Inf(1))
		if lo != nil {
			minV = lo.F[0]
		}
		if hi != nil {
			maxV = hi.F[0]
		}
		for i, v := range x.F {
			out.F[i] = min(max(v, minV), maxV)
		}
	} else {
		minV, maxV := int64(math.MinInt64), int64(math.MaxInt64)
		if lo != nil {
			minV = lo.I[0]
		}
		if hi != nil {
			maxV = hi.I[0]
		}
		for i, v := range x.I {
			out.I[i] = min(max(v, minV), maxV)
		}
	}
	return []*Tensor{out}, nil
}

func opIdentity(c *opCtx) ([]*Tensor, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	return []*Tensor{c.in[0]}, nil
}

var (
	erfNumerator   = [...]float32{-2.72614225801306e-10, 2.77068142495902e-08, -2.10102402082508e-06, -5.69250639462346e-05, -7.34990630326855e-04, -2.95459980854025e-03, -1.60960333262415e-02}
	erfDenominator = [...]float32{-1.45660718464996e-05, -2.13374055278905e-04, -1.68282697438203e-03, -7.37332916720468e-03, -1.42647390514189e-02}
)

// erf32 is a rational approximation of erf with a maximum absolute error near 4e-7.
func erf32(x float32) float32 {
	x = min(max(x, -4), 4)
	x2 := x * x
	num, den := erfNumerator[0], erfDenominator[0]
	for _, c := range erfNumerator[1:] {
		num = num*x2 + c
	}
	for _, c := range erfDenominator[1:] {
		den = den*x2 + c
	}
	return x * num / den
}

func tanh32(x float32) float32 { return float32(math.Tanh(float64(x))) }

func sigmoid32(x float32) float32 { return float32(1 / (1 + math.Exp(-float64(x)))) }

func opErf(c *opCtx) ([]*Tensor, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	x := c.in[0]
	if !x.isFloat() {
		return nil, fmt.Errorf("unsupported for %s", x.DType)
	}
	out := zeros(Float32, x.Shape)
	parallelFor(len(x.F), 12, func(lo, hi int) { erfSlice(out.F[lo:hi], x.F[lo:hi]) })
	return []*Tensor{out}, nil
}
