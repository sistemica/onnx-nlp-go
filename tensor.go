package onnxgo

import "fmt"

type DType uint8

const (
	Float32 DType = iota + 1
	Int64
	Bool
)

func (d DType) String() string {
	switch d {
	case Float32:
		return "float32"
	case Int64:
		return "int64"
	case Bool:
		return "bool"
	}
	return "invalid"
}

// Tensor stores Float32 in F, and Int64 and Bool (0 or 1) in I.
// Tensors are immutable once produced; reshaping ops share storage.
type Tensor struct {
	DType DType
	Shape []int
	F     []float32
	I     []int64

	// static marks weights that never change, so devices may cache them; pack caches their packed panels.
	static bool
	pack   *packCache
}

func numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func NewFloat(shape []int, data []float32) *Tensor {
	return &Tensor{DType: Float32, Shape: shape, F: data}
}

func NewInt(shape []int, data []int64) *Tensor {
	return &Tensor{DType: Int64, Shape: shape, I: data}
}

func NewBool(shape []int, data []bool) *Tensor {
	t := &Tensor{DType: Bool, Shape: shape, I: make([]int64, len(data))}
	for i, v := range data {
		if v {
			t.I[i] = 1
		}
	}
	return t
}

func zeros(dt DType, shape []int) *Tensor {
	n := numel(shape)
	if dt == Float32 {
		return &Tensor{DType: dt, Shape: shape, F: make([]float32, n)}
	}
	return &Tensor{DType: dt, Shape: shape, I: make([]int64, n)}
}

func (t *Tensor) Len() int { return numel(t.Shape) }

func (t *Tensor) isFloat() bool { return t.DType == Float32 }

func (t *Tensor) Bools() []bool {
	out := make([]bool, len(t.I))
	for i, v := range t.I {
		out[i] = v != 0
	}
	return out
}

func (t *Tensor) withShape(shape []int) *Tensor {
	return &Tensor{DType: t.DType, Shape: shape, F: t.F, I: t.I, static: t.static}
}

func (t *Tensor) scalarInt() (int64, error) {
	if t.isFloat() || len(t.I) != 1 {
		return 0, fmt.Errorf("expected integer scalar, got %s %v", t.DType, t.Shape)
	}
	return t.I[0], nil
}

func (t *Tensor) pick(shape []int, offs []int) *Tensor {
	out := zeros(t.DType, shape)
	if t.isFloat() {
		for i, o := range offs {
			out.F[i] = t.F[o]
		}
	} else {
		for i, o := range offs {
			out.I[i] = t.I[o]
		}
	}
	return out
}

func (t *Tensor) copyTo(dst *Tensor, dstOff, srcOff, n int) {
	if t.isFloat() {
		copy(dst.F[dstOff:dstOff+n], t.F[srcOff:srcOff+n])
	} else {
		copy(dst.I[dstOff:dstOff+n], t.I[srcOff:srcOff+n])
	}
}

func stridesOf(shape []int) []int {
	s := make([]int, len(shape))
	acc := 1
	for i := len(shape) - 1; i >= 0; i-- {
		s[i] = acc
		acc *= shape[i]
	}
	return s
}

func normAxis(axis, rank int) (int, error) {
	if axis < -rank || axis >= rank {
		return 0, fmt.Errorf("axis %d out of range for rank %d", axis, rank)
	}
	if axis < 0 {
		axis += rank
	}
	return axis, nil
}

func broadcastShape(a, b []int) ([]int, error) {
	n := max(len(a), len(b))
	out := make([]int, n)
	for i := 0; i < n; i++ {
		da, db := 1, 1
		if j := len(a) - n + i; j >= 0 {
			da = a[j]
		}
		if j := len(b) - n + i; j >= 0 {
			db = b[j]
		}
		switch {
		case da == db || db == 1:
			out[i] = da
		case da == 1:
			out[i] = db
		default:
			return nil, fmt.Errorf("cannot broadcast %v with %v", a, b)
		}
	}
	return out, nil
}

// broadcastOffsets maps every element of out to its source offset in a tensor of shape src.
// It returns nil when src already has the shape of out.
func broadcastOffsets(src, out []int) []int {
	if len(src) == len(out) {
		same := true
		for i := range src {
			if src[i] != out[i] {
				same = false
				break
			}
		}
		if same {
			return nil
		}
	}
	srcStrides := stridesOf(src)
	eff := make([]int, len(out))
	for i := range out {
		if j := len(src) - len(out) + i; j >= 0 && src[j] != 1 {
			eff[i] = srcStrides[j]
		}
	}
	return offsetsFromStrides(out, eff, 0)
}

func offsetsFromStrides(shape, strides []int, base int) []int {
	offs := make([]int, numel(shape))
	if len(offs) == 0 {
		return offs
	}
	pos := 0
	forEachRow(shape, strides, base, func(start, stride, n int) {
		for j := 0; j < n; j++ {
			offs[pos+j] = start + j*stride
		}
		pos += n
	})
	return offs
}

// forEachRow calls fn for every run along the last dimension of shape, in row-major
// order, with the source offset where the run starts and its stride.
func forEachRow(shape, strides []int, base int, fn func(start, stride, n int)) {
	forEachRowRange(shape, strides, base, 0, numel(shape[:max(len(shape)-1, 0)]), fn)
}

// forEachRowRange is forEachRow restricted to the rows [lo, hi).
func forEachRowRange(shape, strides []int, base, lo, hi int, fn func(start, stride, n int)) {
	if len(shape) == 0 {
		fn(base, 0, 1)
		return
	}
	last := len(shape) - 1
	n, stride := shape[last], strides[last]
	idx := rowIndex(lo, shape[:last])
	cur := base + rowOffset(lo, shape[:last], strides)
	for r := lo; r < hi; r++ {
		fn(cur, stride, n)
		for d := last - 1; d >= 0; d-- {
			idx[d]++
			cur += strides[d]
			if idx[d] < shape[d] {
				break
			}
			cur -= strides[d] * shape[d]
			idx[d] = 0
		}
	}
}

// stridedCopy materializes the strided view (shape, strides, base) of t.
func stridedCopy(t *Tensor, shape, strides []int, base int) *Tensor {
	out := zeros(t.DType, shape)
	if out.Len() == 0 {
		return out
	}
	if len(shape) == 0 {
		forEachRow(shape, strides, base, func(start, _, _ int) { copyElem(out, 0, t, start) })
		return out
	}
	n := shape[len(shape)-1]
	rows := out.Len() / n
	parallelFor(rows, n, func(lo, hi int) {
		pos := lo * n
		forEachRowRange(shape, strides, base, lo, hi, func(start, stride, n int) {
			switch {
			case stride == 1 && t.isFloat():
				copy(out.F[pos:pos+n], t.F[start:start+n])
			case stride == 1:
				copy(out.I[pos:pos+n], t.I[start:start+n])
			case t.isFloat():
				for j := 0; j < n; j++ {
					out.F[pos+j] = t.F[start+j*stride]
				}
			default:
				for j := 0; j < n; j++ {
					out.I[pos+j] = t.I[start+j*stride]
				}
			}
			pos += n
		})
	})
	return out
}

func copyElem(dst *Tensor, di int, src *Tensor, si int) {
	if src.isFloat() {
		dst.F[di] = src.F[si]
	} else {
		dst.I[di] = src.I[si]
	}
}

// broadcastStrides returns the strides that read a tensor of shape src as shape out.
func broadcastStrides(src, out []int) []int {
	srcStrides := stridesOf(src)
	eff := make([]int, len(out))
	for i := range out {
		if j := len(src) - len(out) + i; j >= 0 && src[j] != 1 {
			eff[i] = srcStrides[j]
		}
	}
	return eff
}

func rowIndex(r int, outer []int) []int {
	idx := make([]int, len(outer))
	for d := len(outer) - 1; d >= 0; d-- {
		idx[d] = r % outer[d]
		r /= outer[d]
	}
	return idx
}

func rowOffset(r int, outer, strides []int) int {
	off := 0
	for d, i := range rowIndex(r, outer) {
		off += i * strides[d]
	}
	return off
}

// walkRows calls fn for rows [lo, hi) of shape, passing each operand's offset at the start of the row.
func walkRows(shape []int, strides [][]int, lo, hi int, fn func(row int, offsets []int)) {
	last := len(shape) - 1
	outer := shape[:last]
	idx := rowIndex(lo, outer)
	offs := make([]int, len(strides))
	for k, st := range strides {
		offs[k] = rowOffset(lo, outer, st)
	}
	for r := lo; r < hi; r++ {
		fn(r, offs)
		for d := last - 1; d >= 0; d-- {
			idx[d]++
			for k, st := range strides {
				offs[k] += st[d]
			}
			if idx[d] < shape[d] {
				break
			}
			for k, st := range strides {
				offs[k] -= st[d] * shape[d]
			}
			idx[d] = 0
		}
	}
}
