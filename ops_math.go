package onnxgo

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

// packOf returns the pack cache of b when b is one whole matrix.
func packOf(b *Tensor) *packCache {
	if len(b.Shape) == 2 {
		return b.pack
	}
	return nil
}

func opMatMul(c *opCtx) ([]*Tensor, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	a, b := c.in[0], c.in[1]
	if !a.isFloat() || !b.isFloat() {
		return nil, fmt.Errorf("MatMul supports float32 only")
	}
	if len(a.Shape) < 2 || len(b.Shape) < 2 {
		return nil, fmt.Errorf("MatMul needs rank >= 2, got %v and %v", a.Shape, b.Shape)
	}
	ra, rb := len(a.Shape), len(b.Shape)
	m, k, n := a.Shape[ra-2], a.Shape[ra-1], b.Shape[rb-1]
	if b.Shape[rb-2] != k {
		return nil, fmt.Errorf("MatMul shape mismatch %v x %v", a.Shape, b.Shape)
	}
	if rb == 2 && ra > 2 {
		rows := a.Len() / k
		out := zeros(Float32, append(slices.Clone(a.Shape[:ra-1]), n))
		runGemms([]gemmJob{{out.F, a.F, b.F, rows, k, n, b.static, b.pack}})
		return []*Tensor{out}, nil
	}
	batch, err := broadcastShape(a.Shape[:ra-2], b.Shape[:rb-2])
	if err != nil {
		return nil, err
	}
	offA := broadcastOffsets(a.Shape[:ra-2], batch)
	offB := broadcastOffsets(b.Shape[:rb-2], batch)
	out := zeros(Float32, append(slices.Clone(batch), m, n))
	jobs := make([]gemmJob, numel(batch))
	for bi := range jobs {
		ao, bo, oo := at(offA, bi)*m*k, at(offB, bi)*k*n, bi*m*n
		jobs[bi] = gemmJob{out.F[oo : oo+m*n], a.F[ao : ao+m*k], b.F[bo : bo+k*n], m, k, n, b.static, packOf(b)}
	}
	runGemms(jobs)
	return []*Tensor{out}, nil
}

func transpose2D(f []float32, rows, cols int) []float32 {
	out := make([]float32, len(f))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out[j*rows+i] = f[i*cols+j]
		}
	}
	return out
}

func opGemm(c *opCtx) ([]*Tensor, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	a, b := c.in[0], c.in[1]
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("Gemm needs rank-2 inputs")
	}
	af, bf := a.F, b.F
	m, k := a.Shape[0], a.Shape[1]
	if c.node.Int("transA", 0) == 1 {
		af, m, k = transpose2D(af, a.Shape[0], a.Shape[1]), a.Shape[1], a.Shape[0]
	}
	kb, n := b.Shape[0], b.Shape[1]
	if c.node.Int("transB", 0) == 1 {
		bf, kb, n = transpose2D(bf, b.Shape[0], b.Shape[1]), b.Shape[1], b.Shape[0]
	}
	if kb != k {
		return nil, fmt.Errorf("Gemm shape mismatch")
	}
	out := zeros(Float32, []int{m, n})
	runGemms([]gemmJob{{out.F, af, bf, m, k, n, false, nil}})
	alpha, beta := c.node.Float("alpha", 1), c.node.Float("beta", 1)
	if alpha != 1 {
		for i := range out.F {
			out.F[i] *= alpha
		}
	}
	if len(c.in) > 2 && c.in[2] != nil {
		bias := c.in[2]
		offs := broadcastOffsets(bias.Shape, out.Shape)
		for i := range out.F {
			out.F[i] += beta * bias.F[at(offs, i)]
		}
	}
	return []*Tensor{out}, nil
}

func opEinsum(c *opCtx) ([]*Tensor, error) {
	eq := strings.ReplaceAll(c.node.String("equation", ""), " ", "")
	lhs, rhs, ok := strings.Cut(eq, "->")
	if !ok || strings.Contains(eq, "...") {
		return nil, fmt.Errorf("unsupported einsum equation %q", eq)
	}
	terms := strings.Split(lhs, ",")
	if len(terms) != len(c.in) {
		return nil, fmt.Errorf("einsum %q expects %d inputs, got %d", eq, len(terms), len(c.in))
	}
	var labels []rune
	size := map[rune]int{}
	for i, term := range terms {
		if len(term) != len(c.in[i].Shape) || !c.in[i].isFloat() {
			return nil, fmt.Errorf("einsum operand %d does not match %q", i, term)
		}
		for j, l := range term {
			if d, seen := size[l]; seen && d != c.in[i].Shape[j] {
				return nil, fmt.Errorf("einsum label %c has sizes %d and %d", l, d, c.in[i].Shape[j])
			}
			if _, seen := size[l]; !seen {
				labels = append(labels, l)
			}
			size[l] = c.in[i].Shape[j]
		}
	}
	outShape := make([]int, 0, len(rhs))
	for _, l := range rhs {
		d, ok := size[l]
		if !ok {
			return nil, fmt.Errorf("einsum output label %c not in inputs", l)
		}
		outShape = append(outShape, d)
	}
	if out := einsumGemm(terms, rhs, c.in, size); out != nil {
		return out, nil
	}
	labelStrides := func(term string, shape []int) []int {
		st := stridesOf(shape)
		out := make([]int, len(labels))
		for j, l := range term {
			out[slices.Index(labels, l)] += st[j]
		}
		return out
	}
	strides := make([][]int, 0, len(terms)+1)
	for i, term := range terms {
		strides = append(strides, labelStrides(term, c.in[i].Shape))
	}
	strides = append(strides, labelStrides(rhs, outShape))
	dims := make([]int, len(labels))
	total := 1
	for i, l := range labels {
		dims[i] = size[l]
		total *= dims[i]
	}
	out := zeros(Float32, outShape)
	offs := make([]int, len(strides))
	idx := make([]int, len(labels))
	last := len(terms)
	for it := 0; it < total; it++ {
		prod := float32(1)
		for k := 0; k < last; k++ {
			prod *= c.in[k].F[offs[k]]
		}
		out.F[offs[last]] += prod
		for d := len(labels) - 1; d >= 0; d-- {
			idx[d]++
			for k := range offs {
				offs[k] += strides[k][d]
			}
			if idx[d] < dims[d] {
				break
			}
			for k := range offs {
				offs[k] -= strides[k][d] * dims[d]
			}
			idx[d] = 0
		}
	}
	return []*Tensor{out}, nil
}

func opLayerNorm(c *opCtx) ([]*Tensor, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x, scale := c.in[0], c.in[1]
	axis, err := normAxis(int(c.node.Int("axis", -1)), len(x.Shape))
	if err != nil {
		return nil, err
	}
	eps := float64(c.node.Float("epsilon", 1e-5))
	rows, width := numel(x.Shape[:axis]), numel(x.Shape[axis:])
	if len(scale.F) != width {
		return nil, fmt.Errorf("scale has %d values for a normalized width of %d", len(scale.F), width)
	}
	var bias []float32
	if len(c.in) > 2 && c.in[2] != nil {
		if bias = c.in[2].F; len(bias) != width {
			return nil, fmt.Errorf("bias has %d values for a normalized width of %d", len(bias), width)
		}
	}
	out := zeros(Float32, x.Shape)
	mean := zeros(Float32, append(slices.Clone(x.Shape[:axis]), 1))
	invStd := zeros(Float32, mean.Shape)
	parallelFor(rows, width*8, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := x.F[r*width : (r+1)*width]
			var sum float64
			for _, v := range row {
				sum += float64(v)
			}
			mu := sum / float64(width)
			var sq float64
			for _, v := range row {
				d := float64(v) - mu
				sq += d * d
			}
			inv := 1 / math.Sqrt(sq/float64(width)+eps)
			mean.F[r], invStd.F[r] = float32(mu), float32(inv)
			dst := out.F[r*width : (r+1)*width]
			for i, v := range row {
				y := float32((float64(v) - mu) * inv * float64(scale.F[i]))
				if bias != nil {
					y += bias[i]
				}
				dst[i] = y
			}
		}
	})
	return []*Tensor{out, mean, invStd}, nil
}

func opSoftmax(c *opCtx) ([]*Tensor, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	x := c.in[0]
	axis, err := normAxis(int(c.node.Int("axis", -1)), len(x.Shape))
	if err != nil {
		return nil, err
	}
	d := x.Shape[axis]
	outer, inner := numel(x.Shape[:axis]), numel(x.Shape[axis+1:])
	out := zeros(Float32, x.Shape)
	if inner == 1 {
		parallelFor(outer, d*4, func(lo, hi int) {
			for r := lo; r < hi; r++ {
				in, dst := x.F[r*d:(r+1)*d], out.F[r*d:(r+1)*d]
				top := float32(math.Inf(-1))
				for _, v := range in {
					top = max(top, v)
				}
				scaleSlice(dst, 1/expSumShift(dst, in, top))
			}
		})
		return []*Tensor{out}, nil
	}
	parallelFor(outer*inner, d*16, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := (r/inner)*d*inner + r%inner
			top := float32(math.Inf(-1))
			for i := 0; i < d; i++ {
				top = max(top, x.F[base+i*inner])
			}
			var sum float64
			for i := 0; i < d; i++ {
				e := math.Exp(float64(x.F[base+i*inner] - top))
				out.F[base+i*inner] = float32(e)
				sum += e
			}
			for i := 0; i < d; i++ {
				out.F[base+i*inner] = float32(float64(out.F[base+i*inner]) / sum)
			}
		}
	})
	return []*Tensor{out}, nil
}
