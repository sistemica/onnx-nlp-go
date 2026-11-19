package onnxgo

import (
	"fmt"
	"math"
	"slices"
	"sync/atomic"
)

func intList(t *Tensor) ([]int, error) {
	if t == nil || t.isFloat() {
		return nil, fmt.Errorf("expected integer tensor")
	}
	out := make([]int, len(t.I))
	for i, v := range t.I {
		out[i] = int(v)
	}
	return out, nil
}

func opConstant(c *opCtx) ([]*Tensor, error) {
	n := c.node
	switch {
	case n.Attr["value"] != nil:
		return []*Tensor{n.Attr["value"].T}, nil
	case n.Attr["value_float"] != nil:
		return []*Tensor{NewFloat([]int{}, []float32{n.Attr["value_float"].F})}, nil
	case n.Attr["value_floats"] != nil:
		f := n.Attr["value_floats"].Floats
		return []*Tensor{NewFloat([]int{len(f)}, f)}, nil
	case n.Attr["value_int"] != nil:
		return []*Tensor{NewInt([]int{}, []int64{n.Attr["value_int"].I})}, nil
	case n.Attr["value_ints"] != nil:
		v := n.Attr["value_ints"].Ints
		return []*Tensor{NewInt([]int{len(v)}, v)}, nil
	}
	return nil, fmt.Errorf("unsupported Constant attribute")
}

func opConstantOfShape(c *opCtx) ([]*Tensor, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	shape, err := intList(c.in[0])
	if err != nil {
		return nil, err
	}
	val := NewFloat([]int{1}, []float32{0})
	if a := c.node.Attr["value"]; a != nil {
		val = a.T
	}
	out := zeros(val.DType, shape)
	if val.isFloat() {
		for i := range out.F {
			out.F[i] = val.F[0]
		}
	} else {
		for i := range out.I {
			out.I[i] = val.I[0]
		}
	}
	return []*Tensor{out}, nil
}

func opShape(c *opCtx) ([]*Tensor, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	dims := c.in[0].Shape
	start, end := int(c.node.Int("start", 0)), int(c.node.Int("end", int64(len(dims))))
	if start < 0 {
		start += len(dims)
	}
	if end < 0 {
		end += len(dims)
	}
	start, end = min(max(start, 0), len(dims)), min(max(end, 0), len(dims))
	out := zeros(Int64, []int{max(end-start, 0)})
	for i := range out.I {
		out.I[i] = int64(dims[start+i])
	}
	return []*Tensor{out}, nil
}

func opReshape(c *opCtx) ([]*Tensor, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x := c.in[0]
	spec, err := intList(c.in[1])
	if err != nil {
		return nil, err
	}
	allowZero := c.node.Int("allowzero", 0) == 1
	shape := make([]int, len(spec))
	unknown, known := -1, 1
	for i, d := range spec {
		switch {
		case d == -1:
			unknown = i
			continue
		case d == 0 && !allowZero:
			if i >= len(x.Shape) {
				return nil, fmt.Errorf("reshape dimension %d has no counterpart in %v", i, x.Shape)
			}
			d = x.Shape[i]
		}
		shape[i] = d
		known *= d
	}
	if unknown >= 0 {
		if known == 0 {
			return nil, fmt.Errorf("cannot infer dimension in reshape to %v", spec)
		}
		shape[unknown] = x.Len() / known
	}
	if numel(shape) != x.Len() {
		return nil, fmt.Errorf("cannot reshape %v to %v", x.Shape, spec)
	}
	return []*Tensor{x.withShape(shape)}, nil
}

func opSqueeze(c *opCtx) ([]*Tensor, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	x := c.in[0]
	drop := make([]bool, len(x.Shape))
	if len(c.in) > 1 && c.in[1] != nil {
		axes, err := intList(c.in[1])
		if err != nil {
			return nil, err
		}
		for _, a := range axes {
			a, err := normAxis(a, len(x.Shape))
			if err != nil {
				return nil, err
			}
			drop[a] = true
		}
	} else {
		for i, d := range x.Shape {
			drop[i] = d == 1
		}
	}
	shape := []int{}
	for i, d := range x.Shape {
		if !drop[i] {
			shape = append(shape, d)
		}
	}
	return []*Tensor{x.withShape(shape)}, nil
}

func opUnsqueeze(c *opCtx) ([]*Tensor, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x := c.in[0]
	axes, err := intList(c.in[1])
	if err != nil {
		return nil, err
	}
	rank := len(x.Shape) + len(axes)
	add := make([]bool, rank)
	for _, a := range axes {
		a, err := normAxis(a, rank)
		if err != nil {
			return nil, err
		}
		add[a] = true
	}
	shape := make([]int, rank)
	src := 0
	for i := range shape {
		if add[i] {
			shape[i] = 1
		} else {
			shape[i] = x.Shape[src]
			src++
		}
	}
	return []*Tensor{x.withShape(shape)}, nil
}

func opSlice(c *opCtx) ([]*Tensor, error) {
	if err := c.need(3); err != nil {
		return nil, err
	}
	x := c.in[0]
	rank := len(x.Shape)
	starts, err := intList(c.in[1])
	if err != nil {
		return nil, err
	}
	ends, err := intList(c.in[2])
	if err != nil {
		return nil, err
	}
	axes := make([]int, len(starts))
	for i := range axes {
		axes[i] = i
	}
	if len(c.in) > 3 && c.in[3] != nil {
		if axes, err = intList(c.in[3]); err != nil {
			return nil, err
		}
	}
	steps := slices.Repeat([]int{1}, len(starts))
	if len(c.in) > 4 && c.in[4] != nil {
		if steps, err = intList(c.in[4]); err != nil {
			return nil, err
		}
	}
	inStrides := stridesOf(x.Shape)
	outShape := slices.Clone(x.Shape)
	effStrides := slices.Clone(inStrides)
	base := 0
	for i, ax := range axes {
		if ax, err = normAxis(ax, rank); err != nil {
			return nil, err
		}
		d, step := x.Shape[ax], steps[i]
		if step == 0 {
			return nil, fmt.Errorf("slice step is zero")
		}
		start, end := starts[i], ends[i]
		if start < 0 {
			start += d
		}
		if end < 0 {
			end += d
		}
		var n int
		if step > 0 {
			start, end = min(max(start, 0), d), min(max(end, 0), d)
			n = max((end-start+step-1)/step, 0)
		} else {
			start, end = min(max(start, 0), d-1), min(max(end, -1), d-1)
			n = max((start-end-step-1)/(-step), 0)
		}
		outShape[ax] = n
		effStrides[ax] = inStrides[ax] * step
		if n > 0 {
			base += start * inStrides[ax]
		}
	}
	return []*Tensor{stridedCopy(x, outShape, effStrides, base)}, nil
}

func opConcat(c *opCtx) ([]*Tensor, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	first := c.in[0]
	axis, err := normAxis(int(c.node.Int("axis", 0)), len(first.Shape))
	if err != nil {
		return nil, err
	}
	shape := slices.Clone(first.Shape)
	shape[axis] = 0
	for _, t := range c.in {
		if t.DType != first.DType {
			return nil, fmt.Errorf("dtype mismatch: %s vs %s", first.DType, t.DType)
		}
		shape[axis] += t.Shape[axis]
	}
	outer := numel(shape[:axis])
	out := zeros(first.DType, shape)
	pos := 0
	for o := 0; o < outer; o++ {
		for _, t := range c.in {
			chunk := numel(t.Shape[axis:])
			t.copyTo(out, pos, o*chunk, chunk)
			pos += chunk
		}
	}
	return []*Tensor{out}, nil
}

func opSplit(c *opCtx) ([]*Tensor, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	x := c.in[0]
	axis, err := normAxis(int(c.node.Int("axis", 0)), len(x.Shape))
	if err != nil {
		return nil, err
	}
	var sizes []int
	if len(c.in) > 1 && c.in[1] != nil {
		if sizes, err = intList(c.in[1]); err != nil {
			return nil, err
		}
	} else {
		n := max(int(c.node.Int("num_outputs", int64(len(c.node.Out)))), 1)
		per := (x.Shape[axis] + n - 1) / n
		for i, rem := 0, x.Shape[axis]; i < n; i++ {
			sizes = append(sizes, min(per, rem))
			rem -= sizes[i]
		}
	}
	outer := numel(x.Shape[:axis])
	inner := numel(x.Shape[axis+1:])
	rowLen := x.Shape[axis] * inner
	outs := make([]*Tensor, len(sizes))
	offset := 0
	for k, sz := range sizes {
		shape := slices.Clone(x.Shape)
		shape[axis] = sz
		out := zeros(x.DType, shape)
		chunk := sz * inner
		for o := 0; o < outer; o++ {
			x.copyTo(out, o*chunk, o*rowLen+offset*inner, chunk)
		}
		outs[k] = out
		offset += sz
	}
	return outs, nil
}

func opGather(c *opCtx) ([]*Tensor, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x, idx := c.in[0], c.in[1]
	axis, err := normAxis(int(c.node.Int("axis", 0)), len(x.Shape))
	if err != nil {
		return nil, err
	}
	d := x.Shape[axis]
	outer, inner := numel(x.Shape[:axis]), numel(x.Shape[axis+1:])
	shape := slices.Concat(x.Shape[:axis], idx.Shape, x.Shape[axis+1:])
	out := zeros(x.DType, shape)
	pos := 0
	for o := 0; o < outer; o++ {
		for _, ix := range idx.I {
			if ix < 0 {
				ix += int64(d)
			}
			if ix < 0 || ix >= int64(d) {
				return nil, fmt.Errorf("gather index %d out of range for axis size %d", ix, d)
			}
			x.copyTo(out, pos, (o*d+int(ix))*inner, inner)
			pos += inner
		}
	}
	return []*Tensor{out}, nil
}

func opGatherElements(c *opCtx) ([]*Tensor, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x, idx := c.in[0], c.in[1]
	axis, err := normAxis(int(c.node.Int("axis", 0)), len(x.Shape))
	if err != nil {
		return nil, err
	}
	strides := stridesOf(x.Shape)
	walk := slices.Clone(strides)
	walk[axis] = 0
	d := int64(x.Shape[axis])
	out := zeros(x.DType, idx.Shape)
	if out.Len() == 0 {
		return []*Tensor{out}, nil
	}
	n := idx.Shape[len(idx.Shape)-1]
	floats := x.isFloat()
	var failed atomic.Bool
	var failedIndex atomic.Int64
	parallelFor(out.Len()/n, n*4, func(lo, hi int) {
		pos := lo * n
		forEachRowRange(idx.Shape, walk, 0, lo, hi, func(start, stride, n int) {
			for j := 0; j < n; j++ {
				ix := idx.I[pos+j]
				if ix < 0 {
					ix += d
				}
				if ix < 0 || ix >= d {
					failedIndex.Store(ix)
					failed.Store(true)
					continue
				}
				at := start + j*stride + int(ix)*strides[axis]
				if floats {
					out.F[pos+j] = x.F[at]
				} else {
					out.I[pos+j] = x.I[at]
				}
			}
			pos += n
		})
	})
	if failed.Load() {
		return nil, fmt.Errorf("gather index %d out of range for axis size %d", failedIndex.Load(), d)
	}
	return []*Tensor{out}, nil
}

func opTranspose(c *opCtx) ([]*Tensor, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	x := c.in[0]
	rank := len(x.Shape)
	perm := make([]int, rank)
	if p, ok := c.node.Ints("perm"); ok {
		for i, v := range p {
			perm[i] = int(v)
		}
	} else {
		for i := range perm {
			perm[i] = rank - 1 - i
		}
	}
	inStrides := stridesOf(x.Shape)
	shape, strides := make([]int, rank), make([]int, rank)
	for i, p := range perm {
		shape[i], strides[i] = x.Shape[p], inStrides[p]
	}
	return []*Tensor{stridedCopy(x, shape, strides, 0)}, nil
}

func opExpand(c *opCtx) ([]*Tensor, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x := c.in[0]
	target, err := intList(c.in[1])
	if err != nil {
		return nil, err
	}
	shape, err := broadcastShape(x.Shape, target)
	if err != nil {
		return nil, err
	}
	if slices.Equal(x.Shape, shape) {
		return []*Tensor{x}, nil
	}
	return []*Tensor{stridedCopy(x, shape, broadcastStrides(x.Shape, shape), 0)}, nil
}

func opTile(c *opCtx) ([]*Tensor, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x := c.in[0]
	reps, err := intList(c.in[1])
	if err != nil {
		return nil, err
	}
	if len(reps) != len(x.Shape) {
		return nil, fmt.Errorf("Tile repeats %v do not match rank of %v", reps, x.Shape)
	}
	if !slices.ContainsFunc(reps, func(r int) bool { return r != 1 }) {
		return []*Tensor{x}, nil
	}
	rank := len(x.Shape)
	expanded, strides := make([]int, 0, 2*rank), make([]int, 0, 2*rank)
	inStrides := stridesOf(x.Shape)
	outShape := make([]int, rank)
	for i, r := range reps {
		expanded = append(expanded, r, x.Shape[i])
		strides = append(strides, 0, inStrides[i])
		outShape[i] = r * x.Shape[i]
	}
	return []*Tensor{stridedCopy(x, expanded, strides, 0).withShape(outShape)}, nil
}

func opRange(c *opCtx) ([]*Tensor, error) {
	if err := c.need(3); err != nil {
		return nil, err
	}
	start, limit, delta := c.in[0], c.in[1], c.in[2]
	if start.isFloat() {
		s, l, d := start.F[0], limit.F[0], delta.F[0]
		n := max(int(math.Ceil(float64(l-s)/float64(d))), 0)
		out := zeros(Float32, []int{n})
		for i := range out.F {
			out.F[i] = s + float32(i)*d
		}
		return []*Tensor{out}, nil
	}
	s, l, d := start.I[0], limit.I[0], delta.I[0]
	n := max(int(math.Ceil(float64(l-s)/float64(d))), 0)
	out := zeros(Int64, []int{n})
	for i := range out.I {
		out.I[i] = s + int64(i)*d
	}
	return []*Tensor{out}, nil
}
