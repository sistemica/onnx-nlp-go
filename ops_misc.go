package onnxgo

import (
	"fmt"
	"math"
	"slices"
)

func reduce(fF func(acc, x float32) float32, initF float32, fI func(acc, x int64) int64, initI int64) opFunc {
	return func(c *opCtx) ([]*Tensor, error) {
		if err := c.need(1); err != nil {
			return nil, err
		}
		x := c.in[0]
		rank := len(x.Shape)
		var axes []int
		if len(c.in) > 1 && c.in[1] != nil {
			var err error
			if axes, err = intList(c.in[1]); err != nil {
				return nil, err
			}
		} else if a, ok := c.node.Ints("axes"); ok {
			for _, v := range a {
				axes = append(axes, int(v))
			}
		}
		if len(axes) == 0 {
			if c.node.Int("noop_with_empty_axes", 0) == 1 {
				return []*Tensor{x}, nil
			}
			for i := 0; i < rank; i++ {
				axes = append(axes, i)
			}
		}
		reduced := make([]bool, rank)
		for _, a := range axes {
			a, err := normAxis(a, rank)
			if err != nil {
				return nil, err
			}
			reduced[a] = true
		}
		keep := c.node.Int("keepdims", 1) == 1
		var shape []int
		keptShape := make([]int, rank)
		for i, d := range x.Shape {
			keptShape[i] = d
			if reduced[i] {
				keptShape[i] = 1
			}
			if !reduced[i] || keep {
				shape = append(shape, keptShape[i])
			}
		}
		if shape == nil {
			shape = []int{}
		}
		if x.isFloat() {
			if first := trailingAxes(reduced); first >= 0 {
				return []*Tensor{reduceRows(x, shape, first, fF, initF)}, nil
			}
		}
		outStrides := stridesOf(keptShape)
		for i := range outStrides {
			if reduced[i] {
				outStrides[i] = 0
			}
		}
		offs := offsetsFromStrides(x.Shape, outStrides, 0)
		out := zeros(x.DType, shape)
		if x.isFloat() {
			for i := range out.F {
				out.F[i] = initF
			}
			for i, o := range offs {
				out.F[o] = fF(out.F[o], x.F[i])
			}
		} else {
			for i := range out.I {
				out.I[i] = initI
			}
			for i, o := range offs {
				out.I[o] = fI(out.I[o], x.I[i])
			}
		}
		return []*Tensor{out}, nil
	}
}

// trailingAxes returns the first reduced axis if exactly the last axes are reduced, else -1.
func trailingAxes(reduced []bool) int {
	first := len(reduced)
	for first > 0 && reduced[first-1] {
		first--
	}
	for i := 0; i < first; i++ {
		if reduced[i] {
			return -1
		}
	}
	if first == len(reduced) {
		return -1
	}
	return first
}

func reduceRows(x *Tensor, shape []int, first int, fF func(acc, x float32) float32, initF float32) *Tensor {
	out := zeros(Float32, shape)
	width := numel(x.Shape[first:])
	parallelFor(len(out.F), width, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			acc := initF
			for _, v := range x.F[r*width : (r+1)*width] {
				acc = fF(acc, v)
			}
			out.F[r] = acc
		}
	})
	return out
}

func opReduceMean(c *opCtx) ([]*Tensor, error) {
	outs, err := reduceSum(c)
	if err != nil {
		return nil, err
	}
	x, out := c.in[0], outs[0]
	if !x.isFloat() {
		return nil, fmt.Errorf("ReduceMean supports float32 only")
	}
	if out.Len() > 0 && out != x {
		n := float32(x.Len() / out.Len())
		for i := range out.F {
			out.F[i] /= n
		}
	}
	return outs, nil
}

func opCumSum(c *opCtx) ([]*Tensor, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x := c.in[0]
	ax, err := c.in[1].scalarInt()
	if err != nil {
		return nil, err
	}
	axis, err := normAxis(int(ax), len(x.Shape))
	if err != nil {
		return nil, err
	}
	exclusive, reverse := c.node.Int("exclusive", 0) == 1, c.node.Int("reverse", 0) == 1
	d := x.Shape[axis]
	outer, inner := numel(x.Shape[:axis]), numel(x.Shape[axis+1:])
	out := zeros(x.DType, x.Shape)
	for o := 0; o < outer; o++ {
		for in := 0; in < inner; in++ {
			base := o*d*inner + in
			var accF float32
			var accI int64
			for s := 0; s < d; s++ {
				i := s
				if reverse {
					i = d - 1 - s
				}
				p := base + i*inner
				if x.isFloat() {
					if exclusive {
						out.F[p] = accF
						accF += x.F[p]
					} else {
						accF += x.F[p]
						out.F[p] = accF
					}
				} else {
					if exclusive {
						out.I[p] = accI
						accI += x.I[p]
					} else {
						accI += x.I[p]
						out.I[p] = accI
					}
				}
			}
		}
	}
	return []*Tensor{out}, nil
}

func opTopK(c *opCtx) ([]*Tensor, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x := c.in[0]
	kk, err := c.in[1].scalarInt()
	if err != nil {
		return nil, err
	}
	k := int(kk)
	axis, err := normAxis(int(c.node.Int("axis", -1)), len(x.Shape))
	if err != nil {
		return nil, err
	}
	largest := c.node.Int("largest", 1) == 1
	d := x.Shape[axis]
	if k > d {
		return nil, fmt.Errorf("TopK k=%d exceeds axis size %d", k, d)
	}
	outer, inner := numel(x.Shape[:axis]), numel(x.Shape[axis+1:])
	shape := slices.Clone(x.Shape)
	shape[axis] = k
	values, indices := zeros(x.DType, shape), zeros(Int64, shape)
	order := make([]int, d)
	for o := 0; o < outer; o++ {
		for in := 0; in < inner; in++ {
			base := o*d*inner + in
			for i := range order {
				order[i] = i
			}
			less := func(a, b int) bool {
				if x.isFloat() {
					va, vb := x.F[base+a*inner], x.F[base+b*inner]
					if largest {
						return va > vb
					}
					return va < vb
				}
				va, vb := x.I[base+a*inner], x.I[base+b*inner]
				if largest {
					return va > vb
				}
				return va < vb
			}
			slices.SortStableFunc(order, func(a, b int) int {
				switch {
				case less(a, b):
					return -1
				case less(b, a):
					return 1
				}
				return 0
			})
			for r := 0; r < k; r++ {
				dst := o*k*inner + r*inner + in
				src := base + order[r]*inner
				if x.isFloat() {
					values.F[dst] = x.F[src]
				} else {
					values.I[dst] = x.I[src]
				}
				indices.I[dst] = int64(order[r])
			}
		}
	}
	return []*Tensor{values, indices}, nil
}

func opScatterND(c *opCtx) ([]*Tensor, error) {
	if err := c.need(3); err != nil {
		return nil, err
	}
	data, idx, upd := c.in[0], c.in[1], c.in[2]
	if r := c.node.String("reduction", "none"); r != "none" {
		return nil, fmt.Errorf("unsupported reduction %q", r)
	}
	depth := idx.Shape[len(idx.Shape)-1]
	strides := stridesOf(data.Shape)
	slice := numel(data.Shape[depth:])
	out := zeros(data.DType, data.Shape)
	data.copyTo(out, 0, 0, data.Len())
	for q := 0; q < numel(idx.Shape[:len(idx.Shape)-1]); q++ {
		off := 0
		for j := 0; j < depth; j++ {
			ix := idx.I[q*depth+j]
			if ix < 0 {
				ix += int64(data.Shape[j])
			}
			if ix < 0 || ix >= int64(data.Shape[j]) {
				return nil, fmt.Errorf("scatter index %d out of range", ix)
			}
			off += int(ix) * strides[j]
		}
		upd.copyTo(out, off, q*slice, slice)
	}
	return []*Tensor{out}, nil
}

func opIf(c *opCtx) ([]*Tensor, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	branch := "else_branch"
	if c.in[0].I[0] != 0 {
		branch = "then_branch"
	}
	a := c.node.Attr[branch]
	if a == nil || a.G == nil {
		return nil, fmt.Errorf("missing %s", branch)
	}
	return runGraph(a.G, c.scope)
}

var reduceSum = reduce(func(acc, x float32) float32 { return acc + x }, 0,
	func(acc, x int64) int64 { return acc + x }, 0)

func init() {
	defer func() {
		for name, k := range f32Arith {
			ops[name] = withFastFloat32(ops[name], k)
		}
	}()
	ops = map[string]opFunc{
		"Abs":             unary(viaF64(math.Abs), func(x int64) int64 { return max(x, -x) }),
		"Add":             arith(nil, func(a, b int64) int64 { return a + b }),
		"And":             arith(nil, func(a, b int64) int64 { return a & b }),
		"Cast":            opCast,
		"Ceil":            unary(viaF64(math.Ceil), nil),
		"Clip":            opClip,
		"Concat":          opConcat,
		"Constant":        opConstant,
		"ConstantOfShape": opConstantOfShape,
		"CumSum":          opCumSum,
		"Div": arith(nil, func(a, b int64) int64 {
			if b == 0 {
				return 0
			}
			return a / b
		}),
		"Einsum":             opEinsum,
		"Equal":              compare(func(a, b float32) bool { return a == b }, func(a, b int64) bool { return a == b }),
		"Erf":                opErf,
		"Expand":             opExpand,
		"Gather":             opGather,
		"GatherElements":     opGatherElements,
		"Gemm":               opGemm,
		"Greater":            compare(func(a, b float32) bool { return a > b }, func(a, b int64) bool { return a > b }),
		"Identity":           opIdentity,
		"If":                 opIf,
		"IsNaN":              opIsNaN,
		"LayerNormalization": opLayerNorm,
		"Less":               compare(func(a, b float32) bool { return a < b }, func(a, b int64) bool { return a < b }),
		"LessOrEqual":        compare(func(a, b float32) bool { return a <= b }, func(a, b int64) bool { return a <= b }),
		"Log":                unary(viaF64(math.Log), nil),
		"MatMul":             opMatMul,
		"Mod":                opMod,
		"Mul":                arith(nil, func(a, b int64) int64 { return a * b }),
		"Neg":                unary(func(x float32) float32 { return -x }, func(x int64) int64 { return -x }),
		"Not":                unary(nil, func(x int64) int64 { return 1 - x }),
		"Or":                 arith(nil, func(a, b int64) int64 { return a | b }),
		"Range":              opRange,
		"ReduceMax": reduce(func(acc, x float32) float32 { return max(acc, x) }, float32(math.Inf(-1)),
			func(acc, x int64) int64 { return max(acc, x) }, math.MinInt64),
		"ReduceMean":      opReduceMean,
		"ReduceSum":       reduceSum,
		"Pow":             arith(nil, func(a, b int64) int64 { return int64(math.Pow(float64(a), float64(b))) }),
		"Relu":            unary(func(x float32) float32 { return max(x, 0) }, nil),
		"Reshape":         opReshape,
		"ScatterND":       opScatterND,
		"Shape":           opShape,
		"Sigmoid":         unary(sigmoid32, nil),
		"Sign":            unary(sign[float32], sign[int64]),
		"Slice":           opSlice,
		"Softmax":         opSoftmax,
		"Split":           opSplit,
		"Sqrt":            unary(viaF64(math.Sqrt), nil),
		"Squeeze":         opSqueeze,
		"Sub":             arith(nil, func(a, b int64) int64 { return a - b }),
		"Tile":            opTile,
		"TopK":            opTopK,
		"Transpose":       opTranspose,
		"Unsqueeze":       opUnsqueeze,
		"Where":           opWhere,
		"Xor":             arith(nil, func(a, b int64) int64 { return a ^ b }),
		opFusedGelu:       opFusedGeluRun,
		opFusedMatMulBias: opFusedMatMulBiasRun,
	}
}
