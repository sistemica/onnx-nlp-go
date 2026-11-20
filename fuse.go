package onnxnlp

import (
	"fmt"
	"slices"
)

const (
	opFusedGelu       = "FusedGelu"
	opFusedMatMulBias = "FusedMatMulBias"
)

// fuser rewrites chains of operators that exporters emit for GELU, layer normalization and
// linear layers into single operators. A chain is only fused when none of its intermediate
// results is used anywhere else.
type fuser struct {
	g         *Graph
	nodes     []*Node
	producer  map[string]int
	consumers map[string][]int
	uses      map[string]int
	removed   []bool
	replaced  map[int]*Node
}

func fuseNodes(g *Graph) *Graph {
	f := &fuser{
		g: g, nodes: g.Nodes, producer: map[string]int{}, consumers: map[string][]int{}, uses: map[string]int{},
		removed: make([]bool, len(g.Nodes)), replaced: map[int]*Node{},
	}
	for i, n := range g.Nodes {
		for _, out := range n.Out {
			f.producer[out] = i
		}
		for _, in := range n.In {
			f.consumers[in] = append(f.consumers[in], i)
		}
		countUses(n, f.uses)
	}
	for _, out := range g.Outputs {
		f.uses[out]++
	}
	for i := range f.nodes {
		f.fuseGelu(i)
	}
	for i := range f.nodes {
		f.fuseLayerNorm(i)
	}
	for i := range f.nodes {
		f.fuseMatMulBias(i)
	}
	fused := Graph{Name: g.Name, Inputs: g.Inputs, Outputs: g.Outputs, Initializers: g.Initializers}
	for i, n := range g.Nodes {
		switch {
		case f.removed[i]:
		case f.replaced[i] != nil:
			fused.Nodes = append(fused.Nodes, f.replaced[i])
		default:
			fused.Nodes = append(fused.Nodes, n)
		}
	}
	return &fused
}

// countUses counts every reference to a value by a node, including references from nested graphs.
func countUses(n *Node, uses map[string]int) {
	for _, in := range n.In {
		if in != "" {
			uses[in]++
		}
	}
	for _, a := range n.Attr {
		if a.G == nil {
			continue
		}
		for _, inner := range a.G.Nodes {
			countUses(inner, uses)
		}
		for _, out := range a.G.Outputs {
			uses[out]++
		}
	}
}

func (f *fuser) node(i int) (*Node, bool) {
	if i < 0 || i >= len(f.nodes) || f.removed[i] || f.replaced[i] != nil {
		return nil, false
	}
	return f.nodes[i], true
}

// soleUser returns the one node consuming name, provided nothing else uses it.
func (f *fuser) soleUser(name string, op string) (int, *Node, bool) {
	users := f.consumers[name]
	if f.uses[name] != 1 || len(users) != 1 {
		return 0, nil, false
	}
	n, ok := f.node(users[0])
	if !ok || n.Op != op {
		return 0, nil, false
	}
	return users[0], n, true
}

func (f *fuser) scalarConstant(name string) (float32, bool) {
	t, ok := f.g.Initializers[name]
	if !ok || t.DType != Float32 || len(t.F) != 1 {
		return 0, false
	}
	return t.F[0], true
}

func (f *fuser) vectorConstant(name string) bool {
	t, ok := f.g.Initializers[name]
	return ok && t.DType == Float32 && len(t.Shape) == 1
}

// otherInput returns the input of a two-input node that is not name, which the node must consume.
func otherInput(n *Node, name string) (string, bool) {
	if len(n.In) != 2 {
		return "", false
	}
	if n.In[0] == name {
		return n.In[1], true
	}
	return n.In[0], true
}

func (f *fuser) replace(last int, repl *Node, absorbed ...int) {
	for _, i := range absorbed {
		f.removed[i] = true
	}
	f.replaced[last] = repl
}

// fuseGelu matches x * (erf(x / c1) + c2) * c3.
func (f *fuser) fuseGelu(erfIdx int) {
	erf, ok := f.node(erfIdx)
	if !ok || erf.Op != "Erf" || len(erf.In) != 1 {
		return
	}
	divIdx, ok := f.producer[erf.In[0]]
	div, ok2 := f.node(divIdx)
	if !ok || !ok2 || div.Op != "Div" || len(div.In) != 2 {
		return
	}
	if _, _, ok := f.soleUser(div.Out[0], "Erf"); !ok {
		return
	}
	c1, ok := f.scalarConstant(div.In[1])
	if !ok {
		return
	}
	x := div.In[0]
	addIdx, add, ok := f.soleUser(erf.Out[0], "Add")
	if !ok {
		return
	}
	c2name, ok := otherInput(add, erf.Out[0])
	if !ok {
		return
	}
	c2, ok := f.scalarConstant(c2name)
	if !ok {
		return
	}
	mul1Idx, mul1, ok := f.soleUser(add.Out[0], "Mul")
	if !ok {
		return
	}
	if other, ok := otherInput(mul1, add.Out[0]); !ok || other != x {
		return
	}
	mul2Idx, mul2, ok := f.soleUser(mul1.Out[0], "Mul")
	if !ok {
		return
	}
	c3name, ok := otherInput(mul2, mul1.Out[0])
	if !ok {
		return
	}
	c3, ok := f.scalarConstant(c3name)
	if !ok {
		return
	}
	f.replace(mul2Idx, &Node{
		Name: mul2.Name + "_gelu", Op: opFusedGelu, In: []string{x}, Out: mul2.Out,
		Attr: mkFloatAttrs(map[string]float32{"divisor": c1, "offset": c2, "scale": c3}),
	}, divIdx, erfIdx, addIdx, mul1Idx)
}

func mkFloatAttrs(values map[string]float32) map[string]*Attr {
	attrs := make(map[string]*Attr, len(values))
	for name, v := range values {
		attrs[name] = &Attr{F: v}
	}
	return attrs
}

func (f *fuser) lastAxisMean(n *Node) bool {
	if n.Op != "ReduceMean" || len(n.In) != 1 || n.Int("keepdims", 1) != 1 {
		return false
	}
	axes, ok := n.Ints("axes")
	return ok && slices.Equal(axes, []int64{-1})
}

// fuseLayerNorm matches (x - mean(x)) / sqrt(mean((x - mean(x))^2) + eps) * w + b over the last axis.
func (f *fuser) fuseLayerNorm(meanIdx int) {
	mean, ok := f.node(meanIdx)
	if !ok || !f.lastAxisMean(mean) {
		return
	}
	x := mean.In[0]
	subIdx, sub, ok := f.soleUser(mean.Out[0], "Sub")
	if !ok || len(sub.In) != 2 || sub.In[0] != x || sub.In[1] != mean.Out[0] {
		return
	}
	users := f.consumers[sub.Out[0]]
	if f.uses[sub.Out[0]] != 2 || len(users) != 2 {
		return
	}
	var powIdx, divIdx int
	var pow, div *Node
	for _, u := range users {
		switch n := f.nodes[u]; n.Op {
		case "Pow":
			powIdx, pow = u, n
		case "Div":
			divIdx, div = u, n
		}
	}
	if pow == nil || div == nil || len(pow.In) != 2 || pow.In[0] != sub.Out[0] || len(div.In) != 2 || div.In[0] != sub.Out[0] {
		return
	}
	if two, ok := f.scalarConstant(pow.In[1]); !ok || two != 2 {
		return
	}
	var1Idx, var1, ok := f.soleUser(pow.Out[0], "ReduceMean")
	if !ok || !f.lastAxisMean(var1) {
		return
	}
	addIdx, add, ok := f.soleUser(var1.Out[0], "Add")
	if !ok {
		return
	}
	epsName, ok := otherInput(add, var1.Out[0])
	if !ok {
		return
	}
	eps, ok := f.scalarConstant(epsName)
	if !ok {
		return
	}
	sqrtIdx, sqrt, ok := f.soleUser(add.Out[0], "Sqrt")
	if !ok || div.In[1] != sqrt.Out[0] || f.uses[sqrt.Out[0]] != 1 {
		return
	}
	mulIdx, mul, ok := f.soleUser(div.Out[0], "Mul")
	if !ok {
		return
	}
	weight, ok := otherInput(mul, div.Out[0])
	if !ok || !f.vectorConstant(weight) {
		return
	}
	biasIdx, biasAdd, ok := f.soleUser(mul.Out[0], "Add")
	if !ok {
		return
	}
	bias, ok := otherInput(biasAdd, mul.Out[0])
	if !ok || !f.vectorConstant(bias) {
		return
	}
	f.replace(biasIdx, &Node{
		Name: biasAdd.Name + "_layernorm", Op: "LayerNormalization", In: []string{x, weight, bias}, Out: biasAdd.Out,
		Attr: map[string]*Attr{"epsilon": {F: eps}, "axis": {I: -1}},
	}, meanIdx, subIdx, powIdx, var1Idx, addIdx, sqrtIdx, divIdx, mulIdx)
}

// fuseMatMulBias matches x @ W + b where W is a constant matrix and b a constant vector.
func (f *fuser) fuseMatMulBias(mmIdx int) {
	mm, ok := f.node(mmIdx)
	if !ok || mm.Op != "MatMul" || len(mm.In) != 2 {
		return
	}
	w, ok := f.g.Initializers[mm.In[1]]
	if !ok || w.DType != Float32 || len(w.Shape) != 2 {
		return
	}
	addIdx, add, ok := f.soleUser(mm.Out[0], "Add")
	if !ok {
		return
	}
	biasName, ok := otherInput(add, mm.Out[0])
	if !ok || !f.vectorConstant(biasName) || f.g.Initializers[biasName].Shape[0] != w.Shape[1] {
		return
	}
	f.replace(addIdx, &Node{
		Name: add.Name + "_matmulbias", Op: opFusedMatMulBias, In: []string{mm.In[0], mm.In[1], biasName}, Out: add.Out,
	}, mmIdx)
}

func opFusedGeluRun(c *opCtx) ([]*Tensor, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	x := c.in[0]
	if !x.isFloat() {
		return nil, fmt.Errorf("FusedGelu supports float32 only")
	}
	divisor, offset, scale := c.node.Float("divisor", 1), c.node.Float("offset", 0), c.node.Float("scale", 1)
	out := zeros(Float32, x.Shape)
	parallelFor(len(x.F), 12, func(lo, hi int) {
		geluSlice(out.F[lo:hi], x.F[lo:hi], divisor, offset, scale)
	})
	return []*Tensor{out}, nil
}

func opFusedMatMulBiasRun(c *opCtx) ([]*Tensor, error) {
	if err := c.need(3); err != nil {
		return nil, err
	}
	outs, err := opMatMul(&opCtx{node: c.node, in: c.in[:2], scope: c.scope})
	if err != nil {
		return nil, err
	}
	out, bias := outs[0], c.in[2]
	n := len(bias.F)
	if n == 0 || out.Shape[len(out.Shape)-1] != n {
		return nil, fmt.Errorf("bias of length %d does not match output %v", n, out.Shape)
	}
	parallelFor(len(out.F)/n, n, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := out.F[r*n : (r+1)*n]
			for j, b := range bias.F {
				row[j] += b
			}
		}
	})
	return outs, nil
}
