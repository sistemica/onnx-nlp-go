package onnxgo

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func addNode(g *Graph, op string, in []string, out string, attrs map[string]any) *Node {
	n := &Node{Name: out, Op: op, In: in, Out: []string{out}, Attr: mkAttrs(attrs)}
	g.Nodes = append(g.Nodes, n)
	return n
}

func nodeFor(g *Graph, out string) *Node {
	for _, n := range g.Nodes {
		if slices.Contains(n.Out, out) {
			return n
		}
	}
	panic("no node produces " + out)
}

func newTestGraph(inputs ...string) *Graph {
	if len(inputs) == 0 {
		inputs = []string{"x"}
	}
	return &Graph{Inputs: inputs, Initializers: map[string]*Tensor{}, Outputs: []string{"y"}}
}

func scalarInit(g *Graph, name string, v float32) { g.Initializers[name] = fT([]int{}, v) }

func vectorInit(g *Graph, name string, n int, base float32) {
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = base + float32(i)/10
	}
	g.Initializers[name] = fT([]int{n}, vals...)
}

func geluGraph() *Graph {
	g := newTestGraph()
	scalarInit(g, "c1", float32(math.Sqrt2))
	scalarInit(g, "c2", 1)
	scalarInit(g, "c3", 0.5)
	addNode(g, "Div", []string{"x", "c1"}, "d", nil)
	addNode(g, "Erf", []string{"d"}, "e", nil)
	addNode(g, "Add", []string{"e", "c2"}, "a", nil)
	addNode(g, "Mul", []string{"x", "a"}, "m", nil)
	addNode(g, "Mul", []string{"m", "c3"}, "y", nil)
	return g
}

func layerNormGraph(width int) *Graph {
	g := newTestGraph()
	scalarInit(g, "two", 2)
	scalarInit(g, "eps", 1e-5)
	vectorInit(g, "w", width, 1)
	vectorInit(g, "b", width, 0.5)
	last := map[string]any{"axes": []int64{-1}}
	addNode(g, "ReduceMean", []string{"x"}, "mu", last)
	addNode(g, "Sub", []string{"x", "mu"}, "s", nil)
	addNode(g, "Pow", []string{"s", "two"}, "p", nil)
	addNode(g, "ReduceMean", []string{"p"}, "v", last)
	addNode(g, "Add", []string{"v", "eps"}, "ve", nil)
	addNode(g, "Sqrt", []string{"ve"}, "sd", nil)
	addNode(g, "Div", []string{"s", "sd"}, "n", nil)
	addNode(g, "Mul", []string{"n", "w"}, "sc", nil)
	addNode(g, "Add", []string{"sc", "b"}, "y", nil)
	return g
}

func matMulBiasGraph() *Graph {
	g := newTestGraph()
	g.Initializers["W"] = fT([]int{3, 4}, iota32(12)...)
	vectorInit(g, "bias", 4, 1)
	addNode(g, "MatMul", []string{"x", "W"}, "t", nil)
	addNode(g, "Add", []string{"bias", "t"}, "y", nil)
	return g
}

func opNames(g *Graph) []string {
	var ops []string
	for _, n := range g.Nodes {
		ops = append(ops, n.Op)
	}
	return ops
}

func TestFusionRewritesTheKnownPatterns(t *testing.T) {
	cases := []struct {
		name  string
		graph *Graph
		want  []string
	}{
		{"gelu", geluGraph(), []string{opFusedGelu}},
		{"layer norm", layerNormGraph(4), []string{"LayerNormalization"}},
		{"matmul bias", matMulBiasGraph(), []string{opFusedMatMulBias}},
	}
	for _, c := range cases {
		before := len(c.graph.Nodes)
		fused := fuseNodes(c.graph)
		if got := opNames(fused); !slices.Equal(got, c.want) {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
		if len(c.graph.Nodes) != before {
			t.Errorf("%s: the input graph was modified", c.name)
		}
		if fused.Nodes[0].Out[0] != "y" {
			t.Errorf("%s: the fused node must keep the final output name", c.name)
		}
	}
}

func TestFusedGraphsMatchUnfusedResults(t *testing.T) {
	rng := rand.New(rand.NewPCG(91, 92))
	for _, g := range []*Graph{geluGraph(), layerNormGraph(8), matMulBiasGraph()} {
		width := 8
		if g.Nodes[0].Op == "MatMul" {
			width = 3
		}
		fused, err := NewSession(g)
		if err != nil {
			t.Fatal(err)
		}
		plain := &Session{graph: foldConstants(g)}
		if len(fused.graph.Nodes) >= len(plain.graph.Nodes) {
			t.Fatalf("%s: fusion removed nothing", g.Nodes[0].Op)
		}
		shapes := [][]int{{width}, {5, width}, {2, 3, width}}
		if g.Nodes[0].Op == "MatMul" {
			shapes = shapes[1:]
		}
		for _, shape := range shapes {
			x := NewFloat(shape, randomSlice(rng, numel(shape)))
			a, err := fused.Run(map[string]*Tensor{"x": x})
			if err != nil {
				t.Fatal(err)
			}
			b, err := plain.Run(map[string]*Tensor{"x": x})
			if err != nil {
				t.Fatal(err)
			}
			wantShape(t, a["y"], b["y"].Shape)
			for i := range b["y"].F {
				if d := math.Abs(float64(a["y"].F[i] - b["y"].F[i])); d > 1e-5*(1+math.Abs(float64(b["y"].F[i]))) {
					t.Fatalf("%s %v element %d: fused %v, plain %v", g.Nodes[0].Op, shape, i, a["y"].F[i], b["y"].F[i])
				}
			}
		}
	}
}

func TestFusionIgnoresBrokenPatterns(t *testing.T) {
	type mutation struct {
		name   string
		mutate func(g *Graph)
	}
	geluBreaks := []mutation{
		{"erf fed by a non-division", func(g *Graph) { nodeFor(g, "d").Op = "Mul" }},
		{"erf fed by a graph input", func(g *Graph) { nodeFor(g, "e").In = []string{"x"} }},
		{"erf with two inputs", func(g *Graph) { nodeFor(g, "e").In = []string{"d", "d"} }},
		{"division by a graph input", func(g *Graph) { g.Inputs = append(g.Inputs, "c1"); delete(g.Initializers, "c1") }},
		{"division by a vector", func(g *Graph) { vectorInit(g, "c1", 3, 1) }},
		{"division by an integer", func(g *Graph) { g.Initializers["c1"] = iT([]int{}, 2) }},
		{"division result reused", func(g *Graph) { addNode(g, "Neg", []string{"d"}, "extra", nil); g.Outputs = append(g.Outputs, "extra") }},
		{"erf consumed by a non-add", func(g *Graph) { nodeFor(g, "a").Op = "Sub" }},
		{"erf result is a graph output", func(g *Graph) { g.Outputs = append(g.Outputs, "e") }},
		{"add with one input", func(g *Graph) { nodeFor(g, "a").In = []string{"e"} }},
		{"add of a non-constant", func(g *Graph) { g.Inputs = append(g.Inputs, "c2"); delete(g.Initializers, "c2") }},
		{"mul of a different tensor", func(g *Graph) { g.Inputs = append(g.Inputs, "z"); nodeFor(g, "m").In = []string{"z", "a"} }},
		{"mul with a missing operand", func(g *Graph) { nodeFor(g, "m").In = []string{"a"} }},
		{"add result reused", func(g *Graph) { g.Outputs = append(g.Outputs, "a") }},
		{"last step is not a multiplication", func(g *Graph) { nodeFor(g, "y").Op = "Add" }},
		{"last multiplication by a non-constant", func(g *Graph) { g.Inputs = append(g.Inputs, "c3"); delete(g.Initializers, "c3") }},
		{"last multiplication by a vector", func(g *Graph) { vectorInit(g, "c3", 2, 1) }},
		{"last multiplication with one input", func(g *Graph) { nodeFor(g, "y").In = []string{"m"} }},
		{"intermediate used by a nested graph", func(g *Graph) {
			inner := &Graph{Nodes: []*Node{{Op: "Neg", In: []string{"a"}, Out: []string{"q"}}}, Outputs: []string{"q"}}
			g.Inputs = append(g.Inputs, "flag")
			addNode(g, "If", []string{"flag"}, "extra", map[string]any{"then_branch": inner, "else_branch": inner})
			g.Outputs = append(g.Outputs, "extra")
		}},
	}
	for _, m := range geluBreaks {
		g := geluGraph()
		m.mutate(g)
		before := len(g.Nodes)
		if fused := fuseNodes(g); slices.Contains(opNames(fused), opFusedGelu) || len(fused.Nodes) != before {
			t.Errorf("gelu / %s: pattern was fused anyway: %v", m.name, opNames(fused))
		}
	}

	layerBreaks := []mutation{
		{"mean over another axis", func(g *Graph) { nodeFor(g, "mu").Attr = mkAttrs(map[string]any{"axes": []int64{1}}) }},
		{"mean without keepdims", func(g *Graph) { nodeFor(g, "mu").Attr = mkAttrs(map[string]any{"axes": []int64{-1}, "keepdims": 0}) }},
		{"mean without axes", func(g *Graph) { nodeFor(g, "mu").Attr = nil }},
		{"mean with an axes input", func(g *Graph) { nodeFor(g, "mu").In = []string{"x", "two"} }},
		{"subtraction in the wrong order", func(g *Graph) { nodeFor(g, "s").In = []string{"mu", "x"} }},
		{"subtraction of something else", func(g *Graph) { g.Inputs = append(g.Inputs, "z"); nodeFor(g, "s").In = []string{"z", "mu"} }},
		{"centered value used once", func(g *Graph) { nodeFor(g, "n").In = []string{"x", "sd"} }},
		{"centered value used three times", func(g *Graph) { g.Outputs = append(g.Outputs, "s") }},
		{"centered value not squared", func(g *Graph) { nodeFor(g, "p").Op = "Mul" }},
		{"squared twice", func(g *Graph) { nodeFor(g, "n").Op = "Pow" }},
		{"power other than two", func(g *Graph) { scalarInit(g, "two", 3) }},
		{"power of a non-constant", func(g *Graph) { g.Inputs = append(g.Inputs, "two"); delete(g.Initializers, "two") }},
		{"power with one input", func(g *Graph) { nodeFor(g, "p").In = []string{"s"} }},
		{"division by something else", func(g *Graph) { nodeFor(g, "n").In = []string{"s", "x"} }},
		{"division with one input", func(g *Graph) { nodeFor(g, "n").In = []string{"s"} }},
		{"division numerator is not centered", func(g *Graph) { nodeFor(g, "n").In = []string{"x", "sd"} }},
		{"variance mean over another axis", func(g *Graph) { nodeFor(g, "v").Attr = mkAttrs(map[string]any{"axes": []int64{0}}) }},
		{"variance not followed by add", func(g *Graph) { nodeFor(g, "ve").Op = "Sub" }},
		{"epsilon missing", func(g *Graph) { nodeFor(g, "ve").In = []string{"v"} }},
		{"epsilon not constant", func(g *Graph) { g.Inputs = append(g.Inputs, "eps"); delete(g.Initializers, "eps") }},
		{"no square root", func(g *Graph) { nodeFor(g, "sd").Op = "Neg" }},
		{"square root reused", func(g *Graph) { g.Outputs = append(g.Outputs, "sd") }},
		{"no scale", func(g *Graph) { nodeFor(g, "sc").Op = "Add" }},
		{"scale is not a vector", func(g *Graph) { scalarInit(g, "w", 2) }},
		{"scale is not constant", func(g *Graph) { g.Inputs = append(g.Inputs, "w"); delete(g.Initializers, "w") }},
		{"scale with one input", func(g *Graph) { nodeFor(g, "sc").In = []string{"n"} }},
		{"no bias", func(g *Graph) { nodeFor(g, "y").Op = "Mul" }},
		{"bias is not a vector", func(g *Graph) { scalarInit(g, "b", 2) }},
		{"bias with one input", func(g *Graph) { nodeFor(g, "y").In = []string{"sc"} }},
		{"intermediate is a graph output", func(g *Graph) { g.Outputs = append(g.Outputs, "mu") }},
	}
	for _, m := range layerBreaks {
		g := layerNormGraph(4)
		m.mutate(g)
		before := len(g.Nodes)
		if fused := fuseNodes(g); slices.Contains(opNames(fused), "LayerNormalization") || len(fused.Nodes) != before {
			t.Errorf("layer norm / %s: pattern was fused anyway: %v", m.name, opNames(fused))
		}
	}

	biasBreaks := []mutation{
		{"matmul with one input", func(g *Graph) { nodeFor(g, "t").In = []string{"x"} }},
		{"weight is a graph input", func(g *Graph) { g.Inputs = append(g.Inputs, "W"); delete(g.Initializers, "W") }},
		{"weight is a vector", func(g *Graph) { vectorInit(g, "W", 3, 1) }},
		{"weight is an integer matrix", func(g *Graph) { g.Initializers["W"] = iT([]int{3, 4}, make([]int64, 12)...) }},
		{"product consumed by a non-add", func(g *Graph) { nodeFor(g, "y").Op = "Mul" }},
		{"product reused", func(g *Graph) { g.Outputs = append(g.Outputs, "t") }},
		{"add with one input", func(g *Graph) { nodeFor(g, "y").In = []string{"t"} }},
		{"bias is not constant", func(g *Graph) { g.Inputs = append(g.Inputs, "bias"); delete(g.Initializers, "bias") }},
		{"bias is a matrix", func(g *Graph) { g.Initializers["bias"] = fT([]int{1, 4}, 1, 2, 3, 4) }},
		{"bias of the wrong length", func(g *Graph) { vectorInit(g, "bias", 5, 1) }},
	}
	for _, m := range biasBreaks {
		g := matMulBiasGraph()
		m.mutate(g)
		before := len(g.Nodes)
		if fused := fuseNodes(g); slices.Contains(opNames(fused), opFusedMatMulBias) || len(fused.Nodes) != before {
			t.Errorf("matmul bias / %s: pattern was fused anyway: %v", m.name, opNames(fused))
		}
	}
}

func TestFusionLeavesLayerNormAloneWhenItsCenteredValueHasOtherUsers(t *testing.T) {
	g := layerNormGraph(4)
	scalarInit(g, "c1", float32(math.Sqrt2))
	scalarInit(g, "c2", 1)
	scalarInit(g, "c3", 0.5)
	addNode(g, "Div", []string{"s", "c1"}, "gd", nil)
	addNode(g, "Erf", []string{"gd"}, "ge", nil)
	addNode(g, "Add", []string{"ge", "c2"}, "ga", nil)
	addNode(g, "Mul", []string{"s", "ga"}, "gm", nil)
	addNode(g, "Mul", []string{"gm", "c3"}, "gy", nil)
	g.Outputs = append(g.Outputs, "gy")
	ops := opNames(fuseNodes(g))
	if !slices.Contains(ops, opFusedGelu) || slices.Contains(ops, "LayerNormalization") {
		t.Fatalf("the GELU fuses; the layer norm, whose centered value has extra users, stays: %v", ops)
	}
}

func TestFusionHandlesChainedAndOverlappingPatterns(t *testing.T) {
	g := newTestGraph()
	g.Initializers["W1"] = fT([]int{4, 6}, iota32(24)...)
	g.Initializers["W2"] = fT([]int{6, 4}, iota32(24)...)
	vectorInit(g, "b1", 6, 0)
	vectorInit(g, "b2", 4, 0)
	scalarInit(g, "c1", float32(math.Sqrt2))
	scalarInit(g, "c2", 1)
	scalarInit(g, "c3", 0.5)
	addNode(g, "MatMul", []string{"x", "W1"}, "t1", nil)
	addNode(g, "Add", []string{"t1", "b1"}, "h", nil)
	addNode(g, "Div", []string{"h", "c1"}, "d", nil)
	addNode(g, "Erf", []string{"d"}, "e", nil)
	addNode(g, "Add", []string{"e", "c2"}, "a", nil)
	addNode(g, "Mul", []string{"h", "a"}, "m", nil)
	addNode(g, "Mul", []string{"m", "c3"}, "act", nil)
	addNode(g, "MatMul", []string{"act", "W2"}, "t2", nil)
	addNode(g, "Add", []string{"t2", "b2"}, "y", nil)
	fused := fuseNodes(g)
	if got := opNames(fused); !slices.Equal(got, []string{opFusedMatMulBias, opFusedGelu, opFusedMatMulBias}) {
		t.Fatalf("got %v", got)
	}
	s, err := NewSession(g)
	if err != nil {
		t.Fatal(err)
	}
	plain := &Session{graph: foldConstants(g)}
	x := fT([]int{3, 4}, 0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.7, 0.8, -0.9, 1, 1.1, -1.2)
	a, err := s.Run(map[string]*Tensor{"x": x})
	if err != nil {
		t.Fatal(err)
	}
	b, err := plain.Run(map[string]*Tensor{"x": x})
	if err != nil {
		t.Fatal(err)
	}
	for i := range b["y"].F {
		if d := math.Abs(float64(a["y"].F[i] - b["y"].F[i])); d > 1e-3*(1+math.Abs(float64(b["y"].F[i]))) {
			t.Fatalf("element %d: fused %v, plain %v", i, a["y"].F[i], b["y"].F[i])
		}
	}
}

func TestFusedOperatorsReportBadInputs(t *testing.T) {
	gelu := &Node{Op: opFusedGelu, Attr: mkAttrs(map[string]any{"divisor": float32(2), "offset": float32(1), "scale": float32(0.5)})}
	if _, err := ops[opFusedGelu](&opCtx{node: gelu}); err == nil {
		t.Fatal("missing input must fail")
	}
	if _, err := ops[opFusedGelu](&opCtx{node: gelu, in: []*Tensor{iT([]int{1}, 1)}}); err == nil {
		t.Fatal("integer input must fail")
	}
	out, err := ops[opFusedGelu](&opCtx{node: gelu, in: []*Tensor{fT([]int{2}, 0, 2)}})
	if err != nil {
		t.Fatal(err)
	}
	wantF(t, out[0], []int{2}, 0, 2*(float32(math.Erf(1))+1)*0.5)

	bias := &Node{Op: opFusedMatMulBias}
	x, w := fT([]int{2, 3}, iota32(6)...), fT([]int{3, 4}, iota32(12)...)
	if _, err := ops[opFusedMatMulBias](&opCtx{node: bias, in: []*Tensor{x, w}}); err == nil {
		t.Fatal("missing bias must fail")
	}
	if _, err := ops[opFusedMatMulBias](&opCtx{node: bias, in: []*Tensor{x, x, fT([]int{4})}}); err == nil {
		t.Fatal("a failing matrix product must fail the fused operator")
	}
	if _, err := ops[opFusedMatMulBias](&opCtx{node: bias, in: []*Tensor{x, w, fT([]int{3}, 1, 1, 1)}}); err == nil {
		t.Fatal("a bias of the wrong length must fail")
	}
	if _, err := ops[opFusedMatMulBias](&opCtx{node: bias, in: []*Tensor{x, w, fT([]int{0})}}); err == nil {
		t.Fatal("an empty bias must fail")
	}
	got, err := ops[opFusedMatMulBias](&opCtx{node: bias, in: []*Tensor{x, w, fT([]int{4}, 10, 20, 30, 40)}})
	if err != nil {
		t.Fatal(err)
	}
	wantF(t, got[0], []int{2, 4}, 30, 43, 56, 69, 66, 88, 110, 132)
}

func TestLayerNormRejectsMismatchedParameters(t *testing.T) {
	x := fT([]int{2, 4}, iota32(8)...)
	mustFail(t, "LayerNormalization", nil, "scale has 3 values", x, fT([]int{3}, 1, 1, 1))
	mustFail(t, "LayerNormalization", nil, "bias has 2 values", x, fT([]int{4}, 1, 1, 1, 1), fT([]int{2}, 0, 0))
}
