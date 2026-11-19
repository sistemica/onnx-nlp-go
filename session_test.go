package onnxgo

import (
	"strings"
	"testing"
)

func TestSessionRejectsUnsupportedGraphs(t *testing.T) {
	cases := map[string]*Graph{
		"unknown operator": {Nodes: []*Node{{Op: "Frobnicate"}}},
		"foreign domain":   {Nodes: []*Node{{Op: "Add", Domain: "com.microsoft"}}},
		"nested unknown":   {Nodes: []*Node{{Op: "If", Attr: mkAttrs(map[string]any{"then_branch": &Graph{Nodes: []*Node{{Op: "Frobnicate"}}}})}}},
	}
	for name, g := range cases {
		if _, err := NewSession(g); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	if _, err := NewSession(&Graph{Nodes: []*Node{{Op: "Add", Domain: "ai.onnx"}}}); err != nil {
		t.Fatalf("explicit default domain rejected: %v", err)
	}
}

func TestSessionRunErrors(t *testing.T) {
	s, err := NewSession(&Graph{Inputs: []string{"x"}, Nodes: []*Node{{Op: "Relu", In: []string{"x"}, Out: []string{"y"}}}, Outputs: []string{"y"}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Run(map[string]*Tensor{}); err == nil || !strings.Contains(err.Error(), "missing input") {
		t.Fatalf("missing input: %v", err)
	}
	if _, err := s.Run(map[string]*Tensor{"x": iT([]int{1}, 1)}); err == nil || !strings.Contains(err.Error(), "Relu") {
		t.Fatalf("operator failure should name the node: %v", err)
	}
	bad, _ := NewSession(&Graph{Nodes: []*Node{{Op: "Relu", In: []string{"ghost"}, Out: []string{"y"}}}, Outputs: []string{"y"}})
	if _, err := bad.Run(nil); err == nil || !strings.Contains(err.Error(), "not computed") {
		t.Fatalf("uncomputed input: %v", err)
	}
	noOut, _ := NewSession(&Graph{Outputs: []string{"nothing"}})
	if _, err := noOut.Run(nil); err == nil || !strings.Contains(err.Error(), "graph output") {
		t.Fatalf("missing output: %v", err)
	}
}

func TestSessionSkipsEmptyOptionalNames(t *testing.T) {
	g := &Graph{
		Inputs: []string{"x"},
		Nodes: []*Node{
			{Op: "Clip", In: []string{"x", "", "hi"}, Out: []string{"y"}},
			{Op: "TopK", In: []string{"y", "k"}, Out: []string{"v", ""}},
		},
		Initializers: map[string]*Tensor{"hi": fT([]int{}, 2), "k": iT([]int{1}, 1)},
		Outputs:      []string{"v"},
	}
	s, err := NewSession(g)
	if err != nil {
		t.Fatal(err)
	}
	out, err := s.Run(map[string]*Tensor{"x": fT([]int{3}, 1, 5, 3)})
	if err != nil {
		t.Fatal(err)
	}
	wantF(t, out["v"], []int{1}, 2)
}

func TestProfiling(t *testing.T) {
	s, _ := NewSession(&Graph{
		Inputs:  []string{"x"},
		Nodes:   []*Node{{Op: "Relu", In: []string{"x"}, Out: []string{"y"}}, {Op: "Neg", In: []string{"y"}, Out: []string{"z"}}},
		Outputs: []string{"z"},
	})
	if s.OpStats() != nil {
		t.Fatal("stats before EnableProfiling must be nil")
	}
	s.EnableProfiling()
	for i := 0; i < 3; i++ {
		if _, err := s.Run(map[string]*Tensor{"x": fT([]int{1}, 1)}); err != nil {
			t.Fatal(err)
		}
	}
	stats := s.OpStats()
	if len(stats) != 2 || stats[0].Calls != 3 || stats[1].Calls != 3 || stats[0].Total < stats[1].Total {
		t.Fatalf("stats %+v", stats)
	}
}

func TestProfilingFollowsSubgraphs(t *testing.T) {
	g := &Graph{
		Inputs:  []string{"c"},
		Nodes:   []*Node{{Op: "If", In: []string{"c"}, Out: []string{"r"}, Attr: mkAttrs(map[string]any{"then_branch": branch(1), "else_branch": branch(2)})}},
		Outputs: []string{"r"},
	}
	s, _ := NewSession(g)
	s.EnableProfiling()
	if _, err := s.Run(map[string]*Tensor{"c": bT([]int{}, true)}); err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, st := range s.OpStats() {
		seen[st.Op] = true
	}
	if !seen["If"] || !seen["Constant"] {
		t.Fatalf("stats %+v", s.OpStats())
	}
}

func constNode(out string, v *Tensor) *Node {
	return &Node{Op: "Constant", Out: []string{out}, Attr: mkAttrs(map[string]any{"value": v})}
}

func TestConstantFoldingRemovesConstantSubgraphs(t *testing.T) {
	g := &Graph{
		Inputs: []string{"x"},
		Nodes: []*Node{
			constNode("c", fT([]int{2}, 1, 2)),
			{Op: "Add", In: []string{"c", "c"}, Out: []string{"c2"}},
			{Op: "Shape", In: []string{"c2"}, Out: []string{"s"}},
			{Op: "Add", In: []string{"x", "c2"}, Out: []string{"y"}},
			{Op: "Clip", In: []string{"c2", "", "hi"}, Out: []string{"clipped"}},
			{Op: "TopK", In: []string{"c2", "k"}, Out: []string{"v", ""}},
			{Op: "Add", In: []string{"y", "clipped"}, Out: []string{"z"}},
		},
		Initializers: map[string]*Tensor{"hi": fT([]int{}, 3), "k": iT([]int{1}, 1)},
		Outputs:      []string{"z", "s", "v"},
	}
	s, err := NewSession(g)
	if err != nil {
		t.Fatal(err)
	}
	if len(s.graph.Nodes) != 2 {
		t.Fatalf("expected the two input-dependent nodes to remain, got %d", len(s.graph.Nodes))
	}
	if c2 := s.graph.Initializers["c2"]; c2 == nil || !c2.static || c2.pack == nil {
		t.Fatalf("folded results must become static initializers: %+v", c2)
	}
	if len(g.Nodes) != 7 {
		t.Fatal("folding must not modify the caller's graph")
	}
	out, err := s.Run(map[string]*Tensor{"x": fT([]int{2}, 10, 20)})
	if err != nil {
		t.Fatal(err)
	}
	wantF(t, out["z"], []int{2}, 12+2, 24+3)
	wantI(t, out["s"], []int{1}, 2)
	wantF(t, out["v"], []int{1}, 4)
}

func TestConstantFoldingLeavesDynamicAndFailingNodes(t *testing.T) {
	g := &Graph{
		Inputs: []string{"c"},
		Nodes: []*Node{
			{Op: "If", In: []string{"flag"}, Out: []string{"r"}, Attr: mkAttrs(map[string]any{"then_branch": branch(1), "else_branch": branch(2)})},
			{Op: "Reshape", In: []string{"data", "badshape"}, Out: []string{"broken"}},
		},
		Initializers: map[string]*Tensor{"flag": bT([]int{}, true), "data": fT([]int{2}, 1, 2), "badshape": iT([]int{1}, 5)},
		Outputs:      []string{"r"},
	}
	s, err := NewSession(g)
	if err != nil {
		t.Fatal(err)
	}
	if len(s.graph.Nodes) != 2 {
		t.Fatalf("subgraph and failing nodes must stay for run time, got %d nodes", len(s.graph.Nodes))
	}
	if _, err := s.Run(map[string]*Tensor{"c": fT([]int{}, 0)}); err == nil {
		t.Fatal("the failing node must report its error when the graph runs")
	}
}

func TestFoldedConstantsFeedSubgraphs(t *testing.T) {
	inner := &Graph{
		Nodes:   []*Node{{Op: "Add", In: []string{"folded", "folded"}, Out: []string{"y"}}},
		Outputs: []string{"y"},
	}
	g := &Graph{
		Inputs: []string{"cond"},
		Nodes: []*Node{
			constNode("folded", fT([]int{}, 4)),
			{Op: "If", In: []string{"cond"}, Out: []string{"r"}, Attr: mkAttrs(map[string]any{"then_branch": inner, "else_branch": branch(0)})},
		},
		Outputs: []string{"r"},
	}
	s, err := NewSession(g)
	if err != nil {
		t.Fatal(err)
	}
	out, err := s.Run(map[string]*Tensor{"cond": bT([]int{}, true)})
	if err != nil {
		t.Fatal(err)
	}
	wantF(t, out["r"], []int{}, 8)
}
