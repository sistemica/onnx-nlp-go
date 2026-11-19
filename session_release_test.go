package onnxnlp

import (
	"runtime"
	"slices"
	"testing"
	"weak"
)

func TestRunReleasesValuesAfterTheirLastUse(t *testing.T) {
	var first weak.Pointer[Tensor]
	freedDuringRun := false
	ops["ZZMake"] = func(c *opCtx) ([]*Tensor, error) {
		x := zeros(Float32, []int{1 << 18})
		first = weak.Make(x)
		return []*Tensor{x}, nil
	}
	ops["ZZNext"] = func(c *opCtx) ([]*Tensor, error) { return []*Tensor{zeros(Float32, []int{1 << 18})}, nil }
	ops["ZZCheck"] = func(c *opCtx) ([]*Tensor, error) {
		runtime.GC()
		runtime.GC()
		freedDuringRun = first.Value() == nil
		return []*Tensor{zeros(Float32, []int{1})}, nil
	}
	t.Cleanup(func() {
		delete(ops, "ZZMake")
		delete(ops, "ZZNext")
		delete(ops, "ZZCheck")
	})
	s, err := NewSession(&Graph{
		Inputs: []string{"x"}, Outputs: []string{"d"},
		Nodes: []*Node{
			{Op: "ZZMake", In: []string{"x"}, Out: []string{"a"}},
			{Op: "ZZNext", In: []string{"a"}, Out: []string{"b"}},
			{Op: "ZZNext", In: []string{"b"}, Out: []string{"c"}},
			{Op: "ZZCheck", In: []string{"c"}, Out: []string{"d"}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	out, err := s.Run(map[string]*Tensor{"x": zeros(Float32, []int{1})})
	if err != nil || out["d"] == nil {
		t.Fatalf("%v %v", out, err)
	}
	if !freedDuringRun {
		t.Fatal("a tensor whose last reader has run must be freed before the run ends")
	}
}

func TestLastUsesCountSubgraphReadsAndKeepOutputs(t *testing.T) {
	branch := &Graph{Nodes: []*Node{{Op: "Add", In: []string{"outer", "w"}, Out: []string{"r"}}}, Outputs: []string{"r"}}
	g := &Graph{
		Inputs: []string{"x"}, Outputs: []string{"kept"},
		Nodes: []*Node{
			{Op: "Identity", In: []string{"x"}, Out: []string{"outer"}},
			{Op: "Identity", In: []string{"x"}, Out: []string{"kept"}},
			{Op: "If", In: []string{"cond"}, Out: []string{"y"}, Attr: map[string]*Attr{"then_branch": {G: branch}}},
			{Op: "Identity", In: []string{"", "y"}, Out: []string{"z"}},
		},
	}
	last := lastUses(g)
	if last["outer"] != 2 || last["w"] != 2 || last["cond"] != 2 || last["y"] != 3 || last["x"] != 1 {
		t.Fatalf("last uses %v: a value read inside a branch lives until the If node", last)
	}
	if _, used := last["z"]; used {
		t.Fatal("z is never read")
	}
	if uses := nodeUses(g.Nodes[2]); !slices.Contains(uses, "outer") || !slices.Contains(uses, "r") {
		t.Fatalf("uses of the If node: %v", uses)
	}
}

func TestSubgraphsStillSeeOuterValuesAfterRelease(t *testing.T) {
	thenBranch := &Graph{Nodes: []*Node{{Op: "Add", In: []string{"outer", "outer"}, Out: []string{"r"}}}, Outputs: []string{"r"}}
	elseBranch := &Graph{Nodes: []*Node{{Op: "Identity", In: []string{"outer"}, Out: []string{"r"}}}, Outputs: []string{"r"}}
	s, err := NewSession(&Graph{
		Inputs: []string{"x", "cond"}, Outputs: []string{"y"},
		Nodes: []*Node{
			{Op: "Identity", In: []string{"x"}, Out: []string{"outer"}},
			{Op: "If", In: []string{"cond"}, Out: []string{"y"}, Attr: map[string]*Attr{"then_branch": {G: thenBranch}, "else_branch": {G: elseBranch}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		cond bool
		want float32
	}{{true, 6}, {false, 3}} {
		out, err := s.Run(map[string]*Tensor{"x": fT([]int{1}, 3), "cond": bT([]int{1}, tc.cond)})
		if err != nil {
			t.Fatal(err)
		}
		wantF(t, out["y"], []int{1}, tc.want)
	}
}
