package onnxnlp

import "testing"

func TestReduce(t *testing.T) {
	x := fT([]int{2, 3}, 1, 2, 3, 4, 5, 6)
	wantF(t, run1(t, "ReduceSum", nil, x), []int{1, 1}, 21)
	wantF(t, run1(t, "ReduceSum", map[string]any{"keepdims": 0}, x), []int{}, 21)
	wantF(t, run1(t, "ReduceSum", map[string]any{"keepdims": 0}, x, iT([]int{1}, 1)), []int{2}, 6, 15)
	wantF(t, run1(t, "ReduceSum", nil, x, iT([]int{1}, 0)), []int{1, 3}, 5, 7, 9)
	wantF(t, run1(t, "ReduceSum", nil, x, iT([]int{1}, -1)), []int{2, 1}, 6, 15)
	wantF(t, run1(t, "ReduceMax", map[string]any{"keepdims": 0}, x, iT([]int{1}, 0)), []int{3}, 4, 5, 6)
	wantF(t, run1(t, "ReduceMean", map[string]any{"axes": []int64{-1}}, x), []int{2, 1}, 2, 5)
	wantF(t, run1(t, "ReduceMean", map[string]any{"keepdims": 0}, x, iT([]int{2}, 0, 1)), []int{}, 3.5)
	wantI(t, run1(t, "ReduceSum", map[string]any{"keepdims": 0}, iT([]int{2, 2}, 1, 2, 3, 4), iT([]int{1}, 0)), []int{2}, 4, 6)
	wantI(t, run1(t, "ReduceMax", map[string]any{"keepdims": 0}, iT([]int{2, 2}, 1, 9, 3, 4), iT([]int{1}, 1)), []int{2}, 9, 4)
	wantF(t, run1(t, "ReduceSum", nil, fT([]int{}, 5)), []int{}, 5)
	wide := fT([]int{2, 2, 3}, iota32(12)...)
	wantF(t, run1(t, "ReduceSum", map[string]any{"keepdims": 0}, wide, iT([]int{2}, 1, 2)), []int{2}, 15, 51)
	wantF(t, run1(t, "ReduceMean", nil, wide, iT([]int{1}, -1)), []int{2, 2, 1}, 1, 4, 7, 10)
	wantF(t, run1(t, "ReduceMax", map[string]any{"keepdims": 0}, wide, iT([]int{2}, 0, 2)), []int{2}, 8, 11)
	wantF(t, run1(t, "ReduceSum", map[string]any{"keepdims": 0}, wide, iT([]int{2}, 0, 1)), []int{3}, 18, 22, 26)
	same := run1(t, "ReduceSum", map[string]any{"noop_with_empty_axes": 1}, x, iT([]int{0}))
	if same != x {
		t.Fatal("noop_with_empty_axes should return the input")
	}
	mustFail(t, "ReduceSum", nil, "out of range", x, iT([]int{1}, 7))
	mustFail(t, "ReduceSum", nil, "expected integer tensor", x, fT([]int{1}, 0))
	mustFail(t, "ReduceMean", nil, "float32 only", iT([]int{2}, 1, 2))
	mustFail(t, "ReduceMean", map[string]any{"noop_with_empty_axes": 1}, "out of range", x, iT([]int{1}, 5))
	mustFail(t, "ReduceMax", nil, "at least 1 inputs")
	if got := run1(t, "ReduceMean", map[string]any{"noop_with_empty_axes": 1}, x, iT([]int{0})); got != x {
		t.Fatal("ReduceMean without axes and noop should return the input")
	}
}

func TestCumSum(t *testing.T) {
	x := fT([]int{2, 3}, 1, 2, 3, 4, 5, 6)
	wantF(t, run1(t, "CumSum", nil, x, iT([]int{}, 1)), []int{2, 3}, 1, 3, 6, 4, 9, 15)
	wantF(t, run1(t, "CumSum", map[string]any{"exclusive": 1}, x, iT([]int{}, 1)), []int{2, 3}, 0, 1, 3, 0, 4, 9)
	wantF(t, run1(t, "CumSum", map[string]any{"reverse": 1}, x, iT([]int{}, 1)), []int{2, 3}, 6, 5, 3, 15, 11, 6)
	wantF(t, run1(t, "CumSum", nil, x, iT([]int{}, 0)), []int{2, 3}, 1, 2, 3, 5, 7, 9)
	xi := iT([]int{3}, 1, 2, 3)
	wantI(t, run1(t, "CumSum", nil, xi, iT([]int{}, 0)), []int{3}, 1, 3, 6)
	wantI(t, run1(t, "CumSum", map[string]any{"exclusive": 1, "reverse": 1}, xi, iT([]int{}, -1)), []int{3}, 5, 3, 0)
	mustFail(t, "CumSum", nil, "expected integer scalar", x, fT([]int{}, 1))
	mustFail(t, "CumSum", nil, "out of range", x, iT([]int{}, 4))
	mustFail(t, "CumSum", nil, "at least 2 inputs", x)
}

func TestTopK(t *testing.T) {
	x := fT([]int{2, 4}, 1, 5, 3, 5, 9, 2, 8, 2)
	outs, err := callOp("TopK", nil, x, iT([]int{1}, 2))
	if err != nil {
		t.Fatal(err)
	}
	wantF(t, outs[0], []int{2, 2}, 5, 5, 9, 8)
	wantI(t, outs[1], []int{2, 2}, 1, 3, 0, 2)
	outs, err = callOp("TopK", map[string]any{"largest": 0}, x, iT([]int{1}, 2))
	if err != nil {
		t.Fatal(err)
	}
	wantF(t, outs[0], []int{2, 2}, 1, 3, 2, 2)
	wantI(t, outs[1], []int{2, 2}, 0, 2, 1, 3)
	outs, err = callOp("TopK", map[string]any{"axis": 0}, x, iT([]int{1}, 1))
	if err != nil {
		t.Fatal(err)
	}
	wantF(t, outs[0], []int{1, 4}, 9, 5, 8, 5)
	xi := iT([]int{4}, 4, 9, 1, 9)
	outs, err = callOp("TopK", nil, xi, iT([]int{1}, 3))
	if err != nil {
		t.Fatal(err)
	}
	wantI(t, outs[0], []int{3}, 9, 9, 4)
	wantI(t, outs[1], []int{3}, 1, 3, 0)
	outs, err = callOp("TopK", map[string]any{"largest": 0}, xi, iT([]int{1}, 2))
	if err != nil {
		t.Fatal(err)
	}
	wantI(t, outs[0], []int{2}, 1, 4)
	mustFail(t, "TopK", nil, "exceeds axis size", x, iT([]int{1}, 9))
	mustFail(t, "TopK", map[string]any{"axis": 5}, "out of range", x, iT([]int{1}, 1))
	mustFail(t, "TopK", nil, "expected integer scalar", x, fT([]int{1}, 1))
	mustFail(t, "TopK", nil, "at least 2 inputs", x)
}

func TestScatterND(t *testing.T) {
	data := fT([]int{4}, 1, 2, 3, 4)
	wantF(t, run1(t, "ScatterND", nil, data, iT([]int{2, 1}, 3, 0), fT([]int{2}, 10, 20)), []int{4}, 20, 2, 3, 10)
	wantF(t, run1(t, "ScatterND", nil, data, iT([]int{1, 1}, -1), fT([]int{1}, 7)), []int{4}, 1, 2, 3, 7)
	rows := fT([]int{2, 2}, 1, 2, 3, 4)
	wantF(t, run1(t, "ScatterND", nil, rows, iT([]int{1, 1}, 1), fT([]int{1, 2}, 8, 9)), []int{2, 2}, 1, 2, 8, 9)
	wantI(t, run1(t, "ScatterND", nil, iT([]int{2}, 1, 2), iT([]int{1, 1}, 0), iT([]int{1}, 5)), []int{2}, 5, 2)
	mustFail(t, "ScatterND", nil, "out of range", data, iT([]int{1, 1}, 4), fT([]int{1}, 1))
	mustFail(t, "ScatterND", nil, "out of range", data, iT([]int{1, 1}, -5), fT([]int{1}, 1))
	mustFail(t, "ScatterND", map[string]any{"reduction": "add"}, "unsupported reduction", data, iT([]int{1, 1}, 0), fT([]int{1}, 1))
	mustFail(t, "ScatterND", nil, "at least 3 inputs", data)
}

func branch(value float32) *Graph {
	return &Graph{
		Nodes:   []*Node{{Op: "Constant", Out: []string{"v"}, Attr: mkAttrs(map[string]any{"value": fT([]int{}, value)})}},
		Outputs: []string{"v"},
	}
}

func TestIf(t *testing.T) {
	attrs := map[string]any{"then_branch": branch(1), "else_branch": branch(2)}
	wantF(t, run1(t, "If", attrs, bT([]int{}, true)), []int{}, 1)
	wantF(t, run1(t, "If", attrs, bT([]int{}, false)), []int{}, 2)
	mustFail(t, "If", map[string]any{"then_branch": branch(1)}, "missing else_branch", bT([]int{}, false))
	mustFail(t, "If", nil, "at least 1 inputs")
}

func TestIfReadsOuterScope(t *testing.T) {
	inner := &Graph{
		Nodes:   []*Node{{Op: "Add", In: []string{"outer", "outer"}, Out: []string{"y"}}},
		Outputs: []string{"y"},
	}
	g := &Graph{
		Inputs:  []string{"cond", "outer"},
		Nodes:   []*Node{{Op: "If", In: []string{"cond"}, Out: []string{"r"}, Attr: mkAttrs(map[string]any{"then_branch": inner, "else_branch": branch(0)})}},
		Outputs: []string{"r"},
	}
	s, err := NewSession(g)
	if err != nil {
		t.Fatal(err)
	}
	out, err := s.Run(map[string]*Tensor{"cond": bT([]int{}, true), "outer": fT([]int{}, 4)})
	if err != nil {
		t.Fatal(err)
	}
	wantF(t, out["r"], []int{}, 8)
}

func TestRound4AndSign(t *testing.T) {
	if got := round4(0.123456); got != 0.1235 {
		t.Fatalf("round4: %v", got)
	}
	if got := sign[float32](-0.5); got != -1 {
		t.Fatalf("sign: %v", got)
	}
}

func TestTrilu(t *testing.T) {
	x := fT([]int{2, 3}, 1, 2, 3, 4, 5, 6)
	wantF(t, run1(t, "Trilu", nil, x), []int{2, 3}, 1, 2, 3, 0, 5, 6)
	wantF(t, run1(t, "Trilu", map[string]any{"upper": 0}, x), []int{2, 3}, 1, 0, 0, 4, 5, 0)
	wantF(t, run1(t, "Trilu", nil, x, iT(nil, 1)), []int{2, 3}, 0, 2, 3, 0, 0, 6)
	wantF(t, run1(t, "Trilu", map[string]any{"upper": 0}, x, iT(nil, -1)), []int{2, 3}, 0, 0, 0, 4, 0, 0)
	batched := fT([]int{2, 2, 2}, 1, 2, 3, 4, 5, 6, 7, 8)
	wantF(t, run1(t, "Trilu", nil, batched), []int{2, 2, 2}, 1, 2, 0, 4, 5, 6, 0, 8)
	ints := run1(t, "Trilu", nil, iT([]int{2, 2}, 1, 2, 3, 4))
	if ints.I[0] != 1 || ints.I[1] != 2 || ints.I[2] != 0 || ints.I[3] != 4 {
		t.Fatalf("integer Trilu: %v", ints.I)
	}
	mustFail(t, "Trilu", nil, "at least 2 dimensions", fT([]int{3}, 1, 2, 3))
	mustFail(t, "Trilu", nil, "at least 1 inputs")
	mustFail(t, "Trilu", nil, "expected integer scalar", x, fT([]int{1}, 0))
}
