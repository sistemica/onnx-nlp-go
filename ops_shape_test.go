package onnxnlp

import "testing"

func TestConstant(t *testing.T) {
	wantF(t, run1(t, "Constant", map[string]any{"value": fT([]int{2}, 1, 2)}), []int{2}, 1, 2)
	wantF(t, run1(t, "Constant", map[string]any{"value_float": float32(1.5)}), []int{}, 1.5)
	wantF(t, run1(t, "Constant", map[string]any{"value_floats": []float32{1, 2}}), []int{2}, 1, 2)
	wantI(t, run1(t, "Constant", map[string]any{"value_int": 7}), []int{}, 7)
	wantI(t, run1(t, "Constant", map[string]any{"value_ints": []int64{1, 2, 3}}), []int{3}, 1, 2, 3)
	mustFail(t, "Constant", nil, "unsupported Constant attribute")
}

func TestConstantOfShape(t *testing.T) {
	wantF(t, run1(t, "ConstantOfShape", nil, iT([]int{2}, 2, 2)), []int{2, 2}, 0, 0, 0, 0)
	wantI(t, run1(t, "ConstantOfShape", map[string]any{"value": iT([]int{1}, 7)}, iT([]int{1}, 3)), []int{3}, 7, 7, 7)
	wantF(t, run1(t, "ConstantOfShape", map[string]any{"value": fT([]int{1}, 2.5)}, iT([]int{1}, 2)), []int{2}, 2.5, 2.5)
	wantB(t, run1(t, "ConstantOfShape", map[string]any{"value": bT([]int{1}, true)}, iT([]int{1}, 2)), []int{2}, true, true)
	mustFail(t, "ConstantOfShape", nil, "expected integer tensor", fT([]int{1}, 1))
	mustFail(t, "ConstantOfShape", nil, "at least 1 inputs")
}

func TestShape(t *testing.T) {
	x := fT([]int{2, 3, 4})
	wantI(t, run1(t, "Shape", nil, x), []int{3}, 2, 3, 4)
	wantI(t, run1(t, "Shape", map[string]any{"start": 1}, x), []int{2}, 3, 4)
	wantI(t, run1(t, "Shape", map[string]any{"start": -2, "end": -1}, x), []int{1}, 3)
	wantI(t, run1(t, "Shape", map[string]any{"start": 2, "end": 1}, x), []int{0})
	mustFail(t, "Shape", nil, "at least 1 inputs")
}

func TestReshape(t *testing.T) {
	x := fT([]int{2, 3}, iota32(6)...)
	wantF(t, run1(t, "Reshape", nil, x, iT([]int{2}, 3, 2)), []int{3, 2}, 0, 1, 2, 3, 4, 5)
	wantF(t, run1(t, "Reshape", nil, x, iT([]int{2}, -1, 2)), []int{3, 2}, 0, 1, 2, 3, 4, 5)
	wantF(t, run1(t, "Reshape", nil, x, iT([]int{2}, 0, -1)), []int{2, 3}, 0, 1, 2, 3, 4, 5)
	wantF(t, run1(t, "Reshape", map[string]any{"allowzero": 1}, fT([]int{0}), iT([]int{2}, 0, 5)), []int{0, 5})
	mustFail(t, "Reshape", nil, "cannot reshape", x, iT([]int{2}, 4, 2))
	mustFail(t, "Reshape", map[string]any{"allowzero": 1}, "cannot infer", fT([]int{0}), iT([]int{2}, -1, 0))
	mustFail(t, "Reshape", nil, "no counterpart", fT([]int{0}), iT([]int{2}, -1, 0))
	mustFail(t, "Reshape", nil, "expected integer tensor", x, fT([]int{1}, 1))
	mustFail(t, "Reshape", nil, "at least 2 inputs", x)
}

func TestSqueezeUnsqueeze(t *testing.T) {
	x := fT([]int{1, 3, 1}, 1, 2, 3)
	wantF(t, run1(t, "Squeeze", nil, x), []int{3}, 1, 2, 3)
	wantF(t, run1(t, "Squeeze", nil, x, iT([]int{1}, 0)), []int{3, 1}, 1, 2, 3)
	wantF(t, run1(t, "Squeeze", nil, x, iT([]int{1}, -1)), []int{1, 3}, 1, 2, 3)
	mustFail(t, "Squeeze", nil, "out of range", x, iT([]int{1}, 5))
	mustFail(t, "Squeeze", nil, "expected integer tensor", x, fT([]int{1}, 0))
	mustFail(t, "Squeeze", nil, "at least 1 inputs")
	y := fT([]int{2}, 1, 2)
	wantF(t, run1(t, "Unsqueeze", nil, y, iT([]int{1}, 0)), []int{1, 2}, 1, 2)
	wantF(t, run1(t, "Unsqueeze", nil, y, iT([]int{2}, 0, -1)), []int{1, 2, 1}, 1, 2)
	mustFail(t, "Unsqueeze", nil, "out of range", y, iT([]int{1}, 9))
	mustFail(t, "Unsqueeze", nil, "expected integer tensor", y, fT([]int{1}, 0))
	mustFail(t, "Unsqueeze", nil, "at least 2 inputs", y)
}

func TestSlice(t *testing.T) {
	x := fT([]int{2, 4}, iota32(8)...)
	wantF(t, run1(t, "Slice", nil, x, iT([]int{1}, 1), iT([]int{1}, 3)), []int{1, 4}, 4, 5, 6, 7)
	wantF(t, run1(t, "Slice", nil, x, iT([]int{1}, 1), iT([]int{1}, 3), iT([]int{1}, 1)), []int{2, 2}, 1, 2, 5, 6)
	wantF(t, run1(t, "Slice", nil, x, iT([]int{1}, 1), iT([]int{1}, 2), iT([]int{1}, 0)), []int{1, 4}, 4, 5, 6, 7)
	wantF(t, run1(t, "Slice", nil, x, iT([]int{1}, 0), iT([]int{1}, 4), iT([]int{1}, 1), iT([]int{1}, 2)), []int{2, 2}, 0, 2, 4, 6)
	wantF(t, run1(t, "Slice", nil, x, iT([]int{1}, -1), iT([]int{1}, -5), iT([]int{1}, 1), iT([]int{1}, -1)), []int{2, 4}, 3, 2, 1, 0, 7, 6, 5, 4)
	wantF(t, run1(t, "Slice", nil, x, iT([]int{1}, 100), iT([]int{1}, 200), iT([]int{1}, 1)), []int{2, 0})
	wantF(t, run1(t, "Slice", nil, x, iT([]int{1}, -100), iT([]int{1}, 100)), []int{2, 4}, iota32(8)...)
	wantF(t, run1(t, "Slice", nil, x, iT([]int{1}, 3), iT([]int{1}, -100), iT([]int{1}, 1), iT([]int{1}, -3)), []int{2, 2}, 3, 0, 7, 4)
	mustFail(t, "Slice", nil, "step is zero", x, iT([]int{1}, 0), iT([]int{1}, 1), iT([]int{1}, 0), iT([]int{1}, 0))
	mustFail(t, "Slice", nil, "out of range", x, iT([]int{1}, 0), iT([]int{1}, 1), iT([]int{1}, 7))
	mustFail(t, "Slice", nil, "expected integer tensor", x, fT([]int{1}, 0), iT([]int{1}, 1))
	mustFail(t, "Slice", nil, "expected integer tensor", x, iT([]int{1}, 0), fT([]int{1}, 1))
	mustFail(t, "Slice", nil, "expected integer tensor", x, iT([]int{1}, 0), iT([]int{1}, 1), fT([]int{1}, 0))
	mustFail(t, "Slice", nil, "expected integer tensor", x, iT([]int{1}, 0), iT([]int{1}, 1), iT([]int{1}, 0), fT([]int{1}, 1))
	mustFail(t, "Slice", nil, "at least 3 inputs", x, iT([]int{1}, 0))
}

func TestConcatSplit(t *testing.T) {
	a, b := fT([]int{2, 2}, 1, 2, 3, 4), fT([]int{2, 1}, 9, 8)
	wantF(t, run1(t, "Concat", map[string]any{"axis": 1}, a, b), []int{2, 3}, 1, 2, 9, 3, 4, 8)
	wantF(t, run1(t, "Concat", map[string]any{"axis": -1}, a, b), []int{2, 3}, 1, 2, 9, 3, 4, 8)
	wantF(t, run1(t, "Concat", nil, a, a), []int{4, 2}, 1, 2, 3, 4, 1, 2, 3, 4)
	wantI(t, run1(t, "Concat", nil, iT([]int{1}, 1), iT([]int{2}, 2, 3)), []int{3}, 1, 2, 3)
	mustFail(t, "Concat", nil, "dtype mismatch", a, iT([]int{2, 2}, 1, 2, 3, 4))
	mustFail(t, "Concat", map[string]any{"axis": 5}, "out of range", a)
	mustFail(t, "Concat", nil, "at least 1 inputs")

	x := fT([]int{2, 4}, iota32(8)...)
	outs, err := callOp("Split", map[string]any{"axis": 1}, x, iT([]int{2}, 1, 3))
	if err != nil {
		t.Fatal(err)
	}
	wantF(t, outs[0], []int{2, 1}, 0, 4)
	wantF(t, outs[1], []int{2, 3}, 1, 2, 3, 5, 6, 7)
	outs, err = callOp("Split", map[string]any{"axis": 1, "num_outputs": 3}, x)
	if err != nil {
		t.Fatal(err)
	}
	wantF(t, outs[0], []int{2, 2}, 0, 1, 4, 5)
	wantF(t, outs[2], []int{2, 0})
	outs, err = callOp("Split", nil, x)
	if err != nil {
		t.Fatal(err)
	}
	if len(outs) != 3 {
		t.Fatalf("split with 3 declared outputs returned %d tensors", len(outs))
	}
	mustFail(t, "Split", map[string]any{"axis": 5}, "out of range", x)
	mustFail(t, "Split", nil, "expected integer tensor", x, fT([]int{1}, 1))
	mustFail(t, "Split", nil, "at least 1 inputs")
}

func TestGather(t *testing.T) {
	x := fT([]int{3, 2}, 0, 1, 2, 3, 4, 5)
	wantF(t, run1(t, "Gather", nil, x, iT([]int{2}, 2, 0)), []int{2, 2}, 4, 5, 0, 1)
	wantF(t, run1(t, "Gather", nil, x, iT([]int{}, -1)), []int{2}, 4, 5)
	wantF(t, run1(t, "Gather", map[string]any{"axis": 1}, x, iT([]int{1, 2}, 1, 0)), []int{3, 1, 2}, 1, 0, 3, 2, 5, 4)
	wantI(t, run1(t, "Gather", nil, iT([]int{3}, 7, 8, 9), iT([]int{2}, 2, 1)), []int{2}, 9, 8)
	mustFail(t, "Gather", nil, "out of range", x, iT([]int{1}, 3))
	mustFail(t, "Gather", nil, "out of range", x, iT([]int{1}, -4))
	mustFail(t, "Gather", map[string]any{"axis": 4}, "out of range", x, iT([]int{1}, 0))
	mustFail(t, "Gather", nil, "at least 2 inputs", x)
}

func TestGatherElements(t *testing.T) {
	x := fT([]int{2, 3}, 0, 1, 2, 3, 4, 5)
	wantF(t, run1(t, "GatherElements", map[string]any{"axis": 1}, x, iT([]int{2, 2}, 2, 0, 1, -1)), []int{2, 2}, 2, 0, 4, 5)
	wantF(t, run1(t, "GatherElements", map[string]any{"axis": -1}, x, iT([]int{2, 1}, 1, 2)), []int{2, 1}, 1, 5)
	wantF(t, run1(t, "GatherElements", nil, x, iT([]int{1, 3}, 1, 0, 1)), []int{1, 3}, 3, 1, 5)
	wantI(t, run1(t, "GatherElements", map[string]any{"axis": 1}, iT([]int{1, 3}, 5, 6, 7), iT([]int{1, 2}, 2, 2)), []int{1, 2}, 7, 7)
	wantF(t, run1(t, "GatherElements", map[string]any{"axis": 1}, x, iT([]int{2, 0})), []int{2, 0})
	mustFail(t, "GatherElements", map[string]any{"axis": 1}, "out of range", x, iT([]int{2, 1}, 3, 0))
	mustFail(t, "GatherElements", map[string]any{"axis": 1}, "out of range", x, iT([]int{2, 1}, -4, 0))
	mustFail(t, "GatherElements", map[string]any{"axis": 9}, "out of range", x, iT([]int{2, 1}, 0, 0))
	mustFail(t, "GatherElements", nil, "at least 2 inputs", x)
}

func TestTranspose(t *testing.T) {
	x := fT([]int{2, 3}, iota32(6)...)
	wantF(t, run1(t, "Transpose", nil, x), []int{3, 2}, 0, 3, 1, 4, 2, 5)
	y := fT([]int{1, 2, 3}, iota32(6)...)
	wantF(t, run1(t, "Transpose", map[string]any{"perm": []int64{2, 0, 1}}, y), []int{3, 1, 2}, 0, 3, 1, 4, 2, 5)
	wantI(t, run1(t, "Transpose", nil, iT([]int{2, 2}, 1, 2, 3, 4)), []int{2, 2}, 1, 3, 2, 4)
	mustFail(t, "Transpose", nil, "at least 1 inputs")
}

func TestExpandTile(t *testing.T) {
	x := fT([]int{2, 1}, 1, 2)
	wantF(t, run1(t, "Expand", nil, x, iT([]int{2}, 1, 3)), []int{2, 3}, 1, 1, 1, 2, 2, 2)
	if run1(t, "Expand", nil, x, iT([]int{2}, 2, 1)) != x {
		t.Fatal("Expand to the same shape should return the input")
	}
	mustFail(t, "Expand", nil, "cannot broadcast", x, iT([]int{2}, 3, 3))
	mustFail(t, "Expand", nil, "expected integer tensor", x, fT([]int{1}, 3))
	mustFail(t, "Expand", nil, "at least 2 inputs", x)

	wantF(t, run1(t, "Tile", nil, fT([]int{2}, 1, 2), iT([]int{1}, 3)), []int{6}, 1, 2, 1, 2, 1, 2)
	wantF(t, run1(t, "Tile", nil, fT([]int{1, 2}, 1, 2), iT([]int{2}, 2, 2)), []int{2, 4}, 1, 2, 1, 2, 1, 2, 1, 2)
	wantI(t, run1(t, "Tile", nil, iT([]int{2}, 1, 2), iT([]int{1}, 2)), []int{4}, 1, 2, 1, 2)
	same := fT([]int{2}, 1, 2)
	if run1(t, "Tile", nil, same, iT([]int{1}, 1)) != same {
		t.Fatal("Tile with all repeats 1 should return the input")
	}
	mustFail(t, "Tile", nil, "do not match rank", same, iT([]int{2}, 1, 1))
	mustFail(t, "Tile", nil, "expected integer tensor", same, fT([]int{1}, 1))
	mustFail(t, "Tile", nil, "at least 2 inputs", same)
}

func TestRange(t *testing.T) {
	wantI(t, run1(t, "Range", nil, iT([]int{}, 1), iT([]int{}, 8), iT([]int{}, 3)), []int{3}, 1, 4, 7)
	wantI(t, run1(t, "Range", nil, iT([]int{}, 5), iT([]int{}, 0), iT([]int{}, -2)), []int{3}, 5, 3, 1)
	wantI(t, run1(t, "Range", nil, iT([]int{}, 5), iT([]int{}, 0), iT([]int{}, 1)), []int{0})
	wantF(t, run1(t, "Range", nil, fT([]int{}, 0), fT([]int{}, 1), fT([]int{}, 0.25)), []int{4}, 0, 0.25, 0.5, 0.75)
	mustFail(t, "Range", nil, "at least 3 inputs", iT([]int{}, 1))
}
