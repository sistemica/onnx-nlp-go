package onnxnlp

import (
	"math"
	"testing"
)

func TestArithmeticFloat(t *testing.T) {
	a := fT([]int{2, 3}, 1, 2, 3, 4, 5, 6)
	cases := []struct {
		op   string
		b    *Tensor
		want []float32
	}{
		{"Add", fT([]int{3}, 10, 20, 30), []float32{11, 22, 33, 14, 25, 36}},
		{"Sub", fT([]int{2, 1}, 1, 2), []float32{0, 1, 2, 2, 3, 4}},
		{"Mul", fT([]int{}, 2), []float32{2, 4, 6, 8, 10, 12}},
		{"Div", fT([]int{1, 3}, 1, 2, 3), []float32{1, 1, 1, 4, 2.5, 2}},
		{"Pow", fT([]int{}, 2), []float32{1, 4, 9, 16, 25, 36}},
		{"Mod", fT([]int{}, 4), []float32{1, 2, 3, 0, 1, 2}},
	}
	for _, c := range cases {
		t.Run(c.op, func(t *testing.T) { wantF(t, run1(t, c.op, nil, a, c.b), []int{2, 3}, c.want...) })
	}
	wantF(t, run1(t, "Mod", map[string]any{"fmod": 1}, fT([]int{2}, -5, 5), fT([]int{}, 3)), []int{2}, -2, 2)
	wantF(t, run1(t, "Mod", nil, fT([]int{2}, -5, 5), fT([]int{}, 3)), []int{2}, 1, 2)
}

func TestArithmeticInt(t *testing.T) {
	a := iT([]int{3}, 7, -7, 5)
	wantI(t, run1(t, "Add", nil, a, iT([]int{}, 1)), []int{3}, 8, -6, 6)
	wantI(t, run1(t, "Sub", nil, a, iT([]int{}, 1)), []int{3}, 6, -8, 4)
	wantI(t, run1(t, "Mul", nil, a, iT([]int{}, 2)), []int{3}, 14, -14, 10)
	wantI(t, run1(t, "Div", nil, a, iT([]int{3}, 2, 2, 0)), []int{3}, 3, -3, 0)
	wantI(t, run1(t, "Pow", nil, iT([]int{2}, 2, 3), iT([]int{}, 3)), []int{2}, 8, 27)
	wantI(t, run1(t, "Mod", nil, a, iT([]int{}, 3)), []int{3}, 1, 2, 2)
	wantI(t, run1(t, "Mod", map[string]any{"fmod": 1}, a, iT([]int{3}, 3, 3, 0)), []int{3}, 1, -1, 0)
	wantI(t, run1(t, "Mod", nil, a, iT([]int{}, 0)), []int{3}, 0, 0, 0)
}

func TestPowKernels(t *testing.T) {
	x := fT([]int{3}, 1, 4, 9)
	wantF(t, run1(t, "Pow", nil, x, fT([]int{}, 2)), []int{3}, 1, 16, 81)
	wantF(t, run1(t, "Pow", nil, x, fT([]int{}, 0.5)), []int{3}, 1, 2, 3)
	wantF(t, run1(t, "Pow", nil, x, fT([]int{}, 3)), []int{3}, 1, 64, 729)
	wantF(t, run1(t, "Pow", nil, x, fT([]int{3}, 1, 0.5, 0)), []int{3}, 1, 2, 1)
	wantF(t, run1(t, "Pow", nil, fT([]int{}, 2), fT([]int{3}, 1, 2, 3)), []int{3}, 2, 4, 8)
}

func TestArithmeticErrors(t *testing.T) {
	mustFail(t, "Add", nil, "at least 2 inputs", fT([]int{1}, 1))
	mustFail(t, "Add", nil, "input 1 is empty", fT([]int{1}, 1), nil)
	mustFail(t, "Add", nil, "dtype mismatch", fT([]int{1}, 1), iT([]int{1}, 1))
	mustFail(t, "Mul", nil, "dtype mismatch", fT([]int{1}, 1), iT([]int{1}, 1))
	mustFail(t, "Add", nil, "cannot broadcast", fT([]int{2}, 1, 2), fT([]int{3}, 1, 2, 3))
	mustFail(t, "Add", nil, "cannot broadcast", iT([]int{2}, 1, 2), iT([]int{3}, 1, 2, 3))
}

func TestFastFloatBroadcastShapes(t *testing.T) {
	a := fT([]int{2, 1, 3}, iota32(6)...)
	b := fT([]int{1, 4, 1}, 10, 20, 30, 40)
	wantF(t, run1(t, "Add", nil, a, b), []int{2, 4, 3},
		10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42,
		13, 14, 15, 23, 24, 25, 33, 34, 35, 43, 44, 45)
	wantF(t, run1(t, "Sub", nil, fT([]int{}, 5), fT([]int{}, 3)), []int{}, 2)
	wantF(t, run1(t, "Mul", nil, fT([]int{3, 1}, 1, 2, 3), fT([]int{1}, 2)), []int{3, 1}, 2, 4, 6)
	wantF(t, run1(t, "Div", nil, fT([]int{1}, 8), fT([]int{3}, 1, 2, 4)), []int{3}, 8, 4, 2)
	wantF(t, run1(t, "Add", nil, fT([]int{2, 1}, 1, 2), fT([]int{1, 1}, 5)), []int{2, 1}, 6, 7)
	wantF(t, run1(t, "Sub", nil, fT([]int{1, 3}, 1, 2, 3), fT([]int{2, 1}, 1, 10)), []int{2, 3}, 0, 1, 2, -9, -8, -7)
	wantF(t, run1(t, "Div", nil, fT([]int{2, 2}, 2, 4, 6, 8), fT([]int{}, 2)), []int{2, 2}, 1, 2, 3, 4)
	scalarLeft := fT([]int{1}, 10)
	vector := fT([]int{3}, 1, 2, 4)
	wantF(t, run1(t, "Add", nil, scalarLeft, vector), []int{3}, 11, 12, 14)
	wantF(t, run1(t, "Sub", nil, scalarLeft, vector), []int{3}, 9, 8, 6)
	wantF(t, run1(t, "Mul", nil, scalarLeft, vector), []int{3}, 10, 20, 40)
	wantF(t, run1(t, "Div", nil, scalarLeft, vector), []int{3}, 10, 5, 2.5)
	empty := run1(t, "Add", nil, fT([]int{0, 3}), fT([]int{3}, 1, 2, 3))
	wantShape(t, empty, []int{0, 3})
}

func TestFastFloatLargeParallelMatchesGeneric(t *testing.T) {
	n := 1 << 17
	a := fT([]int{4, n / 4}, iota32(n)...)
	b := fT([]int{n / 4}, iota32(n/4)...)
	got := run1(t, "Add", nil, a, b)
	for i := range got.F {
		if want := a.F[i] + b.F[i%(n/4)]; got.F[i] != want {
			t.Fatalf("element %d: got %v, want %v", i, got.F[i], want)
		}
	}
}

func TestComparisonAndLogic(t *testing.T) {
	a, b := fT([]int{3}, 1, 2, 3), fT([]int{3}, 2, 2, 2)
	wantB(t, run1(t, "Equal", nil, a, b), []int{3}, false, true, false)
	wantB(t, run1(t, "Greater", nil, a, b), []int{3}, false, false, true)
	wantB(t, run1(t, "Less", nil, a, b), []int{3}, true, false, false)
	wantB(t, run1(t, "LessOrEqual", nil, a, b), []int{3}, true, true, false)
	ia, ib := iT([]int{3}, 1, 2, 3), iT([]int{}, 2)
	wantB(t, run1(t, "Equal", nil, ia, ib), []int{3}, false, true, false)
	wantB(t, run1(t, "Greater", nil, ia, ib), []int{3}, false, false, true)
	wantB(t, run1(t, "Less", nil, ia, ib), []int{3}, true, false, false)
	wantB(t, run1(t, "LessOrEqual", nil, ia, ib), []int{3}, true, true, false)
	p, q := bT([]int{4}, true, true, false, false), bT([]int{4}, true, false, true, false)
	wantB(t, run1(t, "And", nil, p, q), []int{4}, true, false, false, false)
	wantB(t, run1(t, "Or", nil, p, q), []int{4}, true, true, true, false)
	wantB(t, run1(t, "Xor", nil, p, q), []int{4}, false, true, true, false)
	wantB(t, run1(t, "Not", nil, p), []int{4}, false, false, true, true)
	mustFail(t, "Equal", nil, "dtype mismatch", a, ia)
	mustFail(t, "Not", nil, "unsupported", a)
	mustFail(t, "And", nil, "at least 2 inputs", p)
}

func TestUnaryMath(t *testing.T) {
	x := fT([]int{4}, -2, -0.5, 0, 2.25)
	wantF(t, run1(t, "Abs", nil, x), []int{4}, 2, 0.5, 0, 2.25)
	wantF(t, run1(t, "Neg", nil, x), []int{4}, 2, 0.5, 0, -2.25)
	wantF(t, run1(t, "Sign", nil, x), []int{4}, -1, -1, 0, 1)
	wantF(t, run1(t, "Ceil", nil, x), []int{4}, -2, 0, 0, 3)
	wantF(t, run1(t, "Relu", nil, x), []int{4}, 0, 0, 0, 2.25)
	wantF(t, run1(t, "Sqrt", nil, fT([]int{2}, 4, 2.25)), []int{2}, 2, 1.5)
	wantF(t, run1(t, "Log", nil, fT([]int{2}, 1, float32(math.E))), []int{2}, 0, 1)
	wantF(t, run1(t, "Erf", nil, fT([]int{3}, 0, 1, -1)), []int{3}, 0, 0.8427008, -0.8427008)
	wantF(t, run1(t, "Sigmoid", nil, fT([]int{2}, 0, 100)), []int{2}, 0.5, 1)
	wantF(t, run1(t, "Tanh", nil, fT([]int{3}, 0, 20, -20)), []int{3}, 0, 1, -1)
	if got := run1(t, "Tanh", nil, fT([]int{1}, 0.5)); math.Abs(float64(got.F[0])-math.Tanh(0.5)) > 1e-6 {
		t.Fatalf("tanh(0.5) = %v", got.F[0])
	}
	wantI(t, run1(t, "Abs", nil, iT([]int{2}, -3, 3)), []int{2}, 3, 3)
	wantI(t, run1(t, "Neg", nil, iT([]int{2}, -3, 3)), []int{2}, 3, -3)
	wantI(t, run1(t, "Sign", nil, iT([]int{3}, -3, 0, 3)), []int{3}, -1, 0, 1)
	mustFail(t, "Relu", nil, "unsupported for int64", iT([]int{1}, 1))
	mustFail(t, "Ceil", nil, "unsupported", iT([]int{1}, 1))
	mustFail(t, "Abs", nil, "at least 1 inputs")
	mustFail(t, "Ceil", nil, "input 0 is empty", nil)
}

func TestErfApproximationError(t *testing.T) {
	var worst float64
	for x := -8.0; x <= 8.0; x += 1e-3 {
		worst = max(worst, math.Abs(float64(erf32(float32(x)))-math.Erf(float64(float32(x)))))
	}
	if worst > 5e-7 {
		t.Fatalf("erf32 error %g exceeds 5e-7", worst)
	}
	for _, x := range []float32{float32(math.Inf(1)), float32(math.Inf(-1))} {
		if got := erf32(x); math.Abs(float64(got)-math.Copysign(1, float64(x))) > 1e-6 {
			t.Fatalf("erf32(%v) = %v", x, got)
		}
	}
	if got := erf32(float32(math.NaN())); got == got {
		t.Fatalf("erf32(NaN) = %v, want NaN", got)
	}
}

func TestUnaryWithoutFloatImplementation(t *testing.T) {
	if _, err := unary(nil, func(x int64) int64 { return x })(&opCtx{in: []*Tensor{fT([]int{1}, 1)}}); err == nil {
		t.Fatal("expected an error for a float input without a float implementation")
	}
}

func TestUnaryLargeParallel(t *testing.T) {
	x := fT([]int{1 << 17}, iota32(1<<17)...)
	got := run1(t, "Neg", nil, x)
	for i, v := range got.F {
		if v != -float32(i) {
			t.Fatalf("element %d: %v", i, v)
		}
	}
}

func TestIsNaN(t *testing.T) {
	wantB(t, run1(t, "IsNaN", nil, fT([]int{3}, 1, float32(math.NaN()), float32(math.Inf(1)))), []int{3}, false, true, false)
	mustFail(t, "IsNaN", nil, "at least 1 inputs")
}

func TestCast(t *testing.T) {
	f := fT([]int{4}, -1.7, 0, 2.5, 3)
	wantI(t, run1(t, "Cast", map[string]any{"to": onnxInt64}, f), []int{4}, -1, 0, 2, 3)
	wantI(t, run1(t, "Cast", map[string]any{"to": onnxInt32}, f), []int{4}, -1, 0, 2, 3)
	wantB(t, run1(t, "Cast", map[string]any{"to": onnxBool}, f), []int{4}, true, false, true, true)
	i := iT([]int{3}, 0, 2, -3)
	wantF(t, run1(t, "Cast", map[string]any{"to": onnxFloat}, i), []int{3}, 0, 2, -3)
	wantB(t, run1(t, "Cast", map[string]any{"to": onnxBool}, i), []int{3}, false, true, true)
	b := bT([]int{2}, true, false)
	wantI(t, run1(t, "Cast", map[string]any{"to": onnxInt64}, b), []int{2}, 1, 0)
	wantF(t, run1(t, "Cast", map[string]any{"to": onnxFloat}, b), []int{2}, 1, 0)
	same := run1(t, "Cast", map[string]any{"to": onnxFloat}, f)
	if same != f {
		t.Fatal("casting to the same dtype should return the input")
	}
	mustFail(t, "Cast", map[string]any{"to": 99}, "unsupported cast target", f)
	mustFail(t, "Cast", map[string]any{"to": onnxFloat}, "at least 1 inputs")
}

func TestWhere(t *testing.T) {
	cond := bT([]int{2, 1}, true, false)
	wantF(t, run1(t, "Where", nil, cond, fT([]int{3}, 1, 2, 3), fT([]int{}, 9)), []int{2, 3}, 1, 2, 3, 9, 9, 9)
	wantI(t, run1(t, "Where", nil, cond, iT([]int{1}, 5), iT([]int{2, 2}, 1, 2, 3, 4)), []int{2, 2}, 5, 5, 3, 4)
	mustFail(t, "Where", nil, "dtype mismatch", cond, fT([]int{1}, 1), iT([]int{1}, 1))
	mustFail(t, "Where", nil, "cannot broadcast", bT([]int{2}, true, true), fT([]int{3}, 1, 2, 3), fT([]int{3}, 1, 2, 3))
	mustFail(t, "Where", nil, "cannot broadcast", bT([]int{1}, true), fT([]int{3}, 1, 2, 3), fT([]int{2}, 1, 2))
	mustFail(t, "Where", nil, "at least 3 inputs", cond, fT([]int{1}, 1))
}

func TestWhereScalarAndEmptyShapes(t *testing.T) {
	wantF(t, run1(t, "Where", nil, bT([]int{}, true), fT([]int{}, 1), fT([]int{}, 2)), []int{}, 1)
	wantF(t, run1(t, "Where", nil, bT([]int{}, false), fT([]int{}, 1), fT([]int{}, 2)), []int{}, 2)
	wantI(t, run1(t, "Where", nil, bT([]int{}, false), iT([]int{}, 1), iT([]int{}, 2)), []int{}, 2)
	wantShape(t, run1(t, "Where", nil, bT([]int{0, 2}), fT([]int{2}, 1, 2), fT([]int{2}, 3, 4)), []int{0, 2})
}

func TestErfOperatorErrors(t *testing.T) {
	mustFail(t, "Erf", nil, "at least 1 inputs")
	mustFail(t, "Erf", nil, "unsupported for int64", iT([]int{1}, 1))
}

func TestClip(t *testing.T) {
	x := fT([]int{4}, -5, 0, 5, 10)
	wantF(t, run1(t, "Clip", nil, x, fT([]int{}, -1), fT([]int{}, 6)), []int{4}, -1, 0, 5, 6)
	wantF(t, run1(t, "Clip", nil, x, nil, fT([]int{}, 6)), []int{4}, -5, 0, 5, 6)
	wantF(t, run1(t, "Clip", nil, x, fT([]int{}, 1)), []int{4}, 1, 1, 5, 10)
	wantF(t, run1(t, "Clip", nil, x), []int{4}, -5, 0, 5, 10)
	xi := iT([]int{4}, -5, 0, 5, 10)
	wantI(t, run1(t, "Clip", nil, xi, iT([]int{}, -1), iT([]int{}, 6)), []int{4}, -1, 0, 5, 6)
	wantI(t, run1(t, "Clip", nil, xi, nil, iT([]int{}, 6)), []int{4}, -5, 0, 5, 6)
	wantI(t, run1(t, "Clip", nil, xi), []int{4}, -5, 0, 5, 10)
	mustFail(t, "Clip", nil, "at least 1 inputs")
}

func TestIdentity(t *testing.T) {
	x := fT([]int{1}, 3)
	if run1(t, "Identity", nil, x) != x {
		t.Fatal("Identity must return its input")
	}
	mustFail(t, "Identity", nil, "at least 1 inputs")
}
