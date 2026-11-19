package onnxgo

import (
	"slices"
	"testing"
)

func TestDTypeString(t *testing.T) {
	for dt, want := range map[DType]string{Float32: "float32", Int64: "int64", Bool: "bool", 0: "invalid"} {
		if dt.String() != want {
			t.Errorf("%d: %q, want %q", dt, dt.String(), want)
		}
	}
}

func TestTensorConstructorsAndAccessors(t *testing.T) {
	b := bT([]int{3}, true, false, true)
	if !slices.Equal(b.I, []int64{1, 0, 1}) || !slices.Equal(b.Bools(), []bool{true, false, true}) || b.Len() != 3 {
		t.Fatalf("bool tensor %+v", b)
	}
	if v, err := iT([]int{1}, 7).scalarInt(); err != nil || v != 7 {
		t.Fatalf("scalarInt: %v %v", v, err)
	}
	if _, err := iT([]int{2}, 1, 2).scalarInt(); err == nil {
		t.Fatal("two elements are not a scalar")
	}
	if _, err := fT([]int{1}, 1).scalarInt(); err == nil {
		t.Fatal("float is not an integer scalar")
	}
	if _, err := intList(nil); err == nil {
		t.Fatal("nil tensor must fail intList")
	}
}

func TestNormAxis(t *testing.T) {
	if a, err := normAxis(-1, 3); err != nil || a != 2 {
		t.Fatalf("got %d %v", a, err)
	}
	if _, err := normAxis(3, 3); err == nil {
		t.Fatal("expected an error")
	}
	if _, err := normAxis(-4, 3); err == nil {
		t.Fatal("expected an error")
	}
}

func TestBroadcastHelpers(t *testing.T) {
	if s, err := broadcastShape([]int{2, 1}, []int{3}); err != nil || !slices.Equal(s, []int{2, 3}) {
		t.Fatalf("got %v %v", s, err)
	}
	if _, err := broadcastShape([]int{2}, []int{3}); err == nil {
		t.Fatal("expected an error")
	}
	if broadcastOffsets([]int{2, 3}, []int{2, 3}) != nil {
		t.Fatal("equal shapes need no offsets")
	}
	if got := broadcastOffsets([]int{3}, []int{2, 3}); !slices.Equal(got, []int{0, 1, 2, 0, 1, 2}) {
		t.Fatalf("got %v", got)
	}
	if got := broadcastOffsets([]int{2, 1}, []int{2, 2}); !slices.Equal(got, []int{0, 0, 1, 1}) {
		t.Fatalf("got %v", got)
	}
	if got := broadcastStrides([]int{2, 1}, []int{3, 2, 4}); !slices.Equal(got, []int{0, 1, 0}) {
		t.Fatalf("got %v", got)
	}
}

func TestOffsetsAndRows(t *testing.T) {
	if got := offsetsFromStrides([]int{2, 3}, []int{3, 1}, 0); !slices.Equal(got, []int{0, 1, 2, 3, 4, 5}) {
		t.Fatalf("got %v", got)
	}
	if got := offsetsFromStrides([]int{2, 2}, []int{1, 2}, 10); !slices.Equal(got, []int{10, 12, 11, 13}) {
		t.Fatalf("got %v", got)
	}
	if got := offsetsFromStrides([]int{0, 3}, []int{3, 1}, 0); len(got) != 0 {
		t.Fatalf("got %v", got)
	}
	var calls int
	forEachRow([]int{}, nil, 5, func(start, stride, n int) {
		calls++
		if start != 5 || n != 1 {
			t.Fatalf("scalar row: %d %d %d", start, stride, n)
		}
	})
	if calls != 1 {
		t.Fatalf("scalar rows: %d", calls)
	}
	var rows [][2]int
	forEachRowRange([]int{3, 2}, []int{2, 1}, 0, 1, 3, func(start, stride, n int) { rows = append(rows, [2]int{start, n}) })
	if !slices.Equal(rows, [][2]int{{2, 2}, {4, 2}}) {
		t.Fatalf("got %v", rows)
	}
}

func TestStridedCopy(t *testing.T) {
	x := fT([]int{2, 3}, 0, 1, 2, 3, 4, 5)
	wantF(t, stridedCopy(x, []int{3, 2}, []int{1, 3}, 0), []int{3, 2}, 0, 3, 1, 4, 2, 5)
	wantF(t, stridedCopy(x, []int{2, 2}, []int{3, 1}, 1), []int{2, 2}, 1, 2, 4, 5)
	wantF(t, stridedCopy(x, []int{}, []int{}, 4), []int{}, 4)
	wantI(t, stridedCopy(iT([]int{4}, 1, 2, 3, 4), []int{2}, []int{2}, 0), []int{2}, 1, 3)
	wantI(t, stridedCopy(iT([]int{4}, 1, 2, 3, 4), []int{2, 2}, []int{2, 1}, 0), []int{2, 2}, 1, 2, 3, 4)
	wantShape(t, stridedCopy(x, []int{0, 3}, []int{3, 1}, 0), []int{0, 3})
	big := fT([]int{512, 512}, iota32(512*512)...)
	tr := stridedCopy(big, []int{512, 512}, []int{1, 512}, 0)
	if tr.F[1] != 512 || tr.F[512] != 1 {
		t.Fatalf("parallel transpose wrong: %v %v", tr.F[1], tr.F[512])
	}
}

func TestRowHelpers(t *testing.T) {
	if got := rowIndex(7, []int{2, 2, 3}); !slices.Equal(got, []int{1, 0, 1}) {
		t.Fatalf("got %v", got)
	}
	if got := rowOffset(7, []int{2, 2, 3}, []int{100, 10, 1}); got != 101 {
		t.Fatalf("got %d", got)
	}
	if got := stridesOf([]int{2, 3, 4}); !slices.Equal(got, []int{12, 4, 1}) {
		t.Fatalf("got %v", got)
	}
}

func TestPickAndCopyTo(t *testing.T) {
	x := fT([]int{4}, 1, 2, 3, 4)
	wantF(t, x.pick([]int{2}, []int{3, 0}), []int{2}, 4, 1)
	xi := iT([]int{4}, 1, 2, 3, 4)
	wantI(t, xi.pick([]int{2}, []int{2, 1}), []int{2}, 3, 2)
	dst := zeros(Int64, []int{4})
	xi.copyTo(dst, 1, 2, 2)
	wantI(t, dst, []int{4}, 0, 3, 4, 0)
	wantShape(t, zeros(Float32, []int{0, 2}), []int{0, 2})
	copyElem(dst, 0, xi, 3)
	if dst.I[0] != 4 {
		t.Fatal("copyElem failed")
	}
}
