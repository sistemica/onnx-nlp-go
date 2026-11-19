package onnxgo

import (
	"math"
	"slices"
	"strings"
	"testing"
)

func fT(shape []int, v ...float32) *Tensor { return NewFloat(shape, v) }
func iT(shape []int, v ...int64) *Tensor   { return NewInt(shape, v) }
func bT(shape []int, v ...bool) *Tensor    { return NewBool(shape, v) }

func mkAttrs(kv map[string]any) map[string]*Attr {
	out := map[string]*Attr{}
	for k, v := range kv {
		switch x := v.(type) {
		case int:
			out[k] = &Attr{I: int64(x)}
		case float32:
			out[k] = &Attr{F: x}
		case string:
			out[k] = &Attr{S: x}
		case []int64:
			out[k] = &Attr{Ints: x}
		case []float32:
			out[k] = &Attr{Floats: x}
		case *Tensor:
			out[k] = &Attr{T: x}
		case *Graph:
			out[k] = &Attr{G: x}
		default:
			panic("unsupported attribute type")
		}
	}
	return out
}

func callOp(op string, attrs map[string]any, in ...*Tensor) ([]*Tensor, error) {
	node := &Node{Op: op, Attr: mkAttrs(attrs), Out: []string{"y0", "y1", "y2"}}
	return ops[op](&opCtx{node: node, in: in})
}

func run1(t *testing.T, op string, attrs map[string]any, in ...*Tensor) *Tensor {
	t.Helper()
	outs, err := callOp(op, attrs, in...)
	if err != nil {
		t.Fatalf("%s: %v", op, err)
	}
	return outs[0]
}

func mustFail(t *testing.T, op string, attrs map[string]any, want string, in ...*Tensor) {
	t.Helper()
	_, err := callOp(op, attrs, in...)
	if err == nil {
		t.Fatalf("%s: expected an error containing %q", op, want)
	}
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("%s: error %q does not contain %q", op, err, want)
	}
}

func wantShape(t *testing.T, got *Tensor, shape []int) {
	t.Helper()
	if !slices.Equal(got.Shape, shape) {
		t.Fatalf("shape %v, want %v", got.Shape, shape)
	}
}

func wantF(t *testing.T, got *Tensor, shape []int, vals ...float32) {
	t.Helper()
	if got.DType != Float32 {
		t.Fatalf("dtype %s, want float32", got.DType)
	}
	wantShape(t, got, shape)
	if len(got.F) != len(vals) {
		t.Fatalf("got %d values %v, want %d", len(got.F), got.F, len(vals))
	}
	for i, v := range vals {
		if math.Abs(float64(got.F[i]-v)) > 1e-4*(1+math.Abs(float64(v))) && !(math.IsInf(float64(v), 0) && got.F[i] == v) {
			t.Fatalf("value %d: got %v, want %v (all: %v)", i, got.F[i], v, got.F)
		}
	}
}

func wantI(t *testing.T, got *Tensor, shape []int, vals ...int64) {
	t.Helper()
	if got.DType != Int64 {
		t.Fatalf("dtype %s, want int64", got.DType)
	}
	wantShape(t, got, shape)
	if !slices.Equal(got.I, vals) {
		t.Fatalf("got %v, want %v", got.I, vals)
	}
}

func wantB(t *testing.T, got *Tensor, shape []int, vals ...bool) {
	t.Helper()
	if got.DType != Bool {
		t.Fatalf("dtype %s, want bool", got.DType)
	}
	wantShape(t, got, shape)
	if !slices.Equal(got.Bools(), vals) {
		t.Fatalf("got %v, want %v", got.Bools(), vals)
	}
}

func iota32(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(i)
	}
	return out
}
