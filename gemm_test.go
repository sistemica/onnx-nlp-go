package onnxnlp

import (
	"math"
	"math/rand/v2"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func checkGemmAgainstNaive(t *testing.T) {
	t.Helper()
	rng := rand.New(rand.NewPCG(21, 22))
	sizes := [][3]int{{1, 1, 1}, {3, 5, 7}, {4, 16, 16}, {4, 3, 16}, {5, 17, 33}, {8, 64, 48}, {37, 96, 50}, {130, 70, 129}, {2, 1, 40}}
	for _, sz := range sizes {
		m, k, n := sz[0], sz[1], sz[2]
		a, b := randomSlice(rng, m*k), randomSlice(rng, k*n)
		want := naiveGemm(a, b, m, k, n)
		out := make([]float32, m*n)
		runGemms([]gemmJob{{out: out, a: a, b: b, m: m, k: k, n: n}})
		for i := range want {
			if math.Abs(float64(out[i]-want[i])) > 1e-4*(1+math.Abs(float64(want[i]))) {
				t.Fatalf("%v element %d: got %v, want %v", sz, i, out[i], want[i])
			}
		}
	}
}

func TestGemmMatchesNaive(t *testing.T) { checkGemmAgainstNaive(t) }

func TestGemmWithPrepackedWeights(t *testing.T) {
	rng := rand.New(rand.NewPCG(23, 24))
	for _, sz := range [][3]int{{1, 5, 3}, {3, 17, 20}, {4, 16, 16}, {9, 33, 47}, {130, 70, 129}} {
		m, k, n := sz[0], sz[1], sz[2]
		w := NewFloat([]int{k, n}, randomSlice(rng, k*n))
		w.static, w.pack = true, &packCache{}
		for round := 0; round < 2; round++ {
			x := NewFloat([]int{m, k}, randomSlice(rng, m*k))
			got, err := opMatMul(&opCtx{in: []*Tensor{x, w}})
			if err != nil {
				t.Fatal(err)
			}
			want := naiveGemm(x.F, w.F, m, k, n)
			for i := range want {
				if math.Abs(float64(got[0].F[i]-want[i])) > 1e-4*(1+math.Abs(float64(want[i]))) {
					t.Fatalf("%v round %d element %d: got %v, want %v", sz, round, i, got[0].F[i], want[i])
				}
			}
		}
	}
}

func TestPackIsIgnoredForBatchedWeights(t *testing.T) {
	w := fT([]int{2, 3, 2}, 1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 2, 2)
	w.static, w.pack = true, &packCache{}
	x := fT([]int{2, 1, 3}, 1, 2, 3, 1, 2, 3)
	got := run1(t, "MatMul", nil, x, w)
	wantF(t, got, []int{2, 1, 2}, 4, 5, 8, 10)
}

func TestGemmAccumulatesAndSkipsEmpty(t *testing.T) {
	runGemms(nil)
	out := []float32{10, 10}
	runGemms([]gemmJob{{out: out, a: []float32{1, 2}, b: []float32{3, 4}, m: 1, k: 2, n: 1}})
	if out[0] != 21 || out[1] != 10 {
		t.Fatalf("got %v", out)
	}
}

func TestParallelForSplitsWork(t *testing.T) {
	var covered [100]int
	parallelFor(100, minParallelWork, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			covered[i]++
		}
	})
	for i, c := range covered {
		if c != 1 {
			t.Fatalf("index %d visited %d times", i, c)
		}
	}
	serial := 0
	parallelFor(5, 1, func(lo, hi int) { serial += hi - lo })
	if serial != 5 {
		t.Fatalf("serial path covered %d items", serial)
	}
}

func TestMatMul(t *testing.T) {
	a := fT([]int{2, 3}, 1, 2, 3, 4, 5, 6)
	b := fT([]int{3, 2}, 7, 8, 9, 10, 11, 12)
	wantF(t, run1(t, "MatMul", nil, a, b), []int{2, 2}, 58, 64, 139, 154)
	batched := fT([]int{2, 1, 3}, 1, 0, 0, 0, 1, 0)
	wantF(t, run1(t, "MatMul", nil, batched, b), []int{2, 1, 2}, 7, 8, 9, 10)
	bothBatched := fT([]int{2, 3, 1}, 1, 1, 1, 2, 2, 2)
	wantF(t, run1(t, "MatMul", nil, fT([]int{2, 1, 3}, 1, 2, 3, 1, 2, 3), bothBatched), []int{2, 1, 1}, 6, 12)
	folded := fT([]int{2, 3, 3}, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 2)
	wantF(t, run1(t, "MatMul", nil, folded, b), []int{2, 3, 2}, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24)
	mustFail(t, "MatMul", nil, "float32 only", iT([]int{2, 2}, 1, 2, 3, 4), iT([]int{2, 2}, 1, 2, 3, 4))
	mustFail(t, "MatMul", nil, "rank >= 2", fT([]int{3}, 1, 2, 3), b)
	mustFail(t, "MatMul", nil, "shape mismatch", a, a)
	mustFail(t, "MatMul", nil, "cannot broadcast", fT([]int{2, 1, 1}, 1, 1), fT([]int{3, 1, 1}, 1, 1, 1))
	mustFail(t, "MatMul", nil, "at least 2 inputs", a)
}

func TestGemm(t *testing.T) {
	a := fT([]int{2, 3}, 1, 2, 3, 4, 5, 6)
	b := fT([]int{3, 2}, 1, 0, 0, 1, 1, 1)
	wantF(t, run1(t, "Gemm", nil, a, b), []int{2, 2}, 4, 5, 10, 11)
	wantF(t, run1(t, "Gemm", map[string]any{"alpha": float32(2)}, a, b), []int{2, 2}, 8, 10, 20, 22)
	wantF(t, run1(t, "Gemm", map[string]any{"beta": float32(0.5)}, a, b, fT([]int{2}, 2, 4)), []int{2, 2}, 5, 7, 11, 13)
	wantF(t, run1(t, "Gemm", map[string]any{"transB": 1}, a, fT([]int{2, 3}, 1, 0, 1, 0, 1, 1)), []int{2, 2}, 4, 5, 10, 11)
	wantF(t, run1(t, "Gemm", map[string]any{"transA": 1}, fT([]int{3, 2}, 1, 4, 2, 5, 3, 6), b), []int{2, 2}, 4, 5, 10, 11)
	mustFail(t, "Gemm", nil, "rank-2", fT([]int{3}, 1, 2, 3), b)
	mustFail(t, "Gemm", nil, "shape mismatch", a, a)
	mustFail(t, "Gemm", nil, "at least 2 inputs", a)
}

func TestEinsum(t *testing.T) {
	a := fT([]int{2, 3}, 1, 2, 3, 4, 5, 6)
	b := fT([]int{3, 2}, 1, 0, 0, 1, 1, 1)
	wantF(t, run1(t, "Einsum", map[string]any{"equation": "ij,jk->ik"}, a, b), []int{2, 2}, 4, 5, 10, 11)
	wantF(t, run1(t, "Einsum", map[string]any{"equation": "ij->i"}, a), []int{2}, 6, 15)
	wantF(t, run1(t, "Einsum", map[string]any{"equation": "ij->ji"}, a), []int{3, 2}, 1, 4, 2, 5, 3, 6)
	wantF(t, run1(t, "Einsum", map[string]any{"equation": "bcd,bqd->bcq"}, fT([]int{1, 2, 2}, 1, 0, 0, 1), fT([]int{1, 3, 2}, 1, 2, 3, 4, 5, 6)), []int{1, 2, 3}, 1, 3, 5, 2, 4, 6)
	mustFail(t, "Einsum", map[string]any{"equation": "ij,jk"}, "unsupported einsum", a, b)
	mustFail(t, "Einsum", map[string]any{"equation": "...i->i"}, "unsupported einsum", a)
	mustFail(t, "Einsum", map[string]any{"equation": "ij->i"}, "expects 1 inputs", a, b)
	mustFail(t, "Einsum", map[string]any{"equation": "ijk->i"}, "does not match", a)
	mustFail(t, "Einsum", map[string]any{"equation": "ij->i"}, "does not match", iT([]int{2, 2}, 1, 2, 3, 4))
	mustFail(t, "Einsum", map[string]any{"equation": "ij,ik->jk"}, "has sizes", a, fT([]int{3, 2}, 1, 2, 3, 4, 5, 6))
	mustFail(t, "Einsum", map[string]any{"equation": "ij->z"}, "not in inputs", a)
}

func TestLayerNormalization(t *testing.T) {
	x := fT([]int{2, 4}, 1, 2, 3, 4, 10, 10, 10, 10)
	scale, bias := fT([]int{4}, 1, 1, 1, 1), fT([]int{4}, 0, 0, 0, 1)
	outs, err := callOp("LayerNormalization", map[string]any{"epsilon": float32(0)}, x, scale, bias)
	if err != nil {
		t.Fatal(err)
	}
	r := float32(1 / math.Sqrt(1.25))
	wantF(t, outs[0], []int{2, 4}, -1.5*r, -0.5*r, 0.5*r, 1.5*r+1, 0, 0, 0, 1)
	wantF(t, outs[1], []int{2, 1}, 2.5, 10)
	wantF(t, run1(t, "LayerNormalization", map[string]any{"epsilon": float32(0), "axis": 0}, fT([]int{2, 2}, 1, 3, 1, 3), fT([]int{2, 2}, 1, 1, 1, 1)), []int{2, 2}, -1, 1, -1, 1)
	noBias := run1(t, "LayerNormalization", nil, x, scale)
	wantShape(t, noBias, []int{2, 4})
	mustFail(t, "LayerNormalization", map[string]any{"axis": 5}, "out of range", x, scale)
	mustFail(t, "LayerNormalization", nil, "at least 2 inputs", x)
}

func TestSoftmax(t *testing.T) {
	x := fT([]int{2, 2}, 0, 0, 1, 3)
	e := float32(math.Exp(2))
	wantF(t, run1(t, "Softmax", nil, x), []int{2, 2}, 0.5, 0.5, 1/(1+e), e/(1+e))
	s1, s3 := float32(1/(1+math.E)), float32(1/(1+math.Exp(3)))
	wantF(t, run1(t, "Softmax", map[string]any{"axis": 0}, x), []int{2, 2}, s1, s3, 1-s1, 1-s3)
	wide := fT([]int{2, 3, 20}, iota32(120)...)
	for _, tc := range []struct{ axis, size, inner int }{{-1, 20, 1}, {1, 3, 20}} {
		got := run1(t, "Softmax", map[string]any{"axis": tc.axis}, wide)
		for g := 0; g < 120/tc.size; g++ {
			base := (g/tc.inner)*tc.size*tc.inner + g%tc.inner
			var sum float64
			for i := 0; i < tc.size; i++ {
				sum += float64(got.F[base+i*tc.inner])
			}
			if math.Abs(sum-1) > 1e-5 {
				t.Fatalf("axis %d group %d sums to %v", tc.axis, g, sum)
			}
		}
	}
	mustFail(t, "Softmax", map[string]any{"axis": 4}, "out of range", x)
	mustFail(t, "Softmax", nil, "at least 1 inputs")
}

func TestParallelForCoversEveryItemOnce(t *testing.T) {
	for _, n := range []int{1, 7, 100, 1000, 12345} {
		counts := make([]int32, n)
		parallelFor(n, minParallelWork, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				atomic.AddInt32(&counts[i], 1)
			}
		})
		for i, c := range counts {
			if c != 1 {
				t.Fatalf("n=%d: item %d ran %d times", n, i, c)
			}
		}
	}
}

func TestParallelForNestedAndConcurrentCalls(t *testing.T) {
	var total atomic.Int64
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			parallelFor(64, minParallelWork, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					parallelFor(16, minParallelWork, func(a, b int) { total.Add(int64(b - a)) })
				}
			})
		}()
	}
	wg.Wait()
	if want := int64(8 * 64 * 16); total.Load() != want {
		t.Fatalf("nested parallel regions ran %d items, want %d", total.Load(), want)
	}
}

func TestParallelForGrowsThePoolWithGOMAXPROCS(t *testing.T) {
	saved := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(saved)
	for _, procs := range []int{2, 6, 3} {
		runtime.GOMAXPROCS(procs)
		var ran atomic.Int64
		parallelFor(1000, minParallelWork, func(lo, hi int) { ran.Add(int64(hi - lo)) })
		if ran.Load() != 1000 {
			t.Fatalf("GOMAXPROCS=%d: ran %d items", procs, ran.Load())
		}
	}
	pool.mu.Lock()
	workers := pool.workers
	pool.mu.Unlock()
	if workers < 5 {
		t.Fatalf("pool has %d helpers after running with 6 procs, want at least 5", workers)
	}
}

// bruteEinsum evaluates an explicit-output einsum by visiting every assignment of the labels.
func bruteEinsum(terms []string, rhs string, in []*Tensor, size map[rune]int) *Tensor {
	var labels []rune
	for l := range size {
		labels = append(labels, l)
	}
	shape := make([]int, 0, len(rhs))
	for _, l := range rhs {
		shape = append(shape, size[l])
	}
	out := zeros(Float32, shape)
	value := map[rune]int{}
	offset := func(term string, t *Tensor) int {
		off, stride := 0, 1
		for i := len(term) - 1; i >= 0; i-- {
			off += value[rune(term[i])] * stride
			stride *= t.Shape[i]
		}
		return off
	}
	var walk func(i int)
	walk = func(i int) {
		if i == len(labels) {
			prod := float32(1)
			for k, term := range terms {
				prod *= in[k].F[offset(term, in[k])]
			}
			out.F[offset(rhs, out)] += prod
			return
		}
		for v := 0; v < size[labels[i]]; v++ {
			value[labels[i]] = v
			walk(i + 1)
		}
	}
	walk(0)
	return out
}

func TestEinsumMatchesBruteForce(t *testing.T) {
	rng := rand.New(rand.NewPCG(111, 112))
	sizes := map[rune]int{'a': 2, 'b': 3, 'i': 4, 'j': 5, 'k': 3, 'q': 2}
	equations := []string{
		"ij,jk->ik", "ij,jk->ki", "bij,bjk->bik", "bij,bjk->bki", "bij,bjk->ikb", "abij,abjk->abik",
		"bcd,bqd->bcq", "bld,bqd->bql", "bi,bi->bi", "bi,bj->bij", "i,j->ij", "ij,kj->ik", "i,i->",
		"ij,jk->i", "ij,jk->k", "ij,ik->kj", "ii,ij->j", "ij,j->i", "ab,ab->ba",
	}
	for _, eq := range equations {
		lhs, rhs, _ := strings.Cut(eq, "->")
		terms := strings.Split(lhs, ",")
		in := make([]*Tensor, len(terms))
		size := map[rune]int{}
		for k, term := range terms {
			shape := make([]int, len(term))
			for i, l := range term {
				shape[i] = sizes[l]
				size[l] = sizes[l]
			}
			in[k] = NewFloat(shape, randomSlice(rng, numel(shape)))
		}
		for _, l := range rhs {
			size[l] = sizes[l]
		}
		want := bruteEinsum(terms, rhs, in, size)
		got := run1(t, "Einsum", map[string]any{"equation": eq}, in...)
		wantShape(t, got, want.Shape)
		for i := range want.F {
			if math.Abs(float64(got.F[i]-want.F[i])) > 1e-4*(1+math.Abs(float64(want.F[i]))) {
				t.Fatalf("%s element %d: got %v, want %v", eq, i, got.F[i], want.F[i])
			}
		}
	}
}

func TestEinsumWithEmptyDimension(t *testing.T) {
	got := run1(t, "Einsum", map[string]any{"equation": "ij,jk->ik"}, fT([]int{2, 0}), fT([]int{0, 3}))
	wantF(t, got, []int{2, 3}, 0, 0, 0, 0, 0, 0)
	empty := run1(t, "Einsum", map[string]any{"equation": "ij,jk->ik"}, fT([]int{0, 2}), fT([]int{2, 3}, 1, 2, 3, 4, 5, 6))
	wantShape(t, empty, []int{0, 3})
}
