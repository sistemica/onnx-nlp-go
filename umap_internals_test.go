package onnxnlp

import (
	"math"
	"slices"
	"testing"
)

func TestTridiagonalEigenSolvesSmallMatrices(t *testing.T) {
	values, vectors := tridiagonalEigen([]float64{2, 2}, []float64{1})
	if math.Abs(values[0]-3) > 1e-12 || math.Abs(values[1]-1) > 1e-12 {
		t.Fatalf("eigenvalues of [[2 1] [1 2]]: %v", values)
	}
	if math.Abs(math.Abs(vectors[0][0])-1/math.Sqrt2) > 1e-12 || math.Abs(vectors[0][0]-vectors[0][1]) > 1e-12 {
		t.Fatalf("eigenvector of 3: %v", vectors[0])
	}
	diag := []float64{4, 1, 3, 2}
	off := []float64{0.5, 0.25, 1}
	values, vectors = tridiagonalEigen(diag, off)
	for i := range values {
		residual := 0.0
		for r := range diag {
			row := diag[r] * vectors[i][r]
			if r > 0 {
				row += off[r-1] * vectors[i][r-1]
			}
			if r < len(off) {
				row += off[r] * vectors[i][r+1]
			}
			residual = math.Max(residual, math.Abs(row-values[i]*vectors[i][r]))
		}
		if residual > 1e-12 {
			t.Errorf("eigenpair %d has residual %g", i, residual)
		}
	}
	if !slices.IsSortedFunc(values, func(a, b float64) int { return -cmpFloat(a, b) }) {
		t.Fatalf("eigenvalues must be in descending order: %v", values)
	}
	one, _ := tridiagonalEigen([]float64{5}, nil)
	if one[0] != 5 {
		t.Fatalf("a 1x1 matrix: %v", one)
	}
	deflated, _ := tridiagonalEigen([]float64{1, 2, 3}, []float64{0, 0})
	if !slices.Equal(deflated, []float64{3, 2, 1}) {
		t.Fatalf("a diagonal matrix keeps its entries: %v", deflated)
	}
}

func cmpFloat(a, b float64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

func pathGraph(n int) []graphEdge {
	var edges []graphEdge
	for i := 0; i < n; i++ {
		if i > 0 {
			edges = append(edges, graphEdge{int32(i), int32(i - 1), 1})
		}
		if i < n-1 {
			edges = append(edges, graphEdge{int32(i), int32(i + 1), 1})
		}
	}
	return edges
}

func TestLargestEigenpairsFindsAntisymmetricModes(t *testing.T) {
	const n = 30
	op := newNormalizedAdjacency(pathGraph(n), n)
	values, _ := largestEigenpairs(n, 3, op.apply, 1e-10)
	for k := range values {
		if want := math.Cos(math.Pi * float64(k) / (n - 1)); math.Abs(values[k]-want) > 1e-9 {
			t.Errorf("eigenvalue %d: %.10f, want %.10f; a symmetric start vector skips the antisymmetric mode k=1", k, values[k], want)
		}
	}
}

func TestLargestEigenpairsOfAPathGraph(t *testing.T) {
	const n = 40
	op := newNormalizedAdjacency(pathGraph(n), n)
	values, vectors := largestEigenpairs(n, 4, op.apply, 1e-10)
	for k := range values {
		if want := math.Cos(math.Pi * float64(k) / (n - 1)); math.Abs(values[k]-want) > 1e-9 {
			t.Errorf("eigenvalue %d: %.10f, want %.10f", k, values[k], want)
		}
	}
	scratch := make([]float64, n)
	for i, v := range vectors {
		op.apply(scratch, v)
		for k := range scratch {
			if math.Abs(scratch[k]-values[i]*v[k]) > 1e-8 {
				t.Fatalf("eigenvector %d does not satisfy A v = lambda v at %d", i, k)
			}
		}
	}
}

func TestLargestEigenpairsRestartsWhenTheStartVectorIsAnEigenvector(t *testing.T) {
	const n = 12
	var edges []graphEdge
	for i := 0; i < n; i++ {
		for _, j := range []int{(i + n - 1) % n, (i + 1) % n} {
			edges = append(edges, graphEdge{int32(i), int32(j), 1})
		}
	}
	slices.SortFunc(edges, func(a, b graphEdge) int {
		if a.head != b.head {
			return int(a.head - b.head)
		}
		return int(a.tail - b.tail)
	})
	op := newNormalizedAdjacency(edges, n)
	values, _ := largestEigenpairs(n, 3, op.apply, 1e-10)
	if len(values) != 3 || math.Abs(values[0]-1) > 1e-9 || math.Abs(values[1]-math.Cos(2*math.Pi/n)) > 1e-9 {
		t.Fatalf("a regular graph makes the uniform start vector an eigenvector; the solver must continue: %v", values)
	}
}

func TestLargestEigenpairsStopsWhenTheKrylovSpaceIsExhausted(t *testing.T) {
	diagonal := []float64{5, 3, 1}
	values, _ := largestEigenpairs(3, 2, func(dst, x []float64) {
		for i := range dst {
			dst[i] = diagonal[i] * x[i]
		}
	}, 1e-12)
	if math.Abs(values[0]-5) > 1e-9 || math.Abs(values[1]-3) > 1e-9 {
		t.Fatalf("eigenvalues %v", values)
	}
	same := func(dst, x []float64) { copy(dst, x) }
	values, vectors := largestEigenpairs(50, 2, same, 1e-12)
	if len(values) != 2 || math.Abs(values[0]-1) > 1e-12 || math.Abs(values[1]-1) > 1e-12 {
		t.Fatalf("the identity has 50 equal eigenvalues, found by restarting: %v", values)
	}
	if math.Abs(dot(vectors[0], vectors[1])) > 1e-9 {
		t.Fatal("eigenvectors of a repeated eigenvalue must still be orthogonal")
	}
}

func TestSpectralInitOrdersPathPointsAlongTheFirstCoordinate(t *testing.T) {
	const n = 30
	got := spectralInit(pathGraph(n), n, 2, &splitMix{1})
	if len(got) != n || len(got[0]) != 2 {
		t.Fatalf("shape %d x %d", len(got), len(got[0]))
	}
	// The eigenvectors of the symmetric normalised Laplacian are D^1/2 times the monotone
	// eigenvectors of the random walk; dividing by sqrt(degree) recovers the ordering.
	along := make([]float64, n)
	for i := range along {
		degree := 2.0
		if i == 0 || i == n-1 {
			degree = 1
		}
		along[i] = got[i][0] / math.Sqrt(degree)
	}
	direction := math.Copysign(1, along[n-1]-along[0])
	for i := 1; i < n; i++ {
		if direction*(along[i]-along[i-1]) <= 0 {
			t.Fatalf("the degree-scaled first coordinate must change monotonically along a path; it does not at %d", i)
		}
	}
}

func TestSpectralInitFallsBackToRandomPositions(t *testing.T) {
	two := []graphEdge{{0, 1, 1}, {1, 0, 1}, {2, 3, 1}, {3, 2, 1}}
	for name, edges := range map[string][]graphEdge{"disconnected": two, "fewer points than dimensions": two[:2]} {
		n := 4
		if name != "disconnected" {
			n = 2
		}
		got := spectralInit(edges, n, 3, &splitMix{5})
		for _, p := range got {
			for _, v := range p {
				if v < -10 || v > 10 {
					t.Fatalf("%s: coordinate %v is outside [-10, 10]", name, v)
				}
			}
		}
	}
	if !connected(pathGraph(5), 5) || connected([]graphEdge{{0, 1, 1}, {1, 0, 1}}, 3) {
		t.Fatal("connected misjudges small graphs")
	}
}

func TestLayoutPullsConnectedPointsTogetherAndPushesOthersApart(t *testing.T) {
	init := [][]float64{{0, 0}, {4, 0}, {20, 0}}
	edges := []graphEdge{{0, 1, 1}, {1, 0, 1}}
	a, b := findAB(1, 0)
	attract := newLayout(init, edges, 10, a, b, 0, &splitMix{1})
	attract.run()
	moved := attract.points()
	if d := math.Hypot(moved[0][0]-moved[1][0], moved[0][1]-moved[1][1]); d >= 4 {
		t.Fatalf("connected points should move closer: distance %v", d)
	}
	if moved[2][0] != 20 {
		t.Fatalf("an unconnected point without negative sampling must not move: %v", moved[2])
	}
	towardOther := [][]float64{{0, 0}, {-1, 0}, {-3, 0}}
	pairEdges := []graphEdge{{0, 1, 1}}
	without := newLayout(towardOther, pairEdges, 20, a, b, 0, &splitMix{3})
	without.run()
	with := newLayout(towardOther, pairEdges, 20, a, b, 5, &splitMix{3})
	with.run()
	if with.points()[0][0] <= without.points()[0][0] {
		t.Fatalf("repulsion from the point at -3 must push the head right of where attraction alone leaves it: %v vs %v",
			with.points()[0][0], without.points()[0][0])
	}
}

func TestLayoutCoefficientsAreClippedAndCoincidentPointsAreSafe(t *testing.T) {
	if clip4(10) != 4 || clip4(-10) != -4 || clip4(1.5) != 1.5 {
		t.Fatal("clip4")
	}
	a, b := findAB(1, 0)
	l := newLayout([][]float64{{1, 1}, {1, 1}}, []graphEdge{{0, 1, 1}, {1, 0, 1}}, 5, a, b, 5, &splitMix{2})
	l.run()
	for _, p := range l.points() {
		for _, v := range p {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("coincident points produced %v", v)
			}
		}
	}
	if squaredDist32([]float32{1, 2}, []float32{4, 6}) != 25 {
		t.Fatal("squaredDist32")
	}
}

func TestSplitMixAndNoisyScale(t *testing.T) {
	rng := &splitMix{9}
	var sum, sumSq float64
	const draws = 20000
	for i := 0; i < draws; i++ {
		u := rng.uniform()
		if u < 0 || u >= 1 {
			t.Fatalf("uniform out of range: %v", u)
		}
		g := rng.normal()
		sum += g
		sumSq += g * g
	}
	if mean, variance := sum/draws, sumSq/draws; math.Abs(mean) > 0.05 || math.Abs(variance-1) > 0.05 {
		t.Fatalf("normal draws have mean %.3f variance %.3f", mean, variance)
	}
	scaled := noisyScale([][]float64{{1, -2}, {0.5, 4}}, &splitMix{1})
	if math.Abs(math.Abs(scaled[1][1])-10) > 1e-3 || math.Abs(scaled[0][1]+5) > 1e-3 {
		t.Fatalf("the largest magnitude becomes 10 and the rest scale with it: %v", scaled)
	}
}

func TestGivensRotationAndOrthogonalRestart(t *testing.T) {
	c, s, r := givens(3, 4)
	if math.Abs(c-0.8) > 1e-15 || math.Abs(s-0.6) > 1e-15 || math.Abs(r-5) > 1e-15 {
		t.Fatalf("givens(3, 4) = %v %v %v", c, s, r)
	}
	if c, s, r := givens(0, 0); c != 1 || s != 0 || r != 0 {
		t.Fatalf("two zeros need no rotation: %v %v %v", c, s, r)
	}
	basis := [][]float64{{1, 0, 0, 0}, {0, 1, 0, 0}}
	v := orthogonalVector(basis, &splitMix{3})
	if math.Abs(dot(v, v)-1) > 1e-12 || math.Abs(v[0]) > 1e-12 || math.Abs(v[1]) > 1e-12 {
		t.Fatalf("restart vector %v must be a unit vector orthogonal to the basis", v)
	}
}

func TestCoefficientTableMatchesTheFormulas(t *testing.T) {
	a, b, gamma := 1.9328, 0.7905, 1.0
	attraction := func(d float64) float64 { return -2 * a * b * math.Pow(d, b-1) / (a*math.Pow(d, b) + 1) }
	repulsion := func(d float64) float64 { return 2 * gamma * b / ((0.001 + d) * (a*math.Pow(d, b) + 1)) }
	rng := &splitMix{11}
	for name, f := range map[string]func(float64) float64{"attraction": attraction, "repulsion": repulsion} {
		table := newCoefficientTable(f)
		worst := 0.0
		for i := 0; i < 200000; i++ {
			d := math.Exp(rng.uniform()*40 - 25)
			worst = max(worst, math.Abs(table.at(d)-f(d))/math.Abs(f(d)))
		}
		if worst > 1e-5 {
			t.Errorf("%s: largest relative error %.3g", name, worst)
		}
		if table.at(math.Ldexp(1, tableMinExp-5)) != table.values[0] || table.at(math.Ldexp(1, tableMaxExp+5)) != table.values[len(table.values)-2] {
			t.Errorf("%s: arguments outside the grid must use its end values", name)
		}
	}
}
