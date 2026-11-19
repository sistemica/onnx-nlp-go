package onnxgo

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func TestFindABMatchesUMAPLearn(t *testing.T) {
	for _, tc := range []struct{ spread, minDist, a, b float64 }{
		{1, 0.1, 1.5769434602697652, 0.8950608778515733},
		{1, 0.5, 0.5830300203414425, 1.3341669924314914},
		{2, 0.25, 0.41627349667013186, 0.9218551324072873},
		{1, 1, 0.11497568327441922, 1.929237144081752},
	} {
		a, b := findAB(tc.spread, tc.minDist)
		if math.Abs(a-tc.a) > 1e-4*tc.a || math.Abs(b-tc.b) > 1e-4*tc.b {
			t.Errorf("spread %v min_dist %v: a, b = %.6f, %.6f; umap-learn %.6f, %.6f", tc.spread, tc.minDist, a, b, tc.a, tc.b)
		}
	}
}

func directions(rng *rand.Rand, groups, each, dim int, noise float64) ([][]float32, []int) {
	var data [][]float32
	var truth []int
	for g := 0; g < groups; g++ {
		center := make([]float64, dim)
		for d := range center {
			center[d] = rng.NormFloat64()
		}
		for i := 0; i < each; i++ {
			row := make([]float32, dim)
			for d := range row {
				row[d] = float32(center[d] + noise*rng.NormFloat64())
			}
			data = append(data, row)
			truth = append(truth, g)
		}
	}
	return data, truth
}

func TestUMAPSeparatesGroupsAndIsDeterministic(t *testing.T) {
	data, truth := directions(rand.New(rand.NewPCG(1, 2)), 3, 60, 24, 0.25)
	u := DefaultUMAP()
	u.Epochs = 120
	first, err := u.Fit(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != len(data) || len(first[0]) != 5 {
		t.Fatalf("got %d points of %d dimensions", len(first), len(first[0]))
	}
	again, _ := u.Fit(data)
	if !slices.EqualFunc(first, again, slices.Equal[[]float64]) {
		t.Fatal("the same seed must give the same embedding")
	}
	u.Seed = 7
	other, _ := u.Fit(data)
	if slices.EqualFunc(first, other, slices.Equal[[]float64]) {
		t.Fatal("a different seed should change the embedding")
	}
	labels, _, err := HDBSCAN{MinClusterSize: 10}.Fit(first)
	if err != nil {
		t.Fatal(err)
	}
	if r := adjustedRand(labels, truth); r < 0.95 {
		t.Fatalf("clusters recovered with adjusted Rand %.3f", r)
	}
}

func TestUMAPUsesMoreEpochsForLargeInputs(t *testing.T) {
	data, _ := directions(rand.New(rand.NewPCG(3, 4)), 2, 10, 8, 0.2)
	u := UMAP{Neighbors: 5, Components: 2, Spread: 1, NegativeSamples: 5, Seed: 1}
	if _, err := u.Fit(data); err != nil {
		t.Fatalf("default epoch count: %v", err)
	}
	u.Neighbors = 100
	if _, err := u.Fit(data); err != nil {
		t.Fatalf("more neighbours than points: %v", err)
	}
}

func TestUMAPRejectsBadInput(t *testing.T) {
	good := [][]float32{{1, 0}, {0, 1}, {1, 1}, {2, 1}}
	for name, u := range map[string]UMAP{
		"no components":     {Neighbors: 3, Components: 0, Spread: 1},
		"one neighbour":     {Neighbors: 1, Components: 2, Spread: 1},
		"zero spread":       {Neighbors: 3, Components: 2},
		"negative min dist": {Neighbors: 3, Components: 2, Spread: 1, MinDist: -1},
		"negative samples":  {Neighbors: 3, Components: 2, Spread: 1, NegativeSamples: -1},
	} {
		if _, err := u.Fit(good); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	u := DefaultUMAP()
	if _, err := u.Fit(good[:2]); err == nil {
		t.Error("two points must be rejected")
	}
	if _, err := u.Fit([][]float32{{1, 2}, {3}, {4, 5}}); err == nil {
		t.Error("ragged rows must be rejected")
	}
	if _, err := u.Fit([][]float32{{}, {}, {}}); err == nil {
		t.Error("empty rows must be rejected")
	}
}

func TestCosineNeighborsOrderDuplicatesAndZeroVectors(t *testing.T) {
	data := [][]float32{{1, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}, {-1, 0}}
	idx, dist := cosineNeighbors(data, 4)
	if !slices.Equal(idx[0], []int32{0, 1, 2, 3}) || dist[0][1] != 0 {
		t.Fatalf("a duplicate is at distance exactly 0: %v %v", idx[0], dist[0])
	}
	if idx[1][0] != 1 || idx[1][1] != 0 {
		t.Fatalf("self comes first even next to a duplicate: %v", idx[1])
	}
	if math.Abs(dist[0][2]-(1-1/math.Sqrt2)) > 1e-6 || math.Abs(dist[0][3]-1) > 1e-6 {
		t.Fatalf("distances %v", dist[0])
	}
	if dist[4][1] != 1 {
		t.Fatalf("a zero vector is at distance 1 from everything: %v", dist[4])
	}
	if got := dist[0]; !slices.IsSorted(got) {
		t.Fatalf("distances must be ascending: %v", got)
	}
	short, _ := cosineNeighbors(data[:2], 4)
	if len(short[0]) != 2 {
		t.Fatalf("a row cannot have more neighbours than other points plus itself: %v", short[0])
	}
}

func TestSmoothKNNAndFuzzyGraph(t *testing.T) {
	idx := [][]int32{{0, 1, 2}, {1, 0, 2}, {2, 1, 0}}
	dist := [][]float64{{0, 0.1, 0.4}, {0, 0.1, 0.3}, {0, 0.3, 0.4}}
	sigma, rho := smoothKNN(dist, 3)
	if rho[0] != 0.1 || rho[2] != 0.3 {
		t.Fatalf("rho is the distance to the nearest neighbour: %v", rho)
	}
	for i, s := range sigma {
		var members float64 = 1
		for _, d := range dist[i][2:] {
			members += math.Exp(-(d - rho[i]) / s)
		}
		if math.Abs(members-math.Log2(3)) > 1e-4 {
			t.Errorf("point %d: neighbourhood size %.5f, want log2(3) = %.5f", i, members, math.Log2(3))
		}
	}
	edges := fuzzyGraph(idx, dist, 3)
	weight := map[[2]int32]float32{}
	for _, e := range edges {
		if e.head == e.tail {
			t.Fatal("the graph must not contain self edges")
		}
		if e.weight <= 0 || e.weight > 1 {
			t.Fatalf("weight %v outside (0, 1]", e.weight)
		}
		weight[[2]int32{e.head, e.tail}] = e.weight
	}
	for key, w := range weight {
		if weight[[2]int32{key[1], key[0]}] != w {
			t.Fatalf("edge %v is not symmetric", key)
		}
	}
	if !slices.IsSortedFunc(edges, func(a, b graphEdge) int {
		if a.head != b.head {
			return int(a.head - b.head)
		}
		return int(a.tail - b.tail)
	}) {
		t.Fatal("edges must be sorted by head and tail")
	}
	if weight[[2]int32{0, 1}] != 1 {
		t.Fatalf("the nearest neighbour is fully connected: %v", weight[[2]int32{0, 1}])
	}
}

func TestSmoothKNNWithoutPositiveDistances(t *testing.T) {
	sigma, rho := smoothKNN([][]float64{{0, 0, 0}, {0, 0, 0}}, 3)
	if rho[0] != 0 || sigma[0] <= 0 {
		t.Fatalf("identical points: rho %v sigma %v", rho, sigma)
	}
	edges := fuzzyGraph([][]int32{{0, 1, 2}, {1, 0, 2}}, [][]float64{{0, 0, 0}, {0, 0, 0}}, 3)
	for _, e := range edges {
		if e.weight != 1 {
			t.Fatalf("zero-distance neighbours are fully connected: %+v", e)
		}
	}
}

func TestPruneGraphDropsEdgesTooWeakToBeSampled(t *testing.T) {
	edges := []graphEdge{{0, 1, 1}, {1, 0, 0.001}, {1, 2, 0.5}}
	got := pruneGraph(edges, 100)
	if len(got) != 2 || got[0].weight != 1 || got[1].weight != 0.5 {
		t.Fatalf("pruned graph %v", got)
	}
	if len(pruneGraph(edges, 1000)) != 3 {
		t.Fatal("a weight of 0.001 is sampled once in 1000 epochs when the maximum is 1")
	}
}

func TestDefaultEpochsDependOnInputSize(t *testing.T) {
	if defaultEpochs(10000) != 500 || defaultEpochs(10001) != 200 {
		t.Fatalf("epochs: %d for 10000 points, %d for 10001", defaultEpochs(10000), defaultEpochs(10001))
	}
}

func TestSmoothKNNGrowsTheKernelWidthForDistantNeighbours(t *testing.T) {
	dist := [][]float64{{0, 5, 10, 20, 40}}
	sigma, rho := smoothKNN(dist, 5)
	members := 1.0
	for _, d := range dist[0][2:] {
		members += math.Exp(-(d - rho[0]) / sigma[0])
	}
	if sigma[0] < 2 || math.Abs(members-math.Log2(5)) > 1e-4 {
		t.Fatalf("sigma %v gives a neighbourhood of %.5f, want log2(5) = %.5f", sigma[0], members, math.Log2(5))
	}
}
