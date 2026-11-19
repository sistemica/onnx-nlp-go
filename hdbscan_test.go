package onnxnlp

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func blobs(rng *rand.Rand, centers [][2]float64, each int, spread float64) [][]float64 {
	var pts [][]float64
	for _, c := range centers {
		for i := 0; i < each; i++ {
			pts = append(pts, []float64{c[0] + rng.NormFloat64()*spread, c[1] + rng.NormFloat64()*spread})
		}
	}
	return pts
}

func TestHDBSCANFindsSeparatedBlobsAndNoise(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	pts := blobs(rng, [][2]float64{{0, 0}, {20, 0}, {0, 20}}, 40, 0.5)
	pts = append(pts, []float64{100, 100}, []float64{-80, 60})
	labels, strengths, err := HDBSCAN{MinClusterSize: 10}.Fit(pts)
	if err != nil {
		t.Fatal(err)
	}
	for b := 0; b < 3; b++ {
		first := labels[b*40]
		if first < 0 {
			t.Fatalf("blob %d was labelled noise", b)
		}
		for i := b * 40; i < (b+1)*40; i++ {
			if labels[i] != first {
				t.Fatalf("blob %d is split: point %d has label %d, want %d", b, i, labels[i], first)
			}
		}
	}
	if labels[0] == labels[40] || labels[40] == labels[80] || labels[0] == labels[80] {
		t.Fatalf("different blobs share a label: %v", []int{labels[0], labels[40], labels[80]})
	}
	if labels[120] != -1 || labels[121] != -1 || strengths[120] != 0 || strengths[121] != 0 {
		t.Fatalf("far points must be noise with strength 0: %v %v %v %v", labels[120], labels[121], strengths[120], strengths[121])
	}
	for i, s := range strengths[:120] {
		if s < 0 || s > 1 || math.IsNaN(s) {
			t.Fatalf("strength %d is %v", i, s)
		}
	}
}

func TestHDBSCANMinSamplesChangesDensityEstimate(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	pts := blobs(rng, [][2]float64{{0, 0}, {6, 0}}, 60, 1)
	defaults, _, _ := HDBSCAN{MinClusterSize: 10}.Fit(pts)
	explicit, _, _ := HDBSCAN{MinClusterSize: 10, MinSamples: 10}.Fit(pts)
	if !slices.Equal(defaults, explicit) {
		t.Fatal("MinSamples = 0 must behave like MinSamples = MinClusterSize")
	}
	smooth, _, err := HDBSCAN{MinClusterSize: 10, MinSamples: 40}.Fit(pts)
	if err != nil {
		t.Fatal(err)
	}
	if countValue(smooth, -1) < countValue(defaults, -1) {
		t.Fatalf("a larger MinSamples cannot label fewer points as noise: %d < %d", countValue(smooth, -1), countValue(defaults, -1))
	}
}

func countValue(s []int, v int) int {
	n := 0
	for _, x := range s {
		if x == v {
			n++
		}
	}
	return n
}

func TestHDBSCANHandlesDuplicatesAndTinyInputs(t *testing.T) {
	same := make([][]float64, 25)
	for i := range same {
		same[i] = []float64{1, 2}
	}
	pts := append(slices.Clone(same), blobs(rand.New(rand.NewPCG(5, 6)), [][2]float64{{50, 50}}, 25, 1)...)
	labels, strengths, err := HDBSCAN{MinClusterSize: 5}.Fit(pts)
	if err != nil {
		t.Fatal(err)
	}
	for i := range strengths {
		if math.IsNaN(strengths[i]) {
			t.Fatalf("strength %d is NaN", i)
		}
	}
	if labels[0] < 0 || labels[0] == labels[30] {
		t.Fatalf("duplicates form their own cluster: %v %v", labels[0], labels[30])
	}
	if strengths[0] != 1 {
		t.Fatalf("identical points leave the cluster at infinite density: strength %v", strengths[0])
	}
	for _, tiny := range [][][]float64{nil, {{1, 2}}} {
		labels, strengths, err := HDBSCAN{MinClusterSize: 2}.Fit(tiny)
		if err != nil || len(labels) != len(tiny) || len(strengths) != len(tiny) {
			t.Fatalf("%d points: %v %v %v", len(tiny), labels, strengths, err)
		}
	}
	labels, _, err = HDBSCAN{MinClusterSize: 5}.Fit([][]float64{{0}, {1}, {5}})
	if err != nil || !slices.Equal(labels, []int{-1, -1, -1}) {
		t.Fatalf("fewer points than the minimum cluster size are all noise: %v %v", labels, err)
	}
}

func TestHDBSCANRejectsBadInput(t *testing.T) {
	if _, _, err := (HDBSCAN{MinClusterSize: 1}).Fit([][]float64{{1}, {2}}); err == nil {
		t.Fatal("MinClusterSize 1 must be rejected")
	}
	if _, _, err := (HDBSCAN{MinClusterSize: 2}).Fit([][]float64{{1, 2}, {3}}); err == nil {
		t.Fatal("ragged points must be rejected")
	}
}

func TestSingleLinkageMergesInWeightOrder(t *testing.T) {
	points := [][]float64{{0}, {1}, {3}, {7}}
	rows := singleLinkage(4, minimumSpanningTree(points, []float64{0, 0, 0, 0}))
	want := []linkageRow{{0, 1, 1, 2}, {4, 2, 2, 3}, {5, 3, 4, 4}}
	for i, r := range rows {
		if r.dist != want[i].dist || r.size != want[i].size {
			t.Fatalf("row %d: %+v, want %+v", i, r, want[i])
		}
	}
	if order := bfsFromHierarchy(rows, 4, 6); len(order) != 7 || order[0] != 6 {
		t.Fatalf("breadth-first order %v", order)
	}
}

func TestCoreDistanceIsDistanceToKthOtherPoint(t *testing.T) {
	points := [][]float64{{0}, {1}, {3}, {7}}
	if got := coreDistances(points, 2); !slices.Equal(got, []float64{3, 2, 3, 6}) {
		t.Fatalf("core distances %v", got)
	}
	if got := coreDistances(points, 1); !slices.Equal(got, []float64{1, 1, 2, 4}) {
		t.Fatalf("nearest neighbour distances %v", got)
	}
}

func TestAdjustedRand(t *testing.T) {
	if r := adjustedRand([]int{0, 0, 1, 1, 2}, []int{5, 5, 7, 7, 9}); math.Abs(r-1) > 1e-12 {
		t.Fatalf("relabelling must not change the index: %v", r)
	}
	if r := adjustedRand([]int{0, 0, 1, 1}, []int{0, 1, 0, 1}); r > 0 {
		t.Fatalf("unrelated partitions score %v", r)
	}
}

func TestHDBSCANPrefersStableParentOverItsChildren(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 9))
	pts := blobs(rng, [][2]float64{{0, 0}, {3, 0}}, 80, 1)
	pts = append(pts, blobs(rng, [][2]float64{{60, 60}}, 40, 1)...)
	labels, _, err := HDBSCAN{MinClusterSize: 10}.Fit(pts)
	if err != nil {
		t.Fatal(err)
	}
	merged := map[int]bool{}
	for _, l := range labels[:160] {
		if l >= 0 {
			merged[l] = true
		}
	}
	if len(merged) != 1 {
		t.Fatalf("overlapping blobs should stay one cluster, found %d", len(merged))
	}
	if labels[160] < 0 || merged[labels[160]] {
		t.Fatalf("the distant blob is its own cluster: %d", labels[160])
	}
}
