package onnxgo

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func bertopicRef(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("ONNXGO_BERTOPIC_REF")
	if dir == "" {
		t.Skip("ONNXGO_BERTOPIC_REF not set")
	}
	return dir
}

func readNpyInts(t *testing.T, path string) []int {
	t.Helper()
	raw, shape := readNpyAny(t, path)
	if len(shape) != 1 {
		t.Fatalf("%s: shape %v", path, shape)
	}
	out := make([]int, len(raw))
	for i, v := range raw {
		out[i] = int(v)
	}
	return out
}

func readPoints(t *testing.T, path string) [][]float64 {
	t.Helper()
	flat, shape := readNpyAny(t, path)
	points := make([][]float64, shape[0])
	for i := range points {
		points[i] = flat[i*shape[1] : (i+1)*shape[1]]
	}
	return points
}

// adjustedRand measures how alike two labelings are as partitions; 1 means identical.
func adjustedRand(a, b []int) float64 {
	pairs := func(n int) float64 { return float64(n) * float64(n-1) / 2 }
	joint, rows, cols := map[[2]int]int{}, map[int]int{}, map[int]int{}
	for i := range a {
		joint[[2]int{a[i], b[i]}]++
		rows[a[i]]++
		cols[b[i]]++
	}
	var sumJoint, sumRows, sumCols float64
	for _, n := range joint {
		sumJoint += pairs(n)
	}
	for _, n := range rows {
		sumRows += pairs(n)
	}
	for _, n := range cols {
		sumCols += pairs(n)
	}
	expected := sumRows * sumCols / pairs(len(a))
	return (sumJoint - expected) / ((sumRows+sumCols)/2 - expected)
}

func TestHDBSCANSpanningTreeAgainstReference(t *testing.T) {
	dir := bertopicRef(t)
	points := readPoints(t, filepath.Join(dir, "umap5.npy"))
	linkage, _ := readNpyAny(t, filepath.Join(dir, "hdb_exact_linkage.npy"))
	rows := singleLinkage(len(points), minimumSpanningTree(points, coreDistances(points, 10)))
	worst := 0.0
	for i, r := range rows {
		worst = max(worst, math.Abs(r.dist-linkage[i*4+2]))
	}
	t.Logf("%d merge distances, largest difference from the library %.3g", len(rows), worst)
	if worst > 1e-12 {
		t.Fatalf("merge distances differ from the reference by %g", worst)
	}
}

func TestHDBSCANTreeAgainstReference(t *testing.T) {
	dir := bertopicRef(t)
	linkage, shape := readNpyAny(t, filepath.Join(dir, "hdb_exact_linkage.npy"))
	rows := make([]linkageRow, shape[0])
	for i := range rows {
		rows[i] = linkageRow{int(linkage[i*4]), int(linkage[i*4+1]), linkage[i*4+2], int(linkage[i*4+3])}
	}
	n := len(rows) + 1
	tree := condenseTree(rows, 10)
	labels, strengths := assignClusters(tree, n, selectClusters(tree, computeStability(tree)))
	wantLabels := readNpyInts(t, filepath.Join(dir, "hdb_exact_labels.npy"))
	wantStrengths, _ := readNpyAny(t, filepath.Join(dir, "hdb_exact_probs.npy"))
	wrong, worst := 0, 0.0
	for i := range labels {
		if labels[i] != wantLabels[i] {
			wrong++
		}
		worst = max(worst, math.Abs(strengths[i]-wantStrengths[i]))
	}
	t.Logf("%d points: %d labels differ, largest strength difference %.3g", n, wrong, worst)
	if wrong != 0 || worst > 1e-9 {
		t.Fatalf("the library's own dendrogram gives different labels (%d) or strengths (%g)", wrong, worst)
	}
}

func TestHDBSCANAgainstReference(t *testing.T) {
	dir := bertopicRef(t)
	points := readPoints(t, filepath.Join(dir, "umap5.npy"))
	wantLabels := readNpyInts(t, filepath.Join(dir, "hdb_exact_labels.npy"))
	labels, _, err := HDBSCAN{MinClusterSize: 10}.Fit(points)
	if err != nil {
		t.Fatal(err)
	}
	disagree := 0
	for i := range labels {
		if (labels[i] == -1) != (wantLabels[i] == -1) {
			disagree++
		}
	}
	rand := adjustedRand(labels, wantLabels)
	t.Logf("%d points: adjusted Rand index %.5f, %d noise disagreements", len(labels), rand, disagree)
	if rand < 0.98 {
		t.Fatalf("adjusted Rand index %.4f against the library", rand)
	}
}

func TestHDBSCANOnOtherEmbeddingsAgainstReference(t *testing.T) {
	dir := bertopicRef(t)
	for seed := 1; seed <= 4; seed++ {
		points := readPoints(t, filepath.Join(dir, fmt.Sprintf("umap5_seed%d.npy", seed)))
		want := readNpyInts(t, filepath.Join(dir, fmt.Sprintf("hdb_exact_labels_seed%d.npy", seed)))
		labels, _, err := HDBSCAN{MinClusterSize: 10}.Fit(points)
		if err != nil {
			t.Fatal(err)
		}
		clusters := map[int]bool{}
		for _, l := range labels {
			clusters[l] = true
		}
		wantClusters := map[int]bool{}
		for _, l := range want {
			wantClusters[l] = true
		}
		rand := adjustedRand(labels, want)
		t.Logf("seed %d: go %d labels values (%d noise), library %d (%d noise), adjusted Rand %.4f",
			seed, len(clusters), countValue(labels, -1), len(wantClusters), countValue(want, -1), rand)
		if rand < 0.95 {
			t.Errorf("seed %d: adjusted Rand %.4f", seed, rand)
		}
	}
}

func TestHDBSCANPredictAgainstReference(t *testing.T) {
	dir := bertopicRef(t)
	points := readPoints(t, filepath.Join(dir, "umap5.npy"))
	linkage, shape := readNpyAny(t, filepath.Join(dir, "pred_train_linkage.npy"))
	rows := make([]linkageRow, shape[0])
	for i := range rows {
		rows[i] = linkageRow{int(linkage[i*4]), int(linkage[i*4+1]), linkage[i*4+2], int(linkage[i*4+3])}
	}
	model := &HDBSCANModel{points: points, minSamples: 10}
	model.cluster(rows, 10)
	if want := readNpyInts(t, filepath.Join(dir, "pred_train_labels.npy")); adjustedRand(model.Labels, want) != 1 {
		t.Fatal("the model built from the library's dendrogram must reproduce its labels")
	}
	newPoints := readPoints(t, filepath.Join(dir, "new_umap.npy"))
	wantLabels := readNpyInts(t, filepath.Join(dir, "pred_labels.npy"))
	wantStrengths, _ := readNpyAny(t, filepath.Join(dir, "pred_probs.npy"))
	labels, strengths, err := model.Predict(newPoints)
	if err != nil {
		t.Fatal(err)
	}
	wrong, worst := 0, 0.0
	for i := range labels {
		if labels[i] != wantLabels[i] {
			wrong++
		}
		worst = max(worst, math.Abs(strengths[i]-wantStrengths[i]))
	}
	t.Logf("%d new points: %d labels differ from approximate_predict (%d noise in the library's answer), largest strength difference %.3g",
		len(labels), wrong, countValue(wantLabels, -1), worst)
	if wrong != 0 || worst > 1e-9 {
		t.Fatalf("prediction differs from the library: %d labels, strength %g", wrong, worst)
	}
}
