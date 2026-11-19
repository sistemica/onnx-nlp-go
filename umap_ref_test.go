package onnxgo

import (
	"math"
	"path/filepath"
	"sort"
	"testing"
	"time"
)

func readEdges(t *testing.T, dir, edgesFile, weightsFile string) []graphEdge {
	t.Helper()
	pairs, shape := readNpyAny(t, filepath.Join(dir, edgesFile))
	weights, _ := readNpyAny(t, filepath.Join(dir, weightsFile))
	edges := make([]graphEdge, shape[0])
	for i := range edges {
		edges[i] = graphEdge{int32(pairs[2*i]), int32(pairs[2*i+1]), float32(weights[i])}
	}
	return edges
}

func readEmbeddings(t *testing.T, dir string) [][]float32 {
	t.Helper()
	flat, shape := readNpy(t, filepath.Join(dir, "emb.npy"))
	rows := make([][]float32, shape[0])
	for i := range rows {
		rows[i] = flat[i*shape[1] : (i+1)*shape[1]]
	}
	return rows
}

func TestUMAPGraphAgainstReference(t *testing.T) {
	dir := bertopicRef(t)
	data := readEmbeddings(t, dir)
	want := readEdges(t, dir, "pruned_edges.npy", "pruned_weights.npy")
	idx, dist := cosineNeighbors(data, 15)
	got := pruneGraph(fuzzyGraph(idx, dist, 15), 500)
	wantWeight := map[[2]int32]float32{}
	for _, e := range want {
		wantWeight[[2]int32{e.head, e.tail}] = e.weight
	}
	shared, worst := 0, 0.0
	for _, e := range got {
		if w, ok := wantWeight[[2]int32{e.head, e.tail}]; ok {
			shared++
			worst = max(worst, math.Abs(float64(e.weight-w)))
		}
	}
	jaccard := float64(shared) / float64(len(got)+len(want)-shared)
	t.Logf("edges: %d computed, %d reference, %d shared (Jaccard %.5f); largest weight difference on shared edges %.3g", len(got), len(want), shared, jaccard, worst)
	if shared != len(want) || len(got) != len(want) || worst > 1e-4 {
		t.Fatal("neighbour graph differs from umap-learn's")
	}
}

func TestUMAPSpectralAgainstReference(t *testing.T) {
	dir := bertopicRef(t)
	edges := readEdges(t, dir, "pruned_edges.npy", "pruned_weights.npy")
	wantValues, _ := readNpyAny(t, filepath.Join(dir, "spectral_values.npy"))
	wantFlat, shape := readNpyAny(t, filepath.Join(dir, "spectral_vectors.npy"))
	n := shape[0]
	op := newNormalizedAdjacency(edges, n)
	theta, vectors := largestEigenpairs(n, 6, op.apply, 1e-9)
	for i := 0; i < 6; i++ {
		if d := math.Abs((1 - theta[i]) - wantValues[i]); d > 1e-5 {
			t.Errorf("eigenvalue %d: got %.8f, want %.8f", i, 1-theta[i], wantValues[i])
		}
	}
	for c := 0; c < shape[1]; c++ {
		var inner, normWant float64
		for i := 0; i < n; i++ {
			inner += vectors[c+1][i] * wantFlat[i*shape[1]+c]
			normWant += wantFlat[i*shape[1]+c] * wantFlat[i*shape[1]+c]
		}
		cos := math.Abs(inner) / math.Sqrt(normWant)
		t.Logf("eigenvector %d: |cos| with scipy's = %.9f", c+1, cos)
		if cos < 0.9999 {
			t.Errorf("eigenvector %d differs from scipy's: |cos| = %.6f", c+1, cos)
		}
	}
}

func TestUMAPLayoutAgainstReference(t *testing.T) {
	dir := bertopicRef(t)
	edges := readEdges(t, dir, "sgd_edges.npy", "sgd_weights.npy")
	initFlat, shape := readNpy(t, filepath.Join(dir, "sgd_init.npy"))
	wantFlat, _ := readNpy(t, filepath.Join(dir, "sgd_out.npy"))
	init := make([][]float64, shape[0])
	for i := range init {
		init[i] = make([]float64, shape[1])
		for d := range init[i] {
			init[i][d] = float64(initFlat[i*shape[1]+d])
		}
	}
	a, b := findAB(1, 0)
	l := newLayout(init, edges, 30, a, b, 5, &splitMix{7})
	l.gamma = 0
	l.epochs = 2
	l.run()
	var worst, sumSq, scale float64
	for i, v := range l.coords {
		d := float64(v) - float64(wantFlat[i])
		worst = max(worst, math.Abs(d))
		sumSq += d * d
		scale = max(scale, math.Abs(float64(wantFlat[i])))
	}
	t.Logf("%d coordinates after 2 attraction-only epochs: largest difference %.3g, rms %.3g (coordinates up to %.1f)", len(l.coords), worst, math.Sqrt(sumSq/float64(len(l.coords))), scale)
	if worst > 1e-4 {
		t.Fatalf("layout differs from umap-learn's optimiser by %g", worst)
	}
}

func TestFindABAgainstReference(t *testing.T) {
	a, b := findAB(1, 0)
	t.Logf("a=%.10f b=%.10f", a, b)
	if math.Abs(a-1.93280839734315) > 1e-5 || math.Abs(b-0.7904949732233831) > 1e-5 {
		t.Fatalf("a, b = %v, %v", a, b)
	}
}

// neighbourRecall is the mean share of each sampled point's original k nearest neighbours that
// are also among its k nearest neighbours in the embedding.
func neighbourRecall(data [][]float32, low [][]float64, k, sample int) float64 {
	idx, _ := cosineNeighbors(data, k+1)
	var total float64
	step := len(data) / sample
	count := 0
	for i := 0; i < len(data); i += step {
		type pair struct {
			d float64
			j int
		}
		var near []pair
		for j := range low {
			if j != i {
				near = append(near, pair{math.Sqrt(squaredDistance(low[i], low[j])), j})
			}
		}
		sort.Slice(near, func(a, b int) bool { return near[a].d < near[b].d })
		inLow := map[int]bool{}
		for _, p := range near[:k] {
			inLow[p.j] = true
		}
		hits := 0
		for _, j := range idx[i][1:] {
			if inLow[int(j)] {
				hits++
			}
		}
		total += float64(hits) / float64(k)
		count++
	}
	return total / float64(count)
}

func clusterCount(labels []int) int {
	seen := map[int]bool{}
	for _, l := range labels {
		if l >= 0 {
			seen[l] = true
		}
	}
	return len(seen)
}

func TestUMAPEndToEndAgainstReference(t *testing.T) {
	dir := bertopicRef(t)
	data := readEmbeddings(t, dir)
	ref := readPoints(t, filepath.Join(dir, "umap5.npy"))
	wantLabels := readNpyInts(t, filepath.Join(dir, "hdb_exact_labels.npy"))
	theirs := neighbourRecall(data, ref, 15, 300)
	var totalRand float64
	var clustered, collapsed int
	for seed := uint64(1); seed <= 8; seed++ {
		u := DefaultUMAP()
		u.Seed = seed
		start := time.Now()
		low, err := u.Fit(data)
		if err != nil {
			t.Fatal(err)
		}
		took := time.Since(start).Round(10 * time.Millisecond)
		recall := neighbourRecall(data, low, 15, 300)
		if recall < theirs-0.05 {
			t.Fatalf("seed %d: neighbour preservation %.3f is clearly below umap-learn's %.3f", seed, recall, theirs)
		}
		labels, _, err := HDBSCAN{MinClusterSize: 10}.Fit(low)
		if err != nil {
			t.Fatal(err)
		}
		if clusterCount(labels) <= 3 {
			collapsed++
			t.Logf("seed %d (%v): neighbours kept %.3f; clustering collapsed to %d clusters, %d noise", seed, took, recall, clusterCount(labels), countValue(labels, -1))
			continue
		}
		clustered++
		rand := adjustedRand(labels, wantLabels)
		totalRand += rand
		t.Logf("seed %d (%v): neighbours kept %.3f (umap-learn %.3f); %d clusters, %d noise, adjusted Rand vs umap-learn %.3f",
			seed, took, recall, theirs, clusterCount(labels), countValue(labels, -1), rand)
	}
	t.Logf("%d of 8 seeds collapsed to two clusters (umap-learn: 19 of 40 on a fixed init, same corpus); mean adjusted Rand of the others %.3f (umap-learn against itself across seeds: 0.65-0.70)",
		collapsed, totalRand/float64(max(clustered, 1)))
	if clustered == 0 || totalRand/float64(clustered) < 0.55 {
		t.Fatalf("clustered runs agree with umap-learn only at adjusted Rand %.3f", totalRand/float64(max(clustered, 1)))
	}
}
