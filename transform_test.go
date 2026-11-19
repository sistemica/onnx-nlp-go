package onnxgo

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func TestCosineSearchAgainstATrainingSet(t *testing.T) {
	train := [][]float32{{1, 0}, {0, 1}, {1, 1}, {-1, 0}}
	idx, dist := cosineSearch([][]float32{{2, 0.1}, {0, 0}}, train, 3, false)
	if idx[0][0] != 0 || idx[0][1] != 2 || len(idx[0]) != 3 {
		t.Fatalf("neighbours of a new point, nearest first: %v", idx[0])
	}
	if !slices.IsSorted(dist[0]) || dist[0][0] > 0.01 {
		t.Fatalf("distances %v", dist[0])
	}
	if dist[1][0] != 1 {
		t.Fatalf("a zero vector is at distance 1 from everything: %v", dist[1])
	}
	again, _ := cosineSearch([][]float32{{1, 0}}, train, 2, false)
	if again[0][0] != 0 {
		t.Fatalf("an identical row is its own nearest neighbour, without a self offset: %v", again[0])
	}
}

func TestBipartiteGraphHasNoSelfOffsetAndDropsDisconnectedNeighbours(t *testing.T) {
	idx := [][]int32{{0, 1, 2, 3}, {1, 0, 2, 3}}
	dist := [][]float64{{0, 0.1, 0.3, 2}, {0.05, 0.1, 0.2, 0.4}}
	edges := bipartiteGraph(idx, dist, 4)
	weight := map[[2]int32]float32{}
	for _, e := range edges {
		weight[[2]int32{e.head, e.tail}] = e.weight
	}
	if weight[[2]int32{0, 0}] != 1 {
		t.Fatalf("an identical training point has weight 1: %v", weight)
	}
	if _, kept := weight[[2]int32{0, 3}]; kept {
		t.Fatal("a neighbour at the disconnection distance must be dropped")
	}
	if len(edges) != 7 {
		t.Fatalf("%d edges, want 7", len(edges))
	}
	if weight[[2]int32{0, 1}] <= weight[[2]int32{0, 2}] {
		t.Fatalf("closer neighbours have larger weights: %v", weight)
	}
}

func TestAverageNeighbourPositions(t *testing.T) {
	embedding := [][]float64{{0, 0}, {4, 0}, {0, 8}}
	edges := []graphEdge{{0, 0, 0.5}, {0, 1, 0.5}, {1, 2, 1}, {1, 0, 0.25}, {2, 2, 0.6}, {2, 1, 1}}
	got := averageNeighbourPositions(edges, 4, embedding)
	if !slices.Equal(got[0], []float64{2, 0}) {
		t.Fatalf("a weighted average: %v", got[0])
	}
	if !slices.Equal(got[1], []float64{0, 8}) {
		t.Fatalf("a weight of 1 copies the training point: %v", got[1])
	}
	if !slices.Equal(got[2], []float64{4, 0}) {
		t.Fatalf("the exact match with the smaller index wins among several: %v", got[2])
	}
	if !slices.Equal(got[3], []float64{0, 0}) {
		t.Fatalf("a point without neighbours starts at the origin: %v", got[3])
	}
}

func TestUMAPTransformKeepsNewPointsNearTheirGroup(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	all, truth := directions(rng, 3, 90, 24, 0.25)
	var train, fresh [][]float32
	var trainTruth, freshTruth []int
	for i := range all {
		if i%3 == 0 {
			fresh, freshTruth = append(fresh, all[i]), append(freshTruth, truth[i])
		} else {
			train, trainTruth = append(train, all[i]), append(trainTruth, truth[i])
		}
	}
	u := DefaultUMAP()
	u.Epochs = 150
	model, err := u.FitModel(train)
	if err != nil {
		t.Fatal(err)
	}
	low, err := model.Transform(fresh)
	if err != nil {
		t.Fatal(err)
	}
	if len(low) != len(fresh) || len(low[0]) != 5 {
		t.Fatalf("shape %d x %d", len(low), len(low[0]))
	}
	centre := map[int][]float64{}
	count := map[int]int{}
	for i, p := range model.Embedding() {
		g := trainTruth[i]
		if centre[g] == nil {
			centre[g] = make([]float64, 5)
		}
		for d := range p {
			centre[g][d] += p[d]
		}
		count[g]++
	}
	for g := range centre {
		for d := range centre[g] {
			centre[g][d] /= float64(count[g])
		}
	}
	wrong := 0
	for i, p := range low {
		best, bestDist := -1, math.Inf(1)
		for g, c := range centre {
			if d := squaredDistance(p, c); d < bestDist {
				best, bestDist = g, d
			}
		}
		if best != freshTruth[i] {
			wrong++
		}
	}
	if wrong > 0 {
		t.Fatalf("%d of %d new points landed nearest to another group's centre", wrong, len(low))
	}
	again, _ := model.Transform(fresh)
	if !slices.EqualFunc(low, again, slices.Equal[[]float64]) {
		t.Fatal("transform must be repeatable")
	}
}

func TestUMAPTransformEpochsAndErrors(t *testing.T) {
	data, _ := directions(rand.New(rand.NewPCG(7, 8)), 2, 20, 8, 0.2)
	u := UMAP{Neighbors: 5, Components: 2, Spread: 1, NegativeSamples: 5, Seed: 1, Epochs: 3}
	model, err := u.FitModel(data)
	if err != nil {
		t.Fatal(err)
	}
	if out, err := model.Transform(data[:2]); err != nil || len(out) != 2 {
		t.Fatalf("with fewer than three training epochs a single epoch still runs: %v %v", out, err)
	}
	if out, err := model.Transform(nil); err != nil || out != nil {
		t.Fatalf("no new rows: %v %v", out, err)
	}
	if _, err := model.Transform([][]float32{{1, 2}}); err == nil {
		t.Fatal("rows of the wrong length must be rejected")
	}
	if _, err := u.FitModel(nil); err == nil {
		t.Fatal("FitModel must report bad input")
	}
	if _, err := u.Fit(nil); err == nil {
		t.Fatal("Fit must report bad input")
	}
	u.Epochs = 0
	model, _ = u.FitModel(data)
	if _, err := model.Transform(data[:3]); err != nil {
		t.Fatal(err)
	}
}

func TestTransformLayoutMovesOnlyNewPoints(t *testing.T) {
	training := [][]float64{{0, 0}, {1, 0}, {10, 10}}
	edges := []graphEdge{{0, 0, 1}, {0, 1, 1}}
	a, b := findAB(1, 0)
	l := newTransformLayout([][]float64{{5, 5}}, training, edges, 20, a, b, 5, &splitMix{1})
	l.run()
	moved := l.points()[0]
	if math.Hypot(moved[0], moved[1]) >= math.Hypot(5, 5) {
		t.Fatalf("the new point should be pulled toward its neighbours: %v", moved)
	}
	if !slices.Equal(l.target, flatten32(training)) {
		t.Fatal("training points must not move")
	}
}

func hdbscanBlobs(seed uint64) ([][]float64, [][]float64) {
	rng := rand.New(rand.NewPCG(seed, 3))
	train := blobs(rng, [][2]float64{{0, 0}, {20, 0}}, 40, 0.5)
	fresh := [][]float64{{0.1, 0.2}, {20.2, -0.1}, {10, 40}, {0, 0}}
	return train, fresh
}

func TestHDBSCANPredictPlacesNewPoints(t *testing.T) {
	train, fresh := hdbscanBlobs(1)
	model, err := HDBSCAN{MinClusterSize: 10}.FitModel(train)
	if err != nil {
		t.Fatal(err)
	}
	if labels, _, err := model.Predict(nil); err != nil || len(labels) != 0 {
		t.Fatalf("no points to predict: %v %v", labels, err)
	}
	labels, strengths, err := model.Predict(fresh)
	if err != nil {
		t.Fatal(err)
	}
	if labels[0] != model.Labels[0] || labels[1] != model.Labels[40] || labels[0] == labels[1] {
		t.Fatalf("points inside a blob join its cluster: %v (training labels %d and %d)", labels, model.Labels[0], model.Labels[40])
	}
	if labels[2] != -1 || strengths[2] != 0 {
		t.Fatalf("a far point is noise: %d with strength %v", labels[2], strengths[2])
	}
	for i := range strengths {
		if strengths[i] < 0 || strengths[i] > 1 {
			t.Fatalf("strength %v", strengths[i])
		}
	}
	if _, _, err := model.Predict([][]float64{{1, 2, 3}}); err == nil {
		t.Fatal("a point of another dimension must be rejected")
	}
}

func TestHDBSCANPredictWithoutClustersAndWithCoincidentPoints(t *testing.T) {
	model, err := HDBSCAN{MinClusterSize: 5}.FitModel([][]float64{{0}, {1}, {5}})
	if err != nil {
		t.Fatal(err)
	}
	labels, strengths, err := model.Predict([][]float64{{0.5}})
	if err != nil || labels[0] != -1 || strengths[0] != 0 {
		t.Fatalf("a model without clusters predicts noise: %v %v %v", labels, strengths, err)
	}
	same := make([][]float64, 30)
	for i := range same {
		same[i] = []float64{1, 1}
	}
	rng := rand.New(rand.NewPCG(2, 2))
	pts := append(same, blobs(rng, [][2]float64{{30, 30}}, 30, 0.5)...)
	model, _ = HDBSCAN{MinClusterSize: 5}.FitModel(pts)
	labels, strengths, _ = model.Predict([][]float64{{1, 1}})
	if labels[0] < 0 || math.IsNaN(strengths[0]) || strengths[0] < 0 || strengths[0] > 1 {
		t.Fatalf("a point on top of a duplicated cluster: %v %v", labels, strengths)
	}
}

func TestHDBSCANPredictWalksUpToTheEnclosingCluster(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 9))
	pts := blobs(rng, [][2]float64{{0, 0}, {3, 0}}, 80, 1)
	pts = append(pts, blobs(rng, [][2]float64{{60, 60}}, 40, 1)...)
	model, _ := HDBSCAN{MinClusterSize: 10}.FitModel(pts)
	between := [][]float64{{1.5, 8}, {1.5, 0.2}, {-30, -30}}
	labels, strengths, err := model.Predict(between)
	if err != nil {
		t.Fatal(err)
	}
	if labels[1] < 0 || strengths[2] >= strengths[1] || strengths[2] > 0.1 {
		t.Fatalf("a point inside the merged blobs joins them firmly; a distant one at most joins weakly: %v %v", labels, strengths)
	}
}

func TestTopicModelTransform(t *testing.T) {
	docs, _ := subjectDocuments()
	embedder := &stubEmbedder{}
	b := NewBERTopic(embedder)
	b.UMAP.Epochs = 150
	model, err := b.Fit(docs)
	if err != nil {
		t.Fatal(err)
	}
	fresh := []string{"astronomy telescope and the galaxy", "cooking recipe and the oven", "football goal and the league"}
	topics, strengths, err := model.Transform(fresh)
	if err != nil {
		t.Fatal(err)
	}
	for i, doc := range fresh {
		first := model.Assignments[slices.IndexFunc(docs, func(d string) bool { return d[:6] == doc[:6] })]
		if topics[i] != first || strengths[i] <= 0 {
			t.Errorf("%q: topic %d strength %v, same-subject training documents are in topic %d", doc, topics[i], strengths[i], first)
		}
	}
	vectors := stubVectors(fresh)
	byVector, _, err := model.TransformEmbeddings(vectors)
	if err != nil || !slices.Equal(byVector, topics) {
		t.Fatalf("transforming the embeddings gives the same topics: %v %v %v", byVector, topics, err)
	}
	if _, _, err := model.TransformEmbeddings([][]float32{{1, 2}}); err == nil {
		t.Fatal("embeddings of the wrong length must be rejected")
	}
	embedder.err = errors.New("no model")
	if _, _, err := model.Transform(fresh); err == nil {
		t.Fatal("embedder failures must surface")
	}
	broken := *model
	broken.clusters = &HDBSCANModel{points: [][]float64{{1}}, labelOf: map[int]int{0: 0}}
	if _, _, err := broken.TransformEmbeddings(vectors); err == nil {
		t.Fatal("clusters of another dimension than the reducer's output must be reported")
	}
	fitted, err := NewBERTopic(nil).FitEmbeddings(docs, stubVectors(docs))
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := fitted.Transform(fresh); err == nil {
		t.Fatal("a model fitted from embeddings has no embedder")
	}
}
