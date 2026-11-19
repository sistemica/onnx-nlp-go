package onnxnlp

import (
	"errors"
	"math"
	"sort"
)

// UMAP reduces embeddings with cosine distance to a low-dimensional Euclidean space, following
// the algorithm and defaults of umap-learn's classic (0.5.x compatible) layout.
type UMAP struct {
	Neighbors  int
	Components int
	MinDist    float64
	Spread     float64
	// Epochs of zero picks 500 for up to 10000 points and 200 above.
	Epochs          int
	NegativeSamples int
	Seed            uint64
}

// DefaultUMAP returns the settings BERTopic uses.
func DefaultUMAP() UMAP {
	return UMAP{Neighbors: 15, Components: 5, Spread: 1, NegativeSamples: 5, Seed: 42}
}

const (
	smoothKTolerance = 1e-5
	minKDistScale    = 1e-3
)

type graphEdge struct {
	head, tail int32
	weight     float32
}

// UMAPModel is a fitted UMAP: the training rows, their embedding, and what new points need.
type UMAPModel struct {
	params    UMAP
	data      [][]float32
	embedding [][]float64
	a, b      float64
	neighbors int
}

// Embedding returns the position of every training row.
func (m *UMAPModel) Embedding() [][]float64 { return m.embedding }

// Fit embeds the rows of data and returns one Components-dimensional point per row.
func (u UMAP) Fit(data [][]float32) ([][]float64, error) {
	m, err := u.FitModel(data)
	if err != nil {
		return nil, err
	}
	return m.embedding, nil
}

// FitModel embeds the rows of data and keeps what is needed to embed new rows later.
func (u UMAP) FitModel(data [][]float32) (*UMAPModel, error) {
	n := len(data)
	switch {
	case u.Components < 1 || u.Neighbors < 2 || u.Spread <= 0 || u.MinDist < 0 || u.NegativeSamples < 0:
		return nil, errors.New("umap: invalid parameters")
	case n < 3:
		return nil, errors.New("umap: at least 3 points are needed")
	}
	for _, row := range data {
		if len(row) != len(data[0]) || len(row) == 0 {
			return nil, errors.New("umap: rows differ in length")
		}
	}
	k := min(u.Neighbors, n)
	idx, dist := cosineNeighbors(data, k)
	graph := fuzzyGraph(idx, dist, k)
	epochs := u.Epochs
	if epochs <= 0 {
		epochs = defaultEpochs(n)
	}
	graph = pruneGraph(graph, epochs)
	rng := &splitMix{u.Seed}
	init := noisyScale(spectralInit(graph, n, u.Components, rng), rng)
	a, b := findAB(u.Spread, u.MinDist)
	layout := newLayout(init, graph, epochs, a, b, u.NegativeSamples, rng)
	layout.run()
	return &UMAPModel{params: u, data: data, embedding: layout.points(), a: a, b: b, neighbors: k}, nil
}

// Transform places new rows among the training embedding: each starts at the weighted average of
// its neighbours' positions and is then pulled and pushed for a third of the training epochs
// while the training points stay fixed.
func (m *UMAPModel) Transform(data [][]float32) ([][]float64, error) {
	for _, row := range data {
		if len(row) != len(m.data[0]) {
			return nil, errors.New("umap: new rows differ in length from the training rows")
		}
	}
	if len(data) == 0 {
		return nil, nil
	}
	idx, dist := cosineSearch(data, m.data, m.neighbors, false)
	graph := bipartiteGraph(idx, dist, m.neighbors)
	init := averageNeighbourPositions(graph, len(data), m.embedding)
	epochs := m.params.Epochs / 3
	if m.params.Epochs <= 0 {
		epochs = 30
		if len(data) <= 10000 {
			epochs = 100
		}
	}
	epochs = max(epochs, 1)
	layout := newTransformLayout(init, m.embedding, pruneGraph(graph, epochs), epochs, m.a, m.b, m.params.NegativeSamples, &splitMix{m.params.Seed})
	layout.run()
	return layout.points(), nil
}

// disconnectionDistance is the cosine distance beyond which a neighbour counts as unconnected.
const disconnectionDistance = 2

// bipartiteGraph connects every new point to its training neighbours. Unlike the training graph
// it has no self edges, no nearest-neighbour offset (rho is zero) and is not symmetrised.
func bipartiteGraph(idx [][]int32, dist [][]float64, k int) []graphEdge {
	sigma, rho := smoothKNN(dist, k, false)
	var edges []graphEdge
	for i := range idx {
		for j, nb := range idx[i] {
			if dist[i][j] >= disconnectionDistance {
				continue
			}
			v := 1.0
			if d := dist[i][j] - rho[i]; d > 0 && sigma[i] != 0 {
				v = math.Exp(-d / sigma[i])
			}
			if w := float32(v); w > 0 {
				edges = append(edges, graphEdge{int32(i), nb, w})
			}
		}
	}
	return edges
}

// averageNeighbourPositions starts each new point at the position of a training point it equals,
// or else at the average of its neighbours' positions weighted by the graph. A point without
// neighbours starts at the origin.
func averageNeighbourPositions(edges []graphEdge, count int, embedding [][]float64) [][]float64 {
	dim := len(embedding[0])
	out := make([][]float64, count)
	for i := range out {
		out[i] = make([]float64, dim)
	}
	exact := make([]int32, count)
	for i := range exact {
		exact[i] = -1
	}
	totals := make([]float64, count)
	for _, e := range edges {
		totals[e.head] += float64(e.weight)
		if e.weight == 1 && (exact[e.head] < 0 || e.tail < exact[e.head]) {
			exact[e.head] = e.tail
		}
	}
	for _, e := range edges {
		if exact[e.head] >= 0 {
			continue
		}
		for d := range out[e.head] {
			out[e.head][d] += float64(e.weight) / totals[e.head] * embedding[e.tail][d]
		}
	}
	for i, t := range exact {
		if t >= 0 {
			copy(out[i], embedding[t])
		}
	}
	return out
}

func defaultEpochs(points int) int {
	if points > 10000 {
		return 200
	}
	return 500
}

// findAB fits 1 / (1 + a x^(2b)) to the target membership curve by Levenberg-Marquardt.
func findAB(spread, minDist float64) (a, b float64) {
	const samples = 300
	xs, ys := make([]float64, samples), make([]float64, samples)
	for i := range xs {
		xs[i] = float64(i) * spread * 3 / (samples - 1)
		ys[i] = 1
		if xs[i] >= minDist {
			ys[i] = math.Exp(-(xs[i] - minDist) / spread)
		}
	}
	residuals := func(a, b float64) (cost float64, jtj [3]float64, jtr [2]float64) {
		for i, x := range xs {
			p := math.Pow(x, 2*b)
			f := 1 / (1 + a*p)
			r := f - ys[i]
			da := -p * f * f
			db := 0.0
			if x > 0 {
				db = -a * 2 * math.Log(x) * p * f * f
			}
			cost += r * r
			jtj[0] += da * da
			jtj[1] += da * db
			jtj[2] += db * db
			jtr[0] += da * r
			jtr[1] += db * r
		}
		return cost, jtj, jtr
	}
	a, b = 1, 1
	cost, jtj, jtr := residuals(a, b)
	damping := 1e-3
	for iter := 0; iter < 200; iter++ {
		m00, m11 := jtj[0]*(1+damping), jtj[2]*(1+damping)
		det := m00*m11 - jtj[1]*jtj[1]
		da := -(m11*jtr[0] - jtj[1]*jtr[1]) / det
		db := -(m00*jtr[1] - jtj[1]*jtr[0]) / det
		next, njtj, njtr := residuals(a+da, b+db)
		if next < cost {
			improved := cost - next
			a, b, cost, jtj, jtr = a+da, b+db, next, njtj, njtr
			damping = max(damping/3, 1e-12)
			if improved < 1e-15*max(cost, 1e-30) || math.Abs(da)+math.Abs(db) < 1e-13 {
				break
			}
		} else {
			damping *= 4
			if damping > 1e12 {
				break
			}
		}
	}
	return a, b
}

// cosineNeighbors returns for every row its k nearest rows by cosine distance, itself first.
func cosineNeighbors(data [][]float32, k int) ([][]int32, [][]float64) {
	return cosineSearch(data, data, k, true)
}

// cosineSearch finds for every query row the k nearest reference rows by cosine distance. With
// self set, the queries are the reference rows and each row comes first in its own list.
func cosineSearch(queries, reference [][]float32, k int, self bool) ([][]int32, [][]float64) {
	n, dim := len(reference), len(reference[0])
	unitT := make([]float32, n*dim)
	norms := make([]float64, n)
	unitRows := func(rows [][]float32) ([]float32, []float64) {
		unit, rowNorms := make([]float32, len(rows)*dim), make([]float64, len(rows))
		for i, row := range rows {
			var norm float64
			for _, v := range row {
				norm += float64(v) * float64(v)
			}
			rowNorms[i] = math.Sqrt(norm)
			scale := float32(0)
			if norm > 0 {
				scale = float32(1 / rowNorms[i])
			}
			for d, v := range row {
				unit[i*dim+d] = v * scale
			}
		}
		return unit, rowNorms
	}
	unit, queryNorms := unitRows(queries)
	if self {
		norms = queryNorms
		for i := 0; i < n; i++ {
			for d := 0; d < dim; d++ {
				unitT[d*n+i] = unit[i*dim+d]
			}
		}
	} else {
		refUnit, refNorms := unitRows(reference)
		norms = refNorms
		for i := 0; i < n; i++ {
			for d := 0; d < dim; d++ {
				unitT[d*n+i] = refUnit[i*dim+d]
			}
		}
	}
	idx, dist := make([][]int32, len(queries)), make([][]float64, len(queries))
	const block = 256
	sims := make([]float32, block*n)
	for lo := 0; lo < len(queries); lo += block {
		hi := min(lo+block, len(queries))
		clear(sims)
		runGemms([]gemmJob{{out: sims[:(hi-lo)*n], a: unit[lo*dim : hi*dim], b: unitT, m: hi - lo, k: dim, n: n}})
		parallelFor(hi-lo, n*k, func(from, to int) {
			for r := from; r < to; r++ {
				q := lo + r
				owner := -1
				if self {
					owner = q
				}
				idx[q], dist[q] = nearestByDistance(sims[r*n:(r+1)*n], owner, k)
				refineDistances(queries[q], queryNorms[q], reference, norms, idx[q], dist[q], owner >= 0)
			}
		})
	}
	return idx, dist
}

// refineDistances recomputes the distances of a neighbour row in float64, where identical vectors
// come out as exactly zero, and restores the ascending order. A row that starts with its own
// point keeps it first.
func refineDistances(query []float32, queryNorm float64, reference [][]float32, norms []float64, idx []int32, dist []float64, selfFirst bool) {
	first := 0
	if selfFirst {
		first = 1
	}
	for t := first; t < len(idx); t++ {
		other := int(idx[t])
		var dotProduct float64
		for d, v := range query {
			dotProduct += float64(v) * float64(reference[other][d])
		}
		dist[t] = 1
		if queryNorm > 0 && norms[other] > 0 {
			dist[t] = max(0, 1-dotProduct/(queryNorm*norms[other]))
		}
		if dist[t] < 1e-9 {
			dist[t] = 0
		}
	}
	order := make([]int, len(idx)-first)
	for i := range order {
		order[i] = i + first
	}
	sort.SliceStable(order, func(a, b int) bool { return dist[order[a]] < dist[order[b]] })
	sortedIdx, sortedDist := make([]int32, len(order)), make([]float64, len(order))
	for i, o := range order {
		sortedIdx[i], sortedDist[i] = idx[o], dist[o]
	}
	copy(idx[first:], sortedIdx)
	copy(dist[first:], sortedDist)
}

// nearestByDistance picks the k most similar entries. A row that belongs to the reference
// (self >= 0) is excluded from its own candidates and put first.
func nearestByDistance(sims []float32, self, k int) ([]int32, []float64) {
	idx, dist := make([]int32, k), make([]float64, k)
	first, filled := 0, 0
	if self >= 0 {
		idx[0] = int32(self)
		first, filled = 1, 1
	}
	for j, s := range sims {
		if j == self {
			continue
		}
		d := max(0, 1-float64(s))
		if filled == k && d >= dist[k-1] {
			continue
		}
		at := min(filled, k-1)
		for at > first && dist[at-1] > d {
			idx[at], dist[at] = idx[at-1], dist[at-1]
			at--
		}
		idx[at], dist[at] = int32(j), d
		filled = min(filled+1, k)
	}
	return idx[:filled], dist[:filled]
}

// smoothKNN finds each point's distance to its nearest neighbour (rho) and the kernel width
// (sigma) at which the fuzzy neighbourhood has log2(k) members.
func smoothKNN(dist [][]float64, k int, offsetByNearest bool) (sigma, rho []float64) {
	target := math.Log2(float64(k))
	sigma, rho = make([]float64, len(dist)), make([]float64, len(dist))
	var total float64
	var count int
	for _, row := range dist {
		for _, d := range row {
			total += d
			count++
		}
	}
	meanAll := total / float64(count)
	for i, row := range dist {
		for _, d := range row {
			if d > 0 && offsetByNearest {
				rho[i] = d
				break
			}
		}
		lo, hi, mid := 0.0, math.MaxFloat32, 1.0
		for iter := 0; iter < 64; iter++ {
			var psum float64
			for _, d := range row[1:] {
				if d -= rho[i]; d > 0 {
					psum += math.Exp(-d / mid)
				} else {
					psum++
				}
			}
			if math.Abs(psum-target) < smoothKTolerance {
				break
			}
			if psum > target {
				hi = mid
				mid = (lo + hi) / 2
			} else {
				lo = mid
				if hi >= math.MaxFloat32 {
					mid *= 2
				} else {
					mid = (lo + hi) / 2
				}
			}
		}
		sigma[i] = mid
		mean := meanAll
		if rho[i] > 0 {
			mean = 0
			for _, d := range row {
				mean += d
			}
			mean /= float64(len(row))
		}
		sigma[i] = max(sigma[i], minKDistScale*mean)
	}
	return sigma, rho
}

// fuzzyGraph builds the symmetric weighted neighbour graph: a + aᵀ - a∘aᵀ over the directed
// membership strengths. Edges are sorted by head and then tail.
func fuzzyGraph(idx [][]int32, dist [][]float64, k int) []graphEdge {
	sigma, rho := smoothKNN(dist, k, true)
	directed := map[[2]int32]float64{}
	for i := range idx {
		for j, nb := range idx[i] {
			if int(nb) == i {
				continue
			}
			v := 1.0
			if d := dist[i][j] - rho[i]; d > 0 && sigma[i] != 0 {
				v = math.Exp(-d / sigma[i])
			}
			if v > 0 {
				directed[[2]int32{int32(i), nb}] = v
			}
		}
	}
	merged := make(map[[2]int32]float32, 2*len(directed))
	for key, a := range directed {
		b := directed[[2]int32{key[1], key[0]}]
		w := float32(a + b - a*b)
		merged[key], merged[[2]int32{key[1], key[0]}] = w, w
	}
	edges := make([]graphEdge, 0, len(merged))
	for key, w := range merged {
		if w > 0 {
			edges = append(edges, graphEdge{key[0], key[1], w})
		}
	}
	sort.Slice(edges, func(a, b int) bool {
		if edges[a].head != edges[b].head {
			return edges[a].head < edges[b].head
		}
		return edges[a].tail < edges[b].tail
	})
	return edges
}

// pruneGraph drops edges too weak to be sampled even once in the given number of epochs.
func pruneGraph(edges []graphEdge, epochs int) []graphEdge {
	var strongest float32
	for _, e := range edges {
		strongest = max(strongest, e.weight)
	}
	kept := make([]graphEdge, 0, len(edges))
	for _, e := range edges {
		if e.weight >= strongest/float32(epochs) {
			kept = append(kept, e)
		}
	}
	return kept
}
