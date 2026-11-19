package onnxgo

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

// Fit embeds the rows of data and returns one Components-dimensional point per row.
func (u UMAP) Fit(data [][]float32) ([][]float64, error) {
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
	return layout.points(), nil
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
	n, dim := len(data), len(data[0])
	unit := make([]float32, n*dim)
	unitT := make([]float32, n*dim)
	norms := make([]float64, n)
	for i, row := range data {
		var norm float64
		for _, v := range row {
			norm += float64(v) * float64(v)
		}
		norms[i] = math.Sqrt(norm)
		scale := float32(0)
		if norm > 0 {
			scale = float32(1 / norms[i])
		}
		for d, v := range row {
			unit[i*dim+d] = v * scale
			unitT[d*n+i] = v * scale
		}
	}
	idx, dist := make([][]int32, n), make([][]float64, n)
	const block = 256
	sims := make([]float32, block*n)
	for lo := 0; lo < n; lo += block {
		hi := min(lo+block, n)
		clear(sims)
		runGemms([]gemmJob{{out: sims[:(hi-lo)*n], a: unit[lo*dim : hi*dim], b: unitT, m: hi - lo, k: dim, n: n}})
		parallelFor(hi-lo, n*k, func(from, to int) {
			for r := from; r < to; r++ {
				idx[lo+r], dist[lo+r] = nearestByDistance(sims[r*n:(r+1)*n], lo+r, k)
				refineDistances(data, norms, idx[lo+r], dist[lo+r])
			}
		})
	}
	return idx, dist
}

// refineDistances recomputes the distances of a neighbour row in float64, where identical
// vectors come out as exactly zero, and restores the ascending order after self.
func refineDistances(data [][]float32, norms []float64, idx []int32, dist []float64) {
	self := int(idx[0])
	for t := 1; t < len(idx); t++ {
		other := int(idx[t])
		var dotProduct float64
		for d, v := range data[self] {
			dotProduct += float64(v) * float64(data[other][d])
		}
		dist[t] = 1
		if norms[self] > 0 && norms[other] > 0 {
			dist[t] = max(0, 1-dotProduct/(norms[self]*norms[other]))
		}
		if dist[t] < 1e-9 {
			dist[t] = 0
		}
	}
	order := make([]int, len(idx)-1)
	for i := range order {
		order[i] = i + 1
	}
	sort.SliceStable(order, func(a, b int) bool { return dist[order[a]] < dist[order[b]] })
	sortedIdx, sortedDist := make([]int32, len(order)), make([]float64, len(order))
	for i, o := range order {
		sortedIdx[i], sortedDist[i] = idx[o], dist[o]
	}
	copy(idx[1:], sortedIdx)
	copy(dist[1:], sortedDist)
}

// nearestByDistance picks the k-1 most similar entries other than self and puts self first.
func nearestByDistance(sims []float32, self, k int) ([]int32, []float64) {
	idx, dist := make([]int32, k), make([]float64, k)
	idx[0] = int32(self)
	filled := 1
	for j, s := range sims {
		if j == self {
			continue
		}
		d := max(0, 1-float64(s))
		if filled == k && d >= dist[k-1] {
			continue
		}
		at := min(filled, k-1)
		for at > 1 && dist[at-1] > d {
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
func smoothKNN(dist [][]float64, k int) (sigma, rho []float64) {
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
			if d > 0 {
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
	sigma, rho := smoothKNN(dist, k)
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
