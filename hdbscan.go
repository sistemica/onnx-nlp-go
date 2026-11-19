package onnxgo

import (
	"errors"
	"math"
	"slices"
	"sort"
)

// HDBSCAN clusters points by density with excess-of-mass cluster selection. Label -1 marks noise.
type HDBSCAN struct {
	MinClusterSize int
	// MinSamples defaults to MinClusterSize.
	MinSamples int
}

type linkageRow struct {
	left, right int
	dist        float64
	size        int
}

type condensedRow struct {
	parent, child int
	lambda        float64
	size          int
}

type mstEdge struct {
	a, b   int
	weight float64
}

// Fit returns a label and a membership strength in [0, 1] for every point.
func (h HDBSCAN) Fit(points [][]float64) (labels []int, strengths []float64, err error) {
	if h.MinClusterSize < 2 {
		return nil, nil, errors.New("hdbscan: MinClusterSize must be at least 2")
	}
	n := len(points)
	for _, p := range points {
		if len(p) != len(points[0]) {
			return nil, nil, errors.New("hdbscan: points differ in dimension")
		}
	}
	labels, strengths = make([]int, n), make([]float64, n)
	for i := range labels {
		labels[i] = -1
	}
	if n < 2 {
		return labels, strengths, nil
	}
	k := h.MinSamples
	if k <= 0 {
		k = h.MinClusterSize
	}
	k = max(1, min(n-1, k))
	edges := minimumSpanningTree(points, coreDistances(points, k))
	hierarchy := singleLinkage(n, edges)
	tree := condenseTree(hierarchy, h.MinClusterSize)
	selected := selectClusters(tree, computeStability(tree))
	labels, strengths = assignClusters(tree, n, selected)
	return labels, strengths, nil
}

func squaredDistance(a, b []float64) float64 {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

// coreDistances returns each point's distance to its k-th nearest other point.
func coreDistances(points [][]float64, k int) []float64 {
	core := make([]float64, len(points))
	parallelFor(len(points), len(points)*(len(points[0])+k), func(lo, hi int) {
		nearest := make([]float64, k+1)
		for i := lo; i < hi; i++ {
			for j := range nearest {
				nearest[j] = math.Inf(1)
			}
			for _, q := range points {
				d := squaredDistance(points[i], q)
				if d >= nearest[k] {
					continue
				}
				at := k
				for at > 0 && nearest[at-1] > d {
					nearest[at] = nearest[at-1]
					at--
				}
				nearest[at] = d
			}
			core[i] = math.Sqrt(nearest[k])
		}
	})
	return core
}

// minimumSpanningTree runs Prim's algorithm on the mutual reachability graph and returns the
// edges sorted by weight. Edges of equal weight are ordered by their endpoints: tied merges
// change which clusters the condensed tree sees, so the order must not depend on the traversal.
func minimumSpanningTree(points [][]float64, core []float64) []mstEdge {
	n := len(points)
	inTree := make([]bool, n)
	best := make([]float64, n)
	from := make([]int, n)
	for i := range best {
		best[i] = math.Inf(1)
	}
	edges := make([]mstEdge, 0, n-1)
	current := 0
	for len(edges) < n-1 {
		inTree[current] = true
		next, nextWeight := -1, math.Inf(1)
		for j := 0; j < n; j++ {
			if inTree[j] {
				continue
			}
			w := max(core[current], core[j], math.Sqrt(squaredDistance(points[current], points[j])))
			if w < best[j] {
				best[j], from[j] = w, current
			}
			if best[j] < nextWeight || next < 0 {
				next, nextWeight = j, best[j]
			}
		}
		edges = append(edges, mstEdge{from[next], next, nextWeight})
		current = next
	}
	sort.Slice(edges, func(a, b int) bool {
		x, y := edges[a], edges[b]
		switch {
		case x.weight != y.weight:
			return x.weight < y.weight
		case min(x.a, x.b) != min(y.a, y.b):
			return min(x.a, x.b) < min(y.a, y.b)
		}
		return max(x.a, x.b) < max(y.a, y.b)
	})
	return edges
}

// singleLinkage merges the components of the sorted edges. Leaves are numbered 0..n-1 and the
// merge of row i is node n+i.
func singleLinkage(n int, edges []mstEdge) []linkageRow {
	parent := make([]int, 2*n-1)
	size := make([]int, 2*n-1)
	for i := range parent {
		parent[i] = -1
		if i < n {
			size[i] = 1
		}
	}
	find := func(x int) int {
		root := x
		for parent[root] != -1 {
			root = parent[root]
		}
		for parent[x] != -1 && parent[x] != root {
			x, parent[x] = parent[x], root
		}
		return root
	}
	rows := make([]linkageRow, len(edges))
	for i, e := range edges {
		a, b := find(e.a), find(e.b)
		node := n + i
		size[node] = size[a] + size[b]
		parent[a], parent[b] = node, node
		rows[i] = linkageRow{a, b, e.weight, size[node]}
	}
	return rows
}

func bfsFromHierarchy(rows []linkageRow, leaves, root int) []int {
	var order []int
	level := []int{root}
	for len(level) > 0 {
		order = append(order, level...)
		var next []int
		for _, node := range level {
			if node >= leaves {
				r := rows[node-leaves]
				next = append(next, r.left, r.right)
			}
		}
		level = next
	}
	return order
}

// condenseTree drops splits that shed fewer than minSize points. The root cluster is numbered
// with the leaf count.
func condenseTree(rows []linkageRow, minSize int) []condensedRow {
	leaves := len(rows) + 1
	root := 2 * len(rows)
	relabel := make([]int, root+1)
	relabel[root] = leaves
	next := leaves + 1
	ignore := make([]bool, root+1)
	var out []condensedRow
	subtreeLeaves := func(node int, lambda float64, parent int) {
		for _, sub := range bfsFromHierarchy(rows, leaves, node) {
			if sub < leaves {
				out = append(out, condensedRow{parent, sub, lambda, 1})
			}
			ignore[sub] = true
		}
	}
	sizeOf := func(node int) int {
		if node >= leaves {
			return rows[node-leaves].size
		}
		return 1
	}
	for _, node := range bfsFromHierarchy(rows, leaves, root) {
		if ignore[node] || node < leaves {
			continue
		}
		r := rows[node-leaves]
		lambda := math.Inf(1)
		if r.dist > 0 {
			lambda = 1 / r.dist
		}
		ls, rs := sizeOf(r.left), sizeOf(r.right)
		switch {
		case ls >= minSize && rs >= minSize:
			relabel[r.left], relabel[r.right] = next, next+1
			next += 2
			out = append(out, condensedRow{relabel[node], relabel[r.left], lambda, ls}, condensedRow{relabel[node], relabel[r.right], lambda, rs})
		case ls < minSize && rs < minSize:
			subtreeLeaves(r.left, lambda, relabel[node])
			subtreeLeaves(r.right, lambda, relabel[node])
		case ls < minSize:
			relabel[r.right] = relabel[node]
			subtreeLeaves(r.left, lambda, relabel[node])
		default:
			relabel[r.left] = relabel[node]
			subtreeLeaves(r.right, lambda, relabel[node])
		}
	}
	return out
}

// computeStability sums, per cluster, (lambda - birth) * size over its rows. Clusters are
// numbered from the smallest parent id.
func computeStability(tree []condensedRow) map[int]float64 {
	first, lastParent, last := math.MaxInt, 0, 0
	for _, r := range tree {
		first, lastParent, last = min(first, r.parent), max(lastParent, r.parent), max(last, r.parent, r.child)
	}
	birth := make([]float64, last+1)
	for i := range birth {
		birth[i] = math.NaN()
	}
	for _, r := range tree {
		if math.IsNaN(birth[r.child]) || r.lambda < birth[r.child] {
			birth[r.child] = r.lambda
		}
	}
	birth[first] = 0
	stability := map[int]float64{}
	for p := first; p <= lastParent; p++ {
		stability[p] = 0
	}
	for _, r := range tree {
		stability[r.parent] += (r.lambda - birth[r.parent]) * float64(r.size)
	}
	return stability
}

// selectClusters applies excess-of-mass selection and returns the chosen cluster ids.
func selectClusters(tree []condensedRow, stability map[int]float64) map[int]bool {
	children := map[int][]int{}
	for _, r := range tree {
		if r.size > 1 {
			children[r.parent] = append(children[r.parent], r.child)
		}
	}
	ids := make([]int, 0, len(stability))
	for id := range stability {
		ids = append(ids, id)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(ids)))
	selected := map[int]bool{}
	for _, id := range ids {
		selected[id] = true
	}
	delete(selected, ids[len(ids)-1])
	for _, id := range ids[:len(ids)-1] {
		var below float64
		for _, c := range children[id] {
			below += stability[c]
		}
		if below > stability[id] {
			selected[id] = false
			stability[id] = below
			continue
		}
		for queue := slices.Clone(children[id]); len(queue) > 0; queue = queue[1:] {
			selected[queue[0]] = false
			queue = append(queue, children[queue[0]]...)
		}
	}
	for id, ok := range selected {
		if !ok {
			delete(selected, id)
		}
	}
	return selected
}

// assignClusters labels each point with the selected cluster that contains it, numbering the
// clusters by increasing id, and derives the membership strength from the point's lambda.
func assignClusters(tree []condensedRow, n int, selected map[int]bool) ([]int, []float64) {
	ids := make([]int, 0, len(selected))
	for id := range selected {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	number := make(map[int]int, len(ids))
	for i, id := range ids {
		number[id] = i
	}
	up := map[int]int{}
	death := map[int]float64{}
	for _, r := range tree {
		death[r.parent] = max(death[r.parent], r.lambda)
		if r.child >= n {
			up[r.child] = r.parent
		}
	}
	labels, strengths := make([]int, n), make([]float64, n)
	for i := range labels {
		labels[i] = -1
	}
	for _, r := range tree {
		if r.child >= n {
			continue
		}
		cluster := r.parent
		for !selected[cluster] {
			p, ok := up[cluster]
			if !ok {
				cluster = -1
				break
			}
			cluster = p
		}
		if cluster < 0 {
			continue
		}
		labels[r.child] = number[cluster]
		if math.IsInf(r.lambda, 1) {
			strengths[r.child] = 1
		} else {
			strengths[r.child] = min(r.lambda, death[cluster]) / death[cluster]
		}
	}
	return labels, strengths
}
