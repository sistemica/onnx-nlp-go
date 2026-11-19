package onnxgo

import (
	"math"
	"slices"
)

// spectralInit places points at the eigenvectors of the normalised graph Laplacian that belong
// to the smallest non-trivial eigenvalues. Graphs that are disconnected or too small for the
// solver start from uniform random positions instead.
func spectralInit(edges []graphEdge, n, dim int, rng *splitMix) [][]float64 {
	random := func() [][]float64 {
		out := make([][]float64, n)
		for i := range out {
			out[i] = make([]float64, dim)
			for d := range out[i] {
				out[i][d] = 20*rng.uniform() - 10
			}
		}
		return out
	}
	if n < dim+1 || !connected(edges, n) {
		return random()
	}
	op := newNormalizedAdjacency(edges, n)
	_, vectors := largestEigenpairs(n, dim+1, op.apply, 1e-9)
	out := make([][]float64, n)
	for i := range out {
		out[i] = make([]float64, dim)
		for d := range out[i] {
			out[i][d] = vectors[d+1][i]
		}
	}
	return out
}

func connected(edges []graphEdge, n int) bool {
	adjacent := make([][]int32, n)
	for _, e := range edges {
		adjacent[e.head] = append(adjacent[e.head], e.tail)
	}
	seen := make([]bool, n)
	queue := []int32{0}
	seen[0] = true
	for count := 1; len(queue) > 0; queue = queue[1:] {
		for _, next := range adjacent[queue[0]] {
			if !seen[next] {
				seen[next] = true
				count++
				queue = append(queue, next)
			}
		}
		if count == n {
			return true
		}
	}
	return false
}

// normalizedAdjacency is D^-1/2 A D^-1/2 in compressed row form.
type normalizedAdjacency struct {
	rowStart []int
	column   []int32
	value    []float64
}

func newNormalizedAdjacency(edges []graphEdge, n int) *normalizedAdjacency {
	degree := make([]float64, n)
	for _, e := range edges {
		degree[e.head] += float64(e.weight)
	}
	op := &normalizedAdjacency{rowStart: make([]int, n+1), column: make([]int32, len(edges)), value: make([]float64, len(edges))}
	for i, e := range edges {
		op.rowStart[e.head+1]++
		op.column[i] = e.tail
		op.value[i] = float64(e.weight) / math.Sqrt(degree[e.head]*degree[e.tail])
	}
	for i := 0; i < n; i++ {
		op.rowStart[i+1] += op.rowStart[i]
	}
	return op
}

func (op *normalizedAdjacency) apply(dst, x []float64) {
	for i := range dst {
		var s float64
		for p := op.rowStart[i]; p < op.rowStart[i+1]; p++ {
			s += op.value[p] * x[op.column[p]]
		}
		dst[i] = s
	}
}

func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// largestEigenpairs runs the Lanczos method with full reorthogonalisation on a symmetric
// operator and returns the count largest eigenvalues, in descending order, with unit
// eigenvectors. It stops when every wanted residual is below tol or the Krylov space is as
// large as the operator. When the Krylov space becomes invariant it continues from a fresh
// orthogonal vector, so eigenvalues the start vector has no component in are still found.
// The start vector is random: a symmetric one, such as all ones, has no component along the
// antisymmetric eigenvectors of a symmetric graph and would never find them.
// A repeated eigenvalue yields a single eigenvector, so highly symmetric graphs can return
// fewer independent vectors than a block method would; they are still valid eigenvectors.
func largestEigenpairs(n, count int, apply func(dst, x []float64), tol float64) ([]float64, [][]float64) {
	maxSteps := min(n, 2000)
	basis := make([][]float64, 0, 64)
	alpha, beta := []float64{}, []float64{}
	restarts := &splitMix{1}
	v := make([]float64, n)
	for i := range v {
		v[i] = restarts.normal()
	}
	scale := 1 / math.Sqrt(dot(v, v))
	for i := range v {
		v[i] *= scale
	}
	w := make([]float64, n)
	nextCheck := 2 * count
	for {
		basis = append(basis, slices.Clone(v))
		apply(w, v)
		alpha = append(alpha, dot(w, v))
		for pass := 0; pass < 2; pass++ {
			for _, q := range basis {
				c := dot(w, q)
				for i := range w {
					w[i] -= c * q[i]
				}
			}
		}
		norm := math.Sqrt(dot(w, w))
		beta = append(beta, norm)
		size := len(basis)
		exhausted := size >= maxSteps
		if norm < 1e-12 && !exhausted {
			beta[size-1] = 0
			v = orthogonalVector(basis, restarts)
			continue
		}
		if exhausted || size >= nextCheck {
			values, vectors := tridiagonalEigen(alpha, beta[:size-1])
			wanted := min(count, size)
			done := exhausted
			if !done {
				done = true
				for i := 0; i < wanted; i++ {
					if math.Abs(norm*vectors[i][size-1]) > tol {
						done = false
					}
				}
			}
			if done {
				return ritzPairs(basis, values[:wanted], vectors[:wanted])
			}
			nextCheck = size + max(10, size/5)
		}
		for i := range v {
			v[i] = w[i] / norm
		}
	}
}

// orthogonalVector returns a random unit vector orthogonal to the basis. Callers only ask while
// the basis is smaller than the space, so a vector with a usable component always exists.
func orthogonalVector(basis [][]float64, rng *splitMix) []float64 {
	v := make([]float64, len(basis[0]))
	for attempt := 0; ; attempt++ {
		for i := range v {
			v[i] = rng.normal()
		}
		for pass := 0; pass < 2; pass++ {
			for _, q := range basis {
				c := dot(v, q)
				for i := range v {
					v[i] -= c * q[i]
				}
			}
		}
		if norm := math.Sqrt(dot(v, v)); norm > 1e-8 || attempt == 7 {
			for i := range v {
				v[i] /= max(norm, 1e-300)
			}
			return v
		}
	}
}

func ritzPairs(basis [][]float64, values []float64, vectors [][]float64) ([]float64, [][]float64) {
	out := make([][]float64, len(values))
	for i := range out {
		out[i] = make([]float64, len(basis[0]))
		for j, q := range basis {
			c := vectors[i][j]
			for k := range q {
				out[i][k] += c * q[k]
			}
		}
	}
	return values, out
}

// givens returns the rotation (c, s) that maps (g, f) to (r, 0). Two zeros need no rotation.
func givens(f, g float64) (c, s, r float64) {
	r = math.Hypot(f, g)
	if r == 0 {
		return 1, 0, 0
	}
	return g / r, f / r, r
}

// tridiagonalEigen returns the eigenvalues of the symmetric tridiagonal matrix with diagonal
// diag and off-diagonal off, largest first, with their unit eigenvectors.
func tridiagonalEigen(diag, off []float64) ([]float64, [][]float64) {
	n := len(diag)
	d := slices.Clone(diag)
	e := append(slices.Clone(off), 0)
	z := make([][]float64, n)
	for i := range z {
		z[i] = make([]float64, n)
		z[i][i] = 1
	}
	for l := 0; l < n; l++ {
		for iter := 0; iter < 60; iter++ {
			m := l
			for ; m < n-1; m++ {
				if math.Abs(e[m]) <= 1e-16*(math.Abs(d[m])+math.Abs(d[m+1])) {
					break
				}
			}
			if m == l {
				break
			}
			g := (d[l+1] - d[l]) / (2 * e[l])
			r := math.Hypot(g, 1)
			g = d[m] - d[l] + e[l]/(g+math.Copysign(r, g))
			s, c, p := 1.0, 1.0, 0.0
			i := m - 1
			for ; i >= l; i-- {
				f, b := s*e[i], c*e[i]
				c, s, r = givens(f, g)
				e[i+1] = r
				g = d[i+1] - p
				r = (d[i]-g)*s + 2*c*b
				p = s * r
				d[i+1] = g + p
				g = c*r - b
				for k := 0; k < n; k++ {
					f = z[k][i+1]
					z[k][i+1] = s*z[k][i] + c*f
					z[k][i] = c*z[k][i] - s*f
				}
			}
			d[l] -= p
			e[l] = g
			e[m] = 0
		}
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int {
		switch {
		case d[a] > d[b]:
			return -1
		case d[a] < d[b]:
			return 1
		}
		return 0
	})
	values, vectors := make([]float64, n), make([][]float64, n)
	for i, o := range order {
		values[i], vectors[i] = d[o], make([]float64, n)
		for k := 0; k < n; k++ {
			vectors[i][k] = z[k][o]
		}
	}
	return values, vectors
}
