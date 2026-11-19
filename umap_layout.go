package onnxgo

import "math"

// splitMix is a small deterministic generator for initialisation noise and negative sampling.
type splitMix struct{ state uint64 }

func (r *splitMix) next() uint64 {
	r.state += 0x9e3779b97f4a7c15
	z := r.state
	z = (z ^ z>>30) * 0xbf58476d1ce4e5b9
	z = (z ^ z>>27) * 0x94d049bb133111eb
	return z ^ z>>31
}

func (r *splitMix) uniform() float64 { return float64(r.next()>>11) / (1 << 53) }

func (r *splitMix) normal() float64 {
	u1, u2 := 1-r.uniform(), r.uniform()
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

// noisyScale stretches coordinates to a maximum magnitude of 10 and adds a little noise.
func noisyScale(points [][]float64, rng *splitMix) [][]float64 {
	var largest float64
	for _, p := range points {
		for _, v := range p {
			largest = max(largest, math.Abs(v))
		}
	}
	expansion := 10 / largest
	out := make([][]float64, len(points))
	for i, p := range points {
		out[i] = make([]float64, len(p))
		for d, v := range p {
			out[i][d] = float64(float32(v*expansion)) + 1e-4*rng.normal()
		}
	}
	return out
}

// layout optimises point positions by stochastic gradient descent on the fuzzy set cross
// entropy: connected points attract, randomly drawn points repel.
type layout struct {
	dim, n      int
	coords      []float32
	head, tail  []int32
	a, b, gamma float64
	negatives   int
	epochs      int
	perSample   []float64
	nextSample  []float64
	perNegative []float64
	nextNeg     []float64
	rng         *splitMix
	moveOther   bool
	initialRate float64
	target      []float32
	targetCount int
	attraction  *coefficientTable
	repulsion   *coefficientTable
}

func newLayout(init [][]float64, edges []graphEdge, epochs int, a, b float64, negatives int, rng *splitMix) *layout {
	l := &layout{n: len(init), dim: len(init[0]), a: a, b: b, gamma: 1, negatives: negatives, epochs: epochs, rng: rng, moveOther: true, initialRate: 1}
	l.coords = flatten32(init)
	l.target, l.targetCount = l.coords, l.n
	var strongest float32
	for _, e := range edges {
		strongest = max(strongest, e.weight)
	}
	for _, e := range edges {
		l.head, l.tail = append(l.head, e.head), append(l.tail, e.tail)
		samples := float32(epochs) * (e.weight / strongest)
		per := -1.0
		if samples > 0 {
			per = float64(epochs) / float64(samples)
		}
		l.perSample = append(l.perSample, per)
		l.perNegative = append(l.perNegative, per/float64(max(negatives, 1)))
	}
	l.nextSample = append([]float64(nil), l.perSample...)
	l.nextNeg = append([]float64(nil), l.perNegative...)
	return l
}

func flatten32(points [][]float64) []float32 {
	out := make([]float32, 0, len(points)*len(points[0]))
	for _, p := range points {
		for _, v := range p {
			out = append(out, float32(v))
		}
	}
	return out
}

// newTransformLayout places new points among fixed training points. Edges lead from a new point
// to a training point, only the new points move, and the learning rate starts at a quarter.
func newTransformLayout(init, training [][]float64, edges []graphEdge, epochs int, a, b float64, negatives int, rng *splitMix) *layout {
	l := newLayout(init, edges, epochs, a, b, negatives, rng)
	l.target, l.targetCount = flatten32(training), len(training)
	l.moveOther = false
	l.initialRate = 0.25
	return l
}

func clip4(v float64) float64 { return max(-4, min(4, v)) }

// coefficientTable tabulates a function of the squared distance on a grid with 256 steps per
// octave and interpolates linearly, which costs a few nanoseconds where evaluating the power
// would cost fifty. The relative error is about 1e-6.
type coefficientTable struct{ values []float64 }

const (
	tableBits   = 8
	tableMinExp = -60
	tableMaxExp = 30
)

func newCoefficientTable(f func(dsq float64) float64) *coefficientTable {
	steps := (tableMaxExp - tableMinExp) << tableBits
	t := &coefficientTable{values: make([]float64, steps+2)}
	for i := range t.values {
		octave, within := i>>tableBits, i&(1<<tableBits-1)
		t.values[i] = f(math.Ldexp(1+float64(within)/(1<<tableBits), octave+tableMinExp))
	}
	return t
}

func (t *coefficientTable) at(dsq float64) float64 {
	bits := math.Float64bits(dsq)
	exponent := int(bits>>52&0x7ff) - 1023
	if exponent < tableMinExp {
		return t.values[0]
	}
	if exponent >= tableMaxExp {
		return t.values[len(t.values)-2]
	}
	mantissa := bits & (1<<52 - 1)
	i := (exponent-tableMinExp)<<tableBits | int(mantissa>>(52-tableBits))
	fraction := float64(mantissa&(1<<(52-tableBits)-1)) / (1 << (52 - tableBits))
	return t.values[i] + fraction*(t.values[i+1]-t.values[i])
}

func squaredDist32(x, y []float32) float32 {
	var s float32
	for i := range x {
		d := x[i] - y[i]
		s += d * d
	}
	return s
}

// run performs all epochs. The learning rate stays at 1 for the first two epochs, then falls
// linearly to zero.
func (l *layout) run() {
	l.attraction = newCoefficientTable(func(dsq float64) float64 {
		return -2 * l.a * l.b * math.Pow(dsq, l.b-1) / (l.a*math.Pow(dsq, l.b) + 1)
	})
	l.repulsion = newCoefficientTable(func(dsq float64) float64 {
		return 2 * l.gamma * l.b / ((0.001 + dsq) * (l.a*math.Pow(dsq, l.b) + 1))
	})
	for n := 0; n < l.epochs; n++ {
		alpha := l.initialRate
		if n > 0 {
			alpha = l.initialRate * (1 - float64(n-1)/float64(l.epochs))
		}
		l.epoch(n, alpha)
	}
}

func (l *layout) epoch(n int, alpha float64) {
	dim := l.dim
	for e := range l.head {
		if l.nextSample[e] > float64(n) {
			continue
		}
		j, k := int(l.head[e]), int(l.tail[e])
		current, other := l.coords[j*dim:(j+1)*dim], l.target[k*dim:(k+1)*dim]
		dsq := float64(squaredDist32(current, other))
		coeff := 0.0
		if dsq > 0 {
			coeff = l.attraction.at(dsq)
		}
		for d := range current {
			g := clip4(coeff * float64(current[d]-other[d]))
			current[d] = float32(float64(current[d]) + g*alpha)
			if l.moveOther {
				other[d] = float32(float64(other[d]) - g*alpha)
			}
		}
		l.nextSample[e] += l.perSample[e]
		if l.negatives == 0 {
			continue
		}
		draws := int((float64(n) - l.nextNeg[e]) / l.perNegative[e])
		for p := 0; p < draws; p++ {
			k := int(l.rng.next() % uint64(l.targetCount))
			other := l.target[k*dim : (k+1)*dim]
			dsq := float64(squaredDist32(current, other))
			coeff := 0.0
			switch {
			case dsq > 0:
				coeff = l.repulsion.at(dsq)
			case j == k:
				continue
			}
			if coeff <= 0 {
				continue
			}
			for d := range current {
				current[d] = float32(float64(current[d]) + clip4(coeff*float64(current[d]-other[d]))*alpha)
			}
		}
		l.nextNeg[e] += float64(draws) * l.perNegative[e]
	}
}

func (l *layout) points() [][]float64 {
	out := make([][]float64, l.n)
	for i := range out {
		out[i] = make([]float64, l.dim)
		for d := range out[i] {
			out[i][d] = float64(l.coords[i*l.dim+d])
		}
	}
	return out
}
