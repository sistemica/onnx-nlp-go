package onnxgo

import (
	"slices"
	"strings"
)

func hasRepeatedLabel(term string) bool {
	for i, l := range term {
		if strings.ContainsRune(term[i+len(string(l)):], l) {
			return true
		}
	}
	return false
}

// einsumGemm evaluates a two-operand contraction as batched matrix products. It returns nil when
// the equation has a repeated label or a label that is summed out of only one operand.
func einsumGemm(terms []string, rhs string, in []*Tensor, size map[rune]int) []*Tensor {
	if len(terms) != 2 || hasRepeatedLabel(terms[0]) || hasRepeatedLabel(terms[1]) || hasRepeatedLabel(rhs) {
		return nil
	}
	a, b := terms[0], terms[1]
	var batch, rows, cols, summed []rune
	for _, l := range rhs {
		inA, inB := strings.ContainsRune(a, l), strings.ContainsRune(b, l)
		switch {
		case inA && inB:
			batch = append(batch, l)
		case inA:
			rows = append(rows, l)
		default:
			cols = append(cols, l)
		}
	}
	for _, l := range a {
		if strings.ContainsRune(rhs, l) {
			continue
		}
		if !strings.ContainsRune(b, l) {
			return nil
		}
		summed = append(summed, l)
	}
	for _, l := range b {
		if !strings.ContainsRune(rhs, l) && !strings.ContainsRune(a, l) {
			return nil
		}
	}
	extent := func(labels []rune) int {
		n := 1
		for _, l := range labels {
			n *= size[l]
		}
		return n
	}
	bn, mn, nn, kn := extent(batch), extent(rows), extent(cols), extent(summed)
	if bn*mn*nn*kn == 0 {
		return nil
	}
	permute := func(t *Tensor, term string, order []rune) *Tensor {
		source := stridesOf(t.Shape)
		shape, strides := make([]int, len(order)), make([]int, len(order))
		for i, l := range order {
			shape[i], strides[i] = size[l], source[strings.IndexRune(term, l)]
		}
		return stridedCopy(t, shape, strides, 0)
	}
	left := permute(in[0], a, slices.Concat(batch, rows, summed))
	right := permute(in[1], b, slices.Concat(batch, summed, cols))
	natural := slices.Concat(batch, rows, cols)
	shape := make([]int, len(natural))
	for i, l := range natural {
		shape[i] = size[l]
	}
	out := zeros(Float32, shape)
	jobs := make([]gemmJob, bn)
	for i := range jobs {
		jobs[i] = gemmJob{
			out: out.F[i*mn*nn : (i+1)*mn*nn], a: left.F[i*mn*kn : (i+1)*mn*kn], b: right.F[i*kn*nn : (i+1)*kn*nn],
			m: mn, k: kn, n: nn,
		}
	}
	runGemms(jobs)
	if string(natural) == rhs {
		return []*Tensor{out}
	}
	source := stridesOf(out.Shape)
	wanted, strides := make([]int, len(rhs)), make([]int, len(rhs))
	for i, l := range []rune(rhs) {
		at := slices.Index(natural, l)
		wanted[i], strides[i] = shape[at], source[at]
	}
	return []*Tensor{stridedCopy(out, wanted, strides, 0)}
}
