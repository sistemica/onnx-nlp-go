package onnxnlp

import "fmt"

func sigmoidF32(x float32) float32 { return 1 / (1 + exp32(-x)) }

func tanhF32(x float32) float32 { return 2*sigmoidF32(2*x) - 1 }

func transposed(f []float32, rows, cols int) []float32 {
	out := make([]float32, len(f))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = f[r*cols+c]
		}
	}
	return out
}

type lstmSpec struct {
	dirs, hidden, seq, batch, input int
	reverse                         []bool
	lengths                         []int
}

func lstmSpecFrom(c *opCtx) (lstmSpec, error) {
	var s lstmSpec
	for _, name := range []string{"clip", "input_forget"} {
		if c.node.Int(name, 0) != 0 || c.node.Float(name, 0) != 0 {
			return s, fmt.Errorf("LSTM attribute %s is not supported", name)
		}
	}
	if a := c.node.Attr["activations"]; a != nil {
		return s, fmt.Errorf("LSTM custom activations are not supported")
	}
	switch dir := c.node.String("direction", "forward"); dir {
	case "forward":
		s.dirs, s.reverse = 1, []bool{false}
	case "reverse":
		s.dirs, s.reverse = 1, []bool{true}
	case "bidirectional":
		s.dirs, s.reverse = 2, []bool{false, true}
	default:
		return s, fmt.Errorf("unknown LSTM direction %q", dir)
	}
	s.hidden = int(c.node.Int("hidden_size", 0))
	x, w, r := c.in[0], c.in[1], c.in[2]
	if len(x.Shape) != 3 || len(w.Shape) != 3 || len(r.Shape) != 3 || s.hidden <= 0 {
		return s, fmt.Errorf("LSTM needs rank-3 X, W and R and a positive hidden_size")
	}
	if c.node.Int("layout", 0) == 1 {
		s.batch, s.seq = x.Shape[0], x.Shape[1]
	} else {
		s.seq, s.batch = x.Shape[0], x.Shape[1]
	}
	s.input = x.Shape[2]
	if w.Shape[0] != s.dirs || w.Shape[1] != 4*s.hidden || w.Shape[2] != s.input ||
		r.Shape[0] != s.dirs || r.Shape[1] != 4*s.hidden || r.Shape[2] != s.hidden {
		return s, fmt.Errorf("LSTM weight shapes W%v R%v do not match input %v and hidden size %d", w.Shape, r.Shape, x.Shape, s.hidden)
	}
	if len(c.in) > 7 && c.in[7] != nil {
		return s, fmt.Errorf("LSTM peephole weights are not supported")
	}
	if len(c.in) > 4 && c.in[4] != nil {
		lengths, err := intList(c.in[4])
		if err != nil || len(lengths) != s.batch {
			return s, fmt.Errorf("LSTM sequence_lens must hold %d integers", s.batch)
		}
		for _, l := range lengths {
			if l < 0 || l > s.seq {
				return s, fmt.Errorf("LSTM sequence length %d is outside [0, %d]", l, s.seq)
			}
		}
		s.lengths = lengths
	}
	return s, nil
}

func (s lstmSpec) optional(c *opCtx, i int, want ...int) ([]float32, error) {
	if len(c.in) <= i || c.in[i] == nil {
		return nil, nil
	}
	t := c.in[i]
	if !t.isFloat() || !slicesEqualInts(t.Shape, want) {
		return nil, fmt.Errorf("LSTM input %d has shape %v, want %v", i, t.Shape, want)
	}
	return t.F, nil
}

func slicesEqualInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func opLSTM(c *opCtx) ([]*Tensor, error) {
	if err := c.need(3); err != nil {
		return nil, err
	}
	for _, t := range c.in[:3] {
		if !t.isFloat() {
			return nil, fmt.Errorf("LSTM supports float32 only")
		}
	}
	s, err := lstmSpecFrom(c)
	if err != nil {
		return nil, err
	}
	bias, err := s.optional(c, 3, s.dirs, 8*s.hidden)
	if err != nil {
		return nil, err
	}
	h0, err := s.optional(c, 5, s.dirs, s.batch, s.hidden)
	if err != nil {
		return nil, err
	}
	c0, err := s.optional(c, 6, s.dirs, s.batch, s.hidden)
	if err != nil {
		return nil, err
	}
	x := c.in[0].F
	if c.node.Int("layout", 0) == 1 {
		x = stridedCopy(c.in[0], []int{s.seq, s.batch, s.input}, []int{s.input, s.seq * s.input, 1}, 0).F
	}
	y := zeros(Float32, []int{s.seq, s.dirs, s.batch, s.hidden})
	yh := zeros(Float32, []int{s.dirs, s.batch, s.hidden})
	yc := zeros(Float32, []int{s.dirs, s.batch, s.hidden})
	g4 := 4 * s.hidden
	for d := 0; d < s.dirs; d++ {
		w := transposed(c.in[1].F[d*g4*s.input:(d+1)*g4*s.input], g4, s.input)
		r := transposed(c.in[2].F[d*g4*s.hidden:(d+1)*g4*s.hidden], g4, s.hidden)
		sum := make([]float32, g4)
		if bias != nil {
			for i := range sum {
				sum[i] = bias[d*2*g4+i] + bias[d*2*g4+g4+i]
			}
		}
		projected := make([]float32, s.seq*s.batch*g4)
		runGemms([]gemmJob{{out: projected, a: x, b: w, m: s.seq * s.batch, k: s.input, n: g4}})
		h, cell := make([]float32, s.batch*s.hidden), make([]float32, s.batch*s.hidden)
		if h0 != nil {
			copy(h, h0[d*s.batch*s.hidden:(d+1)*s.batch*s.hidden])
		}
		if c0 != nil {
			copy(cell, c0[d*s.batch*s.hidden:(d+1)*s.batch*s.hidden])
		}
		s.recur(d, projected, sum, r, h, cell, y)
		copy(yh.F[d*s.batch*s.hidden:], h)
		copy(yc.F[d*s.batch*s.hidden:], cell)
	}
	if c.node.Int("layout", 0) == 1 {
		return []*Tensor{
			stridedCopy(y, []int{s.batch, s.seq, s.dirs, s.hidden}, []int{s.hidden, s.dirs * s.batch * s.hidden, s.batch * s.hidden, 1}, 0),
			stridedCopy(yh, []int{s.batch, s.dirs, s.hidden}, []int{s.hidden, s.batch * s.hidden, 1}, 0),
			stridedCopy(yc, []int{s.batch, s.dirs, s.hidden}, []int{s.hidden, s.batch * s.hidden, 1}, 0),
		}, nil
	}
	return []*Tensor{y, yh, yc}, nil
}

// recur runs one direction over all steps, updating the hidden and cell states in place.
func (s lstmSpec) recur(d int, projected, bias, recurrent, h, cell []float32, y *Tensor) {
	g4, hid := 4*s.hidden, s.hidden
	gates := make([]float32, s.batch*g4)
	for step := 0; step < s.seq; step++ {
		active := make([]bool, s.batch)
		for b := 0; b < s.batch; b++ {
			length := s.seq
			if s.lengths != nil {
				length = s.lengths[b]
			}
			if step >= length {
				clear(gates[b*g4 : (b+1)*g4])
				continue
			}
			t := step
			if s.reverse[d] {
				t = length - 1 - step
			}
			active[b] = true
			row := projected[(t*s.batch+b)*g4 : (t*s.batch+b+1)*g4]
			for i := range row {
				gates[b*g4+i] = row[i] + bias[i]
			}
		}
		runGemms([]gemmJob{{out: gates, a: h, b: recurrent, m: s.batch, k: hid, n: g4}})
		for b := 0; b < s.batch; b++ {
			if !active[b] {
				continue
			}
			length := s.seq
			if s.lengths != nil {
				length = s.lengths[b]
			}
			t := step
			if s.reverse[d] {
				t = length - 1 - step
			}
			g := gates[b*g4 : (b+1)*g4]
			hb, cb := h[b*hid:(b+1)*hid], cell[b*hid:(b+1)*hid]
			out := y.F[((t*s.dirs+d)*s.batch+b)*hid:][:hid]
			for k := 0; k < hid; k++ {
				i, o, f, cand := sigmoidF32(g[k]), sigmoidF32(g[hid+k]), sigmoidF32(g[2*hid+k]), tanhF32(g[3*hid+k])
				cb[k] = f*cb[k] + i*cand
				hb[k] = o * tanhF32(cb[k])
				out[k] = hb[k]
			}
		}
	}
}
