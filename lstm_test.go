package onnxgo

import (
	"math"
	"math/rand/v2"
	"testing"
)

type lstmCase struct {
	seq, batch, input, hidden int
	direction                 string
	layout                    int
	bias, initial             bool
	lengths                   []int64
}

func refSigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

// referenceLSTM follows the ONNX equations directly in float64. It returns Y as [seq][dir][batch][hidden].
func referenceLSTM(x [][][]float64, w, r [][][]float64, b [][]float64, h0, c0 [][][]float64, lengths []int, reverse []bool, hidden int) (y [][][][]float64, yh, yc [][][]float64) {
	seq, batch := len(x), len(x[0])
	dirs := len(reverse)
	y = make([][][][]float64, seq)
	for t := range y {
		y[t] = make([][][]float64, dirs)
		for d := range y[t] {
			y[t][d] = make([][]float64, batch)
			for bi := range y[t][d] {
				y[t][d][bi] = make([]float64, hidden)
			}
		}
	}
	yh, yc = make([][][]float64, dirs), make([][][]float64, dirs)
	for d := 0; d < dirs; d++ {
		yh[d], yc[d] = make([][]float64, batch), make([][]float64, batch)
		for bi := 0; bi < batch; bi++ {
			h, c := make([]float64, hidden), make([]float64, hidden)
			if h0 != nil {
				copy(h, h0[d][bi])
				copy(c, c0[d][bi])
			}
			n := seq
			if lengths != nil {
				n = lengths[bi]
			}
			for step := 0; step < n; step++ {
				t := step
				if reverse[d] {
					t = n - 1 - step
				}
				gate := func(k int) []float64 {
					out := make([]float64, hidden)
					for u := 0; u < hidden; u++ {
						row := k*hidden + u
						sum := 0.0
						for i, xv := range x[t][bi] {
							sum += xv * w[d][row][i]
						}
						for i, hv := range h {
							sum += hv * r[d][row][i]
						}
						if b != nil {
							sum += b[d][row] + b[d][4*hidden+row]
						}
						out[u] = sum
					}
					return out
				}
				gi, go_, gf, gc := gate(0), gate(1), gate(2), gate(3)
				for u := 0; u < hidden; u++ {
					c[u] = refSigmoid(gf[u])*c[u] + refSigmoid(gi[u])*math.Tanh(gc[u])
					h[u] = refSigmoid(go_[u]) * math.Tanh(c[u])
					y[t][d][bi][u] = h[u]
				}
			}
			yh[d][bi], yc[d][bi] = h, c
		}
	}
	return y, yh, yc
}

func randTensor3(rng *rand.Rand, a, b, c int) ([][][]float64, *Tensor) {
	out := make([][][]float64, a)
	flat := make([]float32, 0, a*b*c)
	for i := range out {
		out[i] = make([][]float64, b)
		for j := range out[i] {
			out[i][j] = make([]float64, c)
			for k := range out[i][j] {
				v := float32(rng.NormFloat64() * 0.5)
				out[i][j][k] = float64(v)
				flat = append(flat, v)
			}
		}
	}
	return out, NewFloat([]int{a, b, c}, flat)
}

func TestLSTMMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(131, 132))
	cases := []lstmCase{
		{seq: 5, batch: 1, input: 3, hidden: 4, direction: "forward", bias: true},
		{seq: 5, batch: 2, input: 3, hidden: 4, direction: "reverse"},
		{seq: 6, batch: 2, input: 5, hidden: 3, direction: "bidirectional", bias: true, initial: true},
		{seq: 4, batch: 3, input: 2, hidden: 5, direction: "bidirectional", layout: 1, bias: true},
		{seq: 5, batch: 3, input: 3, hidden: 4, direction: "bidirectional", lengths: []int64{5, 2, 0}, bias: true},
		{seq: 5, batch: 2, input: 3, hidden: 4, direction: "reverse", lengths: []int64{3, 5}, initial: true},
		{seq: 1, batch: 1, input: 1, hidden: 1, direction: "forward"},
		{seq: 7, batch: 2, input: 20, hidden: 24, direction: "bidirectional", layout: 1, bias: true, initial: true, lengths: []int64{7, 4}},
	}
	for ci, tc := range cases {
		dirs := 1
		reverse := []bool{tc.direction == "reverse"}
		if tc.direction == "bidirectional" {
			dirs, reverse = 2, []bool{false, true}
		}
		xr, x := randTensor3(rng, tc.seq, tc.batch, tc.input)
		wr, w := randTensor3(rng, dirs, 4*tc.hidden, tc.input)
		rr, r := randTensor3(rng, dirs, 4*tc.hidden, tc.hidden)
		in := []*Tensor{x, w, r, nil, nil, nil, nil}
		var br [][]float64
		if tc.bias {
			var b3 [][][]float64
			b3, in[3] = randTensor3(rng, 1, dirs, 8*tc.hidden)
			in[3] = in[3].withShape([]int{dirs, 8 * tc.hidden})
			br = b3[0]
		}
		var lengths []int
		if tc.lengths != nil {
			in[4] = iT([]int{tc.batch}, tc.lengths...)
			for _, l := range tc.lengths {
				lengths = append(lengths, int(l))
			}
		}
		var h0r, c0r [][][]float64
		if tc.initial {
			h0r, in[5] = randTensor3(rng, dirs, tc.batch, tc.hidden)
			c0r, in[6] = randTensor3(rng, dirs, tc.batch, tc.hidden)
		}
		attrs := map[string]any{"direction": tc.direction, "hidden_size": tc.hidden, "layout": tc.layout}
		node := &Node{Op: "LSTM", Attr: mkAttrs(attrs), Out: []string{"y", "yh", "yc"}}
		if tc.layout == 1 {
			in[0] = stridedCopy(x, []int{tc.batch, tc.seq, tc.input}, []int{tc.input, tc.batch * tc.input, 1}, 0)
		}
		outs, err := ops["LSTM"](&opCtx{node: node, in: in})
		if err != nil {
			t.Fatalf("case %d: %v", ci, err)
		}
		wantY, wantH, wantC := referenceLSTM(xr, wr, rr, br, h0r, c0r, lengths, reverse, tc.hidden)
		yIdx := func(tm, d, b, u int) int {
			if tc.layout == 1 {
				return ((b*tc.seq+tm)*dirs+d)*tc.hidden + u
			}
			return ((tm*dirs+d)*tc.batch+b)*tc.hidden + u
		}
		hIdx := func(d, b, u int) int {
			if tc.layout == 1 {
				return (b*dirs+d)*tc.hidden + u
			}
			return (d*tc.batch+b)*tc.hidden + u
		}
		for tm := 0; tm < tc.seq; tm++ {
			for d := 0; d < dirs; d++ {
				for b := 0; b < tc.batch; b++ {
					for u := 0; u < tc.hidden; u++ {
						if got := float64(outs[0].F[yIdx(tm, d, b, u)]); math.Abs(got-wantY[tm][d][b][u]) > 2e-5 {
							t.Fatalf("case %d Y[%d,%d,%d,%d]: got %v, want %v", ci, tm, d, b, u, got, wantY[tm][d][b][u])
						}
					}
				}
			}
		}
		for d := 0; d < dirs; d++ {
			for b := 0; b < tc.batch; b++ {
				for u := 0; u < tc.hidden; u++ {
					if got := float64(outs[1].F[hIdx(d, b, u)]); math.Abs(got-wantH[d][b][u]) > 2e-5 {
						t.Fatalf("case %d Y_h[%d,%d,%d]: got %v, want %v", ci, d, b, u, got, wantH[d][b][u])
					}
					if got := float64(outs[2].F[hIdx(d, b, u)]); math.Abs(got-wantC[d][b][u]) > 2e-5 {
						t.Fatalf("case %d Y_c[%d,%d,%d]: got %v, want %v", ci, d, b, u, got, wantC[d][b][u])
					}
				}
			}
		}
	}
}

func TestLSTMRejectsUnsupportedOrMalformedInput(t *testing.T) {
	x, w, r := fT([]int{2, 1, 3}, iota32(6)...), fT([]int{1, 8, 3}, iota32(24)...), fT([]int{1, 8, 2}, iota32(16)...)
	ok := map[string]any{"hidden_size": 2}
	mustFail(t, "LSTM", ok, "at least 3 inputs", x, w)
	mustFail(t, "LSTM", ok, "float32 only", iT([]int{2, 1, 3}, 1, 2, 3, 4, 5, 6), w, r)
	mustFail(t, "LSTM", map[string]any{"hidden_size": 2, "clip": float32(1)}, "clip is not supported", x, w, r)
	mustFail(t, "LSTM", map[string]any{"hidden_size": 2, "input_forget": 1}, "input_forget is not supported", x, w, r)
	mustFail(t, "LSTM", map[string]any{"hidden_size": 2, "activations": "Sigmoid"}, "custom activations", x, w, r)
	mustFail(t, "LSTM", map[string]any{"hidden_size": 2, "direction": "sideways"}, "unknown LSTM direction", x, w, r)
	mustFail(t, "LSTM", map[string]any{"hidden_size": 0}, "positive hidden_size", x, w, r)
	mustFail(t, "LSTM", ok, "positive hidden_size", fT([]int{2, 3}), w, r)
	mustFail(t, "LSTM", map[string]any{"hidden_size": 2, "direction": "bidirectional"}, "do not match", x, w, r)
	mustFail(t, "LSTM", map[string]any{"hidden_size": 3}, "do not match", x, w, r)
	mustFail(t, "LSTM", ok, "peephole", x, w, r, nil, nil, nil, nil, fT([]int{1, 6}))
	mustFail(t, "LSTM", ok, "sequence_lens must hold 1", x, w, r, nil, iT([]int{2}, 1, 1))
	mustFail(t, "LSTM", ok, "sequence_lens must hold 1", x, w, r, nil, fT([]int{1}, 1))
	mustFail(t, "LSTM", ok, "outside [0, 2]", x, w, r, nil, iT([]int{1}, 5))
	mustFail(t, "LSTM", ok, "input 3 has shape", x, w, r, fT([]int{1, 3}))
	mustFail(t, "LSTM", ok, "input 5 has shape", x, w, r, nil, nil, fT([]int{1, 2}))
	mustFail(t, "LSTM", ok, "input 6 has shape", x, w, r, nil, nil, fT([]int{1, 1, 2}), iT([]int{1, 1, 2}, 0, 0))
}
