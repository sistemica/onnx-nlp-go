package onnxgo

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
)

type glirelRefCase struct {
	Tokens        []string        `json:"tokens"`
	Labels        []string        `json:"labels"`
	NER           [][]any         `json:"ner"`
	SpanIdx       [][]int         `json:"span_idx"`
	RelationsIdx  [][][]int       `json:"relations_idx"`
	Scores        [][]float64     `json:"scores"`
	Probs         [][]float64     `json:"probs"`
	InputIDs      []int           `json:"input_ids"`
	FirstSubtoken []int           `json:"first_subtoken"`
	NPromptWords  int             `json:"n_prompt_words"`
	Pred05        []glirelRefPred `json:"pred_0.5"`
	Pred005       []glirelRefPred `json:"pred_0.05"`
}

type glirelRefPred struct {
	HeadPos  []int    `json:"head_pos"`
	TailPos  []int    `json:"tail_pos"`
	HeadText []string `json:"head_text"`
	TailText []string `json:"tail_text"`
	Label    string   `json:"label"`
	Score    float64  `json:"score"`
}

type glirelRef struct {
	Labels []string        `json:"labels"`
	Cases  []glirelRefCase `json:"cases"`
}

func glirelEnv(t testing.TB) (model, ref string) {
	t.Helper()
	model, ref = os.Getenv("ONNXGO_GLIREL"), os.Getenv("ONNXGO_GLIREL_REF")
	if model == "" || ref == "" {
		t.Skip("ONNXGO_GLIREL and ONNXGO_GLIREL_REF not set")
	}
	return model, ref
}

func loadGlirelRef(t testing.TB, ref string) glirelRef {
	t.Helper()
	var r glirelRef
	raw, err := os.ReadFile(filepath.Join(ref, "ref.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

var npyShape = regexp.MustCompile(`'shape': \(([^)]*)\)`)

func readNpy(t testing.TB, path string) ([]float32, []int) {
	t.Helper()
	wide, shape := readNpyAny(t, path)
	data := make([]float32, len(wide))
	for i, v := range wide {
		data[i] = float32(v)
	}
	return data, shape
}

func readNpyAny(t testing.TB, path string) ([]float64, []int) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) < 10 || string(b[1:6]) != "NUMPY" {
		t.Fatalf("%s is not an npy file", path)
	}
	headerLen := int(binary.LittleEndian.Uint16(b[8:10]))
	header := string(b[10 : 10+headerLen])
	fortran := strings.Contains(header, "'fortran_order': True")
	var shape []int
	for _, part := range strings.Split(npyShape.FindStringSubmatch(header)[1], ",") {
		if part = strings.TrimSpace(part); part != "" {
			n, _ := strconv.Atoi(part)
			shape = append(shape, n)
		}
	}
	body := b[10+headerLen:]
	data := make([]float64, numel(shape))
	for i := range data {
		switch {
		case strings.Contains(header, "'<f4'"):
			data[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(body[4*i:])))
		case strings.Contains(header, "'<f8'"):
			data[i] = math.Float64frombits(binary.LittleEndian.Uint64(body[8*i:]))
		case strings.Contains(header, "'<i8'"):
			data[i] = float64(int64(binary.LittleEndian.Uint64(body[8*i:])))
		default:
			t.Fatalf("%s: unsupported dtype in %q", path, header)
		}
	}
	if fortran && len(shape) == 2 {
		rows, cols := shape[0], shape[1]
		rowMajor := make([]float64, len(data))
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				rowMajor[r*cols+c] = data[c*rows+r]
			}
		}
		data = rowMajor
	}
	return data, shape
}

func worstDiff(got, want []float32) (float64, int) {
	var worst float64
	at := 0
	for i := range want {
		if d := math.Abs(float64(got[i]) - float64(want[i])); d > worst {
			worst, at = d, i
		}
	}
	return worst, at
}

func TestGLiRELHeadAgainstReference(t *testing.T) {
	model, ref := glirelEnv(t)
	head, err := Load(filepath.Join(model, "onnx", "head.onnx"))
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range loadGlirelRef(t, ref).Cases {
		pooled, shape := readNpy(t, filepath.Join(ref, fmt.Sprintf("case%d_pooled.npy", i)))
		width := shape[1]
		labels := len(c.Labels)
		prompt := NewFloat([]int{1, 2 * labels, width}, pooled[:2*labels*width])
		words := NewFloat([]int{1, shape[0] - 2*labels - 1, width}, pooled[(2*labels+1)*width:])
		spans := map[[2]int]int{}
		spanData := make([]int64, 0, 2*len(c.SpanIdx))
		for k, s := range c.SpanIdx {
			spans[[2]int{s[0], s[1]}] = k
			spanData = append(spanData, int64(s[0]), int64(s[1]))
		}
		var heads, tails []int64
		for _, p := range c.RelationsIdx {
			heads = append(heads, int64(spans[[2]int{p[0][0], p[0][1]}]))
			tails = append(tails, int64(spans[[2]int{p[1][0], p[1][1]}]))
		}
		out, err := head.Run(map[string]*Tensor{
			"prompt_rep": prompt, "word_rep": words, "span_idx": iT([]int{len(c.SpanIdx), 2}, spanData...),
			"head_ids": iT([]int{len(heads)}, heads...), "tail_ids": iT([]int{len(tails)}, tails...),
		})
		if err != nil {
			t.Fatal(err)
		}
		got := out["scores"]
		want := make([]float32, 0, len(heads)*labels)
		for _, row := range c.Scores {
			for _, v := range row {
				want = append(want, float32(v))
			}
		}
		wantShape(t, got, []int{len(heads), labels})
		d, at := worstDiff(got.F, want)
		t.Logf("case %d: scores %v, max abs diff %.3g (at %d, value %.3f)", i, got.Shape, d, at, want[at])
		if d > 2e-3 {
			t.Errorf("case %d: head scores differ from PyTorch by %g", i, d)
		}
	}
}

func promptWords(c glirelRefCase) []string {
	var words []string
	for _, l := range c.Labels {
		words = append(words, "[REL]", l)
	}
	return append(append(words, "[SEP]"), c.Tokens...)
}

func TestGLiRELTokenizationAgainstReference(t *testing.T) {
	model, ref := glirelEnv(t)
	g, err := LoadGLiREL(model)
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range loadGlirelRef(t, ref).Cases {
		ids, first := g.encodeWords(promptWords(c))
		got := make([]int, len(ids))
		for k, id := range ids {
			got[k] = int(id)
		}
		if !slices.Equal(got, c.InputIDs) {
			t.Errorf("case %d: ids\n got %v\nwant %v", i, got, c.InputIDs)
		}
		if !slices.Equal(first, c.FirstSubtoken) {
			t.Errorf("case %d: first subtokens %v, want %v", i, first, c.FirstSubtoken)
		}
	}
}

func TestGLiRELEncoderAgainstReference(t *testing.T) {
	model, ref := glirelEnv(t)
	g, err := LoadGLiREL(model)
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range loadGlirelRef(t, ref).Cases {
		want, shape := readNpy(t, filepath.Join(ref, fmt.Sprintf("case%d_hidden.npy", i)))
		ids := make([]int64, len(c.InputIDs))
		for k, id := range c.InputIDs {
			ids[k] = int64(id)
		}
		out, err := g.encoder.Run(map[string]*Tensor{
			"input_ids":      NewInt([]int{1, len(ids)}, ids),
			"attention_mask": NewInt([]int{1, len(ids)}, slices.Repeat([]int64{1}, len(ids))),
		})
		if err != nil {
			t.Fatal(err)
		}
		got := out["hidden_states"]
		wantShape(t, got, []int{1, shape[0], shape[1]})
		d, at := worstDiff(got.F, want)
		t.Logf("case %d: hidden %v max abs diff %.3g (at %d, value %.3f)", i, got.Shape, d, at, want[at])
		if d > 5e-3 {
			t.Errorf("case %d: encoder differs from PyTorch by %g", i, d)
		}
	}
}

func refSpans(c glirelRefCase) []Span {
	var spans []Span
	for _, e := range c.NER {
		spans = append(spans, Span{int(e[0].(float64)), int(e[1].(float64)) + 1})
	}
	return spans
}

func TestGLiRELEndToEndAgainstReference(t *testing.T) {
	model, ref := glirelEnv(t)
	g, err := LoadGLiREL(model)
	if err != nil {
		t.Fatal(err)
	}
	total, matched := 0, 0
	for i, c := range loadGlirelRef(t, ref).Cases {
		for _, tc := range []struct {
			threshold float32
			want      []glirelRefPred
		}{{0.5, c.Pred05}, {0.05, c.Pred005}} {
			got, err := g.Extract(c.Tokens, refSpans(c), c.Labels, tc.threshold, 0)
			if err != nil {
				t.Fatal(err)
			}
			total += len(tc.want)
			if len(got) != len(tc.want) {
				t.Errorf("case %d threshold %v: got %d relations, want %d", i, tc.threshold, len(got), len(tc.want))
				continue
			}
			for k, w := range tc.want {
				r := got[k]
				if r.Label != w.Label || r.Head != (Span{w.HeadPos[0], w.HeadPos[1]}) || r.Tail != (Span{w.TailPos[0], w.TailPos[1]}) ||
					!slices.Equal(r.HeadText, w.HeadText) || math.Abs(float64(r.Score)-w.Score) > 2e-3 {
					t.Errorf("case %d threshold %v #%d: got %+v, want %+v", i, tc.threshold, k, r, w)
					continue
				}
				matched++
			}
		}
	}
	t.Logf("%d of %d reference relations matched (labels, spans, texts, scores within 2e-3)", matched, total)
}
