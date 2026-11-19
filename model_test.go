package onnxnlp

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/sistemica/onnx-nlp-go/internal/pb"
)

func TestVarintAndFields(t *testing.T) {
	for _, v := range []uint64{0, 1, 127, 128, 300, 1 << 35, math.MaxUint64} {
		got, n, err := readVarint(pb.Varint(v))
		if err != nil || got != v || n != len(pb.Varint(v)) {
			t.Fatalf("varint %d: got %d (%d bytes), err %v", v, got, n, err)
		}
	}
	if _, _, err := readVarint([]byte{0x80}); err == nil {
		t.Fatal("truncated varint should fail")
	}
	msg := slices.Concat(pb.VarintField(1, 5), pb.Fixed64Field(2, 9), pb.StringField(3, "hi"), pb.Fixed32Field(4, 1.5))
	var seen []field
	if err := forEachField(msg, func(f field) error { seen = append(seen, f); return nil }); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 4 || seen[0].v != 5 || seen[1].v != 9 || string(seen[2].data) != "hi" || seen[3].float32() != 1.5 {
		t.Fatalf("unexpected fields %+v", seen)
	}
	stop := errTruncated
	if err := forEachField(msg, func(field) error { return stop }); err != stop {
		t.Fatalf("callback error not propagated: %v", err)
	}
}

func TestForEachFieldErrors(t *testing.T) {
	bad := map[string][]byte{
		"truncated key":        {0x80},
		"truncated varint val": {0x08, 0x80},
		"truncated fixed64":    {0x11, 1, 2},
		"truncated length":     {0x1a, 0x80},
		"short bytes":          {0x1a, 5, 1},
		"truncated fixed32":    {0x25, 1},
		"wire type 3":          {0x0b},
	}
	for name, b := range bad {
		if err := forEachField(b, func(field) error { return nil }); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestAppendHelpers(t *testing.T) {
	packed := field{wire: 2, data: slices.Concat(pb.Varint(1), pb.Varint(300))}
	if got, err := appendVarints(nil, packed); err != nil || !slices.Equal(got, []int64{1, 300}) {
		t.Fatalf("packed: %v %v", got, err)
	}
	if got, err := appendVarints([]int64{9}, field{wire: 0, v: 4}); err != nil || !slices.Equal(got, []int64{9, 4}) {
		t.Fatalf("unpacked: %v %v", got, err)
	}
	if _, err := appendVarints(nil, field{wire: 2, data: []byte{0x80}}); err == nil {
		t.Fatal("bad packed varints should fail")
	}
	if got := appendFloat32s(nil, field{wire: 5, v: uint64(math.Float32bits(2.5))}); !slices.Equal(got, []float32{2.5}) {
		t.Fatalf("unpacked floats: %v", got)
	}
}

func parseTestModel(t *testing.T, nodes [][]byte, inits [][]byte, inputs, outputs []string) *Graph {
	t.Helper()
	g, err := ParseModel(pb.Model(pb.Graph(nodes, inits, inputs, outputs)))
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestParseModelStructure(t *testing.T) {
	body := pb.Graph([][]byte{pb.Node("Constant", nil, []string{"c"}, pb.AttrTensor("value", pb.TensorFloats("", []int64{1}, []float32{1})))}, nil, nil, []string{"c"})
	g := parseTestModel(t,
		[][]byte{
			pb.Node("Add", []string{"x", "w"}, []string{"y"}),
			pb.Node("If", []string{"y"}, []string{"z"}, pb.AttrGraph("then_branch", body), pb.AttrInt("axis", -2), pb.AttrFloat("alpha", 0.5),
				pb.AttrString("mode", "nearest"), pb.AttrInts("perm", 2, 0, 1), pb.AttrFloats("scales", 1, 2)),
			pb.NodeInDomain("Custom", "com.example", []string{"y"}, []string{"q"}),
			pb.Node("Sink", []string{"z"}, nil),
		},
		[][]byte{pb.TensorFloats("w", []int64{1}, []float32{2})},
		[]string{"x", "w"}, []string{"z"})
	if g.Name != "g" || len(g.Nodes) != 4 || len(g.Initializers) != 1 {
		t.Fatalf("unexpected graph %+v", g)
	}
	if !slices.Equal(g.Inputs, []string{"x"}) || !slices.Equal(g.Outputs, []string{"z"}) {
		t.Fatalf("inputs %v outputs %v", g.Inputs, g.Outputs)
	}
	if !g.Initializers["w"].static {
		t.Fatal("initializers must be marked static")
	}
	n := g.Nodes[1]
	if n.Int("axis", 0) != -2 || n.Int("missing", 7) != 7 || n.Float("alpha", 0) != 0.5 || n.Float("missing", 3) != 3 ||
		n.String("mode", "") != "nearest" || n.String("missing", "d") != "d" {
		t.Fatalf("attribute accessors wrong: %+v", n.Attr)
	}
	if p, ok := n.Ints("perm"); !ok || !slices.Equal(p, []int64{2, 0, 1}) {
		t.Fatalf("perm %v %v", p, ok)
	}
	if _, ok := n.Ints("nope"); ok {
		t.Fatal("missing ints attribute reported present")
	}
	if !slices.Equal(n.Attr["scales"].Floats, []float32{1, 2}) || n.Attr["then_branch"].G == nil || len(n.Attr["then_branch"].G.Nodes) != 1 {
		t.Fatal("float list or subgraph attribute not parsed")
	}
	if g.Nodes[2].Domain != "com.example" {
		t.Fatalf("domain %q", g.Nodes[2].Domain)
	}
}

func TestParseUnpackedAttributes(t *testing.T) {
	g := parseTestModel(t, [][]byte{pb.Node("X", nil, []string{"o"}, pb.AttrIntsUnpacked("ints", 3, 4), pb.AttrFloatsUnpacked("floats", 1, 2))}, nil, nil, []string{"o"})
	a := g.Nodes[0].Attr
	if !slices.Equal(a["ints"].Ints, []int64{3, 4}) || !slices.Equal(a["floats"].Floats, []float32{1, 2}) {
		t.Fatalf("unpacked attributes: %+v %+v", a["ints"], a["floats"])
	}
}

func TestParseTensorEncodings(t *testing.T) {
	g := parseTestModel(t, nil, [][]byte{
		pb.TensorFloats("f_raw", []int64{2}, []float32{1, 2}),
		pb.TensorFloatData("f_data", []int64{2}, []float32{3, 4}),
		pb.TensorInt64s("i_raw", []int64{2}, []int64{-1, 5}),
		pb.TensorInt64Data("i_data", []int64{2}, []int64{6, 7}),
		pb.TensorInt32s("i32_raw", []int64{2}, []int32{-2, 8}),
		pb.TensorInt32Data("i32_data", []int64{2}, []int64{9, 10}),
		pb.TensorBools("b_raw", []int64{2}, []bool{true, false}),
		pb.TensorBoolData("b_data", []int64{2}, []int64{0, 1}),
	}, nil, nil)
	in := g.Initializers
	wantF(t, in["f_raw"], []int{2}, 1, 2)
	wantF(t, in["f_data"], []int{2}, 3, 4)
	wantI(t, in["i_raw"], []int{2}, -1, 5)
	wantI(t, in["i_data"], []int{2}, 6, 7)
	wantI(t, in["i32_raw"], []int{2}, -2, 8)
	wantI(t, in["i32_data"], []int{2}, 9, 10)
	wantB(t, in["b_raw"], []int{2}, true, false)
	wantB(t, in["b_data"], []int{2}, false, true)
}

func TestParseErrors(t *testing.T) {
	if _, err := ParseModel(pb.VarintField(1, 8)); err == nil {
		t.Fatal("a model without a graph must fail")
	}
	if _, err := ParseModel([]byte{0x80}); err == nil {
		t.Fatal("garbage must fail")
	}
	bad := map[string][]byte{
		"external data":    pb.TensorExternal("w"),
		"unsupported type": pb.TensorRaw("w", pb.Half, []int64{1}, []byte{0, 0}),
		"raw size":         pb.TensorRaw("w", pb.Float, []int64{2}, []byte{0, 0, 0, 0}),
		"garbage tensor":   {0x80},
	}
	for name, tensor := range bad {
		if _, err := ParseModel(pb.Model(pb.Graph(nil, [][]byte{tensor}, nil, nil))); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	for name, node := range map[string][]byte{
		"garbage node":      {0x80},
		"garbage attribute": pb.BytesField(5, []byte{0x80}),
		"bad attr tensor":   pb.BytesField(5, pb.AttrTensor("value", pb.TensorExternal("w"))),
		"bad attr graph":    pb.BytesField(5, pb.AttrGraph("body", []byte{0x80})),
		"bad packed ints":   pb.BytesField(5, pb.BytesField(8, []byte{0x80})),
	} {
		if _, err := ParseModel(pb.Model(pb.BytesField(1, node))); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestParseGraphFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.onnx")
	if err := os.WriteFile(path, pb.Model(pb.Graph([][]byte{pb.Node("Relu", []string{"x"}, []string{"y"})}, nil, []string{"x"}, []string{"y"})), 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	out, err := s.Run(map[string]*Tensor{"x": fT([]int{2}, -1, 3)})
	if err != nil {
		t.Fatal(err)
	}
	wantF(t, out["y"], []int{2}, 0, 3)
	if !slices.Equal(s.InputNames(), []string{"x"}) || !slices.Equal(s.OutputNames(), []string{"y"}) {
		t.Fatal("input/output names wrong")
	}
	if _, err := Load(filepath.Join(dir, "missing.onnx")); err == nil {
		t.Fatal("missing file must fail")
	}
	if _, err := ParseGraphFile(filepath.Join(dir, "missing.onnx")); err == nil {
		t.Fatal("missing file must fail")
	}
	os.WriteFile(path, []byte{0x80}, 0o644)
	if _, err := Load(path); err == nil {
		t.Fatal("corrupt file must fail")
	}
}

func floatBytes(vals ...float32) []byte {
	var b []byte
	for _, v := range vals {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
	}
	return b
}

func externalModel(t *testing.T, dir string, tensors ...[]byte) string {
	t.Helper()
	g := pb.Graph(
		[][]byte{pb.Node("Add", []string{"a", "b"}, []string{"y"})},
		tensors, []string{"a", "b"}, []string{"y"},
	)
	path := filepath.Join(dir, "model.onnx")
	if err := os.WriteFile(path, pb.Model(g), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestExternalTensorDataIsReadFromTheModelDirectory(t *testing.T) {
	dir := t.TempDir()
	data := append(floatBytes(1, 2, 3, 4), floatBytes(10, 20)...)
	if err := os.WriteFile(filepath.Join(dir, "weights.bin"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	path := externalModel(t, dir,
		pb.TensorExternalData("a", pb.Float, []int64{2, 2}, [][2]string{{"location", "weights.bin"}, {"offset", "0"}, {"length", "16"}}),
		pb.TensorExternalData("b", pb.Float, []int64{2}, [][2]string{{"location", "weights.bin"}, {"offset", "16"}}),
	)
	g, err := ParseGraphFile(path)
	if err != nil {
		t.Fatal(err)
	}
	wantF(t, g.Initializers["a"], []int{2, 2}, 1, 2, 3, 4)
	wantF(t, g.Initializers["b"], []int{2}, 10, 20)
	if _, err := Load(path); err != nil {
		t.Fatalf("Load must resolve external data as well: %v", err)
	}
}

func TestExternalTensorDataErrors(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "weights.bin"), floatBytes(1, 2), 0o644); err != nil {
		t.Fatal(err)
	}
	entries := func(kv ...string) [][2]string {
		var out [][2]string
		for i := 0; i+1 < len(kv); i += 2 {
			out = append(out, [2]string{kv[i], kv[i+1]})
		}
		return out
	}
	for name, kv := range map[string][]string{
		"absolute location": {"location", "/etc/passwd"},
		"escaping location": {"location", "../weights.bin"},
		"missing file":      {"location", "nothing.bin"},
		"negative offset":   {"location", "weights.bin", "offset", "-4"},
		"bad offset":        {"location", "weights.bin", "offset", "x"},
		"past the end":      {"location", "weights.bin", "offset", "4", "length", "40"},
		"no location":       {"offset", "0"},
	} {
		path := externalModel(t, dir, pb.TensorExternalData("a", pb.Float, []int64{2}, entries(kv...)),
			pb.TensorFloats("b", []int64{2}, []float32{1, 2}))
		if _, err := ParseGraphFile(path); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	tensor := pb.TensorExternalData("a", pb.Float, []int64{2}, entries("location", "weights.bin"))
	if _, err := ParseModel(pb.Model(pb.Graph(nil, [][]byte{tensor}, nil, nil))); err == nil || !strings.Contains(err.Error(), "file") {
		t.Errorf("a model parsed from memory cannot resolve external data: %v", err)
	}
	if _, err := ParseGraphFile(externalModel(t, dir, tensor, pb.TensorFloats("b", []int64{2}, []float32{1, 2}))); err != nil {
		t.Errorf("without offset and length the whole file is read: %v", err)
	}
}
