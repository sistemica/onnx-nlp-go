package onnxgo

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

type Attr struct {
	I      int64
	F      float32
	S      string
	Ints   []int64
	Floats []float32
	T      *Tensor
	G      *Graph
}

type Node struct {
	Name   string
	Op     string
	Domain string
	In     []string
	Out    []string
	Attr   map[string]*Attr
}

type Graph struct {
	Name         string
	Nodes        []*Node
	Initializers map[string]*Tensor
	Inputs       []string
	Outputs      []string
}

func (n *Node) Int(name string, def int64) int64 {
	if a, ok := n.Attr[name]; ok {
		return a.I
	}
	return def
}

func (n *Node) Float(name string, def float32) float32 {
	if a, ok := n.Attr[name]; ok {
		return a.F
	}
	return def
}

func (n *Node) String(name, def string) string {
	if a, ok := n.Attr[name]; ok {
		return a.S
	}
	return def
}

func (n *Node) Ints(name string) ([]int64, bool) {
	a, ok := n.Attr[name]
	if !ok {
		return nil, false
	}
	return a.Ints, true
}

func ParseGraphFile(path string) (*Graph, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseModel(b)
}

func ParseModel(b []byte) (*Graph, error) {
	var g *Graph
	err := forEachField(b, func(f field) error {
		if f.num != 7 {
			return nil
		}
		var err error
		g, err = parseGraph(f.data)
		return err
	})
	if err != nil {
		return nil, err
	}
	if g == nil {
		return nil, fmt.Errorf("onnx: model has no graph")
	}
	return g, nil
}

func parseGraph(b []byte) (*Graph, error) {
	g := &Graph{Initializers: map[string]*Tensor{}}
	var inputs []string
	err := forEachField(b, func(f field) error {
		switch f.num {
		case 1:
			n, err := parseNode(f.data)
			if err != nil {
				return err
			}
			g.Nodes = append(g.Nodes, n)
		case 2:
			g.Name = string(f.data)
		case 5:
			name, t, err := parseTensor(f.data)
			if err != nil {
				return err
			}
			t.static = true
			t.pack = &packCache{}
			g.Initializers[name] = t
		case 11:
			inputs = append(inputs, valueInfoName(f.data))
		case 12:
			g.Outputs = append(g.Outputs, valueInfoName(f.data))
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	for _, name := range inputs {
		if _, isInit := g.Initializers[name]; !isInit {
			g.Inputs = append(g.Inputs, name)
		}
	}
	return g, nil
}

func valueInfoName(b []byte) string {
	var name string
	forEachField(b, func(f field) error {
		if f.num == 1 {
			name = string(f.data)
		}
		return nil
	})
	return name
}

func parseNode(b []byte) (*Node, error) {
	n := &Node{Attr: map[string]*Attr{}}
	err := forEachField(b, func(f field) error {
		switch f.num {
		case 1:
			n.In = append(n.In, string(f.data))
		case 2:
			n.Out = append(n.Out, string(f.data))
		case 3:
			n.Name = string(f.data)
		case 4:
			n.Op = string(f.data)
		case 5:
			name, a, err := parseAttr(f.data)
			if err != nil {
				return err
			}
			n.Attr[name] = a
		case 7:
			n.Domain = string(f.data)
		}
		return nil
	})
	return n, err
}

func parseAttr(b []byte) (string, *Attr, error) {
	a := &Attr{}
	var name string
	err := forEachField(b, func(f field) error {
		var err error
		switch f.num {
		case 1:
			name = string(f.data)
		case 2:
			a.F = f.float32()
		case 3:
			a.I = int64(f.v)
		case 4:
			a.S = string(f.data)
		case 5:
			_, a.T, err = parseTensor(f.data)
		case 6:
			a.G, err = parseGraph(f.data)
		case 7:
			a.Floats = appendFloat32s(a.Floats, f)
		case 8:
			a.Ints, err = appendVarints(a.Ints, f)
		}
		return err
	})
	return name, a, err
}

const (
	onnxFloat = 1
	onnxInt32 = 6
	onnxInt64 = 7
	onnxBool  = 9
)

func parseTensor(b []byte) (string, *Tensor, error) {
	var (
		name     string
		dims     []int64
		dtype    int64
		raw      []byte
		floats   []float32
		ints     []int64
		external bool
	)
	err := forEachField(b, func(f field) error {
		var err error
		switch f.num {
		case 1:
			dims, err = appendVarints(dims, f)
		case 2:
			dtype = int64(f.v)
		case 4:
			floats = appendFloat32s(floats, f)
		case 5, 7:
			ints, err = appendVarints(ints, f)
		case 8:
			name = string(f.data)
		case 9:
			raw = f.data
		case 13:
			external = true
		}
		return err
	})
	if err != nil {
		return "", nil, err
	}
	if external {
		return "", nil, fmt.Errorf("onnx: tensor %q uses external data", name)
	}
	shape := make([]int, len(dims))
	for i, d := range dims {
		shape[i] = int(d)
	}
	n := numel(shape)
	t, err := tensorFromProto(dtype, shape, n, raw, floats, ints)
	if err != nil {
		return "", nil, fmt.Errorf("tensor %q: %w", name, err)
	}
	return name, t, nil
}

func tensorFromProto(dtype int64, shape []int, n int, raw []byte, floats []float32, ints []int64) (*Tensor, error) {
	switch dtype {
	case onnxFloat:
		if raw != nil {
			if len(raw) != 4*n {
				return nil, fmt.Errorf("raw size %d for %d floats", len(raw), n)
			}
			data := make([]float32, n)
			for i := range data {
				data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
			}
			return NewFloat(shape, data), nil
		}
		return NewFloat(shape, floats), nil
	case onnxInt64:
		if raw != nil {
			data := make([]int64, n)
			for i := range data {
				data[i] = int64(binary.LittleEndian.Uint64(raw[8*i:]))
			}
			return NewInt(shape, data), nil
		}
		return NewInt(shape, ints), nil
	case onnxInt32:
		data := make([]int64, n)
		if raw != nil {
			for i := range data {
				data[i] = int64(int32(binary.LittleEndian.Uint32(raw[4*i:])))
			}
		} else {
			copy(data, ints)
		}
		return NewInt(shape, data), nil
	case onnxBool:
		t := zeros(Bool, shape)
		if raw != nil {
			for i := range t.I {
				t.I[i] = int64(raw[i])
			}
		} else {
			copy(t.I, ints)
		}
		return t, nil
	}
	return nil, fmt.Errorf("unsupported tensor data type %d", dtype)
}
