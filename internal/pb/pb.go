// Package pb writes the subset of the ONNX protobuf wire format that the tests need.
package pb

import (
	"encoding/binary"
	"math"
)

const (
	Float = 1
	Int32 = 6
	Int64 = 7
	Bool  = 9
	Half  = 10
)

func Varint(v uint64) []byte {
	var out []byte
	for v >= 0x80 {
		out = append(out, byte(v)|0x80)
		v >>= 7
	}
	return append(out, byte(v))
}

func key(num, wire int) []byte { return Varint(uint64(num<<3 | wire)) }

func VarintField(num int, v uint64) []byte { return append(key(num, 0), Varint(v)...) }

func BytesField(num int, b []byte) []byte {
	return append(append(key(num, 2), Varint(uint64(len(b)))...), b...)
}

func StringField(num int, s string) []byte { return BytesField(num, []byte(s)) }

func Fixed32Field(num int, v float32) []byte {
	out := key(num, 5)
	return binary.LittleEndian.AppendUint32(out, math.Float32bits(v))
}

func Fixed64Field(num int, v uint64) []byte {
	return binary.LittleEndian.AppendUint64(key(num, 1), v)
}

func PackedVarints(num int, vs []int64) []byte {
	var body []byte
	for _, v := range vs {
		body = append(body, Varint(uint64(v))...)
	}
	return BytesField(num, body)
}

func cat(parts ...[]byte) []byte {
	var out []byte
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

func tensorHeader(name string, dtype int, dims []int64) []byte {
	return cat(PackedVarints(1, dims), VarintField(2, uint64(dtype)), StringField(8, name))
}

func TensorRaw(name string, dtype int, dims []int64, raw []byte) []byte {
	return cat(tensorHeader(name, dtype, dims), BytesField(9, raw))
}

func TensorFloats(name string, dims []int64, vals []float32) []byte {
	raw := make([]byte, 0, 4*len(vals))
	for _, v := range vals {
		raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(v))
	}
	return TensorRaw(name, Float, dims, raw)
}

func TensorFloatData(name string, dims []int64, vals []float32) []byte {
	var packed []byte
	for _, v := range vals {
		packed = binary.LittleEndian.AppendUint32(packed, math.Float32bits(v))
	}
	return cat(tensorHeader(name, Float, dims), BytesField(4, packed))
}

func TensorInt64s(name string, dims []int64, vals []int64) []byte {
	raw := make([]byte, 0, 8*len(vals))
	for _, v := range vals {
		raw = binary.LittleEndian.AppendUint64(raw, uint64(v))
	}
	return TensorRaw(name, Int64, dims, raw)
}

func TensorInt64Data(name string, dims []int64, vals []int64) []byte {
	return cat(tensorHeader(name, Int64, dims), PackedVarints(7, vals))
}

func TensorInt32s(name string, dims []int64, vals []int32) []byte {
	raw := make([]byte, 0, 4*len(vals))
	for _, v := range vals {
		raw = binary.LittleEndian.AppendUint32(raw, uint32(v))
	}
	return TensorRaw(name, Int32, dims, raw)
}

func TensorInt32Data(name string, dims []int64, vals []int64) []byte {
	return cat(tensorHeader(name, Int32, dims), PackedVarints(5, vals))
}

func TensorBools(name string, dims []int64, vals []bool) []byte {
	raw := make([]byte, len(vals))
	for i, v := range vals {
		if v {
			raw[i] = 1
		}
	}
	return TensorRaw(name, Bool, dims, raw)
}

func TensorBoolData(name string, dims []int64, vals []int64) []byte {
	return cat(tensorHeader(name, Bool, dims), PackedVarints(5, vals))
}

func TensorExternal(name string) []byte {
	return cat(tensorHeader(name, Float, []int64{1}), BytesField(13, []byte("location")))
}

func AttrInt(name string, v int64) []byte {
	return cat(StringField(1, name), VarintField(3, uint64(v)))
}

func AttrFloat(name string, v float32) []byte {
	return cat(StringField(1, name), Fixed32Field(2, v))
}

func AttrString(name, v string) []byte { return cat(StringField(1, name), StringField(4, v)) }

func AttrInts(name string, vs ...int64) []byte {
	return cat(StringField(1, name), PackedVarints(8, vs))
}

func AttrIntsUnpacked(name string, vs ...int64) []byte {
	out := StringField(1, name)
	for _, v := range vs {
		out = append(out, VarintField(8, uint64(v))...)
	}
	return out
}

func AttrFloats(name string, vs ...float32) []byte {
	var packed []byte
	for _, v := range vs {
		packed = binary.LittleEndian.AppendUint32(packed, math.Float32bits(v))
	}
	return cat(StringField(1, name), BytesField(7, packed))
}

func AttrFloatsUnpacked(name string, vs ...float32) []byte {
	out := StringField(1, name)
	for _, v := range vs {
		out = append(out, Fixed32Field(7, v)...)
	}
	return out
}

func AttrTensor(name string, tensor []byte) []byte {
	return cat(StringField(1, name), BytesField(5, tensor))
}

func AttrGraph(name string, graph []byte) []byte {
	return cat(StringField(1, name), BytesField(6, graph))
}

func Node(op string, in, out []string, attrs ...[]byte) []byte {
	var b []byte
	for _, s := range in {
		b = append(b, StringField(1, s)...)
	}
	for _, s := range out {
		b = append(b, StringField(2, s)...)
	}
	b = append(b, StringField(3, op+"_"+firstOr(out))...)
	b = append(b, StringField(4, op)...)
	for _, a := range attrs {
		b = append(b, BytesField(5, a)...)
	}
	return b
}

func NodeInDomain(op, domain string, in, out []string) []byte {
	return append(Node(op, in, out), StringField(7, domain)...)
}

func firstOr(s []string) string {
	if len(s) == 0 {
		return ""
	}
	return s[0]
}

func ValueInfo(name string) []byte { return StringField(1, name) }

// Graph assembles a GraphProto. Initializer names are also listed as inputs, as ONNX exporters do.
func Graph(nodes [][]byte, initializers [][]byte, inputs, outputs []string) []byte {
	b := StringField(2, "g")
	for _, n := range nodes {
		b = append(b, BytesField(1, n)...)
	}
	for _, t := range initializers {
		b = append(b, BytesField(5, t)...)
	}
	for _, in := range inputs {
		b = append(b, BytesField(11, ValueInfo(in))...)
	}
	for _, out := range outputs {
		b = append(b, BytesField(12, ValueInfo(out))...)
	}
	return b
}

func Model(graph []byte) []byte {
	return cat(VarintField(1, 8), StringField(2, "test"), BytesField(7, graph))
}
