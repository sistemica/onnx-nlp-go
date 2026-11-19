package onnxnlp

import (
	"encoding/binary"
	"errors"
	"math"
)

var errTruncated = errors.New("onnx: truncated protobuf")

type field struct {
	num  int
	wire int
	v    uint64
	data []byte
}

func readVarint(b []byte) (uint64, int, error) {
	var v uint64
	for i := 0; i < len(b) && i < 10; i++ {
		v |= uint64(b[i]&0x7f) << (7 * uint(i))
		if b[i] < 0x80 {
			return v, i + 1, nil
		}
	}
	return 0, 0, errTruncated
}

func forEachField(b []byte, fn func(field) error) error {
	for len(b) > 0 {
		key, n, err := readVarint(b)
		if err != nil {
			return err
		}
		b = b[n:]
		f := field{num: int(key >> 3), wire: int(key & 7)}
		switch f.wire {
		case 0:
			f.v, n, err = readVarint(b)
			if err != nil {
				return err
			}
			b = b[n:]
		case 1:
			if len(b) < 8 {
				return errTruncated
			}
			f.v = binary.LittleEndian.Uint64(b)
			b = b[8:]
		case 2:
			var l uint64
			l, n, err = readVarint(b)
			if err != nil {
				return err
			}
			b = b[n:]
			if uint64(len(b)) < l {
				return errTruncated
			}
			f.data = b[:l]
			b = b[l:]
		case 5:
			if len(b) < 4 {
				return errTruncated
			}
			f.v = uint64(binary.LittleEndian.Uint32(b))
			b = b[4:]
		default:
			return errors.New("onnx: unsupported protobuf wire type")
		}
		if err := fn(f); err != nil {
			return err
		}
	}
	return nil
}

func (f field) float32() float32 { return math.Float32frombits(uint32(f.v)) }

func appendVarints(dst []int64, f field) ([]int64, error) {
	if f.wire == 0 {
		return append(dst, int64(f.v)), nil
	}
	b := f.data
	for len(b) > 0 {
		v, n, err := readVarint(b)
		if err != nil {
			return nil, err
		}
		dst = append(dst, int64(v))
		b = b[n:]
	}
	return dst, nil
}

func appendFloat32s(dst []float32, f field) []float32 {
	if f.wire == 5 {
		return append(dst, f.float32())
	}
	for b := f.data; len(b) >= 4; b = b[4:] {
		dst = append(dst, math.Float32frombits(binary.LittleEndian.Uint32(b)))
	}
	return dst
}
