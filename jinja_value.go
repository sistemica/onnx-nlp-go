package onnxnlp

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Template values are nil (none), bool, int64, float64, string, []any, *jinjaDict, *jinjaNamespace,
// *jinjaLoop, jinjaFunc and jinjaUndefined.

type jinjaUndefined struct{ name string }

type jinjaFunc func(args []any, kw map[string]any) (any, error)

type jinjaDict struct {
	keys []string
	m    map[string]any
}

func newJinjaDict() *jinjaDict { return &jinjaDict{m: map[string]any{}} }

func (d *jinjaDict) set(k string, v any) {
	if _, ok := d.m[k]; !ok {
		d.keys = append(d.keys, k)
	}
	d.m[k] = v
}

type jinjaNamespace struct{ m map[string]any }

type jinjaLoop struct {
	index0 int
	items  []any
}

func (l *jinjaLoop) attr(name string) (any, bool) {
	n := len(l.items)
	switch name {
	case "index":
		return int64(l.index0 + 1), true
	case "index0":
		return int64(l.index0), true
	case "revindex":
		return int64(n - l.index0), true
	case "revindex0":
		return int64(n - l.index0 - 1), true
	case "first":
		return l.index0 == 0, true
	case "last":
		return l.index0 == n-1, true
	case "length":
		return int64(n), true
	case "previtem":
		if l.index0 == 0 {
			return jinjaUndefined{"previtem"}, true
		}
		return l.items[l.index0-1], true
	case "nextitem":
		if l.index0 == n-1 {
			return jinjaUndefined{"nextitem"}, true
		}
		return l.items[l.index0+1], true
	}
	return nil, false
}

func truthy(v any) bool {
	switch x := v.(type) {
	case nil, jinjaUndefined:
		return false
	case bool:
		return x
	case int64:
		return x != 0
	case float64:
		return x != 0
	case string:
		return x != ""
	case []any:
		return len(x) > 0
	case *jinjaDict:
		return len(x.keys) > 0
	}
	return true
}

func typeName(v any) string {
	switch v.(type) {
	case nil:
		return "none"
	case bool:
		return "bool"
	case int64:
		return "int"
	case float64:
		return "float"
	case string:
		return "str"
	case []any:
		return "list"
	case *jinjaDict:
		return "dict"
	case jinjaUndefined:
		return "undefined"
	}
	return "object"
}

// printValue is what {{ value }} prints.
func printValue(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case jinjaUndefined:
		return ""
	}
	return pyRepr(v)
}

// pyRepr formats a value the way Python's repr does.
func pyRepr(v any) string {
	switch x := v.(type) {
	case nil:
		return "None"
	case bool:
		if x {
			return "True"
		}
		return "False"
	case int64:
		return strconv.FormatInt(x, 10)
	case float64:
		return pyFloat(x)
	case string:
		return pyQuote(x)
	case []any:
		parts := make([]string, len(x))
		for i, e := range x {
			parts[i] = pyRepr(e)
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case *jinjaDict:
		parts := make([]string, len(x.keys))
		for i, k := range x.keys {
			parts[i] = pyQuote(k) + ": " + pyRepr(x.m[k])
		}
		return "{" + strings.Join(parts, ", ") + "}"
	case jinjaUndefined:
		return ""
	}
	return fmt.Sprintf("<%s>", typeName(v))
}

func pyQuote(s string) string {
	quote := byte('\'')
	if strings.Contains(s, "'") && !strings.Contains(s, `"`) {
		quote = '"'
	}
	var b strings.Builder
	b.WriteByte(quote)
	for _, r := range s {
		switch {
		case r == rune(quote) || r == '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r < 0x20 || r == 0x7f:
			fmt.Fprintf(&b, `\x%02x`, r)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte(quote)
	return b.String()
}

func pyFloat(f float64) string {
	switch {
	case math.IsNaN(f):
		return "nan"
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	}
	sci := strconv.FormatFloat(f, 'e', -1, 64)
	exp, _ := strconv.Atoi(sci[strings.IndexByte(sci, 'e')+1:])
	if exp < -4 || exp >= 16 {
		return sci
	}
	s := strconv.FormatFloat(f, 'f', -1, 64)
	if !strings.Contains(s, ".") {
		s += ".0"
	}
	return s
}

// jsonDumps formats v like Python's json.dumps with ensure_ascii=False; indent < 0 means one line.
func jsonDumps(v any, indent int) (string, error) {
	var b strings.Builder
	if err := writeJSON(&b, v, indent, 0); err != nil {
		return "", err
	}
	return b.String(), nil
}

func writeJSON(b *strings.Builder, v any, indent, depth int) error {
	newline := func(level int) {
		if indent >= 0 {
			b.WriteString("\n" + strings.Repeat(" ", indent*level))
		}
	}
	separator := ", "
	if indent >= 0 {
		separator = ","
	}
	switch x := v.(type) {
	case nil, jinjaUndefined:
		b.WriteString("null")
	case bool:
		b.WriteString(strconv.FormatBool(x))
	case int64:
		b.WriteString(strconv.FormatInt(x, 10))
	case float64:
		switch {
		case math.IsNaN(x):
			b.WriteString("NaN")
		case math.IsInf(x, 1):
			b.WriteString("Infinity")
		case math.IsInf(x, -1):
			b.WriteString("-Infinity")
		default:
			b.WriteString(pyFloat(x))
		}
	case string:
		writeJSONString(b, x)
	case []any:
		if len(x) == 0 {
			b.WriteString("[]")
			return nil
		}
		b.WriteByte('[')
		for i, e := range x {
			if i > 0 {
				b.WriteString(separator)
			}
			newline(depth + 1)
			if err := writeJSON(b, e, indent, depth+1); err != nil {
				return err
			}
		}
		newline(depth)
		b.WriteByte(']')
	case *jinjaDict:
		if len(x.keys) == 0 {
			b.WriteString("{}")
			return nil
		}
		b.WriteByte('{')
		for i, k := range x.keys {
			if i > 0 {
				b.WriteString(separator)
			}
			newline(depth + 1)
			writeJSONString(b, k)
			b.WriteString(": ")
			if err := writeJSON(b, x.m[k], indent, depth+1); err != nil {
				return err
			}
		}
		newline(depth)
		b.WriteByte('}')
	default:
		return fmt.Errorf("a %s is not JSON serializable", typeName(v))
	}
	return nil
}

func writeJSONString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch {
		case r == '"':
			b.WriteString(`\"`)
		case r == '\\':
			b.WriteString(`\\`)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r == '\b':
			b.WriteString(`\b`)
		case r == '\f':
			b.WriteString(`\f`)
		case r < 0x20:
			fmt.Fprintf(b, `\u%04x`, r)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
}

func asNumber(v any) (float64, bool, bool) {
	switch x := v.(type) {
	case bool:
		if x {
			return 1, true, true
		}
		return 0, true, true
	case int64:
		return float64(x), true, true
	case float64:
		return x, false, true
	}
	return 0, false, false
}

func valuesEqual(a, b any) bool {
	if x, xInt, ok := asNumber(a); ok {
		if y, yInt, ok := asNumber(b); ok {
			if xInt && yInt {
				return int64Of(a) == int64Of(b)
			}
			return x == y
		}
		return false
	}
	switch x := a.(type) {
	case nil:
		return b == nil
	case string:
		y, ok := b.(string)
		return ok && x == y
	case []any:
		y, ok := b.([]any)
		if !ok || len(x) != len(y) {
			return false
		}
		for i := range x {
			if !valuesEqual(x[i], y[i]) {
				return false
			}
		}
		return true
	case *jinjaDict:
		y, ok := b.(*jinjaDict)
		if !ok || len(x.keys) != len(y.keys) {
			return false
		}
		for _, k := range x.keys {
			other, ok := y.m[k]
			if !ok || !valuesEqual(x.m[k], other) {
				return false
			}
		}
		return true
	case jinjaUndefined:
		_, ok := b.(jinjaUndefined)
		return ok
	}
	return false
}

func int64Of(v any) int64 {
	switch x := v.(type) {
	case bool:
		if x {
			return 1
		}
		return 0
	case int64:
		return x
	}
	f, _ := v.(float64)
	return int64(f)
}

// compareValues orders numbers or strings; ok is false for other combinations.
func compareValues(a, b any) (int, bool) {
	if x, _, ok := asNumber(a); ok {
		if y, _, ok := asNumber(b); ok {
			switch {
			case x < y:
				return -1, true
			case x > y:
				return 1, true
			}
			return 0, true
		}
		return 0, false
	}
	x, okX := a.(string)
	y, okY := b.(string)
	if okX && okY {
		return strings.Compare(x, y), true
	}
	return 0, false
}

func runeLen(s string) int { return utf8.RuneCountInString(s) }
