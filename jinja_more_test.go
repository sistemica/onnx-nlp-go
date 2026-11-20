package onnxnlp

import (
	"math"
	"strings"
	"testing"
	"time"
)

func TestJinjaValueHelpers(t *testing.T) {
	d := newJinjaDict()
	d.set("k", int64(1))
	loop := &jinjaLoop{items: []any{1, 2}}
	values := []any{nil, true, int64(1), 1.5, "s", []any{}, d, jinjaUndefined{"u"}, jinjaFunc(nil), loop, &jinjaNamespace{}}
	names := []string{"none", "bool", "int", "float", "str", "list", "dict", "undefined", "object", "object", "object"}
	for i, v := range values {
		if got := typeName(v); got != names[i] {
			t.Errorf("typeName(%v) = %s, want %s", v, got, names[i])
		}
	}
	for v, want := range map[any]bool{nil: false, true: true, false: false, int64(0): false, int64(2): true, 0.0: false, 0.5: true, "": false, "x": true} {
		if truthy(v) != want {
			t.Errorf("truthy(%v) != %v", v, want)
		}
	}
	if truthy([]any{}) || !truthy([]any{1}) || truthy(newJinjaDict()) || !truthy(d) || truthy(jinjaUndefined{}) || !truthy(loop) {
		t.Error("truthiness of containers")
	}
	if _, ok := loop.attr("nosuch"); ok {
		t.Error("an unknown loop attribute must be reported")
	}

	for _, tc := range []struct {
		v    any
		want string
	}{
		{nil, "None"}, {true, "True"}, {false, "False"}, {int64(7), "7"}, {-0.5, "-0.5"}, {1e16, "1e+16"}, {1e-5, "1e-05"}, {math.NaN(), "nan"}, {math.Inf(1), "inf"},
		{math.Inf(-1), "-inf"}, {100.0, "100.0"}, {"it's", `"it's"`}, {"a'\"", `'a\'"'`}, {"a\nb\rc\td\x01\x7f", `'a\nb\rc\td\x01\x7f'`}, {"back\\slash", `'back\\slash'`},
		{[]any{int64(1), "a"}, "[1, 'a']"}, {d, "{'k': 1}"}, {jinjaUndefined{}, ""}, {jinjaFunc(nil), "<object>"},
	} {
		if got := pyRepr(tc.v); got != tc.want {
			t.Errorf("pyRepr(%v) = %s, want %s", tc.v, got, tc.want)
		}
	}
	if printValue("raw") != "raw" || printValue(jinjaUndefined{}) != "" || printValue(nil) != "None" {
		t.Error("printValue")
	}

	json := func(v any, indent int) string {
		out, err := jsonDumps(v, indent)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	if got := json([]any{math.NaN(), math.Inf(1), math.Inf(-1), 1.5, nil, jinjaUndefined{}}, -1); got != "[NaN, Infinity, -Infinity, 1.5, null, null]" {
		t.Errorf("special floats: %s", got)
	}
	if got := json("\\\n\r\t\b\f\x01\"é", -1); got != `"\\\n\r\t\b\f\u0001\"é"` {
		t.Errorf("escapes: %s", got)
	}
	for _, bad := range []any{jinjaFunc(nil), []any{jinjaFunc(nil)}, wrap(jinjaFunc(nil)), []any{1, wrap(jinjaFunc(nil))}} {
		if _, err := jsonDumps(bad, -1); err == nil {
			t.Errorf("%v is not JSON serializable", bad)
		}
	}

	same := []struct {
		a, b any
		want bool
	}{
		{1.5, 1.5, true}, {int64(2), 2.0, true}, {true, int64(1), true}, {int64(1), "1", false}, {nil, nil, true}, {nil, int64(0), false}, {"a", int64(1), false}, {"a", "a", true},
		{[]any{int64(1)}, []any{int64(1)}, true}, {[]any{int64(1)}, []any{int64(2)}, false}, {[]any{int64(1)}, []any{int64(1), int64(2)}, false}, {[]any{int64(1)}, "x", false},
		{d, d, true}, {d, newJinjaDict(), false}, {d, wrap(int64(2)), false}, {d, "x", false}, {jinjaUndefined{}, jinjaUndefined{"other"}, true}, {jinjaUndefined{}, nil, false}, {loop, loop, false},
	}
	for _, tc := range same {
		if got := valuesEqual(tc.a, tc.b); got != tc.want {
			t.Errorf("valuesEqual(%v, %v) = %v", tc.a, tc.b, got)
		}
	}
	if _, ok := compareValues(int64(1), "a"); ok {
		t.Error("a number and a string cannot be ordered")
	}
	if _, ok := compareValues([]any{}, []any{}); ok {
		t.Error("lists cannot be ordered")
	}
	if c, ok := compareValues("a", "b"); !ok || c >= 0 {
		t.Error("strings are ordered")
	}
	if c, ok := compareValues(2.5, int64(2)); !ok || c <= 0 {
		t.Error("numbers are ordered")
	}
}

func wrap(v any) *jinjaDict {
	d := newJinjaDict()
	d.set("k", v)
	return d
}

func TestJinjaTokenizerEdges(t *testing.T) {
	if _, err := tokenizeJinja("'abc"); err == nil {
		t.Error("an unterminated string must fail")
	}
	if _, err := tokenizeJinja("99999999999999999999"); err == nil {
		t.Error("an integer that does not fit must fail")
	}
	if _, err := tokenizeJinja("1e999"); err == nil {
		t.Error("a float that does not fit must fail")
	}
	toks, err := tokenizeJinja(`'a\xzzé\`)
	if err == nil {
		t.Errorf("a string ending in a backslash is unterminated: %v", toks)
	}
	toks, err = tokenizeJinja(`'a\xzz\q'`)
	if err != nil || toks[0].s != `a\xzz\q` {
		t.Errorf("bad escapes stay as written: %q %v", toks[0].s, err)
	}
	toks, err = tokenizeJinja("1.5e+3 2e 7.x")
	if err != nil || toks[0].num != 1500.0 || toks[1].num != int64(2) || toks[3].num != int64(7) {
		t.Errorf("numbers: %+v %v", toks, err)
	}
}

func TestMoreTemplateFailures(t *testing.T) {
	for source, want := range map[string]string{
		"{{ 99999999999999999999 }}":          "out of range",
		"{{ x is eq [ }}":                     "unexpected",
		"{{ x[+] }}":                          "unexpected",
		"{{ (+) }}":                           "unexpected",
		"{{ (1 2) }}":                         "expected",
		"{{ [+] }}":                           "unexpected",
		"{{ {+: 1} }}":                        "unexpected",
		"{{ {'a': +} }}":                      "unexpected",
		"{% if 1 $ %}{% endif %}":             "unexpected character",
		"{% if x %}{% elif 1 $ %}{% endif %}": "unexpected character",
		"{% generation %}{{ + }}{% endgeneration %}":    "unexpected",
		"{% for x in y %}{{ + }}{% endfor %}":           "unexpected",
		"{% for x in y %}{% else %}{{ + }}{% endfor %}": "unexpected",
		"{% macro m() %}{{ + }}{% endmacro %}":          "unexpected",
		"{% if x %}{{ + }}{% endif %}":                  "unexpected",
		"{% set a. = 1 %}":                              "expected a name",
		"{% for a, b in [[1]] if true %}{% endfor %}":   "cannot unpack",
		"{{ 'abc'[::0] }}":                              "zero",
		"{{ xs[missing.x]() }}":                         "undefined",
		"{% for x in 5 %}{% endfor %}":                  "not iterable",
		"{{ nothing|last }}":                            "not iterable",
		"{{ 5|items }}":                                 "needs a dict",
		"{{ 5|tojson(indent='a') }}":                    "integer",
		"{{ [3, 'a']|sort }}":                           "cannot sort",
		"{{ xs|selectattr() }}":                         "attribute name",
		"{{ 5|map('upper') }}":                          "not iterable",
		"{{ xs|map() }}":                                "filter name",
		"{{ 5|selectattr('x') }}":                       "not iterable",
		"{{ xs|selectattr('x', 'nosuch') }}":            "unknown test",
		"{{ [d]|join(attribute='zz.y') }}":              "undefined",
		"{{ [d]|map(attribute='zz.y')|list }}":          "undefined",
		"{{ xs|map('nosuch') }}":                        "unknown filter",
		"{{ 5 is divisibleby }}":                        "divisibleby",
		"{{ 5 is divisibleby 0 }}":                      "divisibleby",
		"{{ 5 is in }}":                                 "needs one argument",
		"{{ 5 is eq }}":                                 "needs one argument",
		"{{ 5 is nosuch }}":                             "unknown test",
		"{{ strftime_now(1) }}":                         "needs a string",
		"{{ strftime_now() }}":                          "needs argument",
		"{{ namespace(1) }}":                            "namespace",
		"{{ dict(1) }}":                                 "dict",
		"{{ range('a') }}":                              "integers",
		"{{ range() }}":                                 "one to three",
		"{{ range(0, 5, 0) }}":                          "empty or too large",
		"{{ range(1000000000) }}":                       "too large",
		"{% macro m(a) %}{% endmacro %}{{ m(1, 2) }}":   "takes 1 arguments",
		"{% macro m(a) %}{% endmacro %}{{ m(b=2) }}":    "no argument",
		"{{ xs[1.5:] }}":                                "integers",
		"{{ 5[1:] }}":                                   "cannot be sliced",
		"{{ missing[1:] }}":                             "undefined",
		"{{ 5() }}":                                     "not callable",
		"{{ -'a' }}":                                    "unary",
		"{{ 1 // 0 }}":                                  "division",
		"{{ 1.5 % 0 }}":                                 "modulo",
		"{{ 'a'.split(1) }}":                            "needs a string",
		"{{ 'a'.replace(1, 2) }}":                       "needs a string",
		"{{ 'a'.replace('a') }}":                        "needs argument",
		"{{ 'a'.join() }}":                              "one argument",
		"{{ 'x'.strip(1) }}":                            "needs a string",
		"{{ 'x'.startswith(1) }}":                       "needs a string",
		"{{ d.get(1) }}":                                "needs a string",
		"{{ 'abc'.split('') }}":                         "empty separator",
		"{{ 1 in 'abc' }}":                              "needs a string",
		"{{ 1 in 5 }}":                                  "not iterable",
		"{{ [1]|sum('a') }}{{ ['a']|sum }}":             "unsupported",
		"{{ 'a'|abs }}":                                 "abs",
		"{{ x|replace(1, 2) }}{{ 5|replace(1) }}":       "needs",
		"{{ 5|join }}":                                  "not iterable",
		"{{ {}|nosuch }}":                               "unknown filter",
		"{{ 5|trim(1) }}":                               "needs a string",
		"{{ 1 < 'a' }}":                                 "not supported",
		"{{ {'a': 1}|tojson(indent=none) }}{{ nothing|length }}": "no length",
		"{% set x = 1 %}{% set x.a = 2 %}":                       "cannot assign",
	} {
		_, err := render(t, source, testVars())
		if err == nil {
			t.Errorf("%q: expected an error mentioning %q", source, want)
		} else if !strings.Contains(err.Error(), want) {
			t.Errorf("%q: error %q does not mention %q", source, err, want)
		}
	}
}

func TestStrftimeZonesAndRealClock(t *testing.T) {
	moment := time.Date(2024, 7, 5, 14, 3, 9, 0, time.UTC)
	if got := strftime(moment, "%Z %z"); got != "UTC +0000" {
		t.Errorf("zones: %q", got)
	}
	got, err := render(t, "{{ strftime_now('%Y-%m') }}", nil)
	if err != nil || got != time.Now().Format("2006-01") {
		t.Errorf("strftime_now = %q %v", got, err)
	}
}

func TestDictAndLoopHelpers(t *testing.T) {
	out, err := render(t, "{{ dict(b=2, a=1)|tojson }}{{ dict(d, c=3)|tojson }}{% for x in xs %}{{ loop.previtem }}{{ loop.nextitem }}{% endfor %}", testVars())
	if err != nil || out != `{"a": 1, "b": 2}{"a": 1, "c": 3}21` {
		t.Errorf("%q %v", out, err)
	}
}
