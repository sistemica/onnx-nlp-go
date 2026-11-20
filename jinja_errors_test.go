package onnxnlp

import (
	"strings"
	"testing"
)

// failing is an expression that always raises: attribute access on an undefined name.
const failing = "missing.x"

func render(t *testing.T, source string, vars map[string]any) (string, error) {
	t.Helper()
	tpl, err := parseJinja(source)
	if err != nil {
		return "", err
	}
	return tpl.render(vars)
}

func testVars() map[string]any {
	d := newJinjaDict()
	d.set("a", int64(1))
	return map[string]any{"xs": []any{int64(1), int64(2)}, "d": d, "yes": true, "no": false, "nothing": nil}
}

func TestErrorsPropagateFromEveryPosition(t *testing.T) {
	for _, pattern := range []string{
		"{{ E }}", "{{ E.y }}", "{{ E[0] }}", "{{ xs[E] }}", "{{ E[1:] }}", "{{ xs[E:] }}", "{{ xs[:E] }}", "{{ xs[::E] }}", "{{ xs[1:2:E] }}",
		"{{ E() }}", "{{ d.get(E) }}", "{{ d.get(a=E) }}", "{{ E.f() }}", "{{ range(E) }}", "{{ -E }}", "{{ not E }}", "{{ E + 1 }}", "{{ 1 + E }}",
		"{{ E and 1 }}", "{{ 1 and E }}", "{{ 0 or E }}", "{{ E or 1 }}", "{{ E == 1 }}", "{{ 1 == E }}", "{{ 1 < 2 < E }}", "{{ E in xs }}", "{{ 1 in E }}",
		"{{ E if yes else 1 }}", "{{ 1 if E else 2 }}", "{{ 1 if no else E }}", "{{ [E] }}", "{{ {E: 1} }}", "{{ {'a': E} }}", "{{ E|upper }}",
		"{{ 1|default(E) }}", "{{ 1|default(x=E) }}", "{{ E is defined }}", "{{ 1 is eq(E) }}", "{{ 1 is in(E) }}", "{{ (E)|tojson }}",
		"{% if E %}{% endif %}", "{% if no %}{% elif E %}{% endif %}", "{% if yes %}{{ E }}{% endif %}", "{% if no %}{% else %}{{ E }}{% endif %}",
		"{% for x in E %}{% endfor %}", "{% for x in xs if E %}{% endfor %}", "{% for x in xs %}{{ E }}{% endfor %}", "{% for x in [] %}{% else %}{{ E }}{% endfor %}",
		"{% for x in xs %}{% if E %}{% endif %}{% endfor %}", "{% set a = E %}", "{% set ns = namespace() %}{% set ns.a = E %}", "{% set missing.a = 1 %}",
		"{% set x %}{{ E }}{% endset %}", "{% macro m() %}{{ E }}{% endmacro %}{{ m() }}", "{% macro m(a=E) %}{% endmacro %}", "{% macro m(a) %}{% endmacro %}{{ m(E) }}",
		"{% macro m(a) %}{% endmacro %}{{ m(a=E) }}", "{% generation %}{{ E }}{% endgeneration %}", "{{ xs|map(attribute=E) }}", "{{ xs|selectattr(E) }}",
		"{{ xs|join(E) }}", "{{ [E]|first }}", "{{ E|map('upper') }}", "{{ xs|selectattr('x.y.z')|list }}", "{{ E.items() }}",
		"{{ (1, E) }}", "{{ [1, 2][E] }}", "{{ E[E] }}", "{{ d[E] }}", "{{ d['a'][E] }}", "{{ ('a')[E:E] }}",
	} {
		source := strings.ReplaceAll(pattern, "E", failing)
		if _, err := render(t, source, testVars()); err == nil {
			t.Errorf("%s: expected an error", pattern)
		}
	}
}

func TestErrorsAreReportedWithTheirLine(t *testing.T) {
	_, err := render(t, "a\nb\n{{ missing.x }}", nil)
	if err == nil || !strings.Contains(err.Error(), "line 3") {
		t.Fatalf("want the line of the failing tag, got %v", err)
	}
	_, err = render(t, "a\n{{ raise_exception('stop') }}", nil)
	if err == nil || err.Error() != "stop" {
		t.Fatalf("raise_exception must not be prefixed: %v", err)
	}
	_, err = render(t, "{{ raise_exception() }}", nil)
	if err == nil || err.Error() != "template raised an exception" {
		t.Fatalf("raise_exception without a message: %v", err)
	}
}

func TestParseErrorsNameTheProblem(t *testing.T) {
	for source, want := range map[string]string{
		"{% if x %}":                            "never closed",
		"{% for x in y %}":                      "never closed",
		"{% for x in y %}{% else %}":            "not closed",
		"{% if x %}{% else %}":                  "not closed",
		"{% endif %}":                           "unexpected",
		"{% foo %}":                             "unsupported tag",
		"{% %}":                                 "malformed",
		"{% 1 %}":                               "malformed",
		"{{ 1 +* 2 }}":                          "unexpected",
		"{{ 1 2 }}":                             "unexpected",
		"{{ [1, 2 }}":                           "expected",
		"{{ (1, 2 }}":                           "expected",
		"{{ {'a' 1} }}":                         "expected",
		"{{ {'a': 1 'b': 2} }}":                 "expected",
		"{{ f(1 2) }}":                          "expected",
		"{{ 'open }}":                           "unterminated",
		"{{ x":                                  "unterminated",
		"{# x":                                  "unterminated",
		"{{ x.1 }}":                             "expected a name",
		"{{ x|1 }}":                             "expected a name",
		"{{ x|a.1 }}":                           "expected a name",
		"{{ x|f(1 2) }}":                        "expected",
		"{{ x is 1 }}":                          "expected a name",
		"{{ x is f(1 2) }}":                     "expected",
		"{{ x[] }}":                             "empty",
		"{{ x[1 2] }}":                          "expected",
		"{{ x[1:2:3:4] }}":                      "colons",
		"{{ 1 $ 2 }}":                           "unexpected character",
		"{% set 1 = 2 %}":                       "cannot assign",
		"{% set a.b.c = 2 %}":                   "expected",
		"{% set a b %}":                         "expected",
		"{% set a = %}":                         "unexpected",
		"{% set a = 1 2 %}":                     "unexpected",
		"{% set x %}a":                          "not closed",
		"{% if %}{% endif %}":                   "unexpected",
		"{% if x y %}{% endif %}":               "unexpected",
		"{% elif x %}":                          "unexpected",
		"{% if x %}{% elif %}{% endif %}":       "unexpected",
		"{% if x %}{% else y %}{% endif %}":     "unexpected",
		"{% for %}{% endfor %}":                 "name",
		"{% for x y %}{% endfor %}":             "\"in\"",
		"{% for (x in y %}{% endfor %}":         "expected",
		"{% for x in %}{% endfor %}":            "unexpected",
		"{% for x in y if %}{% endfor %}":       "unexpected",
		"{% for x in y z %}{% endfor %}":        "unexpected",
		"{% break 1 %}":                         "unexpected",
		"{% macro %}{% endmacro %}":             "name",
		"{% macro m %}{% endmacro %}":           "\"(\"",
		"{% macro m(a b) %}{% endmacro %}":      "expected",
		"{% macro m(1) %}{% endmacro %}":        "name",
		"{% macro m(a=) %}{% endmacro %}":       "unexpected",
		"{% macro m() x %}{% endmacro %}":       "unexpected",
		"{% macro m() %}":                       "not closed",
		"{% generation %}":                      "not closed",
		"{{ 5 if }}":                            "unexpected",
		"{{ 5 if x else }}":                     "unexpected",
		"{{ x|f( }}":                            "unexpected",
		"{{ x if y z }}":                        "unexpected",
		"{{ not }}":                             "unexpected",
		"{{ - }}":                               "unexpected",
		"{{ x.y( }}":                            "unexpected",
		"{{ 1 < }}":                             "unexpected",
		"{{ 1 not 2 }}":                         "unexpected",
		"{{ f(a=) }}":                           "unexpected",
		"{% if x %}{% for y in z %}{% endif %}": "never closed",
	} {
		_, err := parseJinja(source)
		if err == nil {
			t.Errorf("%q: expected a parse error mentioning %q", source, want)
		} else if !strings.Contains(err.Error(), want) {
			t.Errorf("%q: error %q does not mention %q", source, err, want)
		}
	}
}
