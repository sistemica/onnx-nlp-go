package onnxnlp

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
)

var jinjaGlobals = map[string]any{
	"raise_exception": jinjaFunc(func(args []any, _ map[string]any) (any, error) {
		msg := "template raised an exception"
		if len(args) > 0 {
			msg = printValue(args[0])
		}
		return nil, &jinjaRaise{msg}
	}),
	"strftime_now": jinjaFunc(func(args []any, _ map[string]any) (any, error) {
		format, err := stringArg(args, 0, "strftime_now")
		return strftime(time.Now(), format), err
	}),
	"namespace": jinjaFunc(func(args []any, kw map[string]any) (any, error) {
		ns := &jinjaNamespace{m: map[string]any{}}
		if len(args) > 0 {
			d, ok := args[0].(*jinjaDict)
			if !ok {
				return nil, errors.New("namespace() takes a dict or keyword arguments")
			}
			for _, k := range d.keys {
				ns.m[k] = d.m[k]
			}
		}
		for k, v := range kw {
			ns.m[k] = v
		}
		return ns, nil
	}),
	"dict": jinjaFunc(func(args []any, kw map[string]any) (any, error) {
		d := newJinjaDict()
		if len(args) > 0 {
			src, ok := args[0].(*jinjaDict)
			if !ok {
				return nil, errors.New("dict() takes a dict or keyword arguments")
			}
			for _, k := range src.keys {
				d.set(k, src.m[k])
			}
		}
		for _, k := range sortedKeys(kw) {
			d.set(k, kw[k])
		}
		return d, nil
	}),
	"range": jinjaFunc(func(args []any, _ map[string]any) (any, error) {
		bounds := make([]int64, len(args))
		for i, a := range args {
			n, ok := a.(int64)
			if !ok {
				return nil, errors.New("range() takes integers")
			}
			bounds[i] = n
		}
		lo, hi, step := int64(0), int64(0), int64(1)
		switch len(bounds) {
		case 1:
			hi = bounds[0]
		case 2:
			lo, hi = bounds[0], bounds[1]
		case 3:
			lo, hi, step = bounds[0], bounds[1], bounds[2]
		default:
			return nil, errors.New("range() takes one to three arguments")
		}
		if step == 0 || (hi-lo)/step > 1<<20 {
			return nil, errors.New("range() is empty or too large")
		}
		out := []any{}
		for i := lo; (step > 0 && i < hi) || (step < 0 && i > hi); i += step {
			out = append(out, i)
		}
		return out, nil
	}),
}

func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func stringArg(args []any, i int, who string) (string, error) {
	if i >= len(args) {
		return "", fmt.Errorf("%s needs argument %d", who, i+1)
	}
	s, ok := args[i].(string)
	if !ok {
		return "", fmt.Errorf("%s needs a string, not %s", who, typeName(args[i]))
	}
	return s, nil
}

func intArg(args []any, i int, def int64) int64 {
	if i < len(args) {
		if n, ok := args[i].(int64); ok {
			return n
		}
	}
	return def
}

func strftime(t time.Time, format string) string {
	var b strings.Builder
	for i := 0; i < len(format); i++ {
		if format[i] != '%' || i+1 == len(format) {
			b.WriteByte(format[i])
			continue
		}
		i++
		switch format[i] {
		case 'Y':
			b.WriteString(t.Format("2006"))
		case 'y':
			b.WriteString(t.Format("06"))
		case 'm':
			b.WriteString(t.Format("01"))
		case 'd':
			b.WriteString(t.Format("02"))
		case 'e':
			b.WriteString(fmt.Sprintf("%2d", t.Day()))
		case 'H':
			b.WriteString(t.Format("15"))
		case 'I':
			b.WriteString(t.Format("03"))
		case 'M':
			b.WriteString(t.Format("04"))
		case 'S':
			b.WriteString(t.Format("05"))
		case 'p':
			b.WriteString(t.Format("PM"))
		case 'b':
			b.WriteString(t.Format("Jan"))
		case 'B':
			b.WriteString(t.Format("January"))
		case 'a':
			b.WriteString(t.Format("Mon"))
		case 'A':
			b.WriteString(t.Format("Monday"))
		case 'j':
			b.WriteString(fmt.Sprintf("%03d", t.YearDay()))
		case 'Z':
			b.WriteString(t.Format("MST"))
		case 'z':
			b.WriteString(t.Format("-0700"))
		case '%':
			b.WriteByte('%')
		default:
			b.WriteByte('%')
			b.WriteByte(format[i])
		}
	}
	return b.String()
}

func dictMethod(d *jinjaDict, name string) (jinjaFunc, bool) {
	switch name {
	case "items":
		return func([]any, map[string]any) (any, error) {
			out := make([]any, len(d.keys))
			for i, k := range d.keys {
				out[i] = []any{k, d.m[k]}
			}
			return out, nil
		}, true
	case "keys":
		return func([]any, map[string]any) (any, error) { return iterate(d) }, true
	case "values":
		return func([]any, map[string]any) (any, error) {
			out := make([]any, len(d.keys))
			for i, k := range d.keys {
				out[i] = d.m[k]
			}
			return out, nil
		}, true
	case "get":
		return func(args []any, _ map[string]any) (any, error) {
			k, err := stringArg(args, 0, "get")
			if err != nil {
				return nil, err
			}
			if v, ok := d.m[k]; ok {
				return v, nil
			}
			if len(args) > 1 {
				return args[1], nil
			}
			return nil, nil
		}, true
	}
	return nil, false
}

func stringMethod(s, name string) (jinjaFunc, bool) {
	simple := map[string]func(string) string{
		"upper": strings.ToUpper, "lower": strings.ToLower, "title": titleCase, "capitalize": capitalize,
	}
	if f, ok := simple[name]; ok {
		return func([]any, map[string]any) (any, error) { return f(s), nil }, true
	}
	switch name {
	case "strip", "lstrip", "rstrip":
		return func(args []any, _ map[string]any) (any, error) { return trimWith(s, name, args) }, true
	case "startswith", "endswith":
		return func(args []any, _ map[string]any) (any, error) {
			affixes := []string{}
			if len(args) > 0 {
				if list, ok := args[0].([]any); ok {
					for _, a := range list {
						affixes = append(affixes, printValue(a))
					}
				} else if str, ok := args[0].(string); ok {
					affixes = append(affixes, str)
				} else {
					return nil, fmt.Errorf("%s needs a string", name)
				}
			}
			for _, a := range affixes {
				if (name == "startswith" && strings.HasPrefix(s, a)) || (name == "endswith" && strings.HasSuffix(s, a)) {
					return true, nil
				}
			}
			return false, nil
		}, true
	case "split":
		return func(args []any, _ map[string]any) (any, error) { return splitString(s, args) }, true
	case "replace":
		return func(args []any, _ map[string]any) (any, error) { return replaceString(s, args) }, true
	case "join":
		return func(args []any, _ map[string]any) (any, error) {
			if len(args) != 1 {
				return nil, errors.New("join needs one argument")
			}
			return joinValues(args[0], s, nil)
		}, true
	}
	return nil, false
}

func titleCase(s string) string {
	out, startOfWord := []rune{}, true
	for _, r := range s {
		if unicode.IsLetter(r) {
			if startOfWord {
				r = unicode.ToUpper(r)
			} else {
				r = unicode.ToLower(r)
			}
			startOfWord = false
		} else {
			startOfWord = true
		}
		out = append(out, r)
	}
	return string(out)
}

func capitalize(s string) string {
	runes := []rune(strings.ToLower(s))
	if len(runes) > 0 {
		runes[0] = unicode.ToUpper(runes[0])
	}
	return string(runes)
}

func trimWith(s, how string, args []any) (any, error) {
	chars := ""
	if len(args) > 0 && args[0] != nil {
		c, err := stringArg(args, 0, how)
		if err != nil {
			return nil, err
		}
		chars = c
	}
	trimLeft := func(s string) string {
		if chars == "" {
			return strings.TrimLeftFunc(s, unicode.IsSpace)
		}
		return strings.TrimLeft(s, chars)
	}
	trimRight := func(s string) string {
		if chars == "" {
			return strings.TrimRightFunc(s, unicode.IsSpace)
		}
		return strings.TrimRight(s, chars)
	}
	switch how {
	case "lstrip":
		return trimLeft(s), nil
	case "rstrip":
		return trimRight(s), nil
	}
	return trimLeft(trimRight(s)), nil
}

func splitString(s string, args []any) (any, error) {
	limit := int(intArg(args, 1, -1))
	var parts []string
	if len(args) == 0 || args[0] == nil {
		parts = strings.Fields(s)
		if limit >= 0 && len(parts) > limit+1 {
			rest := strings.TrimLeftFunc(s, unicode.IsSpace)
			for i := 0; i < limit; i++ {
				rest = strings.TrimLeftFunc(strings.TrimLeftFunc(rest, func(r rune) bool { return !unicode.IsSpace(r) }), unicode.IsSpace)
			}
			parts = append(parts[:limit], rest)
		}
	} else {
		sep, err := stringArg(args, 0, "split")
		if err != nil {
			return nil, err
		}
		if sep == "" {
			return nil, errors.New("empty separator")
		}
		n := -1
		if limit >= 0 {
			n = limit + 1
		}
		parts = strings.SplitN(s, sep, n)
	}
	out := make([]any, len(parts))
	for i, p := range parts {
		out[i] = p
	}
	return out, nil
}

func replaceString(s string, args []any) (any, error) {
	old, err := stringArg(args, 0, "replace")
	if err != nil {
		return nil, err
	}
	repl, err := stringArg(args, 1, "replace")
	if err != nil {
		return nil, err
	}
	return strings.Replace(s, old, repl, int(intArg(args, 2, -1))), nil
}

func joinValues(list any, sep string, attribute any) (any, error) {
	items, err := iterate(list)
	if err != nil {
		return nil, err
	}
	parts := make([]string, len(items))
	for i, item := range items {
		if attribute != nil {
			if item, err = attributePath(item, printValue(attribute)); err != nil {
				return nil, err
			}
		}
		parts[i] = printValue(item)
	}
	return strings.Join(parts, sep), nil
}

// attributePath follows a dotted name such as "function.name" through dicts and objects.
func attributePath(v any, path string) (any, error) {
	var err error
	for _, part := range strings.Split(path, ".") {
		if d, ok := v.(*jinjaDict); ok {
			if got, found := d.m[part]; found {
				v = got
				continue
			}
			v = jinjaUndefined{part}
			continue
		}
		if v, err = getAttr(v, part); err != nil {
			return nil, err
		}
	}
	return v, nil
}

type jinjaFilter func(x any, args []any, kw map[string]any) (any, error)

func runFilter(name string, x any, args []any, kw map[string]any) (any, error) {
	f, ok := jinjaFilters[name]
	if !ok {
		return nil, fmt.Errorf("unknown filter %q", name)
	}
	return f(x, args, kw)
}

func lengthOf(x any) (int, error) {
	switch v := x.(type) {
	case string:
		return runeLen(v), nil
	case []any:
		return len(v), nil
	case *jinjaDict:
		return len(v.keys), nil
	case jinjaUndefined:
		return 0, nil
	}
	return 0, fmt.Errorf("a %s has no length", typeName(x))
}

func kwOr(kw map[string]any, name string, args []any, i int) any {
	if v, ok := kw[name]; ok {
		return v
	}
	if i < len(args) {
		return args[i]
	}
	return nil
}

var jinjaFilters map[string]jinjaFilter

func init() {
	length := func(x any, _ []any, _ map[string]any) (any, error) {
		n, err := lengthOf(x)
		return int64(n), err
	}
	defaultFilter := func(x any, args []any, kw map[string]any) (any, error) {
		fallback := kwOr(kw, "default_value", args, 0)
		if fallback == nil && len(args) == 0 {
			fallback = ""
		}
		if _, undefined := x.(jinjaUndefined); undefined || (truthy(kwOr(kw, "boolean", args, 1)) && !truthy(x)) {
			return fallback, nil
		}
		return x, nil
	}
	jinjaFilters = map[string]jinjaFilter{
		"length":  length,
		"count":   length,
		"default": defaultFilter,
		"d":       defaultFilter,
		"safe":    func(x any, _ []any, _ map[string]any) (any, error) { return x, nil },
		"string":  func(x any, _ []any, _ map[string]any) (any, error) { return printValue(x), nil },
		"upper":   func(x any, _ []any, _ map[string]any) (any, error) { return strings.ToUpper(printValue(x)), nil },
		"lower":   func(x any, _ []any, _ map[string]any) (any, error) { return strings.ToLower(printValue(x)), nil },
		"title":   func(x any, _ []any, _ map[string]any) (any, error) { return titleCase(printValue(x)), nil },
		"capitalize": func(x any, _ []any, _ map[string]any) (any, error) {
			return capitalize(printValue(x)), nil
		},
		"trim": func(x any, args []any, _ map[string]any) (any, error) { return trimWith(printValue(x), "strip", args) },
		"replace": func(x any, args []any, _ map[string]any) (any, error) {
			return replaceString(printValue(x), args)
		},
		"list": func(x any, _ []any, _ map[string]any) (any, error) {
			items, err := iterate(x)
			return append([]any{}, items...), err
		},
		"first": func(x any, _ []any, _ map[string]any) (any, error) {
			items, err := iterate(x)
			if err != nil || len(items) == 0 {
				return jinjaUndefined{"first"}, err
			}
			return items[0], nil
		},
		"last": func(x any, _ []any, _ map[string]any) (any, error) {
			items, err := iterate(x)
			if err != nil || len(items) == 0 {
				return jinjaUndefined{"last"}, err
			}
			return items[len(items)-1], nil
		},
		"reverse": func(x any, _ []any, _ map[string]any) (any, error) {
			if s, ok := x.(string); ok {
				r := []rune(s)
				slices.Reverse(r)
				return string(r), nil
			}
			items, err := iterate(x)
			out := append([]any{}, items...)
			slices.Reverse(out)
			return out, err
		},
		"join": func(x any, args []any, kw map[string]any) (any, error) {
			sep := ""
			if v := kwOr(kw, "d", args, 0); v != nil {
				sep = printValue(v)
			}
			return joinValues(x, sep, kwOr(kw, "attribute", args, 1))
		},
		"items": func(x any, _ []any, _ map[string]any) (any, error) {
			d, ok := x.(*jinjaDict)
			if !ok {
				return nil, fmt.Errorf("items needs a dict, not %s", typeName(x))
			}
			m, _ := dictMethod(d, "items")
			return m(nil, nil)
		},
		"tojson":     filterToJSON,
		"int":        filterInt,
		"float":      filterFloat,
		"abs":        filterAbs,
		"sum":        filterSum,
		"sort":       filterSort,
		"indent":     filterIndent,
		"selectattr": func(x any, args []any, _ map[string]any) (any, error) { return selectBy(x, args, true) },
		"rejectattr": func(x any, args []any, _ map[string]any) (any, error) { return selectBy(x, args, false) },
		"map":        filterMap,
	}
}

func filterToJSON(x any, _ []any, kw map[string]any) (any, error) {
	indent := -1
	if v, ok := kw["indent"]; ok && v != nil {
		n, isInt := v.(int64)
		if !isInt {
			return nil, errors.New("tojson indent must be an integer")
		}
		indent = int(n)
	}
	return jsonDumps(x, indent)
}

func filterInt(x any, args []any, _ map[string]any) (any, error) {
	def := intArg(args, 0, 0)
	switch v := x.(type) {
	case int64:
		return v, nil
	case bool:
		return int64Of(v), nil
	case float64:
		return int64(v), nil
	case string:
		if n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64); err == nil {
			return n, nil
		}
		if f, err := strconv.ParseFloat(strings.TrimSpace(v), 64); err == nil {
			return int64(f), nil
		}
	}
	return def, nil
}

func filterFloat(x any, args []any, _ map[string]any) (any, error) {
	if f, _, ok := asNumber(x); ok {
		return f, nil
	}
	if str, ok := x.(string); ok {
		if f, err := strconv.ParseFloat(strings.TrimSpace(str), 64); err == nil {
			return f, nil
		}
	}
	if len(args) > 0 {
		return args[0], nil
	}
	return 0.0, nil
}

func filterAbs(x any, _ []any, _ map[string]any) (any, error) {
	switch v := x.(type) {
	case int64:
		if v < 0 {
			return -v, nil
		}
		return v, nil
	case float64:
		return math.Abs(v), nil
	}
	return nil, fmt.Errorf("abs needs a number, not %s", typeName(x))
}

func filterSum(x any, _ []any, _ map[string]any) (any, error) {
	items, err := iterate(x)
	var total any = int64(0)
	for _, item := range items {
		if total, err = arithmetic("+", total, item); err != nil {
			return nil, err
		}
	}
	return total, err
}

func filterSort(x any, _ []any, kw map[string]any) (any, error) {
	items, err := iterate(x)
	if err != nil {
		return nil, err
	}
	out := append([]any{}, items...)
	var sortErr error
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if attr := kw["attribute"]; attr != nil {
			var errA, errB error
			a, errA = attributePath(a, printValue(attr))
			b, errB = attributePath(b, printValue(attr))
			sortErr = errors.Join(sortErr, errA, errB)
		}
		c, ok := compareValues(a, b)
		if !ok {
			sortErr = errors.Join(sortErr, fmt.Errorf("cannot sort %s and %s", typeName(a), typeName(b)))
		}
		return c < 0
	})
	if truthy(kw["reverse"]) {
		slices.Reverse(out)
	}
	return out, sortErr
}

func filterIndent(x any, args []any, kw map[string]any) (any, error) {
	width := intArg(args, 0, 4)
	if n, ok := kw["width"].(int64); ok {
		width = n
	}
	first := truthy(kwOr(kw, "first", args, 1))
	blank := truthy(kwOr(kw, "blank", args, 2))
	pad := strings.Repeat(" ", int(max(width, 0)))
	lines := strings.Split(printValue(x), "\n")
	for i, line := range lines {
		if (i == 0 && !first) || (line == "" && !blank) {
			continue
		}
		lines[i] = pad + line
	}
	return strings.Join(lines, "\n"), nil
}

func selectBy(x any, args []any, keep bool) (any, error) {
	items, err := iterate(x)
	if err != nil {
		return nil, err
	}
	if len(args) == 0 {
		return nil, errors.New("selectattr needs an attribute name")
	}
	path := printValue(args[0])
	test := ""
	var testArgs []any
	if len(args) > 1 {
		test = printValue(args[1])
		testArgs = args[2:]
	}
	out := []any{}
	for _, item := range items {
		v, err := attributePath(item, path)
		if err != nil {
			return nil, err
		}
		pass := truthy(v)
		if test != "" {
			if pass, err = runTest(test, v, testArgs); err != nil {
				return nil, err
			}
		}
		if pass == keep {
			out = append(out, item)
		}
	}
	return out, nil
}

func filterMap(x any, args []any, kw map[string]any) (any, error) {
	items, err := iterate(x)
	if err != nil {
		return nil, err
	}
	out := make([]any, len(items))
	for i, item := range items {
		switch {
		case kw["attribute"] != nil:
			if out[i], err = attributePath(item, printValue(kw["attribute"])); err != nil {
				return nil, err
			}
			if _, undefined := out[i].(jinjaUndefined); undefined && kw["default"] != nil {
				out[i] = kw["default"]
			}
		case len(args) > 0:
			if out[i], err = runFilter(printValue(args[0]), item, args[1:], nil); err != nil {
				return nil, err
			}
		default:
			return nil, errors.New("map needs a filter name or attribute=")
		}
	}
	return out, nil
}

func runTest(name string, x any, args []any) (bool, error) {
	compare := func(op string) (bool, error) {
		if len(args) != 1 {
			return false, fmt.Errorf("test %q needs one argument", name)
		}
		return compareOp(op, x, args[0])
	}
	_, isUndefined := x.(jinjaUndefined)
	switch name {
	case "defined":
		return !isUndefined, nil
	case "undefined":
		return isUndefined, nil
	case "none":
		return x == nil, nil
	case "string":
		_, ok := x.(string)
		return ok, nil
	case "number":
		_, ok := x.(int64)
		_, isFloat := x.(float64)
		return ok || isFloat, nil
	case "integer":
		_, ok := x.(int64)
		return ok, nil
	case "float":
		_, ok := x.(float64)
		return ok, nil
	case "boolean":
		_, ok := x.(bool)
		return ok, nil
	case "true":
		return x == true, nil
	case "false":
		return x == false, nil
	case "mapping":
		_, ok := x.(*jinjaDict)
		return ok, nil
	case "iterable":
		switch x.(type) {
		case []any, string, *jinjaDict, *jinjaLoop, jinjaUndefined:
			return true, nil
		}
		return false, nil
	case "sequence":
		switch x.(type) {
		case []any, string, *jinjaDict:
			return true, nil
		}
		return false, nil
	case "callable":
		_, ok := x.(jinjaFunc)
		return ok || isUndefined, nil
	case "odd", "even":
		n, ok := x.(int64)
		return ok && (n%2 != 0) == (name == "odd"), nil
	case "divisibleby":
		d, isInt := intArgument(args)
		if !isInt || d == 0 {
			return false, errors.New("test \"divisibleby\" needs a nonzero integer argument")
		}
		n, ok := x.(int64)
		return ok && n%d == 0, nil
	case "eq", "equalto", "sameas":
		return compare("==")
	case "ne":
		return compare("!=")
	case "lt", "lessthan":
		return compare("<")
	case "le":
		return compare("<=")
	case "gt", "greaterthan":
		return compare(">")
	case "ge":
		return compare(">=")
	case "in":
		if len(args) != 1 {
			return false, errors.New("test \"in\" needs one argument")
		}
		return contains(args[0], x)
	}
	return false, fmt.Errorf("unknown test %q", name)
}

func intArgument(args []any) (int64, bool) {
	if len(args) != 1 {
		return 0, false
	}
	n, ok := args[0].(int64)
	return n, ok
}
