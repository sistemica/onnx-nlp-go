package onnxnlp

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
)

// jinjaRaise is what raise_exception(message) in a template produces.
type jinjaRaise struct{ msg string }

func (e *jinjaRaise) Error() string { return e.msg }

type jinjaScope struct {
	vars   map[string]any
	parent *jinjaScope
}

func (s *jinjaScope) lookup(name string) (any, bool) {
	for ; s != nil; s = s.parent {
		if v, ok := s.vars[name]; ok {
			return v, true
		}
	}
	return nil, false
}

func newScope(parent *jinjaScope) *jinjaScope {
	return &jinjaScope{vars: map[string]any{}, parent: parent}
}

type loopControl int

const (
	flowNext loopControl = iota
	flowBreak
	flowContinue
)

type jinjaRun struct {
	out strings.Builder
}

// render evaluates the template with the given variables.
func (t *jinjaTemplate) render(vars map[string]any) (string, error) {
	root := newScope(nil)
	for name, v := range jinjaGlobals {
		root.vars[name] = v
	}
	for name, v := range vars {
		root.vars[name] = v
	}
	r := &jinjaRun{}
	if _, err := r.execNodes(t.nodes, root); err != nil {
		return "", err
	}
	return r.out.String(), nil
}

func (r *jinjaRun) execNodes(nodes []jinjaNode, sc *jinjaScope) (loopControl, error) {
	for _, n := range nodes {
		flow, err := r.exec(n, sc)
		if err != nil || flow != flowNext {
			return flow, err
		}
	}
	return flowNext, nil
}

func atLine(line int, err error) error {
	var raised *jinjaRaise
	if errors.As(err, &raised) || strings.HasPrefix(err.Error(), "line ") {
		return err
	}
	return fmt.Errorf("line %d: %w", line, err)
}

func (r *jinjaRun) exec(n jinjaNode, sc *jinjaScope) (loopControl, error) {
	switch n := n.(type) {
	case textNode:
		r.out.WriteString(n.s)
	case outputNode:
		v, err := r.eval(n.x, sc)
		if err != nil {
			return flowNext, atLine(n.line, err)
		}
		r.out.WriteString(printValue(v))
	case blockNode:
		return r.execNodes(n.body, sc)
	case setBlockNode:
		sub := &jinjaRun{}
		if _, err := sub.execNodes(n.body, newScope(sc)); err != nil {
			return flowNext, err
		}
		sc.vars[n.name] = sub.out.String()
	case macroNode:
		return flowNext, r.defineMacro(n, sc)
	case loopControlNode:
		if n.isBreak {
			return flowBreak, nil
		}
		return flowContinue, nil
	case setNode:
		if err := r.assign(n, sc); err != nil {
			return flowNext, atLine(n.line, err)
		}
	case ifNode:
		for i, cond := range n.conds {
			v, err := r.eval(cond, sc)
			if err != nil {
				return flowNext, atLine(n.line, err)
			}
			if truthy(v) {
				return r.execNodes(n.bodies[i], sc)
			}
		}
		return r.execNodes(n.other, sc)
	case forNode:
		return r.execFor(n, sc)
	}
	return flowNext, nil
}

func (r *jinjaRun) assign(n setNode, sc *jinjaScope) error {
	v, err := r.eval(n.value, sc)
	if err != nil {
		return err
	}
	switch target := n.target.(type) {
	case nameExpr:
		sc.vars[target.name] = v
	case attrExpr:
		holder, _ := sc.lookup(target.obj.(nameExpr).name)
		ns, ok := holder.(*jinjaNamespace)
		if !ok {
			return fmt.Errorf("cannot assign attribute %q on a %s", target.name, typeName(holder))
		}
		ns.m[target.name] = v
	}
	return nil
}

func (r *jinjaRun) execFor(n forNode, sc *jinjaScope) (loopControl, error) {
	iterable, err := r.eval(n.iter, sc)
	if err != nil {
		return flowNext, atLine(n.line, err)
	}
	items, err := iterate(iterable)
	if err != nil {
		return flowNext, atLine(n.line, err)
	}
	if n.cond != nil {
		var kept []any
		for _, item := range items {
			inner := newScope(sc)
			if err := bindTargets(n.targets, item, inner); err != nil {
				return flowNext, atLine(n.line, err)
			}
			v, err := r.eval(n.cond, inner)
			if err != nil {
				return flowNext, atLine(n.line, err)
			}
			if truthy(v) {
				kept = append(kept, item)
			}
		}
		items = kept
	}
	if len(items) == 0 {
		return r.execNodes(n.other, sc)
	}
	loop := &jinjaLoop{items: items}
	for i, item := range items {
		loop.index0 = i
		inner := newScope(sc)
		inner.vars["loop"] = loop
		if err := bindTargets(n.targets, item, inner); err != nil {
			return flowNext, atLine(n.line, err)
		}
		flow, err := r.execNodes(n.body, inner)
		if err != nil {
			return flowNext, err
		}
		if flow == flowBreak {
			break
		}
	}
	return flowNext, nil
}

func bindTargets(targets []string, item any, sc *jinjaScope) error {
	if len(targets) == 1 {
		sc.vars[targets[0]] = item
		return nil
	}
	parts, ok := item.([]any)
	if !ok || len(parts) != len(targets) {
		return fmt.Errorf("cannot unpack %s into %d names", typeName(item), len(targets))
	}
	for i, name := range targets {
		sc.vars[name] = parts[i]
	}
	return nil
}

func iterate(v any) ([]any, error) {
	switch x := v.(type) {
	case []any:
		return x, nil
	case *jinjaDict:
		out := make([]any, len(x.keys))
		for i, k := range x.keys {
			out[i] = k
		}
		return out, nil
	case string:
		var out []any
		for _, r := range x {
			out = append(out, string(r))
		}
		return out, nil
	case jinjaUndefined:
		return nil, nil
	}
	return nil, fmt.Errorf("a %s is not iterable", typeName(v))
}

func (r *jinjaRun) eval(x jinjaExpr, sc *jinjaScope) (any, error) {
	switch x := x.(type) {
	case litExpr:
		return x.v, nil
	case nameExpr:
		if v, ok := sc.lookup(x.name); ok {
			return v, nil
		}
		return jinjaUndefined{x.name}, nil
	case attrExpr:
		obj, err := r.eval(x.obj, sc)
		if err != nil {
			return nil, err
		}
		return getAttr(obj, x.name)
	case indexExpr:
		obj, err := r.eval(x.obj, sc)
		if err != nil {
			return nil, err
		}
		index, err := r.eval(x.index, sc)
		if err != nil {
			return nil, err
		}
		return getItem(obj, index)
	case sliceExpr:
		return r.evalSlice(x, sc)
	case callExpr:
		return r.evalCall(x, sc)
	case filterExpr:
		return r.evalFilter(x, sc)
	case unaryExpr:
		return r.evalUnary(x, sc)
	case binaryExpr:
		return r.evalBinary(x, sc)
	case compareExpr:
		return r.evalCompare(x, sc)
	case condExpr:
		test, err := r.eval(x.test, sc)
		if err != nil {
			return nil, err
		}
		if truthy(test) {
			return r.eval(x.yes, sc)
		}
		return r.eval(x.no, sc)
	case listExpr:
		return r.evalList(x.items, sc)
	default:
		literal := x.(dictExpr)
		d := newJinjaDict()
		for i := range literal.keys {
			k, err := r.eval(literal.keys[i], sc)
			if err != nil {
				return nil, err
			}
			v, err := r.eval(literal.vals[i], sc)
			if err != nil {
				return nil, err
			}
			d.set(printValue(k), v)
		}
		return d, nil
	}
}

func (r *jinjaRun) evalList(items []jinjaExpr, sc *jinjaScope) ([]any, error) {
	out := make([]any, len(items))
	for i, item := range items {
		v, err := r.eval(item, sc)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

func (r *jinjaRun) evalSlice(x sliceExpr, sc *jinjaScope) (any, error) {
	obj, err := r.eval(x.obj, sc)
	if err != nil {
		return nil, err
	}
	var bounds [3]*int
	for i, e := range []jinjaExpr{x.lo, x.hi, x.step} {
		if e == nil {
			continue
		}
		v, err := r.eval(e, sc)
		if err != nil {
			return nil, err
		}
		if v == nil {
			continue
		}
		n, ok := v.(int64)
		if !ok {
			return nil, fmt.Errorf("slice indices must be integers, not %s", typeName(v))
		}
		k := int(n)
		bounds[i] = &k
	}
	switch o := obj.(type) {
	case []any:
		idx, err := sliceIndices(len(o), bounds)
		if err != nil {
			return nil, err
		}
		out := make([]any, len(idx))
		for i, j := range idx {
			out[i] = o[j]
		}
		return out, nil
	case string:
		runes := []rune(o)
		idx, err := sliceIndices(len(runes), bounds)
		if err != nil {
			return nil, err
		}
		out := make([]rune, len(idx))
		for i, j := range idx {
			out[i] = runes[j]
		}
		return string(out), nil
	case jinjaUndefined:
		return nil, fmt.Errorf("%q is undefined", o.name)
	}
	return nil, fmt.Errorf("a %s cannot be sliced", typeName(obj))
}

// sliceIndices lists the positions Python's [lo:hi:step] selects from a sequence of length n.
func sliceIndices(n int, b [3]*int) ([]int, error) {
	step := 1
	if b[2] != nil {
		step = *b[2]
	}
	if step == 0 {
		return nil, errors.New("slice step cannot be zero")
	}
	clamp := func(p *int, def, lo, hi int) int {
		if p == nil {
			return def
		}
		v := *p
		if v < 0 {
			v += n
		}
		return min(max(v, lo), hi)
	}
	var out []int
	if step > 0 {
		for i := clamp(b[0], 0, 0, n); i < clamp(b[1], n, 0, n); i += step {
			out = append(out, i)
		}
		return out, nil
	}
	for i := clamp(b[0], n-1, -1, n-1); i > clamp(b[1], -1, -1, n-1); i += step {
		out = append(out, i)
	}
	return out, nil
}

func (r *jinjaRun) evalCall(x callExpr, sc *jinjaScope) (any, error) {
	args, err := r.evalList(x.args, sc)
	if err != nil {
		return nil, err
	}
	kw := map[string]any{}
	for i, k := range x.kwKeys {
		if kw[k], err = r.eval(x.kwVals[i], sc); err != nil {
			return nil, err
		}
	}
	if attr, ok := x.fn.(attrExpr); ok {
		obj, err := r.eval(attr.obj, sc)
		if err != nil {
			return nil, err
		}
		method, err := getAttr(obj, attr.name)
		if err != nil {
			return nil, err
		}
		return callValue(method, attr.name, args, kw)
	}
	fn, err := r.eval(x.fn, sc)
	if err != nil {
		return nil, err
	}
	name := "value"
	if n, ok := x.fn.(nameExpr); ok {
		name = n.name
	}
	return callValue(fn, name, args, kw)
}

func callValue(fn any, name string, args []any, kw map[string]any) (any, error) {
	switch f := fn.(type) {
	case jinjaFunc:
		return f(args, kw)
	case jinjaUndefined:
		return nil, fmt.Errorf("%q is undefined", f.name)
	}
	return nil, fmt.Errorf("%s is not callable", name)
}

func (r *jinjaRun) evalUnary(x unaryExpr, sc *jinjaScope) (any, error) {
	v, err := r.eval(x.x, sc)
	if err != nil {
		return nil, err
	}
	switch x.op {
	case "not":
		return !truthy(v), nil
	case "+":
		if _, _, ok := asNumber(v); ok {
			return v, nil
		}
	case "-":
		switch n := v.(type) {
		case int64:
			return -n, nil
		case float64:
			return -n, nil
		case bool:
			return -int64Of(n), nil
		}
	}
	return nil, fmt.Errorf("bad operand type for unary %s: %s", x.op, typeName(v))
}

func (r *jinjaRun) evalBinary(x binaryExpr, sc *jinjaScope) (any, error) {
	l, err := r.eval(x.l, sc)
	if err != nil {
		return nil, err
	}
	switch x.op {
	case "and":
		if !truthy(l) {
			return l, nil
		}
		return r.eval(x.r, sc)
	case "or":
		if truthy(l) {
			return l, nil
		}
		return r.eval(x.r, sc)
	}
	right, err := r.eval(x.r, sc)
	if err != nil {
		return nil, err
	}
	return arithmetic(x.op, l, right)
}

func arithmetic(op string, a, b any) (any, error) {
	if op == "~" {
		return printValue(a) + printValue(b), nil
	}
	if res, ok, err := sequenceArithmetic(op, a, b); ok || err != nil {
		return res, err
	}
	x, xInt, okX := asNumber(a)
	y, yInt, okY := asNumber(b)
	if !okX || !okY {
		return nil, fmt.Errorf("unsupported operand types for %s: %s and %s", op, typeName(a), typeName(b))
	}
	ints := xInt && yInt
	ia, ib := int64Of(a), int64Of(b)
	switch op {
	case "+":
		if ints {
			return ia + ib, nil
		}
		return x + y, nil
	case "-":
		if ints {
			return ia - ib, nil
		}
		return x - y, nil
	case "*":
		if ints {
			return ia * ib, nil
		}
		return x * y, nil
	case "/":
		if y == 0 {
			return nil, errors.New("division by zero")
		}
		return x / y, nil
	case "//":
		if y == 0 {
			return nil, errors.New("division by zero")
		}
		if ints {
			q := ia / ib
			if (ia%ib != 0) && ((ia < 0) != (ib < 0)) {
				q--
			}
			return q, nil
		}
		return math.Floor(x / y), nil
	case "%":
		if y == 0 {
			return nil, errors.New("modulo by zero")
		}
		if ints {
			m := ia % ib
			if m != 0 && (m < 0) != (ib < 0) {
				m += ib
			}
			return m, nil
		}
		m := math.Mod(x, y)
		if m != 0 && (m < 0) != (y < 0) {
			m += y
		}
		return m, nil
	}
	if ints && ib >= 0 {
		result := int64(1)
		for i := int64(0); i < ib; i++ {
			result *= ia
		}
		return result, nil
	}
	return math.Pow(x, y), nil
}

// sequenceArithmetic handles + and * on strings and lists.
func sequenceArithmetic(op string, a, b any) (any, bool, error) {
	switch x := a.(type) {
	case string:
		switch y := b.(type) {
		case string:
			if op == "+" {
				return x + y, true, nil
			}
		case int64:
			if op == "*" {
				return strings.Repeat(x, max(int(y), 0)), true, nil
			}
		}
	case []any:
		switch y := b.(type) {
		case []any:
			if op == "+" {
				return append(append([]any{}, x...), y...), true, nil
			}
		case int64:
			if op == "*" {
				var out []any
				for i := int64(0); i < y; i++ {
					out = append(out, x...)
				}
				return out, true, nil
			}
		}
	case int64:
		if s, ok := b.(string); ok && op == "*" {
			return strings.Repeat(s, max(int(x), 0)), true, nil
		}
	}
	return nil, false, nil
}

func (r *jinjaRun) evalCompare(x compareExpr, sc *jinjaScope) (any, error) {
	left, err := r.eval(x.first, sc)
	if err != nil {
		return nil, err
	}
	for i, op := range x.ops {
		right, err := r.eval(x.rest[i], sc)
		if err != nil {
			return nil, err
		}
		ok, err := compareOp(op, left, right)
		if err != nil || !ok {
			return false, err
		}
		left = right
	}
	return true, nil
}

func compareOp(op string, a, b any) (bool, error) {
	switch op {
	case "==":
		return valuesEqual(a, b), nil
	case "!=":
		return !valuesEqual(a, b), nil
	case "in", "not in":
		found, err := contains(b, a)
		return found == (op == "in"), err
	}
	c, ok := compareValues(a, b)
	if !ok {
		return false, fmt.Errorf("%q not supported between %s and %s", op, typeName(a), typeName(b))
	}
	switch op {
	case "<":
		return c < 0, nil
	case "<=":
		return c <= 0, nil
	case ">":
		return c > 0, nil
	}
	return c >= 0, nil
}

func contains(container, item any) (bool, error) {
	switch c := container.(type) {
	case string:
		s, ok := item.(string)
		if !ok {
			return false, fmt.Errorf("'in <string>' needs a string, not %s", typeName(item))
		}
		return strings.Contains(c, s), nil
	case []any:
		for _, e := range c {
			if valuesEqual(e, item) {
				return true, nil
			}
		}
		return false, nil
	case *jinjaDict:
		s, ok := item.(string)
		_, found := c.m[s]
		return ok && found, nil
	case jinjaUndefined:
		return false, nil
	}
	return false, fmt.Errorf("argument of type %s is not iterable", typeName(container))
}

func (r *jinjaRun) evalFilter(x filterExpr, sc *jinjaScope) (any, error) {
	v, err := r.eval(x.x, sc)
	if err != nil {
		return nil, err
	}
	args, err := r.evalList(x.call.args, sc)
	if err != nil {
		return nil, err
	}
	kw := map[string]any{}
	for i, k := range x.call.kwKeys {
		if kw[k], err = r.eval(x.call.kwVals[i], sc); err != nil {
			return nil, err
		}
	}
	if x.isTest {
		ok, err := runTest(x.name, v, args)
		return ok != x.negate, err
	}
	return runFilter(x.name, v, args, kw)
}

// getAttr is Jinja's value.name: methods first, then keys.
func getAttr(obj any, name string) (any, error) {
	switch o := obj.(type) {
	case *jinjaDict:
		if m, ok := dictMethod(o, name); ok {
			return m, nil
		}
		if v, ok := o.m[name]; ok {
			return v, nil
		}
	case *jinjaNamespace:
		if v, ok := o.m[name]; ok {
			return v, nil
		}
	case *jinjaLoop:
		if v, ok := o.attr(name); ok {
			return v, nil
		}
	case string:
		if m, ok := stringMethod(o, name); ok {
			return m, nil
		}
	case jinjaUndefined:
		return nil, fmt.Errorf("%q is undefined", o.name)
	}
	return jinjaUndefined{name}, nil
}

// getItem is Jinja's value[index].
func getItem(obj, index any) (any, error) {
	switch o := obj.(type) {
	case *jinjaDict:
		if k, ok := index.(string); ok {
			if v, found := o.m[k]; found {
				return v, nil
			}
		}
	case *jinjaNamespace:
		if k, ok := index.(string); ok {
			if v, found := o.m[k]; found {
				return v, nil
			}
		}
	case []any:
		if i, ok := index.(int64); ok {
			if i < 0 {
				i += int64(len(o))
			}
			if i >= 0 && i < int64(len(o)) {
				return o[i], nil
			}
		}
	case string:
		if i, ok := index.(int64); ok {
			runes := []rune(o)
			if i < 0 {
				i += int64(len(runes))
			}
			if i >= 0 && i < int64(len(runes)) {
				return string(runes[i]), nil
			}
		}
	case jinjaUndefined:
		return nil, fmt.Errorf("%q is undefined", o.name)
	}
	return jinjaUndefined{printValue(index)}, nil
}

// defineMacro binds the macro to its name; a call renders the body in a scope below the one that
// defined it, with arguments by position, by keyword or from their defaults.
func (r *jinjaRun) defineMacro(n macroNode, sc *jinjaScope) error {
	defaults := make([]any, len(n.params))
	for i, d := range n.defaults {
		if d == nil {
			continue
		}
		v, err := r.eval(d, sc)
		if err != nil {
			return err
		}
		defaults[i] = v
	}
	sc.vars[n.name] = jinjaFunc(func(args []any, kw map[string]any) (any, error) {
		if len(args) > len(n.params) {
			return nil, fmt.Errorf("macro %q takes %d arguments, not %d", n.name, len(n.params), len(args))
		}
		inner := newScope(sc)
		for i, name := range n.params {
			switch v, ok := kw[name]; {
			case i < len(args):
				inner.vars[name] = args[i]
			case ok:
				inner.vars[name] = v
			case n.defaults[i] != nil:
				inner.vars[name] = defaults[i]
			default:
				inner.vars[name] = jinjaUndefined{name}
			}
		}
		for name := range kw {
			if !slices.Contains(n.params, name) {
				return nil, fmt.Errorf("macro %q has no argument %q", n.name, name)
			}
		}
		sub := &jinjaRun{}
		if _, err := sub.execNodes(n.body, inner); err != nil {
			return nil, err
		}
		return sub.out.String(), nil
	})
	return nil
}
