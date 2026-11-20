package onnxnlp

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

type tokenKind int

const (
	tokEOF tokenKind = iota
	tokName
	tokNumber
	tokString
	tokOp
)

type jinjaToken struct {
	kind tokenKind
	s    string
	num  any
}

var jinjaOperators = []string{"**", "//", "==", "!=", "<=", ">=", "+", "-", "*", "/", "%", "~", "<", ">", "=", "(", ")", "[", "]", "{", "}", ",", ".", ":", "|"}

func tokenizeJinja(src string) ([]jinjaToken, error) {
	var out []jinjaToken
	for i := 0; i < len(src); {
		c := src[i]
		switch {
		case c == ' ' || c == '\t' || c == '\n' || c == '\r':
			i++
		case c == '"' || c == '\'':
			s, n, err := readJinjaString(src[i:])
			if err != nil {
				return nil, err
			}
			out = append(out, jinjaToken{kind: tokString, s: s})
			i += n
		case c >= '0' && c <= '9':
			tok, n, err := readJinjaNumber(src[i:])
			if err != nil {
				return nil, err
			}
			out = append(out, tok)
			i += n
		case c == '_' || c >= 0x80 || unicode.IsLetter(rune(c)):
			j := i
			for j < len(src) && (src[j] == '_' || src[j] >= 0x80 || unicode.IsLetter(rune(src[j])) || unicode.IsDigit(rune(src[j]))) {
				j++
			}
			out = append(out, jinjaToken{kind: tokName, s: src[i:j]})
			i = j
		default:
			op := ""
			for _, candidate := range jinjaOperators {
				if strings.HasPrefix(src[i:], candidate) {
					op = candidate
					break
				}
			}
			if op == "" {
				return nil, fmt.Errorf("unexpected character %q", c)
			}
			out = append(out, jinjaToken{kind: tokOp, s: op})
			i += len(op)
		}
	}
	return append(out, jinjaToken{kind: tokEOF}), nil
}

func readJinjaNumber(src string) (jinjaToken, int, error) {
	i := 0
	for i < len(src) && src[i] >= '0' && src[i] <= '9' {
		i++
	}
	isFloat := false
	if i+1 < len(src) && src[i] == '.' && src[i+1] >= '0' && src[i+1] <= '9' {
		isFloat = true
		i++
		for i < len(src) && src[i] >= '0' && src[i] <= '9' {
			i++
		}
	}
	if i < len(src) && (src[i] == 'e' || src[i] == 'E') {
		j := i + 1
		if j < len(src) && (src[j] == '+' || src[j] == '-') {
			j++
		}
		if j < len(src) && src[j] >= '0' && src[j] <= '9' {
			for j < len(src) && src[j] >= '0' && src[j] <= '9' {
				j++
			}
			isFloat, i = true, j
		}
	}
	if isFloat {
		f, err := strconv.ParseFloat(src[:i], 64)
		return jinjaToken{kind: tokNumber, num: f}, i, err
	}
	n, err := strconv.ParseInt(src[:i], 10, 64)
	return jinjaToken{kind: tokNumber, num: n}, i, err
}

// readJinjaString decodes a quoted literal with Python escapes and returns the bytes it spans.
func readJinjaString(src string) (string, int, error) {
	quote := src[0]
	var b strings.Builder
	for i := 1; i < len(src); i++ {
		c := src[i]
		switch {
		case c == quote:
			return b.String(), i + 1, nil
		case c != '\\' || i+1 == len(src):
			b.WriteByte(c)
		default:
			i++
			switch e := src[i]; e {
			case 'n':
				b.WriteByte('\n')
			case 't':
				b.WriteByte('\t')
			case 'r':
				b.WriteByte('\r')
			case 'b':
				b.WriteByte('\b')
			case 'f':
				b.WriteByte('\f')
			case 'v':
				b.WriteByte('\v')
			case 'a':
				b.WriteByte('\a')
			case '0':
				b.WriteByte(0)
			case '\\', '\'', '"':
				b.WriteByte(e)
			case '\n':
			case 'x', 'u', 'U':
				digits := map[byte]int{'x': 2, 'u': 4, 'U': 8}[e]
				if i+digits < len(src) {
					if r, err := strconv.ParseUint(src[i+1:i+1+digits], 16, 32); err == nil {
						b.WriteRune(rune(r))
						i += digits
						continue
					}
				}
				b.WriteByte('\\')
				b.WriteByte(e)
			default:
				b.WriteByte('\\')
				b.WriteByte(e)
			}
		}
	}
	return "", 0, fmt.Errorf("unterminated string %q", src)
}

// Expressions.

type jinjaExpr interface{}

type (
	litExpr  struct{ v any }
	nameExpr struct{ name string }
	attrExpr struct {
		obj  jinjaExpr
		name string
	}
	indexExpr struct{ obj, index jinjaExpr }
	sliceExpr struct{ obj, lo, hi, step jinjaExpr }
	callExpr  struct {
		fn     jinjaExpr
		args   []jinjaExpr
		kwKeys []string
		kwVals []jinjaExpr
	}
	filterExpr struct {
		x      jinjaExpr
		name   string
		call   callExpr
		isTest bool
		negate bool
	}
	unaryExpr struct {
		op string
		x  jinjaExpr
	}
	binaryExpr struct {
		op   string
		l, r jinjaExpr
	}
	compareExpr struct {
		first jinjaExpr
		ops   []string
		rest  []jinjaExpr
	}
	condExpr struct{ test, yes, no jinjaExpr }
	listExpr struct{ items []jinjaExpr }
	dictExpr struct{ keys, vals []jinjaExpr }
)

type exprParser struct {
	toks []jinjaToken
	pos  int
}

func (p *exprParser) peek() jinjaToken { return p.toks[p.pos] }

func (p *exprParser) next() jinjaToken {
	t := p.toks[p.pos]
	if t.kind != tokEOF {
		p.pos++
	}
	return t
}

func (p *exprParser) isOp(s string) bool {
	t := p.peek()
	return t.kind == tokOp && t.s == s
}

func (p *exprParser) isName(s string) bool {
	t := p.peek()
	return t.kind == tokName && t.s == s
}

func (p *exprParser) acceptOp(s string) bool {
	if p.isOp(s) {
		p.pos++
		return true
	}
	return false
}

func (p *exprParser) acceptName(s string) bool {
	if p.isName(s) {
		p.pos++
		return true
	}
	return false
}

func (p *exprParser) expectOp(s string) error {
	if !p.acceptOp(s) {
		return fmt.Errorf("expected %q, found %s", s, p.describe())
	}
	return nil
}

func (p *exprParser) describe() string {
	t := p.peek()
	if t.kind == tokEOF {
		return "the end of the tag"
	}
	if t.kind == tokString {
		return strconv.Quote(t.s)
	}
	if t.kind == tokNumber {
		return pyRepr(t.num)
	}
	return strconv.Quote(t.s)
}

func (p *exprParser) expectEnd() error {
	if p.peek().kind != tokEOF {
		return fmt.Errorf("unexpected %s", p.describe())
	}
	return nil
}

func (p *exprParser) expectName() (string, error) {
	t := p.next()
	if t.kind != tokName {
		p.pos--
		return "", fmt.Errorf("expected a name, found %s", p.describe())
	}
	return t.s, nil
}

// parseExpr parses a full expression, including a trailing "a if b else c".
func (p *exprParser) parseExpr() (jinjaExpr, error) {
	x, err := p.parseOr()
	if err != nil || !p.isName("if") {
		return x, err
	}
	p.pos++
	test, err := p.parseOr()
	if err != nil {
		return nil, err
	}
	var no jinjaExpr = litExpr{jinjaUndefined{}}
	if p.acceptName("else") {
		if no, err = p.parseExpr(); err != nil {
			return nil, err
		}
	}
	return condExpr{test, x, no}, nil
}

func (p *exprParser) parseOr() (jinjaExpr, error) {
	x, err := p.parseAnd()
	for err == nil && p.acceptName("or") {
		var y jinjaExpr
		if y, err = p.parseAnd(); err == nil {
			x = binaryExpr{"or", x, y}
		}
	}
	return x, err
}

func (p *exprParser) parseAnd() (jinjaExpr, error) {
	x, err := p.parseNot()
	for err == nil && p.acceptName("and") {
		var y jinjaExpr
		if y, err = p.parseNot(); err == nil {
			x = binaryExpr{"and", x, y}
		}
	}
	return x, err
}

func (p *exprParser) parseNot() (jinjaExpr, error) {
	if p.acceptName("not") {
		x, err := p.parseNot()
		return unaryExpr{"not", x}, err
	}
	return p.parseCompare()
}

func (p *exprParser) parseCompare() (jinjaExpr, error) {
	first, err := p.parseAdd()
	if err != nil {
		return nil, err
	}
	cmp := compareExpr{first: first}
	for {
		op := ""
		switch t := p.peek(); {
		case t.kind == tokOp && (t.s == "==" || t.s == "!=" || t.s == "<" || t.s == ">" || t.s == "<=" || t.s == ">="):
			op = t.s
			p.pos++
		case t.kind == tokName && t.s == "in":
			op = "in"
			p.pos++
		case t.kind == tokName && t.s == "not" && p.toks[p.pos+1].kind == tokName && p.toks[p.pos+1].s == "in":
			op = "not in"
			p.pos += 2
		}
		if op == "" {
			break
		}
		y, err := p.parseAdd()
		if err != nil {
			return nil, err
		}
		cmp.ops, cmp.rest = append(cmp.ops, op), append(cmp.rest, y)
	}
	if len(cmp.ops) == 0 {
		return first, nil
	}
	return cmp, nil
}

func (p *exprParser) binaryLevel(ops []string, operand func() (jinjaExpr, error)) (jinjaExpr, error) {
	x, err := operand()
	for err == nil {
		op := ""
		for _, candidate := range ops {
			if p.isOp(candidate) {
				op = candidate
			}
		}
		if op == "" {
			break
		}
		p.pos++
		var y jinjaExpr
		if y, err = operand(); err == nil {
			x = binaryExpr{op, x, y}
		}
	}
	return x, err
}

func (p *exprParser) parseAdd() (jinjaExpr, error) {
	return p.binaryLevel([]string{"+", "-"}, p.parseConcat)
}

func (p *exprParser) parseConcat() (jinjaExpr, error) {
	return p.binaryLevel([]string{"~"}, p.parseMul)
}

func (p *exprParser) parseMul() (jinjaExpr, error) {
	return p.binaryLevel([]string{"*", "/", "//", "%"}, p.parsePow)
}

func (p *exprParser) parsePow() (jinjaExpr, error) {
	return p.binaryLevel([]string{"**"}, p.parseUnary)
}

func (p *exprParser) parseUnary() (jinjaExpr, error) { return p.parseUnaryWith(true) }

// parseUnaryWith follows Jinja: the operand of a sign takes attribute access, subscripts and calls
// but not filters or tests, which then apply to the signed value, so -x|abs is abs(-x).
func (p *exprParser) parseUnaryWith(filters bool) (jinjaExpr, error) {
	var x jinjaExpr
	if p.isOp("-") || p.isOp("+") {
		op := p.next().s
		operand, err := p.parseUnaryWith(false)
		if err != nil {
			return nil, err
		}
		x = unaryExpr{op, operand}
	} else {
		var err error
		if x, err = p.parsePrimary(); err != nil {
			return nil, err
		}
	}
	return p.parsePostfix(x, filters)
}

func (p *exprParser) parsePostfix(x jinjaExpr, filters bool) (jinjaExpr, error) {
	for {
		switch {
		case p.acceptOp("."):
			name, err := p.expectName()
			if err != nil {
				return nil, err
			}
			x = attrExpr{x, name}
		case p.acceptOp("["):
			var err error
			if x, err = p.parseSubscript(x); err != nil {
				return nil, err
			}
		case p.acceptOp("("):
			call, err := p.parseArguments(callExpr{fn: x}, ")")
			if err != nil {
				return nil, err
			}
			x = call
		case filters && p.acceptOp("|"):
			name, err := p.expectName()
			if err != nil {
				return nil, err
			}
			for p.acceptOp(".") {
				part, err := p.expectName()
				if err != nil {
					return nil, err
				}
				name += "." + part
			}
			f := filterExpr{x: x, name: name}
			if p.acceptOp("(") {
				if f.call, err = p.parseArguments(callExpr{}, ")"); err != nil {
					return nil, err
				}
			}
			x = f
		case filters && p.isName("is"):
			p.pos++
			f := filterExpr{x: x, isTest: true, negate: p.acceptName("not")}
			name, err := p.expectName()
			if err != nil {
				return nil, err
			}
			f.name = name
			if p.acceptOp("(") {
				if f.call, err = p.parseArguments(callExpr{}, ")"); err != nil {
					return nil, err
				}
			} else if p.startsOperand() {
				arg, err := p.parsePrimary()
				if err != nil {
					return nil, err
				}
				f.call.args = []jinjaExpr{arg}
			}
			x = f
		default:
			return x, nil
		}
	}
}

// startsOperand reports whether the next token can begin the single argument of a test such as
// "is divisibleby 3".
func (p *exprParser) startsOperand() bool {
	t := p.peek()
	switch t.kind {
	case tokNumber, tokString:
		return true
	case tokName:
		return !map[string]bool{"and": true, "or": true, "if": true, "else": true, "is": true, "in": true, "not": true}[t.s]
	case tokOp:
		return t.s == "[" || t.s == "(" || t.s == "{"
	}
	return false
}

func (p *exprParser) parseSubscript(obj jinjaExpr) (jinjaExpr, error) {
	var parts [3]jinjaExpr
	colons := 0
	for !p.isOp("]") {
		switch {
		case p.acceptOp(":"):
			colons++
			if colons > 2 {
				return nil, fmt.Errorf("too many colons in a slice")
			}
		case parts[colons] != nil:
			return nil, fmt.Errorf("expected ']', found %s", p.describe())
		default:
			x, err := p.parseExpr()
			if err != nil {
				return nil, err
			}
			parts[colons] = x
		}
	}
	p.pos++
	if colons == 0 {
		if parts[0] == nil {
			return nil, fmt.Errorf("empty subscript")
		}
		return indexExpr{obj, parts[0]}, nil
	}
	return sliceExpr{obj, parts[0], parts[1], parts[2]}, nil
}

func (p *exprParser) parseArguments(c callExpr, closer string) (callExpr, error) {
	for !p.acceptOp(closer) {
		if len(c.args)+len(c.kwKeys) > 0 {
			if err := p.expectOp(","); err != nil {
				return c, err
			}
			if p.acceptOp(closer) {
				break
			}
		}
		if t := p.peek(); t.kind == tokName && p.toks[p.pos+1].kind == tokOp && p.toks[p.pos+1].s == "=" {
			p.pos += 2
			v, err := p.parseExpr()
			if err != nil {
				return c, err
			}
			c.kwKeys, c.kwVals = append(c.kwKeys, t.s), append(c.kwVals, v)
			continue
		}
		v, err := p.parseExpr()
		if err != nil {
			return c, err
		}
		c.args = append(c.args, v)
	}
	return c, nil
}

func (p *exprParser) parsePrimary() (jinjaExpr, error) {
	t := p.next()
	switch t.kind {
	case tokNumber:
		return litExpr{t.num}, nil
	case tokString:
		s := t.s
		for p.peek().kind == tokString {
			s += p.next().s
		}
		return litExpr{s}, nil
	case tokName:
		switch t.s {
		case "true", "True":
			return litExpr{true}, nil
		case "false", "False":
			return litExpr{false}, nil
		case "none", "None":
			return litExpr{nil}, nil
		}
		return nameExpr{t.s}, nil
	case tokOp:
		switch t.s {
		case "(":
			return p.parseGroup()
		case "[":
			items, err := p.parseItems("]")
			return listExpr{items}, err
		case "{":
			return p.parseDict()
		}
	}
	p.pos--
	return nil, fmt.Errorf("unexpected %s", p.describe())
}

func (p *exprParser) parseGroup() (jinjaExpr, error) {
	if p.acceptOp(")") {
		return listExpr{}, nil
	}
	first, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	if p.acceptOp(")") {
		return first, nil
	}
	if err := p.expectOp(","); err != nil {
		return nil, err
	}
	rest, err := p.parseItems(")")
	return listExpr{append([]jinjaExpr{first}, rest...)}, err
}

func (p *exprParser) parseItems(closer string) ([]jinjaExpr, error) {
	var items []jinjaExpr
	for !p.acceptOp(closer) {
		if len(items) > 0 {
			if err := p.expectOp(","); err != nil {
				return nil, err
			}
			if p.acceptOp(closer) {
				break
			}
		}
		x, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		items = append(items, x)
	}
	return items, nil
}

func (p *exprParser) parseDict() (jinjaExpr, error) {
	d := dictExpr{}
	for !p.acceptOp("}") {
		if len(d.keys) > 0 {
			if err := p.expectOp(","); err != nil {
				return nil, err
			}
			if p.acceptOp("}") {
				break
			}
		}
		k, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		if err := p.expectOp(":"); err != nil {
			return nil, err
		}
		v, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		d.keys, d.vals = append(d.keys, k), append(d.vals, v)
	}
	return d, nil
}

// Statements.

type jinjaNode interface{}

type (
	textNode   struct{ s string }
	outputNode struct {
		x    jinjaExpr
		line int
	}
	ifNode struct {
		conds  []jinjaExpr
		bodies [][]jinjaNode
		other  []jinjaNode
		line   int
	}
	forNode struct {
		targets []string
		iter    jinjaExpr
		cond    jinjaExpr
		body    []jinjaNode
		other   []jinjaNode
		line    int
	}
	setNode struct {
		target jinjaExpr
		value  jinjaExpr
		line   int
	}
	loopControlNode struct{ isBreak bool }
	blockNode       struct{ body []jinjaNode }
	setBlockNode    struct {
		name string
		body []jinjaNode
	}
	macroNode struct {
		name     string
		params   []string
		defaults []jinjaExpr
		body     []jinjaNode
	}
)

type jinjaTemplate struct{ nodes []jinjaNode }

type templateParser struct {
	chunks []jinjaChunk
	pos    int
}

// parseJinja compiles a template.
func parseJinja(src string) (*jinjaTemplate, error) {
	chunks, err := scanJinja(src)
	if err != nil {
		return nil, err
	}
	p := &templateParser{chunks: chunks}
	nodes, end, err := p.parseBody()
	if err != nil {
		return nil, err
	}
	if end != "" {
		return nil, fmt.Errorf("line %d: unexpected {%% %s %%}", p.chunks[p.pos-1].line, end)
	}
	return &jinjaTemplate{nodes}, nil
}

func (p *templateParser) fail(c jinjaChunk, err error) error {
	return fmt.Errorf("line %d: %w", c.line, err)
}

// parseBody reads nodes until a closing tag (endif, else, ...) and returns that tag's first word
// with its tokens left for the caller in p.chunks[p.pos-1].
func (p *templateParser) parseBody() ([]jinjaNode, string, error) {
	var nodes []jinjaNode
	for p.pos < len(p.chunks) {
		c := p.chunks[p.pos]
		p.pos++
		switch c.kind {
		case chunkText:
			nodes = append(nodes, textNode{c.text})
		case chunkVar:
			toks, err := tokenizeJinja(c.text)
			if err != nil {
				return nil, "", p.fail(c, err)
			}
			ep := &exprParser{toks: toks}
			x, err := ep.parseExpr()
			if err == nil {
				err = ep.expectEnd()
			}
			if err != nil {
				return nil, "", p.fail(c, err)
			}
			nodes = append(nodes, outputNode{x, c.line})
		default:
			toks, err := tokenizeJinja(c.text)
			if err != nil {
				return nil, "", p.fail(c, err)
			}
			p.chunks[p.pos-1].toks = toks
			if len(toks) == 1 || toks[0].kind != tokName {
				return nil, "", p.fail(c, fmt.Errorf("empty or malformed tag"))
			}
			word := toks[0].s
			switch word {
			case "endif", "elif", "else", "endfor", "endgeneration", "endmacro", "endset":
				return nodes, word, nil
			}
			node, err := p.parseStatement(c, word, &exprParser{toks: toks, pos: 1})
			if err != nil {
				return nil, "", err
			}
			nodes = append(nodes, node)
		}
	}
	return nodes, "", nil
}

// closeWith parses a body that must end with the given tag.
func (p *templateParser) closeWith(c jinjaChunk, want string) ([]jinjaNode, error) {
	body, end, err := p.parseBody()
	if err != nil {
		return nil, err
	}
	if end != want {
		return nil, p.fail(c, fmt.Errorf("{%% %s %%} is not closed by {%% %s %%}", strings.Fields(c.text)[0], want))
	}
	return body, nil
}

func (p *templateParser) parseStatement(c jinjaChunk, word string, ep *exprParser) (jinjaNode, error) {
	switch word {
	case "if":
		return p.parseIf(c, ep)
	case "for":
		return p.parseFor(c, ep)
	case "set":
		return p.parseSet(c, ep)
	case "break", "continue":
		if err := ep.expectEnd(); err != nil {
			return nil, p.fail(c, err)
		}
		return loopControlNode{word == "break"}, nil
	case "macro":
		return p.parseMacro(c, ep)
	case "generation":
		body, err := p.closeWith(c, "endgeneration")
		return blockNode{body}, err
	}
	return nil, p.fail(c, fmt.Errorf("unsupported tag %q", word))
}

func (p *templateParser) parseIf(c jinjaChunk, ep *exprParser) (jinjaNode, error) {
	n := ifNode{line: c.line}
	for {
		cond, err := ep.parseExpr()
		if err == nil {
			err = ep.expectEnd()
		}
		if err != nil {
			return nil, p.fail(c, err)
		}
		body, end, err := p.parseBody()
		if err != nil {
			return nil, err
		}
		n.conds, n.bodies = append(n.conds, cond), append(n.bodies, body)
		last := p.chunks[p.pos-1]
		switch end {
		case "elif":
			c, ep = last, &exprParser{toks: last.toks, pos: 1}
			continue
		case "else":
			if err := (&exprParser{toks: last.toks, pos: 1}).expectEnd(); err != nil {
				return nil, p.fail(last, err)
			}
			if n.other, err = p.closeWith(last, "endif"); err != nil {
				return nil, err
			}
			return n, nil
		case "endif":
			return n, nil
		}
		return nil, p.fail(c, fmt.Errorf("{%% if %%} is never closed"))
	}
}

func (p *templateParser) parseFor(c jinjaChunk, ep *exprParser) (jinjaNode, error) {
	n := forNode{line: c.line}
	paren := ep.acceptOp("(")
	for {
		name, err := ep.expectName()
		if err != nil {
			return nil, p.fail(c, err)
		}
		n.targets = append(n.targets, name)
		if !ep.acceptOp(",") {
			break
		}
	}
	if paren {
		if err := ep.expectOp(")"); err != nil {
			return nil, p.fail(c, err)
		}
	}
	if !ep.acceptName("in") {
		return nil, p.fail(c, fmt.Errorf("expected \"in\", found %s", ep.describe()))
	}
	var err error
	if n.iter, err = ep.parseOr(); err != nil {
		return nil, p.fail(c, err)
	}
	if ep.acceptName("if") {
		if n.cond, err = ep.parseExpr(); err != nil {
			return nil, p.fail(c, err)
		}
	}
	if err := ep.expectEnd(); err != nil {
		return nil, p.fail(c, err)
	}
	body, end, err := p.parseBody()
	if err != nil {
		return nil, err
	}
	n.body = body
	switch end {
	case "endfor":
	case "else":
		if n.other, err = p.closeWith(p.chunks[p.pos-1], "endfor"); err != nil {
			return nil, err
		}
	default:
		return nil, p.fail(c, fmt.Errorf("{%% for %%} is never closed"))
	}
	return n, nil
}

func (p *templateParser) parseMacro(c jinjaChunk, ep *exprParser) (jinjaNode, error) {
	n := macroNode{}
	var err error
	if n.name, err = ep.expectName(); err == nil {
		err = ep.expectOp("(")
	}
	for err == nil && !ep.acceptOp(")") {
		if len(n.params) > 0 {
			err = ep.expectOp(",")
		}
		var name string
		if err == nil {
			name, err = ep.expectName()
		}
		var def jinjaExpr
		if err == nil && ep.acceptOp("=") {
			def, err = ep.parseExpr()
		}
		n.params, n.defaults = append(n.params, name), append(n.defaults, def)
	}
	if err == nil {
		err = ep.expectEnd()
	}
	if err != nil {
		return nil, p.fail(c, err)
	}
	if n.body, err = p.closeWith(c, "endmacro"); err != nil {
		return nil, err
	}
	return n, nil
}

func (p *templateParser) parseSet(c jinjaChunk, ep *exprParser) (jinjaNode, error) {
	target, err := ep.parsePrimary()
	if err == nil {
		target, err = ep.parsePostfixAssignable(target)
	}
	if name, ok := target.(nameExpr); ok && err == nil && ep.peek().kind == tokEOF {
		body, err := p.closeWith(c, "endset")
		return setBlockNode{name.name, body}, err
	}
	if err == nil {
		err = ep.expectOp("=")
	}
	if err != nil {
		return nil, p.fail(c, err)
	}
	value, err := ep.parseExpr()
	if err == nil {
		err = ep.expectEnd()
	}
	if err != nil {
		return nil, p.fail(c, err)
	}
	return setNode{target, value, c.line}, nil
}

// parsePostfixAssignable accepts name or name.attribute on the left of a set.
func (p *exprParser) parsePostfixAssignable(x jinjaExpr) (jinjaExpr, error) {
	if _, ok := x.(nameExpr); !ok {
		return nil, fmt.Errorf("cannot assign to this expression")
	}
	if p.acceptOp(".") {
		name, err := p.expectName()
		if err != nil {
			return nil, err
		}
		return attrExpr{x, name}, nil
	}
	return x, nil
}
