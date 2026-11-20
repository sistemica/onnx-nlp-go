package onnxnlp

import (
	"errors"
	"fmt"
	"strings"
	"unicode"
)

type chunkKind int

const (
	chunkText chunkKind = iota
	chunkVar
	chunkBlock
)

type jinjaChunk struct {
	kind chunkKind
	text string
	line int
	toks []jinjaToken
}

// scanJinja splits a template into text, {{ }} and {% %} chunks the way Jinja does with
// trim_blocks and lstrip_blocks on, which is how transformers renders chat templates.
func scanJinja(src string) ([]jinjaChunk, error) {
	src = strings.TrimSuffix(src, "\n")
	var out []jinjaChunk
	pos, lineStart := 0, true
	emit := func(kind chunkKind, text string, at int) {
		if kind == chunkText && text == "" {
			return
		}
		out = append(out, jinjaChunk{kind: kind, text: text, line: 1 + strings.Count(src[:at], "\n")})
	}
	for pos < len(src) {
		open, kind := nextTag(src, pos)
		if open < 0 {
			emit(chunkText, src[pos:], pos)
			break
		}
		text := src[pos:open]
		sign := byte(0)
		if open+2 < len(src) && (src[open+2] == '-' || src[open+2] == '+') {
			sign = src[open+2]
		}
		switch {
		case sign == '-':
			text = strings.TrimRightFunc(text, unicode.IsSpace)
		case sign != '+' && kind != chunkVar:
			text = stripLineIndent(text, lineStart)
		}
		emit(chunkText, text, pos)

		bodyFrom := open + 2
		if sign != 0 {
			bodyFrom++
		}
		closer := map[chunkKind]string{chunkVar: "}}", chunkBlock: "%}"}[kind]
		isComment := src[open+1] == '#'
		if isComment {
			closer = "#}"
		}
		end, err := findTagEnd(src, bodyFrom, closer, !isComment)
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", 1+strings.Count(src[:open], "\n"), err)
		}
		body := src[bodyFrom:end]
		next := end + 2
		if strings.HasSuffix(body, "-") {
			body = body[:len(body)-1]
			for next < len(src) && unicode.IsSpace(rune(src[next])) {
				next++
			}
		} else if kind != chunkVar && strings.HasPrefix(src[next:], "\n") {
			next++
		}
		if !isComment {
			emit(kind, body, open)
		}
		lineStart = src[next-1] == '\n'
		pos = next
	}
	return out, nil
}

// nextTag finds the next opening delimiter at or after pos.
func nextTag(src string, pos int) (int, chunkKind) {
	for i := pos; i+1 < len(src); i++ {
		if src[i] != '{' {
			continue
		}
		switch src[i+1] {
		case '{':
			return i, chunkVar
		case '%', '#':
			return i, chunkBlock
		}
	}
	return -1, chunkText
}

// stripLineIndent removes the indentation before a block tag that is alone at the start of its line.
func stripLineIndent(text string, lineStart bool) string {
	cut := strings.LastIndex(text, "\n") + 1
	if cut == 0 && !lineStart {
		return text
	}
	if strings.Trim(text[cut:], " \t") != "" {
		return text
	}
	return text[:cut]
}

// findTagEnd returns where closer starts, skipping string literals and nested braces.
func findTagEnd(src string, from int, closer string, literals bool) (int, error) {
	var quote byte
	depth := 0
	for i := from; i < len(src); i++ {
		c := src[i]
		switch {
		case quote != 0:
			if c == '\\' {
				i++
			} else if c == quote {
				quote = 0
			}
		case literals && (c == '"' || c == '\''):
			quote = c
		case literals && c == '{':
			depth++
		case literals && c == '}' && depth > 0:
			depth--
		case depth == 0 && strings.HasPrefix(src[i:], closer):
			return i, nil
		}
	}
	return 0, errors.New("unterminated tag, expected " + closer)
}
