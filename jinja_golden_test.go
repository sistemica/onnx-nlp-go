package onnxnlp

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
)

type templateGold struct {
	FixedDate string `json:"fixed_date"`
	Models    []struct {
		Name     string            `json:"name"`
		Template string            `json:"template"`
		Tokens   map[string]string `json:"tokens"`
		Cases    []struct {
			Title      string    `json:"title"`
			Generation bool      `json:"generation"`
			Messages   []Message `json:"messages"`
			goldResult
		} `json:"cases"`
	} `json:"models"`
	Strftime struct {
		Moment string `json:"moment"`
		Cases  []struct{ Format, Output string }
	} `json:"strftime"`
	Snippets struct {
		Context json.RawMessage `json:"context"`
		Cases   []struct {
			Template string `json:"template"`
			goldResult
		} `json:"cases"`
	} `json:"snippets"`
}

type goldResult struct {
	Output *string `json:"output"`
	Error  *string `json:"error"`
	Type   string  `json:"type"`
}

func (m *Message) UnmarshalJSON(b []byte) error {
	var raw struct{ Role, Content string }
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	m.Role, m.Content = raw.Role, raw.Content
	return nil
}

// orderedValue decodes JSON into template values, keeping the order of object keys.
func orderedValue(t *testing.T, dec *json.Decoder) any {
	t.Helper()
	tok, err := dec.Token()
	if err != nil {
		t.Fatal(err)
	}
	switch v := tok.(type) {
	case json.Delim:
		if v == '[' {
			list := []any{}
			for dec.More() {
				list = append(list, orderedValue(t, dec))
			}
			dec.Token()
			return list
		}
		d := newJinjaDict()
		for dec.More() {
			key, _ := dec.Token()
			d.set(key.(string), orderedValue(t, dec))
		}
		dec.Token()
		return d
	case json.Number:
		if n, err := v.Int64(); err == nil && !strings.ContainsAny(v.String(), ".eE") {
			return n
		}
		f, _ := v.Float64()
		return f
	}
	return tok
}

func goldVariables(t *testing.T, raw json.RawMessage) map[string]any {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	dec.Token()
	vars := map[string]any{}
	for dec.More() {
		key, _ := dec.Token()
		vars[key.(string)] = orderedValue(t, dec)
	}
	return vars
}

func fixedDate(t *testing.T, date string) {
	t.Helper()
	saved := jinjaGlobals["strftime_now"]
	jinjaGlobals["strftime_now"] = jinjaFunc(func([]any, map[string]any) (any, error) { return date, nil })
	t.Cleanup(func() { jinjaGlobals["strftime_now"] = saved })
}

func checkAgainstGold(t *testing.T, got string, err error, want goldResult) {
	t.Helper()
	switch {
	case want.Error != nil:
		if err == nil {
			t.Errorf("jinja2 failed with %q, but we produced %q", *want.Error, got)
		} else if want.Type == "TemplateError" && !strings.Contains(err.Error(), *want.Error) {
			t.Errorf("error %q, want one containing %q", err, *want.Error)
		}
	case err != nil:
		t.Errorf("unexpected error: %v", err)
	case got != *want.Output:
		t.Errorf("got %q\nwant %q", got, *want.Output)
	}
}

func runTemplateGold(t *testing.T, path string, snippets bool) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var gold templateGold
	if err := json.Unmarshal(raw, &gold); err != nil {
		t.Fatal(err)
	}
	fixedDate(t, gold.FixedDate)
	if snippets {
		moment, err := time.Parse("2006-01-02T15:04:05", gold.Strftime.Moment)
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range gold.Strftime.Cases {
			if got := strftime(moment, c.Format); got != c.Output {
				t.Errorf("strftime(%q) = %q, want %q", c.Format, got, c.Output)
			}
		}
	}
	for _, m := range gold.Models {
		tpl, err := ParseChatTemplate(m.Template)
		if err != nil {
			t.Errorf("%s: %v", m.Name, err)
			continue
		}
		tpl.BOS, tpl.EOS, tpl.PAD, tpl.UNK = m.Tokens["bos_token"], m.Tokens["eos_token"], m.Tokens["pad_token"], m.Tokens["unk_token"]
		for _, c := range m.Cases {
			t.Run(m.Name+"/"+c.Title, func(t *testing.T) {
				got, err := tpl.Render(c.Messages, c.Generation)
				checkAgainstGold(t, got, err, c.goldResult)
			})
		}
	}
	if !snippets {
		return
	}
	vars := goldVariables(t, gold.Snippets.Context)
	for _, c := range gold.Snippets.Cases {
		t.Run(c.Template, func(t *testing.T) {
			var got string
			tpl, err := parseJinja(c.Template)
			if err == nil {
				got, err = tpl.render(vars)
			}
			checkAgainstGold(t, got, err, c.goldResult)
		})
	}
}

func TestChatTemplatesAgainstJinja2(t *testing.T) {
	runTemplateGold(t, "testdata/chat_templates.json", true)
}

func TestMoreChatTemplatesAgainstJinja2(t *testing.T) {
	path := os.Getenv("ONNXNLP_TEMPLATE_GOLD")
	if path == "" {
		t.Skip("ONNXNLP_TEMPLATE_GOLD not set")
	}
	runTemplateGold(t, path, false)
}
