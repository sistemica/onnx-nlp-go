package onnxnlp

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func templateDir(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

var hello = []Message{{"user", "hi"}}

func TestLoadChatTemplateSources(t *testing.T) {
	const source = "{{ bos_token }}{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}{{ eos_token }}{{ pad_token }}{{ unk_token }}"
	for name, tc := range map[string]struct {
		files map[string]string
		want  string
	}{
		"in the tokenizer config": {
			map[string]string{"tokenizer_config.json": `{"chat_template": "` + source + `", "bos_token": "<s>", "eos_token": {"content": "</s>"}, "pad_token": "<pad>", "unk_token": null}`},
			"<s>user:hi;</s><pad>",
		},
		"as a jinja file": {
			map[string]string{"chat_template.jinja": source, "tokenizer_config.json": `{"chat_template": "wrong", "eos_token": "E"}`},
			"user:hi;E",
		},
		"as a jinja file without a config": {
			map[string]string{"chat_template.jinja": source},
			"user:hi;",
		},
		"in chat_template.json": {
			map[string]string{"chat_template.json": `{"chat_template": "` + source + `"}`},
			"user:hi;",
		},
		"as a list of named templates": {
			map[string]string{"tokenizer_config.json": `{"chat_template": [{"name": "rag", "template": "rag"}, {"name": "default", "template": "default {{ messages|length }}"}]}`},
			"default 1",
		},
		"as a list without a default": {
			map[string]string{"tokenizer_config.json": `{"chat_template": [{"name": "rag", "template": "first"}]}`},
			"first",
		},
	} {
		tpl, err := LoadChatTemplate(templateDir(t, tc.files))
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		if got, err := tpl.Render(hello, true); err != nil || got != tc.want {
			t.Errorf("%s: got %q %v, want %q", name, got, err, tc.want)
		}
	}
}

func TestLoadChatTemplateFailures(t *testing.T) {
	for name, files := range map[string]map[string]string{
		"no files":                     {},
		"a config without a template":  {"tokenizer_config.json": `{"eos_token": "x"}`},
		"an empty template string":     {"tokenizer_config.json": `{"chat_template": ""}`},
		"an empty list":                {"tokenizer_config.json": `{"chat_template": []}`},
		"a template of the wrong type": {"tokenizer_config.json": `{"chat_template": 5}`},
	} {
		if _, err := LoadChatTemplate(templateDir(t, files)); !errors.Is(err, ErrNoChatTemplate) {
			t.Errorf("%s: got %v, want ErrNoChatTemplate", name, err)
		}
	}
	for name, files := range map[string]map[string]string{
		"a broken config":        {"tokenizer_config.json": `{`},
		"a broken template file": {"chat_template.jinja": "{% if x %}"},
		"a broken template":      {"tokenizer_config.json": `{"chat_template": "{{ 1 +* 2 }}"}`},
		"a broken json template": {"chat_template.json": `{`},
	} {
		if _, err := LoadChatTemplate(templateDir(t, files)); err == nil || errors.Is(err, ErrNoChatTemplate) {
			t.Errorf("%s: got %v, want a parse or read error", name, err)
		}
	}
	blocked := t.TempDir()
	if err := os.Mkdir(filepath.Join(blocked, "chat_template.jinja"), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadChatTemplate(blocked); err == nil || errors.Is(err, ErrNoChatTemplate) {
		t.Errorf("an unreadable template file must be reported: %v", err)
	}
	if _, err := ParseChatTemplate("{% nope %}"); err == nil || !strings.Contains(err.Error(), "chat template") {
		t.Errorf("a parse error names the chat template: %v", err)
	}
}

func TestChatTemplateRenderReportsErrors(t *testing.T) {
	tpl, err := ParseChatTemplate("{{ raise_exception('System role not supported') }}")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tpl.Render(hello, false); err == nil || !strings.Contains(err.Error(), "System role not supported") {
		t.Fatalf("got %v", err)
	}
}

func TestChatUsesAModelTemplateAndReportsItsErrors(t *testing.T) {
	c := echoChat(t)
	tpl, err := ParseChatTemplate("{% for m in messages %}{{ m.content }}{% endfor %}{% if messages|length > 2 %}{{ raise_exception('too long') }}{% endif %}")
	if err != nil {
		t.Fatal(err)
	}
	c.Template = tpl.Render
	if reply, err := c.Say("a", nil); err != nil || reply != "bcdefg" {
		t.Fatalf("%q %v", reply, err)
	}
	if _, err := c.Say("a", nil); err == nil || !strings.Contains(err.Error(), "too long") {
		t.Fatalf("a template error must reach the caller: %v", err)
	}
	if len(c.History()) != 2 {
		t.Fatalf("the failed question must not stay in the history: %v", c.History())
	}
}

func TestLoadChatTemplateFile(t *testing.T) {
	dir := templateDir(t, map[string]string{"tokenizer_config.json": `{"chat_template": "model's own", "eos_token": "E"}`})
	file := filepath.Join(t.TempDir(), "other.jinja")
	if err := os.WriteFile(file, []byte("other {{ eos_token }}"), 0o644); err != nil {
		t.Fatal(err)
	}
	tpl, err := LoadChatTemplateFile(file, dir)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := tpl.Render(hello, false); got != "other E" {
		t.Fatalf("got %q", got)
	}
	if _, err := LoadChatTemplateFile("", dir); err == nil {
		t.Fatal("no file given must fail")
	}
	if _, err := LoadChatTemplateFile(filepath.Join(t.TempDir(), "missing"), dir); err == nil {
		t.Fatal("a missing file must fail")
	}
}
