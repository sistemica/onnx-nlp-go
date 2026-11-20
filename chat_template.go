package onnxnlp

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// ErrNoChatTemplate is returned by LoadChatTemplate for a directory that defines no chat template.
var ErrNoChatTemplate = errors.New("no chat template found")

// ChatTemplate turns a conversation into the prompt text a model was trained on. It is the Jinja
// template that models ship in tokenizer_config.json, supporting the subset such templates use:
// conditionals, loops, set (also on a namespace), the usual filters and tests, string and dict
// methods, raise_exception and strftime_now. Unsupported tags are rejected when it is parsed.
type ChatTemplate struct {
	// Special tokens made available to the template as bos_token, eos_token, pad_token and
	// unk_token; empty ones stay undefined.
	BOS, EOS, PAD, UNK string

	tpl *jinjaTemplate
}

// ParseChatTemplate compiles Jinja source.
func ParseChatTemplate(source string) (*ChatTemplate, error) {
	tpl, err := parseJinja(source)
	if err != nil {
		return nil, fmt.Errorf("chat template: %w", err)
	}
	return &ChatTemplate{tpl: tpl}, nil
}

// LoadChatTemplate reads the template of a model directory: chat_template.jinja, chat_template.json
// or the chat_template entry of tokenizer_config.json, in that order, together with the special
// tokens from tokenizer_config.json. It returns ErrNoChatTemplate when there is none.
func LoadChatTemplate(dir string) (*ChatTemplate, error) { return loadChatTemplate(dir, "") }

// LoadChatTemplateFile renders with the Jinja template in file instead of the model's own, and
// still takes the special tokens from the model directory.
func LoadChatTemplateFile(file, dir string) (*ChatTemplate, error) {
	if file == "" {
		return nil, errors.New("chat template: no file given")
	}
	return loadChatTemplate(dir, file)
}

func loadChatTemplate(dir, file string) (*ChatTemplate, error) {
	var cfg struct {
		Template json.RawMessage `json:"chat_template"`
		BOS      json.RawMessage `json:"bos_token"`
		EOS      json.RawMessage `json:"eos_token"`
		PAD      json.RawMessage `json:"pad_token"`
		UNK      json.RawMessage `json:"unk_token"`
	}
	if err := readJSON(filepath.Join(dir, "tokenizer_config.json"), &cfg); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	var source string
	var err error
	if file != "" {
		var b []byte
		if b, err = os.ReadFile(file); err == nil {
			source = string(b)
		}
	} else {
		source, err = templateSource(dir, cfg.Template)
	}
	if err != nil {
		return nil, err
	}
	t, err := ParseChatTemplate(source)
	if err != nil {
		return nil, err
	}
	t.BOS, t.EOS, t.PAD, t.UNK = tokenText(cfg.BOS), tokenText(cfg.EOS), tokenText(cfg.PAD), tokenText(cfg.UNK)
	return t, nil
}

func templateSource(dir string, inConfig json.RawMessage) (string, error) {
	if b, err := os.ReadFile(filepath.Join(dir, "chat_template.jinja")); err == nil {
		return string(b), nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	var file struct {
		Template json.RawMessage `json:"chat_template"`
	}
	if err := readJSON(filepath.Join(dir, "chat_template.json"), &file); err == nil {
		inConfig = file.Template
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	var single string
	if json.Unmarshal(inConfig, &single) == nil && single != "" {
		return single, nil
	}
	var named []struct{ Name, Template string }
	if json.Unmarshal(inConfig, &named) == nil {
		for _, n := range named {
			if n.Name == "default" {
				return n.Template, nil
			}
		}
		if len(named) > 0 {
			return named[0].Template, nil
		}
	}
	return "", ErrNoChatTemplate
}

// tokenText reads a special token that tokenizer_config.json writes either as a string or as an
// object with a content field.
func tokenText(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var obj struct{ Content string }
	json.Unmarshal(raw, &obj)
	return obj.Content
}

// Render produces the prompt for messages; with generationPrompt set it ends where the
// assistant's reply begins.
func (t *ChatTemplate) Render(messages []Message, generationPrompt bool) (string, error) {
	list := make([]any, len(messages))
	for i, m := range messages {
		d := newJinjaDict()
		d.set("role", m.Role)
		d.set("content", m.Content)
		list[i] = d
	}
	vars := map[string]any{"messages": list, "add_generation_prompt": generationPrompt}
	for name, token := range map[string]string{"bos_token": t.BOS, "eos_token": t.EOS, "pad_token": t.PAD, "unk_token": t.UNK} {
		if token != "" {
			vars[name] = token
		}
	}
	out, err := t.tpl.render(vars)
	if err != nil {
		return "", fmt.Errorf("chat template: %w", err)
	}
	return out, nil
}
