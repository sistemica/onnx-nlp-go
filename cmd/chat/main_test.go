package main

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	onnxnlp "github.com/sistemica/onnx-nlp-go"
	"github.com/sistemica/onnx-nlp-go/internal/fixture"
)

// fixtureChat is a conversation with a model that answers "a" with "bcdefg".
func fixtureChat(t *testing.T, dir string) *onnxnlp.Chat {
	t.Helper()
	fixture.Must(fixture.WriteLM(dir, true))
	fixture.Must(fixture.WriteLMTokenizer(dir))
	fixture.Must(os.WriteFile(filepath.Join(dir, "generation_config.json"), []byte(`{"eos_token_id":7}`), 0o644))
	m, err := onnxnlp.LoadLanguageModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := onnxnlp.LoadBPE(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	return &onnxnlp.Chat{
		Model: m, Tokenizer: tok, Options: m.Defaults(),
		Template: func(messages []onnxnlp.Message, _ bool) (string, error) {
			var b strings.Builder
			for _, msg := range messages {
				b.WriteString(msg.Content)
			}
			return b.String(), nil
		},
	}
}

func withFixtureChat(t *testing.T) **onnxnlp.Chat {
	t.Helper()
	saved := loadChat
	var chat *onnxnlp.Chat
	loadChat = func(dir, system, template string, _ io.Writer) (*onnxnlp.Chat, error) {
		chat = fixtureChat(t, t.TempDir())
		chat.System = system
		return chat, nil
	}
	t.Cleanup(func() { loadChat = saved })
	return &chat
}

func TestRunAnswersOnce(t *testing.T) {
	chat := withFixtureChat(t)
	var out, errOut bytes.Buffer
	if code := run([]string{"-model", "x", "-prompt", "a", "-stats", "-turns", "2", "-max-tokens", "100", "-seed", "5"}, strings.NewReader(""), &out, &errOut); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	if out.String() != "bcdefg\n" || !strings.Contains(errOut.String(), "6 new tokens") {
		t.Fatalf("output %q, stderr %q", out.String(), errOut.String())
	}
	if (*chat).MaxTurns != 2 || (*chat).Options.MaxNewTokens != 100 || (*chat).Options.Seed != 5 {
		t.Fatalf("flags must reach the chat: %+v", (*chat).Options)
	}
}

func TestRunConversation(t *testing.T) {
	chat := withFixtureChat(t)
	input := "a\n\n/help\n/history\n/reset\n/history\na\n/exit\na\n"
	var out, errOut bytes.Buffer
	if code := run([]string{"-model", "x"}, strings.NewReader(input), &out, &errOut); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	want := "bcdefg\nuser: a\nassistant: bcdefg\nbcdefg\n"
	if out.String() != want {
		t.Fatalf("output %q, want %q", out.String(), want)
	}
	if !strings.Contains(errOut.String(), "/reset") || !strings.Contains(errOut.String(), "conversation cleared") {
		t.Fatalf("stderr %q", errOut.String())
	}
	if len((*chat).History()) != 2 {
		t.Fatalf("after /reset and one more question the history holds one exchange: %v", (*chat).History())
	}
	out.Reset()
	if code := run([]string{"-model", "x"}, strings.NewReader("a"), &out, &errOut); code != 0 || out.String() != "bcdefg\n" {
		t.Fatalf("the last line needs no newline: %d %q", code, out.String())
	}
}

func TestInteractiveSessionShowsPromptAndHelp(t *testing.T) {
	withFixtureChat(t)
	saved := isTerminal
	isTerminal = func(io.Reader) bool { return true }
	defer func() { isTerminal = saved }()
	var out, errOut bytes.Buffer
	if code := run([]string{"-model", "x"}, strings.NewReader("a\n"), &out, &errOut); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	if out.String() != "> bcdefg\n> " || !strings.Contains(errOut.String(), "/exit quits") {
		t.Fatalf("output %q, stderr %q", out.String(), errOut.String())
	}
}

func TestIsTerminal(t *testing.T) {
	if isTerminal(strings.NewReader("")) {
		t.Fatal("a reader that is not a file is no terminal")
	}
	regular, err := os.Create(filepath.Join(t.TempDir(), "input.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if isTerminal(regular) {
		t.Fatal("a regular file is no terminal")
	}
	regular.Close()
	if isTerminal(regular) {
		t.Fatal("a closed file cannot be a terminal")
	}
	null, err := os.Open(os.DevNull)
	if err != nil {
		t.Skip("no null device")
	}
	defer null.Close()
	if !isTerminal(null) {
		t.Fatal("a character device counts as a terminal")
	}
}

func TestRunSamplingFlags(t *testing.T) {
	chat := withFixtureChat(t)
	var out, errOut bytes.Buffer
	args := []string{"-model", "x", "-prompt", "a", "-temperature", "0.5", "-top-k", "1", "-top-p", "0.9", "-repetition-penalty", "1.2"}
	if code := run(args, strings.NewReader(""), &out, &errOut); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	if o := (*chat).Options; o.Temperature != 0.5 || o.TopK != 1 || o.TopP != 0.9 || o.RepetitionPenalty != 1.2 {
		t.Fatalf("options %+v", o)
	}
	if code := run(append(args, "-greedy"), strings.NewReader(""), &out, &errOut); code != 0 || (*chat).Options.Temperature != 0 {
		t.Fatalf("-greedy wins over a temperature: %+v", (*chat).Options)
	}
}

type failingReader struct{}

func (failingReader) Read([]byte) (int, error) { return 0, errors.New("read failed") }

func TestRunErrors(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(nil, strings.NewReader(""), &out, &errOut); code != 2 {
		t.Errorf("no model: exit %d", code)
	}
	if code := run([]string{"-nope"}, strings.NewReader(""), &out, &errOut); code != 2 {
		t.Errorf("bad flag: exit %d", code)
	}
	if code := run([]string{"-model", t.TempDir()}, strings.NewReader(""), &out, &errOut); code != 1 {
		t.Errorf("a directory that is no model: exit %d", code)
	}
	chat := withFixtureChat(t)
	_ = chat
	if code := run([]string{"-model", "x"}, failingReader{}, &out, &errOut); code != 1 || !strings.Contains(errOut.String(), "read failed") {
		t.Errorf("unreadable input: exit %d, stderr %q", code, errOut.String())
	}
	saved := loadChat
	loadChat = func(dir, system, template string, _ io.Writer) (*onnxnlp.Chat, error) {
		c := fixtureChat(t, t.TempDir())
		c.Template = func([]onnxnlp.Message, bool) (string, error) { return "", nil }
		return c, nil
	}
	defer func() { loadChat = saved }()
	errOut.Reset()
	if code := run([]string{"-model", "x", "-prompt", "a"}, strings.NewReader(""), &out, &errOut); code != 1 || !strings.Contains(errOut.String(), "prompt is empty") {
		t.Errorf("an empty prompt: exit %d, stderr %q", code, errOut.String())
	}
	if code := run([]string{"-model", "x"}, strings.NewReader("hello\n"), &out, &errOut); code != 1 {
		t.Errorf("a failing turn ends the conversation: exit %d", code)
	}
}

func TestDefaultLoaderReadsAModelDirectory(t *testing.T) {
	dir := t.TempDir()
	fixture.Must(fixture.WriteLM(dir, true))
	var notice bytes.Buffer
	if _, err := loadChat(dir, "", "", &notice); err == nil {
		t.Fatal("a directory without tokenizer.json must fail")
	}
	fixture.Must(fixture.WriteLMTokenizer(dir))
	chat, err := loadChat(dir, "be brief", "", &notice)
	if err != nil {
		t.Fatal(err)
	}
	if chat.System != "be brief" || len(chat.Options.StopTokens) != 1 || chat.Options.StopTokens[0] != 7 {
		t.Fatalf("without eos in the model's config, <|im_end|> from the tokenizer stops the reply: %+v", chat.Options)
	}
	if chat.Template != nil || !strings.Contains(notice.String(), "using ChatML") {
		t.Fatalf("a model without a template falls back to ChatML and says so: %q", notice.String())
	}
	if _, err := loadChat(t.TempDir(), "", "", &notice); err == nil {
		t.Fatal("an empty directory must fail")
	}
	os.Remove(filepath.Join(dir, "onnx", "model.onnx"))
	if _, err := loadChat(dir, "", "", &notice); err == nil {
		t.Fatal("a missing model must fail")
	}
}

func TestLoaderUsesTheModelsChatTemplate(t *testing.T) {
	dir := t.TempDir()
	fixture.Must(fixture.WriteLM(dir, true))
	fixture.Must(fixture.WriteLMTokenizer(dir))
	fixture.Must(os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"eos_token": "<|im_end|>", "chat_template": "[{{ messages[0].content }}]"}`), 0o644))
	var notice bytes.Buffer
	chat, err := loadChat(dir, "sys", "", &notice)
	if err != nil || chat.Template == nil || notice.Len() != 0 {
		t.Fatalf("%v %v %q", err, chat.Template == nil, notice.String())
	}
	if prompt, err := chat.Template([]onnxnlp.Message{{Role: "system", Content: "x"}}, true); err != nil || prompt != "[x]" {
		t.Fatalf("the model's template renders the prompt: %q %v", prompt, err)
	}
	if got := chat.Options.StopTokens; len(got) != 1 || got[0] != 7 {
		t.Fatalf("the template's eos_token stops the reply: %v", got)
	}

	forced, err := loadChat(dir, "", "chatml", &notice)
	if err != nil || forced.Template != nil {
		t.Fatalf("-template chatml ignores the model's template: %v", err)
	}

	file := filepath.Join(t.TempDir(), "custom.jinja")
	fixture.Must(os.WriteFile(file, []byte("custom {{ eos_token }}"), 0o644))
	custom, err := loadChat(dir, "", file, &notice)
	if err != nil {
		t.Fatal(err)
	}
	if prompt, _ := custom.Template(nil, false); prompt != "custom <|im_end|>" {
		t.Fatalf("a template file gets the model's special tokens: %q", prompt)
	}
	if _, err := loadChat(dir, "", filepath.Join(t.TempDir(), "missing.jinja"), &notice); err == nil {
		t.Fatal("a missing template file must fail")
	}

	fixture.Must(os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"chat_template": "{% macro %}"}`), 0o644))
	if _, err := loadChat(dir, "", "", &notice); err == nil || !strings.Contains(err.Error(), "-template chatml") {
		t.Fatalf("a template that cannot be used names the way out: %v", err)
	}
	fixture.Must(os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"chat_template": "x", "eos_token": "<missing>"}`), 0o644))
	loose, err := loadChat(dir, "", "", &notice)
	if err != nil || len(loose.Options.StopTokens) != 1 || loose.Options.StopTokens[0] != 7 {
		t.Fatalf("an eos_token the vocabulary lacks falls back to <|im_end|>: %v %v", loose.Options.StopTokens, err)
	}
}

func TestMainExitsWithRunStatus(t *testing.T) {
	savedExit, savedArgs := exit, os.Args
	defer func() { exit, os.Args = savedExit, savedArgs }()
	code := -1
	exit = func(c int) { code = c }
	os.Args = []string{"chat"}
	main()
	if code != 2 {
		t.Fatalf("exit code %d", code)
	}
}
