package onnxnlp

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/sistemica/onnx-nlp-go/internal/fixture"
)

func TestRenderChatMLMatchesQwen(t *testing.T) {
	// The text transformers' chat template produces for Qwen2.5 and this conversation.
	want := "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n" +
		"<|im_start|>user\nWie viele Beine hat eine Spinne?<|im_end|>\n<|im_start|>assistant\n"
	got := RenderChatML([]Message{
		{"system", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
		{"user", "Wie viele Beine hat eine Spinne?"},
	}, true)
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	if got := RenderChatML([]Message{{"user", "hi"}, {"assistant", "yo"}}, false); got != "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nyo<|im_end|>\n" {
		t.Fatalf("without a generation prompt: %q", got)
	}
}

// echoChat is a conversation with the successor model, whose answer to "a" is "bcdefg".
func echoChat(t *testing.T) *Chat {
	t.Helper()
	dir := lmDir(t, true)
	fixture.Must(fixture.WriteLMTokenizer(dir))
	fixture.Must(os.WriteFile(filepath.Join(dir, "generation_config.json"), []byte(`{"eos_token_id":7}`), 0o644))
	m, err := LoadLanguageModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := LoadBPE(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	return &Chat{
		Model: m, Tokenizer: tok, Options: m.Defaults(),
		Template: func(messages []Message, _ bool) (string, error) {
			var b strings.Builder
			for _, msg := range messages {
				b.WriteString(msg.Content)
			}
			return b.String(), nil
		},
	}
}

func TestChatRemembersTheConversation(t *testing.T) {
	c := echoChat(t)
	var streamed []string
	reply, err := c.Say("a", func(text string) { streamed = append(streamed, text) })
	if err != nil || reply != "bcdefg" || strings.Join(streamed, "") != reply || len(streamed) < 2 {
		t.Fatalf("reply %q, streamed %q, %v", reply, streamed, err)
	}
	if h := c.History(); len(h) != 2 || h[0] != (Message{"user", "a"}) || h[1] != (Message{"assistant", "bcdefg"}) {
		t.Fatalf("history %v", h)
	}
	if c.Last.NewTokens != 6 || c.Last.PromptTokens != 1 || c.Last.Elapsed <= 0 {
		t.Fatalf("stats %+v", c.Last)
	}
	if _, err := c.Say("a", nil); err != nil || len(c.History()) != 4 || c.Last.PromptTokens != 8 {
		t.Fatalf("the second prompt holds the whole first exchange: %v %+v %v", c.History(), c.Last, err)
	}
	c.Reset()
	if len(c.History()) != 0 {
		t.Fatal("Reset forgets everything")
	}
}

func TestChatKeepsOnlyTheLastTurns(t *testing.T) {
	c := echoChat(t)
	c.MaxTurns = 2
	for i := 0; i < 4; i++ {
		if _, err := c.Say("a", nil); err != nil {
			t.Fatal(err)
		}
	}
	if h := c.History(); len(h) != 4 || h[0].Role != "user" {
		t.Fatalf("two exchanges are kept, starting with a question: %v", h)
	}
	c = echoChat(t)
	c.MaxTurns = 1
	for i := 0; i < 3; i++ {
		c.Say("a", nil)
	}
	if h := c.History(); len(h) != 2 || h[0].Role != "user" || c.Last.PromptTokens != 1 {
		t.Fatalf("one turn means no memory of earlier ones: %v, prompt %d tokens", h, c.Last.PromptTokens)
	}
}

func TestChatDropsOldTurnsToFitThePrompt(t *testing.T) {
	c := echoChat(t)
	c.MaxPromptTokens = 3
	c.Say("a", nil)
	if _, err := c.Say("a", nil); err != nil {
		t.Fatal(err)
	}
	if h := c.History(); len(h) != 2 || c.Last.PromptTokens != 1 {
		t.Fatalf("the first exchange no longer fits next to the new question: %v, prompt %d tokens", h, c.Last.PromptTokens)
	}
}

func TestChatSendsTheSystemMessageFirst(t *testing.T) {
	c := echoChat(t)
	c.System = "bb"
	var seen []Message
	inner := c.Template
	c.Template = func(messages []Message, gen bool) (string, error) {
		seen = messages
		return inner(messages, gen)
	}
	if _, err := c.Say("a", nil); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 2 || seen[0] != (Message{"system", "bb"}) {
		t.Fatalf("messages %v", seen)
	}
	if h := c.History(); len(h) != 2 {
		t.Fatalf("the system message is not part of the history: %v", h)
	}
}

func TestChatErrors(t *testing.T) {
	if _, err := (&Chat{}).Say("hi", nil); err == nil {
		t.Fatal("a chat without a model must fail")
	}
	c := echoChat(t)
	c.Model.maxPositions = 1
	if _, err := c.Say("a", nil); err == nil {
		t.Fatal("a prompt that does not fit must fail")
	}
	if len(c.History()) != 0 {
		t.Fatalf("a failed question is not remembered: %v", c.History())
	}
	c = echoChat(t)
	c.Template = nil
	c.Options.StopTokens = nil
	c.Model.defaults.StopTokens = []int{7}
	if _, err := c.Say("a", nil); err != nil {
		t.Fatalf("the default template is ChatML: %v", err)
	}
	if slices.Equal(c.Options.StopTokens, []int{7}) {
		t.Fatal("stop tokens from the model are used without being written into the options")
	}
}
