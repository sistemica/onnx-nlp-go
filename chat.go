package onnxnlp

import (
	"errors"
	"strings"
	"time"
)

// Message is one turn of a conversation. Role is "system", "user" or "assistant".
type Message struct {
	Role, Content string
}

// RenderChatML writes messages in the ChatML format that Qwen and SmolLM2 are trained on, and with
// generationPrompt set ends with the start of the assistant's turn.
func RenderChatML(messages []Message, generationPrompt bool) string {
	var b strings.Builder
	for _, m := range messages {
		b.WriteString("<|im_start|>" + m.Role + "\n" + m.Content + "<|im_end|>\n")
	}
	if generationPrompt {
		b.WriteString("<|im_start|>assistant\n")
	}
	return b.String()
}

// Chat is a conversation with a model that remembers only the last few exchanges.
type Chat struct {
	Model     *LanguageModel
	Tokenizer *BPE
	// System is sent first on every turn. It is not part of the history.
	System  string
	Options GenerateOptions
	// MaxTurns is how many user and assistant exchanges are kept; zero keeps four.
	MaxTurns int
	// MaxPromptTokens drops the oldest exchanges until the prompt fits; zero means 2048.
	MaxPromptTokens int
	// Template renders the messages into the prompt text; the default is ChatML.
	Template func(messages []Message, generationPrompt bool) string

	// Last describes the most recent reply.
	Last ChatStats

	history []Message
}

// ChatStats describes one reply: the prompt it was given, the tokens it took, and how long.
type ChatStats struct {
	PromptTokens, NewTokens int
	Elapsed                 time.Duration
}

// History returns the remembered messages.
func (c *Chat) History() []Message { return append([]Message(nil), c.history...) }

// Reset forgets the conversation.
func (c *Chat) Reset() { c.history = nil }

func (c *Chat) messages() []Message {
	var out []Message
	if c.System != "" {
		out = append(out, Message{"system", c.System})
	}
	return append(out, c.history...)
}

func (c *Chat) render(generationPrompt bool) string {
	template := c.Template
	if template == nil {
		template = RenderChatML
	}
	return template(c.messages(), generationPrompt)
}

// Say adds the user's message, streams the reply through emit, remembers it and returns it.
func (c *Chat) Say(user string, emit func(text string)) (string, error) {
	if c.Model == nil || c.Tokenizer == nil {
		return "", errors.New("chat: no model or tokenizer")
	}
	turns, budget := c.MaxTurns, c.MaxPromptTokens
	if turns <= 0 {
		turns = 4
	}
	if budget <= 0 {
		budget = 2048
	}
	c.history = append(c.history, Message{"user", user})
	// Each exchange is a user message followed by the reply; the last message is the new question.
	if over := len(c.history) - (2*turns - 1); over > 0 {
		c.history = c.history[over:]
	}
	prompt := c.Tokenizer.Encode(c.render(true))
	for len(prompt) > budget && len(c.history) > 1 {
		c.history = c.history[min(2, len(c.history)-1):]
		prompt = c.Tokenizer.Encode(c.render(true))
	}
	stream := c.Tokenizer.NewDecodeStream()
	var reply strings.Builder
	write := func(text string) {
		if text != "" {
			reply.WriteString(text)
			if emit != nil {
				emit(text)
			}
		}
	}
	opts := c.Options
	if len(opts.StopTokens) == 0 {
		opts.StopTokens = c.Model.Defaults().StopTokens
	}
	start := time.Now()
	tokens, err := c.Model.Generate(prompt, opts, func(token int) bool {
		write(stream.Push(token))
		return true
	})
	if err != nil {
		c.history = c.history[:len(c.history)-1]
		return "", err
	}
	write(stream.Flush())
	c.Last = ChatStats{PromptTokens: len(prompt), NewTokens: len(tokens), Elapsed: time.Since(start)}
	c.history = append(c.history, Message{"assistant", reply.String()})
	return reply.String(), nil
}
