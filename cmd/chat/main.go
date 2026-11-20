package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"

	onnxnlp "github.com/sistemica/onnx-nlp-go"
)

var exit = os.Exit

// loadChat builds the conversation for a model directory with config.json, tokenizer.json and
// onnx/model.onnx. Tests replace it.
var loadChat = func(dir, system string) (*onnxnlp.Chat, error) {
	model, err := onnxnlp.LoadLanguageModel(dir)
	if err != nil {
		return nil, err
	}
	tok, err := onnxnlp.LoadBPE(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	chat := &onnxnlp.Chat{Model: model, Tokenizer: tok, System: system, Options: model.Defaults()}
	if len(chat.Options.StopTokens) == 0 {
		if id, ok := tok.TokenID("<|im_end|>"); ok {
			chat.Options.StopTokens = []int{id}
		}
	}
	return chat, nil
}

// trimMemory frees what loading left behind and makes the collector run when the heap has grown a
// quarter past the weights, instead of waiting for it to double.
func trimMemory() {
	debug.SetGCPercent(25)
	runtime.GC()
	debug.FreeOSMemory()
}

// isTerminal reports whether r is a character device such as a terminal; tests replace it.
var isTerminal = func(r io.Reader) bool {
	f, ok := r.(*os.File)
	if !ok {
		return false
	}
	info, err := f.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}

func main() {
	exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

const help = "commands: /reset forgets the conversation, /history shows it, /exit quits"

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("chat", flag.ContinueOnError)
	fs.SetOutput(stderr)
	model := fs.String("model", "", "directory of the language model (config.json, tokenizer.json, onnx/model.onnx)")
	system := fs.String("system", "You are a helpful assistant.", "system message sent with every turn")
	turns := fs.Int("turns", 4, "exchanges to remember")
	maxTokens := fs.Int("max-tokens", 256, "longest reply in tokens")
	temperature := fs.Float64("temperature", 0, "sampling temperature (default: the model's own setting)")
	topK := fs.Int("top-k", 0, "sample only from the k most likely tokens (default: the model's own setting)")
	topP := fs.Float64("top-p", 0, "sample only from tokens that together reach this probability (default: the model's own setting)")
	penalty := fs.Float64("repetition-penalty", 0, "penalty for repeated tokens (default: the model's own setting)")
	greedy := fs.Bool("greedy", false, "always pick the most likely token")
	seed := fs.Uint64("seed", 1, "random seed for sampling")
	prompt := fs.String("prompt", "", "answer this once and exit")
	stats := fs.Bool("stats", false, "report tokens and speed after every reply")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *model == "" {
		fmt.Fprintln(stderr, "usage: chat -model DIR [-prompt TEXT] [-system TEXT] [-turns 4] [-max-tokens 256] [-greedy] [-temperature T] [-top-k K] [-top-p P] [-stats]")
		return 2
	}
	chat, err := loadChat(*model, *system)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	trimMemory()
	chat.MaxTurns = *turns
	chat.Options.MaxNewTokens, chat.Options.Seed = *maxTokens, *seed
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "temperature":
			chat.Options.Temperature = *temperature
		case "top-k":
			chat.Options.TopK = *topK
		case "top-p":
			chat.Options.TopP = *topP
		case "repetition-penalty":
			chat.Options.RepetitionPenalty = *penalty
		}
	})
	if *greedy {
		chat.Options.Temperature = 0
	}
	ask := func(line string) int {
		if _, err := chat.Say(line, func(text string) { fmt.Fprint(stdout, text) }); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		fmt.Fprintln(stdout)
		if *stats {
			l := chat.Last
			fmt.Fprintf(stderr, "[%d prompt tokens, %d new tokens, %.1f tokens/s]\n", l.PromptTokens, l.NewTokens, float64(l.NewTokens)/l.Elapsed.Seconds())
		}
		return 0
	}
	if *prompt != "" {
		return ask(*prompt)
	}
	interactive := isTerminal(stdin)
	if interactive {
		fmt.Fprintln(stderr, help)
	}
	scanner := bufio.NewScanner(stdin)
	scanner.Buffer(make([]byte, 1<<20), 1<<24)
	for {
		if interactive {
			fmt.Fprint(stdout, "> ")
		}
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		switch {
		case line == "":
		case line == "/exit" || line == "/quit":
			return 0
		case line == "/reset":
			chat.Reset()
			fmt.Fprintln(stderr, "conversation cleared")
		case line == "/history":
			for _, m := range chat.History() {
				fmt.Fprintf(stdout, "%s: %s\n", m.Role, m.Content)
			}
		case strings.HasPrefix(line, "/"):
			fmt.Fprintln(stderr, help)
		default:
			if code := ask(line); code != 0 {
				return code
			}
		}
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}
