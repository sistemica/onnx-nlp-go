package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"onnxgo"
)

var exit = os.Exit

func main() {
	exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func parseSpans(s string) ([]onnxgo.Span, error) {
	var spans []onnxgo.Span
	for _, part := range strings.Split(s, ",") {
		from, to, ok := strings.Cut(strings.TrimSpace(part), ":")
		start, err1 := strconv.Atoi(from)
		end, err2 := strconv.Atoi(to)
		if !ok || err1 != nil || err2 != nil {
			return nil, fmt.Errorf("entity %q is not START:END", part)
		}
		spans = append(spans, onnxgo.Span{Start: start, End: end})
	}
	return spans, nil
}

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("glirel", flag.ContinueOnError)
	fs.SetOutput(stderr)
	model := fs.String("model", "", "directory of the GLiREL ONNX export")
	labels := fs.String("labels", "", "comma-separated relation labels")
	entities := fs.String("entities", "", "comma-separated entity token ranges START:END (END exclusive)")
	threshold := fs.Float64("threshold", 0.5, "minimum relation probability")
	topK := fs.Int("topk", 0, "keep at most this many labels per entity pair (0 keeps all)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *model == "" || *labels == "" || *entities == "" || fs.NArg() == 0 {
		fmt.Fprintln(stderr, "usage: glirel -model DIR -labels a,b -entities 0:2,5:6 [-threshold 0.5] [-topk N] TOKEN...")
		return 2
	}
	spans, err := parseSpans(*entities)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	g, err := onnxgo.LoadGLiREL(*model)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	relations, err := g.Extract(fs.Args(), spans, strings.Split(*labels, ","), float32(*threshold), *topK)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	for _, r := range relations {
		fmt.Fprintf(stdout, "%.3f  %s | %s | %s\n", r.Score, strings.Join(r.HeadText, " "), r.Label, strings.Join(r.TailText, " "))
	}
	return 0
}
