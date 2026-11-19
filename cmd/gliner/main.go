package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"onnxgo"
)

var exit = os.Exit

func main() {
	exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("gliner", flag.ContinueOnError)
	fs.SetOutput(stderr)
	model := fs.String("model", "", "directory of the GLiNER2.5 ONNX export")
	labels := fs.String("labels", "", "comma-separated entity labels")
	threshold := fs.Float64("threshold", 0.5, "minimum entity probability")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *model == "" || *labels == "" || fs.NArg() != 1 {
		fmt.Fprintln(stderr, "usage: gliner -model DIR -labels a,b,c [-threshold 0.5] TEXT")
		return 2
	}
	g, err := onnxgo.LoadGLiNER(*model)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	entities, err := g.Extract(fs.Arg(0), strings.Split(*labels, ","), float32(*threshold))
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	for _, e := range entities {
		fmt.Fprintf(stdout, "%-14s %.3f  [%d,%d)  %s\n", e.Label, e.Score, e.Start, e.End, e.Text)
	}
	return 0
}
