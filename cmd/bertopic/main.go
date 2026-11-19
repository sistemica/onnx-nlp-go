package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"onnxgo"
)

var (
	exit         = os.Exit
	loadEmbedder = func(dir string) (onnxgo.TextEmbedder, error) { return onnxgo.LoadEmbedder(dir) }
)

func main() {
	exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

func readDocuments(r io.Reader) ([]string, error) {
	var docs []string
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 1<<20), 1<<26)
	for scanner.Scan() {
		if line := strings.TrimSpace(scanner.Text()); line != "" {
			docs = append(docs, line)
		}
	}
	return docs, scanner.Err()
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("bertopic", flag.ContinueOnError)
	fs.SetOutput(stderr)
	model := fs.String("model", "", "directory of the sentence embedding model (ONNX export)")
	docsFile := fs.String("docs", "", "file with one document per line (default: standard input)")
	minSize := fs.Int("min-topic-size", 10, "smallest number of documents that forms a topic")
	topWords := fs.Int("top-words", 10, "words listed per topic")
	stopWords := fs.Bool("stop-words", false, "drop English stop words before counting")
	seed := fs.Uint64("seed", 42, "random seed of the dimensionality reduction")
	assignments := fs.Bool("assignments", false, "also print the topic of every document")
	assignFile := fs.String("assign", "", "file with one document per line to assign to the topics found (printed as topic, strength, document)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *model == "" {
		fmt.Fprintln(stderr, "usage: bertopic -model DIR [-docs FILE] [-min-topic-size 10] [-top-words 10] [-stop-words] [-seed 42] [-assignments] [-assign FILE]")
		return 2
	}
	input := stdin
	if *docsFile != "" {
		f, err := os.Open(*docsFile)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		defer f.Close()
		input = f
	}
	docs, err := readDocuments(input)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	embedder, err := loadEmbedder(*model)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	topics := onnxgo.NewBERTopic(embedder)
	topics.MinTopicSize, topics.TopNWords, topics.RemoveStopWords, topics.UMAP.Seed = *minSize, *topWords, *stopWords, *seed
	result, err := topics.Fit(docs)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	for _, topic := range result.Topics {
		words := make([]string, 0, len(topic.Words))
		for _, w := range topic.Words {
			words = append(words, w.Word)
		}
		name := fmt.Sprintf("topic %d", topic.ID)
		if topic.ID < 0 {
			name = "outliers"
		}
		fmt.Fprintf(stdout, "%s (%d documents): %s\n", name, topic.Size, strings.Join(words, ", "))
	}
	if *assignments {
		for i, id := range result.Assignments {
			fmt.Fprintf(stdout, "%d\t%.3f\t%s\n", id, result.Strengths[i], docs[i])
		}
	}
	if *assignFile != "" {
		return assign(result, *assignFile, stdout, stderr)
	}
	return 0
}

func assign(model *onnxgo.TopicModel, path string, stdout, stderr io.Writer) int {
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	defer f.Close()
	docs, err := readDocuments(f)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	topics, strengths, err := model.Transform(docs)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	for i, id := range topics {
		fmt.Fprintf(stdout, "%d\t%.3f\t%s\n", id, strengths[i], docs[i])
	}
	return 0
}
