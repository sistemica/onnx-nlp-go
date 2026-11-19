package main

import (
	"bytes"
	"errors"
	"hash/fnv"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"onnxgo"
)

type stubEmbedder struct{}

func (stubEmbedder) Embed(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		h := fnv.New64a()
		h.Write([]byte(strings.Fields(text)[0]))
		rng := rand.New(rand.NewPCG(h.Sum64(), 1))
		noise := rand.New(rand.NewPCG(uint64(i), 7))
		out[i] = make([]float32, 16)
		for d := range out[i] {
			out[i][d] = float32(rng.NormFloat64() + 0.2*noise.NormFloat64())
		}
	}
	return out, nil
}

func withStub(t *testing.T) {
	t.Helper()
	saved := loadEmbedder
	loadEmbedder = func(string) (onnxgo.TextEmbedder, error) { return stubEmbedder{}, nil }
	t.Cleanup(func() { loadEmbedder = saved })
}

func corpus() string {
	var b strings.Builder
	for _, subject := range [][]string{{"astronomy", "telescope", "galaxy"}, {"cooking", "recipe", "butter"}, {"football", "goal", "league"}} {
		for i := 0; i < 30; i++ {
			b.WriteString(subject[0] + " " + subject[1+i%2] + " and the " + subject[2-i%2] + "\n")
		}
	}
	return b.String()
}

func TestRunPrintsTopicsAndAssignments(t *testing.T) {
	withStub(t)
	var out, errOut bytes.Buffer
	args := []string{"-model", "unused", "-stop-words", "-assignments", "-top-words", "3", "-min-topic-size", "10"}
	if code := run(args, strings.NewReader("\n"+corpus()+"   \n"), &out, &errOut); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	text := out.String()
	if strings.Count(text, "topic ") < 3 || !strings.Contains(text, "documents):") {
		t.Fatalf("expected three topics in the output:\n%s", text)
	}
	if lines := strings.Count(text, "\n"); lines < 90 {
		t.Fatalf("expected one line per document after the topics, got %d lines", lines)
	}
}

func TestRunReportsDocumentsOutsideEveryTopicAsOutliers(t *testing.T) {
	withStub(t)
	var out, errOut bytes.Buffer
	if code := run([]string{"-model", "unused", "-min-topic-size", "31"}, strings.NewReader(corpus()), &out, &errOut); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	if !strings.HasPrefix(out.String(), "outliers (") || strings.Contains(out.String(), "topic 0") {
		t.Fatalf("subjects of 30 documents cannot form topics of at least 31:\n%s", out.String())
	}
}

func TestRunReadsDocumentsFromAFile(t *testing.T) {
	withStub(t)
	path := filepath.Join(t.TempDir(), "docs.txt")
	if err := os.WriteFile(path, []byte(corpus()), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := run([]string{"-model", "unused", "-docs", path}, strings.NewReader(""), &out, &errOut); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "documents):") {
		t.Fatalf("output %q", out.String())
	}
}

type brokenReader struct{}

func (brokenReader) Read([]byte) (int, error) { return 0, errors.New("read failed") }

func TestRunUsageAndErrors(t *testing.T) {
	withStub(t)
	var out, errOut bytes.Buffer
	if code := run(nil, strings.NewReader(""), &out, &errOut); code != 2 {
		t.Errorf("missing model: exit %d", code)
	}
	if code := run([]string{"-nope"}, strings.NewReader(""), &out, &errOut); code != 2 {
		t.Errorf("bad flag: exit %d", code)
	}
	if code := run([]string{"-model", "x", "-docs", filepath.Join(t.TempDir(), "missing.txt")}, strings.NewReader(""), &out, &errOut); code != 1 {
		t.Errorf("missing file: exit %d", code)
	}
	if code := run([]string{"-model", "x"}, brokenReader{}, &out, &errOut); code != 1 {
		t.Errorf("unreadable input: exit %d", code)
	}
	if code := run([]string{"-model", "x"}, strings.NewReader("only one document\n"), &out, &errOut); code != 1 {
		t.Errorf("too few documents: exit %d", code)
	}
	loadEmbedder = func(string) (onnxgo.TextEmbedder, error) { return nil, errors.New("no such model") }
	if code := run([]string{"-model", "x"}, strings.NewReader(corpus()), &out, &errOut); code != 1 || !strings.Contains(errOut.String(), "no such model") {
		t.Errorf("model failure: exit %d, stderr %q", code, errOut.String())
	}
}

func TestDefaultLoaderRejectsAMissingModel(t *testing.T) {
	if _, err := loadEmbedder(t.TempDir()); err == nil {
		t.Fatal("an empty directory is not a model")
	}
}

func TestMainExitsWithRunStatus(t *testing.T) {
	savedExit, savedArgs := exit, os.Args
	defer func() { exit, os.Args = savedExit, savedArgs }()
	code := -1
	exit = func(c int) { code = c }
	os.Args = []string{"bertopic"}
	main()
	if code != 2 {
		t.Fatalf("exit code %d", code)
	}
}
