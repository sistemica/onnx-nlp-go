package main

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/sistemica/onnx-nlp-go/internal/fixture"
)

func TestRun(t *testing.T) {
	dir := t.TempDir()
	fixture.Must(fixture.WriteGLiNER(dir))
	var out, errOut bytes.Buffer
	if code := run([]string{"-model", dir, "-labels", fixture.GLiNERLabel, fixture.GLiNERText}, &out, &errOut); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "person") || !strings.Contains(out.String(), "[0,5)  alice") {
		t.Fatalf("output %q", out.String())
	}
	out.Reset()
	if code := run([]string{"-model", dir, "-labels", fixture.GLiNERLabel, "-threshold", "0.99", fixture.GLiNERText}, &out, &errOut); code != 0 || out.Len() != 0 {
		t.Fatalf("threshold: exit %d, output %q", code, out.String())
	}
}

func TestRunUsageAndErrors(t *testing.T) {
	var out, errOut bytes.Buffer
	for name, args := range map[string][]string{
		"no arguments": nil,
		"no text":      {"-model", "x", "-labels", "a"},
		"bad flag":     {"-nope"},
	} {
		errOut.Reset()
		if code := run(args, &out, &errOut); code != 2 {
			t.Errorf("%s: exit %d", name, code)
		}
	}
	errOut.Reset()
	if code := run([]string{"-model", t.TempDir(), "-labels", "a", "text"}, &out, &errOut); code != 1 || errOut.Len() == 0 {
		t.Fatalf("missing model: exit %d, stderr %q", code, errOut.String())
	}
}

func TestRunReportsExtractionFailure(t *testing.T) {
	dir := t.TempDir()
	fixture.Must(fixture.WriteGLiNER(dir))
	var out, errOut bytes.Buffer
	// the fixture's boundary graph describes exactly one label
	code := run([]string{"-model", dir, "-labels", "a,b", fixture.GLiNERText}, &out, &errOut)
	if code != 1 || !strings.Contains(errOut.String(), "do not describe 2 labels") {
		t.Fatalf("exit %d, stderr %q", code, errOut.String())
	}
}

func TestMainExitsWithRunStatus(t *testing.T) {
	savedExit, savedArgs := exit, os.Args
	defer func() { exit, os.Args = savedExit, savedArgs }()
	code := -1
	exit = func(c int) { code = c }
	os.Args = []string{"gliner"}
	main()
	if code != 2 {
		t.Fatalf("exit code %d", code)
	}
}
