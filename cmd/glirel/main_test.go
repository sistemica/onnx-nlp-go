package main

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"onnxgo/internal/fixture"
)

func fixtureDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	fixture.Must(fixture.WriteGLiREL(dir, [][]float32{{3, -3}, {-2, 2}}))
	return dir
}

func TestRun(t *testing.T) {
	var out, errOut bytes.Buffer
	args := []string{"-model", fixtureDir(t), "-labels", "knows,met", "-entities", "0:1, 2:3", "alice", "met", "bob"}
	if code := run(args, &out, &errOut); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	if lines := strings.Split(strings.TrimSpace(out.String()), "\n"); len(lines) != 2 ||
		!strings.Contains(lines[0], "alice | knows | bob") || !strings.Contains(lines[1], "bob | met | alice") {
		t.Fatalf("output %q", out.String())
	}
	out.Reset()
	if code := run(append([]string{"-topk", "1", "-threshold", "0.99"}, args...), &out, &errOut); code != 0 || out.Len() != 0 {
		t.Fatalf("threshold 0.99: exit %d, output %q", code, out.String())
	}
}

func TestRunUsageAndErrors(t *testing.T) {
	var out, errOut bytes.Buffer
	for name, args := range map[string][]string{
		"no arguments": nil,
		"no tokens":    {"-model", "x", "-labels", "a", "-entities", "0:1"},
		"bad flag":     {"-nope"},
		"bad entity":   {"-model", "x", "-labels", "a", "-entities", "zero:one", "tok"},
		"no colon":     {"-model", "x", "-labels", "a", "-entities", "3", "tok"},
	} {
		errOut.Reset()
		if code := run(args, &out, &errOut); code != 2 {
			t.Errorf("%s: exit %d", name, code)
		}
	}
	errOut.Reset()
	if code := run([]string{"-model", t.TempDir(), "-labels", "a", "-entities", "0:1,1:2", "x", "y"}, &out, &errOut); code != 1 || errOut.Len() == 0 {
		t.Fatalf("missing model: exit %d, stderr %q", code, errOut.String())
	}
	errOut.Reset()
	if code := run([]string{"-model", fixtureDir(t), "-labels", "knows,met", "-entities", "0:1,5:9", "alice", "met", "bob"}, &out, &errOut); code != 1 || !strings.Contains(errOut.String(), "outside") {
		t.Fatalf("bad entity range: exit %d, stderr %q", code, errOut.String())
	}
}

func TestMainExitsWithRunStatus(t *testing.T) {
	savedExit, savedArgs := exit, os.Args
	defer func() { exit, os.Args = savedExit, savedArgs }()
	code := -1
	exit = func(c int) { code = c }
	os.Args = []string{"glirel"}
	main()
	if code != 2 {
		t.Fatalf("exit code %d", code)
	}
}
