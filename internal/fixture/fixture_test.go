package fixture

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWritersProduceExpectedFiles(t *testing.T) {
	dir := t.TempDir()
	Must(WriteGLiNER(dir))
	for _, f := range []string{"tokenizer.json", "gliner2_config.json", "onnx/encoder.onnx", "onnx/boundary.onnx"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			t.Errorf("GLiNER fixture lacks %s", f)
		}
	}
	mini := t.TempDir()
	Must(WriteMiniLM(mini, 0, false))
	if _, err := os.Stat(filepath.Join(mini, "sentence_bert_config.json")); err == nil {
		t.Error("max_seq_length 0 must omit sentence_bert_config.json")
	}
	if len(EmbeddingTable(5)) != 5*Hidden {
		t.Error("embedding table size")
	}
}

func TestWritersReportFailures(t *testing.T) {
	gliner := []string{"tokenizer.json", "gliner2_config.json", "onnx/encoder.onnx", "onnx/boundary.onnx"}
	for _, blocked := range gliner {
		dir := t.TempDir()
		if err := os.MkdirAll(filepath.Join(dir, blocked), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := WriteGLiNER(dir); err == nil {
			t.Errorf("blocking %s did not fail WriteGLiNER", blocked)
		}
	}
	mini := []string{"tokenizer.json", "modules.json", "1_Pooling/config.json", "sentence_bert_config.json", "onnx/model.onnx"}
	for _, blocked := range mini {
		dir := t.TempDir()
		if err := os.MkdirAll(filepath.Join(dir, blocked), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := WriteMiniLM(dir, 8, true); err == nil {
			t.Errorf("blocking %s did not fail WriteMiniLM", blocked)
		}
	}
}

func TestGLiRELWriterReportsFailures(t *testing.T) {
	for _, blocked := range []string{"tokenizer.json", "glirel_export.json", "onnx/encoder.onnx", "onnx/head.onnx"} {
		dir := t.TempDir()
		if err := os.MkdirAll(filepath.Join(dir, blocked), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := WriteGLiREL(dir, [][]float32{{1}}); err == nil {
			t.Errorf("blocking %s did not fail WriteGLiREL", blocked)
		}
	}
}

func TestRerankerWriterReportsFailures(t *testing.T) {
	for _, blocked := range []string{"tokenizer.json", "onnx/model.onnx"} {
		dir := t.TempDir()
		if err := os.MkdirAll(filepath.Join(dir, blocked), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := WriteReranker(dir, 1, true); err == nil {
			t.Errorf("blocking %s did not fail WriteReranker", blocked)
		}
	}
	if err := WriteReranker(t.TempDir(), 2, false); err != nil {
		t.Fatal(err)
	}
}

func TestWritersReportUnusableDirectories(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "occupied")
	if err := os.WriteFile(file, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := WriteGLiNER(file); err == nil {
		t.Error("a file used as the target directory must fail")
	}
	nested := t.TempDir()
	if err := os.WriteFile(filepath.Join(nested, "onnx"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := WriteGLiNER(nested); err == nil {
		t.Error("a file where onnx/ belongs must fail")
	}
}

func TestMustPanicsOnError(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("Must did not panic")
		}
	}()
	Must(os.ErrNotExist)
}
