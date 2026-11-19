package onnxgo

import (
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
)

func benchEnv(b *testing.B, name string) string {
	b.Helper()
	dir := os.Getenv(name)
	if dir == "" {
		b.Skipf("%s not set", name)
	}
	return dir
}

func randomIDs(rng *rand.Rand, rows, cols, vocab int) *Tensor {
	t := zeros(Int64, []int{rows, cols})
	for i := range t.I {
		t.I[i] = int64(5 + rng.IntN(vocab-5))
	}
	return t
}

func randomFloats(rng *rand.Rand, shape ...int) *Tensor {
	t := zeros(Float32, shape)
	for i := range t.F {
		t.F[i] = float32(rng.NormFloat64())
	}
	return t
}

func benchRun(b *testing.B, s *Session, in map[string]*Tensor) {
	b.Helper()
	if _, err := s.Run(in); err != nil {
		b.Fatal(err)
	}
	if os.Getenv("ONNXGO_OPSTATS") != "" {
		s.EnableProfiling()
		defer func() {
			for i, st := range s.OpStats() {
				if i < 8 {
					b.Logf("%-16s %5d calls %9.2f ms/run", st.Op, st.Calls/b.N, float64(st.Total.Microseconds())/1000/float64(b.N))
				}
			}
		}()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.Run(in); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMul(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, sz := range [][3]int{{128, 768, 768}, {128, 768, 3072}, {512, 768, 768}, {1024, 1024, 1024}} {
		m, k, n := sz[0], sz[1], sz[2]
		x, w := randomFloats(rng, m, k), randomFloats(rng, k, n)
		b.Run(fmt.Sprintf("%dx%dx%d", m, k, n), func(b *testing.B) {
			ctx := &opCtx{in: []*Tensor{x, w}}
			for i := 0; i < b.N; i++ {
				if _, err := opMatMul(ctx); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

func BenchmarkGLiNEREncoder(b *testing.B) {
	s, err := Load(filepath.Join(benchEnv(b, "ONNXGO_MODELS"), "onnx", "encoder.onnx"))
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(3, 4))
	for _, seq := range []int{37, 120, 256} {
		in := map[string]*Tensor{
			"input_ids":      randomIDs(rng, 1, seq, 200000),
			"attention_mask": NewInt([]int{1, seq}, onesInt(seq)),
		}
		b.Run(fmt.Sprintf("seq%d", seq), func(b *testing.B) { benchRun(b, s, in) })
	}
}

func onesInt(n int) []int64 {
	out := make([]int64, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

func BenchmarkGLiNERBoundary(b *testing.B) {
	s, err := Load(filepath.Join(benchEnv(b, "ONNXGO_MODELS"), "onnx", "boundary.onnx"))
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(5, 6))
	for _, sz := range [][2]int{{9, 3}, {40, 8}} {
		words, queries := sz[0], sz[1]
		in := map[string]*Tensor{
			"token_states": randomFloats(rng, 1, words, 768),
			"text_mask":    onesBool([]int{1, words}),
			"query_states": randomFloats(rng, 1, queries, 768),
			"query_mask":   onesBool([]int{1, queries}),
		}
		b.Run(fmt.Sprintf("words%dq%d", words, queries), func(b *testing.B) { benchRun(b, s, in) })
	}
}

func BenchmarkMiniLM(b *testing.B) {
	s, err := Load(filepath.Join(benchEnv(b, "ONNXGO_MINILM"), "onnx", "model.onnx"))
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(7, 8))
	for _, sz := range [][2]int{{1, 16}, {32, 16}, {1, 128}, {8, 256}} {
		batch, seq := sz[0], sz[1]
		in := map[string]*Tensor{
			"input_ids":      randomIDs(rng, batch, seq, 30000),
			"attention_mask": NewInt([]int{batch, seq}, onesInt(batch*seq)),
			"token_type_ids": zeros(Int64, []int{batch, seq}),
		}
		b.Run(fmt.Sprintf("batch%dseq%d", batch, seq), func(b *testing.B) { benchRun(b, s, in) })
	}
}

func BenchmarkExtractEntities(b *testing.B) {
	g, err := LoadGLiNER(benchEnv(b, "ONNXGO_MODELS"))
	if err != nil {
		b.Fatal(err)
	}
	text := "Apple CEO Tim Cook met Satya Nadella of Microsoft in Seattle on Monday."
	labels := []string{"person", "company", "city"}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := g.Extract(text, labels, 0.5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkKeyBERTExtract(b *testing.B) {
	e, err := LoadEmbedder(benchEnv(b, "ONNXGO_MINILM"))
	if err != nil {
		b.Fatal(err)
	}
	doc := "Supervised learning is the machine learning task of learning a function that maps an input to an output " +
		"based on example input-output pairs. It infers a function from labeled training data consisting of a set of " +
		"training examples. A supervised learning algorithm analyzes the training data and produces an inferred function."
	kb := NewKeyBERT(e)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := kb.Extract([]string{doc}, KeyBERTOptions{NgramMax: 2}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTokenizers(b *testing.B) {
	b.Run("unigram", func(b *testing.B) {
		tok, err := LoadTokenizer(filepath.Join(benchEnv(b, "ONNXGO_MODELS"), "tokenizer.json"), nil)
		if err != nil {
			b.Fatal(err)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tok.Tokenize("internationalization")
		}
	})
	b.Run("wordpiece", func(b *testing.B) {
		wp, err := LoadWordPiece(filepath.Join(benchEnv(b, "ONNXGO_MINILM"), "tokenizer.json"))
		if err != nil {
			b.Fatal(err)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			wp.Encode("Supervised learning is the machine learning task of learning a function.", 256)
		}
	})
}

func BenchmarkLoad(b *testing.B) {
	dir := benchEnv(b, "ONNXGO_MODELS")
	for i := 0; i < b.N; i++ {
		if _, err := Load(filepath.Join(dir, "onnx", "encoder.onnx")); err != nil {
			b.Fatal(err)
		}
	}
}
