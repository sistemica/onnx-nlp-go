package onnxgo

import (
	"encoding/json"
	"math"
	"os"
	"testing"
	"unicode/utf8"
)

type refEntity struct {
	Text       string  `json:"text"`
	Confidence float64 `json:"confidence"`
	Start      int     `json:"start"`
	End        int     `json:"end"`
}

func TestGLiNEREndToEnd(t *testing.T) {
	models, ref := os.Getenv("ONNXGO_MODELS"), os.Getenv("ONNXGO_E2E")
	if models == "" || ref == "" {
		t.Skip("ONNXGO_MODELS and ONNXGO_E2E not set")
	}
	raw, err := os.ReadFile(ref)
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Text   string   `json:"text"`
		Labels []string `json:"labels"`
		Result struct {
			Entities map[string][]refEntity `json:"entities"`
		} `json:"result"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	g, err := LoadGLiNER(models)
	if err != nil {
		t.Fatal(err)
	}
	total, matched := 0, 0
	for _, c := range cases {
		got, err := g.Extract(c.Text, c.Labels, 0.5)
		if err != nil {
			t.Fatal(err)
		}
		byLabel := map[string][]Entity{}
		for _, e := range got {
			byLabel[e.Label] = append(byLabel[e.Label], e)
		}
		for _, label := range c.Labels {
			want, have := c.Result.Entities[label], byLabel[label]
			total += len(want)
			if len(want) != len(have) {
				t.Errorf("%q label %q: got %d entities %v, want %d %v", c.Text, label, len(have), have, len(want), want)
				continue
			}
			for i, w := range want {
				h := have[i]
				start := utf8.RuneCountInString(c.Text[:h.Start])
				end := utf8.RuneCountInString(c.Text[:h.End])
				if h.Text != w.Text || start != w.Start || end != w.End || math.Abs(float64(h.Score)-w.Confidence) > 2e-3 {
					t.Errorf("%q label %q #%d: got %q [%d,%d) %.4f, want %q [%d,%d) %.4f",
						c.Text, label, i, h.Text, start, end, h.Score, w.Text, w.Start, w.End, w.Confidence)
					continue
				}
				matched++
			}
		}
	}
	t.Logf("%d of %d reference entities matched (text, offsets, score within 2e-3)", matched, total)
}
