package onnxnlp

import "testing"

func TestPlanBatchesBoundsMemoryByLength(t *testing.T) {
	lengths := []int{1500, 1500, 600, 512, 512, 100, 100, 100, 100, 100, 100, 100}
	order := make([]int, len(lengths))
	for i := range order {
		order[i] = i
	}
	batches := planBatches(order, func(i int) int { return lengths[i] }, 8)
	seen := 0
	for _, b := range batches {
		width := lengths[b[0]]
		if len(b) > 8 || (len(b) > 1 && len(b)*width*width > maxBatchCost) {
			t.Fatalf("batch %v of width %d exceeds the limits", b, width)
		}
		seen += len(b)
	}
	if seen != len(lengths) {
		t.Fatalf("%d of %d rows were batched", seen, len(lengths))
	}
	if len(batches) != 2 || len(batches[0]) != 7 || len(batches[1]) != 5 {
		t.Fatalf("the first batch is padded to 1500 tokens and so holds 7 rows; the other 5 rows of 100 tokens fit in one batch: %v", batches)
	}
	if got := planBatches([]int{0}, func(int) int { return 1 << 20 }, 8); len(got) != 1 || len(got[0]) != 1 {
		t.Fatalf("a single huge row still forms a batch: %v", got)
	}
	if got := planBatches(nil, func(int) int { return 1 }, 8); len(got) != 0 {
		t.Fatalf("no rows: %v", got)
	}
	if got := planBatches([]int{0, 1}, func(int) int { return 0 }, 8); len(got) != 1 || len(got[0]) != 2 {
		t.Fatalf("empty rows count as one token: %v", got)
	}
}
